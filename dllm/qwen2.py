"""Independent Qwen2 executor using PyTorch CUDA kernels, not HF generation.

Initial scope: dense Qwen2, FP16, batch size one, greedy decoding. CUDA graphs
remove Python dispatch from each decode step. Prefill remains eager.
"""

import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from safetensors import safe_open
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment


def weight_files(path):
    path = Path(path)
    index = path / "model.safetensors.index.json"
    names = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("weight shard escapes model directory")
    return [path / name for name in names]


def weight_bytes(path):
    """Conservative FP16 weight estimate from headers, before loading tensors."""
    total = 0
    for file in weight_files(path):
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                total += 2 * math.prod(reader.get_slice(key).get_shape())
    return total


class ModelWeights:
    def __init__(self, path, max_context):
        self.path = Path(path).resolve()
        self.config = json.loads((self.path / "config.json").read_text())
        c = self.config
        if (c.get("model_type") != "qwen2" or c.get("use_sliding_window", False)
                or c.get("rope_scaling") or c.get("quantization_config")
                or c.get("hidden_act") != "silu"):
            raise ValueError("only dense, unquantized Qwen2 with standard RoPE is supported")
        if not 1 <= max_context <= c["max_position_embeddings"]:
            raise ValueError("context exceeds model position limit")
        self.max_context = max_context
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        tokenizer_config = json.loads((self.path / "tokenizer_config.json").read_text())
        template = tokenizer_config.get("chat_template")
        if not isinstance(template, str):
            raise ValueError("model needs one text chat template")
        self.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template)
        self.special_tokens = {key: tokenizer_config.get(key) for key in ("bos_token", "eos_token")}
        self.eos = {c["eos_token_id"]}
        generation = self.path / "generation_config.json"
        if generation.exists():
            eos = json.loads(generation.read_text()).get("eos_token_id", [])
            self.eos.update(eos if isinstance(eos, list) else [eos])
        state = {}
        for candidate in weight_files(self.path):
            state.update(load_file(str(candidate), device="cpu"))
        weights = {}
        def take(key):
            return state.pop(key).to(torch.float16)
        weights["embed"] = take("model.embed_tokens.weight")
        weights["norm"] = take("model.norm.weight")
        weights["head"] = weights["embed"] if c.get("tie_word_embeddings") else take("lm_head.weight")
        for i in range(c["num_hidden_layers"]):
            prefix = f"model.layers.{i}."
            for kind in ("weight", "bias"):
                weights[f"{i}.qkv.{kind}"] = torch.cat([
                    take(prefix + f"self_attn.{p}_proj.{kind}") for p in ("q", "k", "v")
                ])
            weights[f"{i}.out"] = take(prefix + "self_attn.o_proj.weight")
            weights[f"{i}.gate_up"] = torch.cat([
                take(prefix + f"mlp.{p}_proj.weight") for p in ("gate", "up")
            ])
            weights[f"{i}.down"] = take(prefix + "mlp.down_proj.weight")
            weights[f"{i}.norm1"] = take(prefix + "input_layernorm.weight")
            weights[f"{i}.norm2"] = take(prefix + "post_attention_layernorm.weight")
        # Keep host backing pageable: pinning whole multi-GB models can harm the
        # machine. Transfers are synchronous in this first executor.
        self.weights = weights
        self.bytes = sum(t.numel() * t.element_size() for t in {id(t): t for t in weights.values()}.values())

    def chat_tokens(self, messages):
        if not messages:
            raise ValueError("at least one message is required")
        for message in messages:
            if message["role"] not in ("system", "user", "assistant") or not isinstance(message["content"], str):
                raise ValueError("only text system/user/assistant messages are supported")
        rendered = self.template.render(messages=messages, add_generation_prompt=True,
                                        tools=None, **self.special_tokens)
        return self.tokenizer.encode(rendered, add_special_tokens=False).ids


class Executor:
    def __init__(self, model, use_graph=True, fuse_projections=True):
        self.model = model
        self.c = model.config
        self.context = model.max_context
        self.fuse_projections = fuse_projections
        c = self.c
        self.heads = c["num_attention_heads"]
        self.kv_heads = c["num_key_value_heads"]
        self.dim = c["hidden_size"] // self.heads
        self.eps = c["rms_norm_eps"]
        kv_bytes = 2 * c["num_hidden_layers"] * self.kv_heads * self.context * self.dim * 2
        # A prior failed construction can leave unused allocator blocks cached
        # after its traceback is released. Return those before querying free VRAM.
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        if model.bytes + kv_bytes + 512 * 1024**2 > free:
            raise MemoryError("insufficient GPU memory for weights, KV cache, and workspace reserve")
        self.w = {}
        copies = {}
        for name, tensor in model.weights.items():
            if id(tensor) not in copies:
                copies[id(tensor)] = tensor.to("cuda")
            self.w[name] = copies[id(tensor)]
        self.kv = [torch.zeros((2, 1, self.kv_heads, self.context, self.dim),
                               dtype=torch.float16, device="cuda")
                   for _ in range(c["num_hidden_layers"])]
        inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, self.dim, 2, device="cuda", dtype=torch.float32) / self.dim))
        angles = torch.outer(torch.arange(self.context, device="cuda", dtype=torch.float32), inv)
        angles = torch.cat((angles, angles), dim=-1)
        self.cos, self.sin = angles.cos().half(), angles.sin().half()
        self.arange = torch.arange(self.context, device="cuda")
        self.token = torch.zeros(1, dtype=torch.long, device="cuda")
        self.position = torch.zeros(1, dtype=torch.long, device="cuda")
        self.graph = None
        if use_graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    self.forward(self.token, self.position, prefill=False)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.inference_mode():
                self.graph_logits = self.forward(self.token, self.position, prefill=False)
                self.graph_next = self.graph_logits.argmax(dim=-1)
        self.length = 0

    def norm(self, x, weight):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * weight

    def rotate(self, x, cos, sin):
        left, right = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-right, left), dim=-1) * sin

    @torch.inference_mode()
    def forward(self, ids, positions, prefill):
        w = self.w
        x = F.embedding(ids, w["embed"])
        count = ids.shape[0]
        cos = self.cos[positions].view(1, 1, count, self.dim)
        sin = self.sin[positions].view(1, 1, count, self.dim)
        mask = None if prefill else (self.arange <= positions[0]).view(1, 1, 1, -1)
        for i in range(self.c["num_hidden_layers"]):
            hidden = self.norm(x, w[f"{i}.norm1"])
            sizes = [self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim]
            if self.fuse_projections:
                q, k, v = F.linear(hidden, w[f"{i}.qkv.weight"], w[f"{i}.qkv.bias"]).split(sizes, dim=-1)
            else:
                q, k, v = [F.linear(hidden, weight, bias) for weight, bias in zip(
                    w[f"{i}.qkv.weight"].split(sizes), w[f"{i}.qkv.bias"].split(sizes))]
            q = self.rotate(q.view(1, count, self.heads, self.dim).transpose(1, 2), cos, sin)
            k = self.rotate(k.view(1, count, self.kv_heads, self.dim).transpose(1, 2), cos, sin)
            v = v.view(1, count, self.kv_heads, self.dim).transpose(1, 2)
            cache_k, cache_v = self.kv[i].unbind(0)
            cache_k.index_copy_(2, positions, k)
            cache_v.index_copy_(2, positions, v)
            if not prefill:
                k, v = cache_k, cache_v
            attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                       is_causal=prefill, enable_gqa=True)
            x = x + F.linear(attention.transpose(1, 2).reshape(count, -1), w[f"{i}.out"])
            hidden = self.norm(x, w[f"{i}.norm2"])
            if self.fuse_projections:
                gate, up = F.linear(hidden, w[f"{i}.gate_up"]).chunk(2, -1)
            else:
                gate, up = [F.linear(hidden, weight) for weight in w[f"{i}.gate_up"].chunk(2)]
            x = x + F.linear(F.silu(gate) * up, w[f"{i}.down"])
        return F.linear(self.norm(x[-1:], w["norm"]), w["head"])

    @torch.inference_mode()
    def prefill(self, tokens):
        if not tokens or len(tokens) > self.context:
            raise ValueError("prompt is empty or exceeds context")
        ids = torch.tensor(tokens, device="cuda", dtype=torch.long)
        self.length = len(tokens)
        return self.forward(ids, torch.arange(len(tokens), device="cuda"), prefill=True)

    @torch.inference_mode()
    def decode(self, token):
        if self.length >= self.context:
            raise ValueError("context exhausted")
        self.token.fill_(token)
        self.position.fill_(self.length)
        self.length += 1
        if self.graph is not None:
            self.graph.replay()
            return self.graph_logits, self.graph_next
        logits = self.forward(self.token, self.position, prefill=False)
        return logits, logits.argmax(dim=-1)
