"""Isolated Q4 OLMoE construction; normal model registration remains unchanged."""
import hashlib
import json
from pathlib import Path
import torch
from scripts.experiments.moe_q4 import routed_linear


def install_validation_dispatch():
    import dllm.qwen2 as qwen2
    from dllm.moe import OlmoeExecutor
    from dllm import moe_kernels
    from dllm.config import decoder_config
    from dllm.moe_prefill import qualified_scope
    from dllm.quantization import mode, quantize_q4
    import reference_model
    if torch.cuda.get_device_name() != 'NVIDIA A2':
        raise ValueError('Q4 MoE model experiment is limited to A2')
    report = dict(kind='experimental Q4 OLMoE; FP16 routers and embeddings; FP32 arithmetic',
                  model_loads=0, reference_loads=0, routed_calls=0, host_weight_bytes=[],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  kernel_sha256=hashlib.sha256(Path('scripts/experiments/moe_q4.py').read_bytes()).hexdigest())
    original_model = qwen2.ModelWeights
    original_linear = OlmoeExecutor._linear
    original_scales = OlmoeExecutor._expert_scales
    original_expert = OlmoeExecutor._expert_linear
    original_routed = moe_kernels.routed_linear
    original_reference = reference_model.load_reference

    def check_scope(path, context=2048):
        config = decoder_config(json.loads((Path(path)/'config.json').read_text()))
        # Reuse the measured geometry/context gate; this does not register Q4.
        if not qualified_scope(config,'NVIDIA A2','q8_0',context,4):
            raise ValueError('unsupported experimental Q4 OLMoE geometry')

    def model_weights(path, max_context, quantization=None):
        if mode(quantization) != 'q4_0':
            return original_model(path,max_context,quantization)
        check_scope(path,max_context)
        model = original_model(path,max_context,quantization='none')
        scales = {}
        for name in model.weights:
            weight = model.weights[name]
            # Routing matrices are tiny and remain FP16 rather than losing
            # routing precision to save a negligible amount of storage.
            if weight.ndim == 2 and name != 'embed' and not name.endswith('.router'):
                model.weights[name],scales[name] = quantize_q4(weight)
        model.quantization = model.weight_format = 'q4_0'
        model.finalize_weights(model.weights,prepared_scales=scales)
        model.weight_cache_status = 'experimental conversion from native FP16 host weights'
        report['model_loads'] += 1
        report['host_weight_bytes'].append(model.bytes)
        return model

    def linear(self, x, weight, bias=None):
        if weight.dtype == torch.uint8:
            return self.quant_linear(x,weight,bias)
        return original_linear(self,x,weight,bias)

    def expert_scales(self, weight):
        if weight.dtype == torch.uint8:
            return self.quant_linear.scales[id(weight)]
        return original_scales(self,weight)

    def expert_linear(self, x, weight, expert, outputs):
        if weight.dtype != torch.uint8:
            return original_expert(self,x,weight,expert,outputs)
        from dllm.quantized_kernels_q4 import linear
        matrix = weight.view(self.c['num_experts'],outputs,-1)[expert]
        scales = self._expert_scales(weight).view(self.c['num_experts'],outputs,-1)[expert]
        return linear(x,matrix,scales)

    def routed(x, weight, scales, selected, outputs, shared_input):
        if weight.dtype == torch.uint8:
            report['routed_calls'] += 1
            return routed_linear(x,weight,scales,selected,outputs,shared_input)
        return original_routed(x,weight,scales,selected,outputs,shared_input)

    def load_reference(path, reference_quantization='none', **kwargs):
        if reference_quantization != 'q4_0':
            return original_reference(path,reference_quantization=reference_quantization,**kwargs)
        check_scope(path)
        # Independently construct the documented mixed storage policy using
        # upstream GGUF conversion, never dllm's quantizer/dequantizer.
        from gguf.quants import quantize, dequantize
        from gguf.constants import GGMLQuantizationType
        model = original_reference(path,reference_quantization='none',**kwargs)
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Embedding):
                module.weight = torch.nn.Parameter(module.weight.detach().half().to(module.weight.dtype),
                                                   requires_grad=False)
                continue
            if not isinstance(module,torch.nn.Linear):
                continue
            dtype = module.weight.dtype
            if name.endswith('.mlp.gate'):
                restored = module.weight.detach().half().to(dtype)
            else:
                # Native checkpoint loading stores matrices in FP16 before
                # Q4 conversion. Preserve that policy even for subnormals;
                # GGUF remains the independent quantizer/dequantizer.
                weight = module.weight.detach().half().float().cpu().numpy()
                restored = torch.from_numpy(dequantize(quantize(weight,GGMLQuantizationType.Q4_0),
                                                       GGMLQuantizationType.Q4_0)).to(dtype)
            module.weight = torch.nn.Parameter(restored,requires_grad=False)
        report['reference_loads'] += 1
        return model

    qwen2.ModelWeights = model_weights
    OlmoeExecutor._linear = linear
    OlmoeExecutor._expert_scales = expert_scales
    OlmoeExecutor._expert_linear = expert_linear
    moe_kernels.routed_linear = routed
    reference_model.load_reference = load_reference
    return report
