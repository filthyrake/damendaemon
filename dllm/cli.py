import argparse
import json
import os
from pathlib import Path
import re
import sys


def download(repository, revision="main"):
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info(repository, revision=revision).sha
    path = snapshot_download(repository, revision=revision,
        allow_patterns=["*.json", "*.safetensors", "LICENSE*", "README.md", "*.jinja"])
    return {"repository": repository, "revision": revision, "path": path}


def resolve_model(value, revision="main"):
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", str(value)):
        return Path(download(str(value), revision)["path"])
    raise ValueError("model must be a local directory or a Hugging Face owner/repository")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        from .bench import main as bench_main
        return bench_main()
    parser = argparse.ArgumentParser(description="dllm: independent GPU inference runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bench", help="benchmark a streaming HTTP endpoint")
    pull = commands.add_parser("pull", help="download a safetensors model from Hugging Face")
    pull.add_argument("repository")
    pull.add_argument("--revision", default="main")
    run = commands.add_parser("run", help="generate using a supported dense model name or local path")
    run.add_argument("path", help="local model directory or Hugging Face owner/repository")
    run.add_argument("--revision", default="main")
    run.add_argument("--prompt", required=True)
    run.add_argument('--raw', action='store_true', help='use the prompt directly without a chat template')
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                     help="Qwen3 thinking mode; auto uses the model template default")
    server = commands.add_parser("serve", help="serve named local models on loopback")
    server.add_argument("--model", action="append", metavar="NAME=PATH")
    server.add_argument("--port", type=int, default=8088)
    server.add_argument("--batch-size", type=int, choices=range(1, 9), default=4)
    server.add_argument("--prefill-chunk-size",type=int,default=0,help="opt-in prompt token budget per decode step; 0 disables")
    server.add_argument("--prefix-cache-tokens",type=int,default=0,help="opt-in GPU prefix snapshot token budget per model")
    for command in (run, server):
        command.add_argument("--device", choices=("cuda", "mps"), default=os.environ.get("DLLM_DEVICE", "cuda"))
        command.add_argument("--context", type=int, default=1024)
        command.add_argument("--gpu-models", type=int, default=1)
        command.add_argument("--host-models", type=int, default=2)
        command.add_argument("--host-gib", type=float, default=8)
        command.add_argument("--no-graph", action="store_true")
    for name in ('models', 'add', 'load', 'unload'):
        management = commands.add_parser(name, help='manage models on a running dllm server')
        management.add_argument('--server', default='http://127.0.0.1:8088')
        if name != 'models':
            management.add_argument('model')
        if name == 'add':
            management.add_argument('path', help='local snapshot or Hugging Face repository on this host')
            management.add_argument('--revision', default='main')
            management.add_argument('--quantization', choices=('none','q8_0','q4_0'))
        if name == 'unload':
            management.add_argument('--release-host', action='store_true')
    args = parser.parse_args()
    if args.command in ('models', 'add', 'load', 'unload'):
        from urllib.request import Request, urlopen
        action = {'models':'status', 'add':'register'}.get(args.command, args.command)
        body = None if action == 'status' else {'model': args.model}
        if action == 'register':
            body['path'] = str(resolve_model(args.path, args.revision))
            if args.quantization is not None:
                body['quantization'] = args.quantization
        if action == 'unload':
            body['release_host'] = args.release_host
        request = Request(args.server.rstrip('/')+'/v1/models/'+action,
                          data=None if body is None else json.dumps(body).encode(),
                          headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=300) as response:
            print(json.dumps(json.load(response), indent=2))
        return
    if args.command == 'run' and args.raw and args.thinking != 'auto':
        parser.error('--raw cannot use chat thinking controls')
    if args.command == "pull":
        # Resolve the revision before downloading; report its immutable identity.
        print(json.dumps(download(args.repository, args.revision)))
        return
    os.environ["DLLM_DEVICE"] = args.device
    from .runtime import Runtime
    if args.command == "run":
        models = {"model": resolve_model(args.path, args.revision)}
    else:
        models = {}
        for entry in args.model or []:
            alias, separator, path = entry.partition("=")
            if not separator or not alias or not path or alias in models:
                parser.error("models need unique NAME=PATH entries")
            # Resolve repository names up front; explicit local paths remain
            # lazy so failed loads can be reported without taking down the API.
            models[alias] = resolve_model(path) if not path.startswith(("/", ".", "~")) else Path(path).expanduser()
    runtime_type = Runtime
    options = {}
    if args.command == "serve":
        from .scheduler import BatchedRuntime
        runtime_type = BatchedRuntime
        options["batch_size"] = args.batch_size
        options["prefill_chunk_size"] = args.prefill_chunk_size
        options["prefix_cache_tokens"] = args.prefix_cache_tokens
    runtime = runtime_type(models, **options, context=args.context, gpu_models=args.gpu_models,
                      host_models=args.host_models, host_gib=args.host_gib, use_graph=not args.no_graph)
    if args.command == "serve":
        from .server import serve
        serve(runtime, args.port)
    else:
        for event in runtime.generate("model", None if args.raw else [{"role": "user", "content": args.prompt}], args.max_tokens,
                                      enable_thinking=None if args.thinking == "auto" else args.thinking == "on",
                                      raw_prompt=args.prompt if args.raw else None):
            if "text" in event:
                print(event["text"], end="", flush=True)
            else:
                print()
                print(json.dumps(event), file=sys.stderr)
