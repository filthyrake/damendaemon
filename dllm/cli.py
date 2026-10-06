import argparse
import json
import os
from pathlib import Path
import re
import sys


def download(repository, revision="main", filename=None):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from .download import INDEX, snapshot_files, gguf_file
    info = HfApi().model_info(repository, revision=revision)
    revision = info.sha
    if not revision:raise ValueError("repository revision could not be resolved")
    filenames = [entry.rfilename for entry in info.siblings or []]
    if filename is not None:
        filename = gguf_file(filenames, filename)
        path = hf_hub_download(repository, filename, revision=revision)
        return {"repository": repository, "revision": revision, "filename": filename, "path": path}
    index = None
    if INDEX in filenames:
        index = json.loads(Path(hf_hub_download(repository, INDEX, revision=revision)).read_text())
    path = snapshot_download(repository, revision=revision,
        allow_patterns=snapshot_files(filenames, index))
    return {"repository": repository, "revision": revision, "path": path}


def resolve_model(value, revision="main", filename=None):
    from .models import is_gguf
    path = Path(value).expanduser()
    if path.is_dir() or is_gguf(path):
        if filename is not None:
            raise ValueError('--filename applies to a Hugging Face repository, not a local path')
        return path.resolve()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", str(value)):
        result = download(str(value), revision) if filename is None else download(str(value), revision, filename=filename)
        return Path(result["path"])
    raise ValueError("model must be a local directory, GGUF file or a Hugging Face owner/repository")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "distributed-serve":
        from .distributed_server_cli import main as distributed_server_main
        return distributed_server_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "distributed-run":
        from .distributed_cli import main as distributed_main
        return distributed_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        from .bench import main as bench_main
        return bench_main()
    parser = argparse.ArgumentParser(description="dllm: independent GPU inference runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bench", help="benchmark a streaming HTTP endpoint")
    commands.add_parser("distributed-run", help="experimental two-rank greedy generation under torchrun")
    commands.add_parser("distributed-serve", help="experimental resident two-rank greedy HTTP service")
    pull = commands.add_parser("pull", help="download a safetensors snapshot or one GGUF file from Hugging Face")
    pull.add_argument("repository")
    pull.add_argument("--revision", default="main")
    pull.add_argument("--filename", help="exact GGUF filename within the repository")
    run = commands.add_parser("run", help="generate using a supported dense model name or local path")
    run.add_argument("path", help="local model directory, GGUF file or Hugging Face owner/repository")
    run.add_argument("--revision", default="main")
    run.add_argument("--filename", help="exact GGUF filename within the repository")
    run.add_argument("--chat-template", metavar="FILE", help="override the model chat template with local UTF-8 text")
    run.add_argument("--prompt", required=True)
    run.add_argument('--raw', action='store_true', help='use the prompt directly without a chat template')
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument("--reasoning-effort",choices=("low","medium","high"),help="GPT-OSS Harmony reasoning effort")
    run.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                     help="Qwen3 thinking mode; auto uses the model template default")
    server = commands.add_parser("serve", help="serve named local models on loopback")
    server.add_argument("--model", action="append", metavar="NAME=PATH")
    server.add_argument('--model-config', metavar='FILE', help='versioned JSON model manifest for repeatable startup')
    server.add_argument("--chat-template", action="append", metavar="NAME=FILE", help="override a named model chat template")
    server.add_argument("--port", type=int, default=8088)
    server.add_argument("--batch-size", type=int, choices=range(1, 9), default=4)
    server.add_argument("--batch-wait-ms",type=float,default=0,help="idle prompt coalescing delay, 0..10 ms; default 0 disables")
    server.add_argument("--prefill-chunk-size",type=int,default=0,help="opt-in prompt token budget per decode step; 0 disables")
    server.add_argument("--prefix-cache-tokens",type=int,default=0,help="opt-in GPU prefix snapshot token budget per model")
    for command in (run, server):
        command.add_argument("--device", choices=("cuda", "rocm", "mps"), default=os.environ.get("DLLM_DEVICE", "cuda"))
        command.add_argument("--context", type=int, default=1024)
        command.add_argument("--kv-cache-tokens",type=int,help="shared CUDA KV token budget, multiple of 16; 0 selects fixed slots")
        command.add_argument("--gpu-models", type=int, default=1)
        command.add_argument("--host-models", type=int, default=2)
        command.add_argument("--host-gib", type=float, default=8)
        command.add_argument("--no-graph", action="store_true")
        command.add_argument('--layer-split',type=int,help='experimental two-GPU layer split; qualified value: 12 (context <=512)')
    for name in ('status', 'drain', 'models', 'add', 'load', 'unload'):
        management = commands.add_parser(name, help={'status': 'show scheduler queue status', 'drain': 'stop admission and finish queued work'}.get(name, 'manage models on a running dllm server'))
        management.add_argument('--server', default='http://127.0.0.1:8088')
        if name == 'models':
            management.add_argument('--save-config', metavar='FILE', help='save registered models as a restart manifest')
        if name not in ('status', 'drain', 'models'):
            management.add_argument('model')
        if name == 'add':
            management.add_argument('path', help='local snapshot or Hugging Face repository on this host')
            management.add_argument('--revision', default='main')
            management.add_argument('--filename', help='exact GGUF filename within the repository')
            management.add_argument('--chat-template', metavar='FILE')
            management.add_argument('--quantization', choices=('none','q8_0','q4_0'))
        if name == 'unload':
            management.add_argument('--release-host', action='store_true')
    args = parser.parse_args()
    manifest = None
    if args.command == 'serve' and args.model_config:
        if args.model or args.chat_template:
            parser.error('--model-config cannot combine with --model or --chat-template')
        from .model_manifest import load
        try:
            manifest = load(args.model_config, resolve_model)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if args.command in ('status', 'drain', 'models', 'add', 'load', 'unload'):
        from urllib.request import Request, urlopen
        action = {'status':'scheduler', 'models':'status', 'add':'register'}.get(args.command, args.command)
        if args.command == 'models' and args.save_config:
            action = 'config'
        body = ({} if action == 'drain' else None if action in ('scheduler', 'status', 'config') else {'model': args.model})
        if action == 'register':
            body['path'] = str(resolve_model(args.path, args.revision, args.filename))
            if args.chat_template is not None:
                from .chat_template import read_override
                body['chat_template'] = read_override(args.chat_template)
            if args.quantization is not None:
                body['quantization'] = args.quantization
        if action == 'unload':
            body['release_host'] = args.release_host
        endpoint = ('/v1/scheduler/status' if action == 'scheduler' else
                    '/v1/scheduler/drain' if action == 'drain' else '/v1/models/'+action)
        request = Request(args.server.rstrip('/')+endpoint,
                          data=None if body is None else json.dumps(body).encode(),
                          headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=10 if action in ('scheduler', 'drain') else 300) as response:
            result = json.load(response)
        if action == 'config':
            from .model_manifest import save
            save(args.save_config, result)
            print(json.dumps({'saved': str(Path(args.save_config).expanduser()), 'models': list(result['models'])}))
        else:
            print(json.dumps(result, indent=2))
        return
    if args.command == 'run' and args.raw and args.thinking != 'auto':
        parser.error('--raw cannot use chat thinking controls')
    if args.command == "pull":
        # Resolve the revision before downloading; report its immutable identity.
        print(json.dumps(download(args.repository, args.revision, args.filename)))
        return
    os.environ["DLLM_DEVICE"] = args.device
    if args.layer_split is not None:
        if args.layer_split<=0:parser.error('layer split must be positive')
        os.environ['DLLM_EXPERIMENTAL_LAYER_SPLIT']=str(args.layer_split)
    if args.kv_cache_tokens is not None:
        if args.kv_cache_tokens < 0 or args.kv_cache_tokens % 16:
            parser.error('KV token budget must be a nonnegative multiple of 16')
        if args.kv_cache_tokens:
            os.environ['DLLM_KV_CACHE_TOKENS'] = str(args.kv_cache_tokens)
        else:
            os.environ.pop('DLLM_KV_CACHE_TOKENS',None)
    from .runtime import Runtime
    if args.command == "run":
        models = {"model": resolve_model(args.path, args.revision, args.filename)}
    else:
        models = {}
        for entry in args.model or []:
            alias, separator, path = entry.partition("=")
            if not separator or not alias or not path or alias in models:
                parser.error("models need unique NAME=PATH entries")
            # Resolve repository names up front; explicit local paths remain
            # lazy so failed loads can be reported without taking down the API.
            models[alias] = resolve_model(path) if not path.startswith(("/", ".", "~")) else Path(path).expanduser()
    from .chat_template import read_override
    templates = {}
    if args.command == "run" and args.chat_template:
        templates["model"] = read_override(args.chat_template)
    elif args.command == "serve":
        for entry in args.chat_template or []:
            alias, separator, path = entry.partition("=")
            if not separator or alias not in models or not path or alias in templates:
                parser.error("chat templates need unique NAME=FILE entries for configured models")
            templates[alias] = read_override(path)
    runtime_type = Runtime
    options = {}
    if args.command == "serve":
        from .scheduler import BatchedRuntime
        runtime_type = BatchedRuntime
        options["batch_size"] = args.batch_size
        options["batch_wait_ms"] = args.batch_wait_ms
        options["prefill_chunk_size"] = args.prefill_chunk_size
        options["prefix_cache_tokens"] = args.prefix_cache_tokens
        if manifest is not None:
            models, options['quantizations'], templates = manifest
    runtime = runtime_type(models, chat_templates=templates, **options, context=args.context, gpu_models=args.gpu_models,
                      host_models=args.host_models, host_gib=args.host_gib, use_graph=not args.no_graph)
    if args.command == "serve":
        from .server import serve
        serve(runtime, args.port)
    else:
        for event in runtime.generate("model", None if args.raw else [{"role": "user", "content": args.prompt}], args.max_tokens,
                                      enable_thinking=None if args.thinking == "auto" else args.thinking == "on",
                                      raw_prompt=args.prompt if args.raw else None,reasoning_effort=args.reasoning_effort):
            if "text" in event:
                print(event["text"], end="", flush=True)
            else:
                print()
                print(json.dumps(event), file=sys.stderr)
