"""Compare standalone GGUF tokenization with llama.cpp's independent tokenizer."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen

from tokenizers import Tokenizer
from dllm.gguf_model import plan


CASES = [
    '', 'Hello, world!', "We're here; I'M READY. isn't it?", '1234567890 1.25 -500',
    'a  b\t\n\r\n  c   ', 'café e\u0301 é Å A\u030a', '你好，世界！こんにちは 한국어',
    '🦊 👨\u200d👩\u200d👧\u200d👦 👍🏽', 'عَرَبِيّ हिन्दी',
    '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n',
    '<tool_call>{"name":"weather","arguments":{"city":"東京"}}</tool_call>',
    'def hello(x):\n    return f"value={x}"\n', '\x00x\x01\x1f\x7f',
    'Z͑͗a̐͊l̚g̕ō end', '\u00a0\u2009\u2028\u2029 \n',
]


def main():
    p = argparse.ArgumentParser(); p.add_argument('gguf'); p.add_argument('--llama-server', required=True)
    p.add_argument('--hf-tokenizer'); args = p.parse_args()
    source = plan(args.gguf); tokenizer = Tokenizer.from_str(source.tokenizer_json)
    hf = Tokenizer.from_file(args.hf_tokenizer) if args.hf_tokenizer else None
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'server.log'
        with path.open('w') as log:
            process = subprocess.Popen([args.llama_server, '-m', args.gguf, '-ngl', '0',
                '-c', '512', '-np', '1', '-t', '2', '--port', str(port), '--host', '127.0.0.1',
                '--no-warmup'], stdout=log, stderr=log, env={**os.environ, 'CUDA_VISIBLE_DEVICES':''})
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        with urlopen(base+'/health', timeout=1) as response:
                            if response.status == 200: break
                    except OSError: time.sleep(.1)
                else: raise RuntimeError(path.read_text()[-5000:])
                cases = []
                for text in CASES:
                    body = {'content':text, 'add_special':False, 'parse_special':True}
                    with urlopen(Request(base+'/tokenize', data=json.dumps(body).encode(),
                        headers={'Content-Type':'application/json'}), timeout=30) as response:
                        expected = json.load(response)['tokens']
                    actual = tokenizer.encode(text, add_special_tokens=False).ids
                    comparison = dict(text=text, expected=expected, actual=actual, equal=actual == expected)
                    if hf:
                        comparison['hf_tokens'] = hf.encode(text, add_special_tokens=False).ids
                        comparison['hf_equal'] = comparison['hf_tokens'] == actual
                    cases.append(comparison)
                report = dict(passed=all(c['equal'] for c in cases), gguf=str(Path(args.gguf).resolve()), cases=cases)
                print(json.dumps(report, ensure_ascii=False, indent=2))
                assert report['passed'], 'standalone tokenizer differs from llama.cpp'
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=20)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()


if __name__ == '__main__': main()
