"""Live raw-completion API proof, including a checkpoint without chat metadata."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from tokenizers import Tokenizer
from dllm.bench import events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('--device', choices=('cuda','mps'), default='cuda')
    parser.add_argument('--host-gib', type=float, default=8)
    args = parser.parse_args()
    source = args.path.resolve()
    text = 'def binary_search(values, target):\n    '
    ids = Tokenizer.from_file(str(source/'tokenizer.json')).encode(text, add_special_tokens=True).ids
    with tempfile.TemporaryDirectory(prefix='dllm-raw-') as directory:
        root = Path(directory)
        model = root/'base'
        model.mkdir()
        for item in source.iterdir():
            if item.name not in ('tokenizer_config.json','chat_template.jinja'):
                (model/item.name).symlink_to(item)
        config = json.loads((source/'tokenizer_config.json').read_text())
        config.pop('chat_template',None)
        (model/'tokenizer_config.json').write_text(json.dumps(config))
        logpath = root/'server.log'
        with logpath.open('w') as log:
            process = subprocess.Popen([sys.executable,'-m','dllm','serve','--port','0',
                '--context','512','--device',args.device,'--host-gib',str(args.host_gib),'--model','a='+str(model)],stdout=log,stderr=log)
            try:
                deadline = time.monotonic()+180
                while time.monotonic()<deadline and process.poll() is None:
                    ready=[line for line in logpath.read_text().splitlines() if line.startswith('dllm listening on ')]
                    if ready:
                        base=ready[-1].split()[-1]
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError(logpath.read_text())
                def post(body, route='/v1/completions'):
                    return urlopen(Request(base+route,data=json.dumps(body).encode(),
                        headers={'Content-Type':'application/json'}),timeout=180)
                def request(prompt=text, **options):
                    with post({'model':'a','prompt':prompt,'max_tokens':24,'temperature':0,**options}) as response:
                        return json.load(response)
                first=request()
                tokenized=request(ids)
                content=first['choices'][0]['text']
                assert content and content==tokenized['choices'][0]['text']
                assert first['usage']==tokenized['usage']
                assert first['usage']['prompt_tokens']==len(ids)
                with post({'model':'a','prompt':text,'max_tokens':24,'temperature':0,
                           'stream':True,'stream_options':{'include_usage':True}}) as response:
                    chunks=list(events(response, require_terminated=True))
                assert chunks[-1]=='[DONE]'
                parsed=[json.loads(value) for value in chunks[:-1]]
                assert all(item['object']=='text_completion' for item in parsed)
                assert ''.join(c['text'] for item in parsed for c in item['choices'])==content
                assert parsed[-1]['usage']==first['usage']
                with ThreadPoolExecutor(4) as pool:
                    concurrent=list(pool.map(lambda _:request(),range(4)))
                assert all(item['choices'][0]['text']==content for item in concurrent)
                assert max(item['dllm_timing']['max_batch_size'] for item in concurrent)==4
                seeded=[request(temperature=.8,top_k=20,seed=17) for _ in range(2)]
                assert seeded[0]['choices']==seeded[1]['choices']
                stopped=request(stop=content[:8])
                assert stopped['choices'][0]['text']=='' and stopped['choices'][0]['finish_reason']=='stop'
                for body,route in (({'model':'a','messages':[{'role':'user','content':'hi'}]},'/v1/chat/completions'),
                    ({'model':'a','prompt':[10**9]},'/v1/completions'),
                    ({'model':'a','prompt':[True]},'/v1/completions'),
                    ({'model':'a','prompt':[1]*513},'/v1/completions')):
                    try:
                        post(body,route).close()
                        raise AssertionError('invalid input accepted')
                    except HTTPError as error:
                        assert error.code==400
                        error.close()
                assert request()['choices'][0]['text']==content
                print(json.dumps({'passed':True,'model_path':str(source),'device':args.device,
                    'raw_text_equals_token_ids':True,'stream_matches_nonstream':True,
                    'concurrent_batch':4,'seed_and_stop_passed':True,'base_checkpoint_without_template':True,
                    'rejection_recovery':True,'sample':first},indent=2))
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__=='__main__':
    main()
