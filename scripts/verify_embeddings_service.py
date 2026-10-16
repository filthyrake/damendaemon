"""Live embeddings API and encoder/decoder residency proof."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import base64
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from tokenizers import Tokenizer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('encoder',type=Path)
    p.add_argument('decoder',type=Path)
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    p.add_argument('--decoder-quantization',choices=('none','q8_0'),default='none')
    args=p.parse_args()
    encoder,decoder=str(args.encoder.resolve()),str(args.decoder.resolve())
    inputs=['A dog plays in the park.','こんにちは — café 🦙']
    tokenizer=Tokenizer.from_file(str(args.encoder/'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    ids=[tokenizer.encode(text,add_special_tokens=True).ids for text in inputs]
    with tempfile.TemporaryDirectory(prefix='dllm-embedding-api-') as directory:
        logpath=Path(directory)/'server.log'
        with logpath.open('w') as log:
            process=subprocess.Popen([sys.executable,'-m','dllm','serve','--port','0','--device',args.device,
                '--context','512','--gpu-models','1','--host-models','2','--model','chat='+decoder],stdout=log,stderr=log,
                env={**os.environ,'DLLM_QUANTIZATION':args.decoder_quantization})
            try:
                deadline=time.monotonic()+120
                while time.monotonic()<deadline and process.poll() is None:
                    ready=[s for s in logpath.read_text().splitlines() if s.startswith('dllm listening on ')]
                    if ready:base=ready[-1].split()[-1];break
                    time.sleep(.1)
                else:raise RuntimeError(logpath.read_text())
                def post(route,body):
                    with urlopen(Request(base+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:return json.load(response)
                def embed(**options):return post('/v1/embeddings',{'model':'encoder','input':inputs,**options})
                post('/v1/models/register',{'model':'encoder','path':encoder,'quantization':'none'})
                first=embed()
                assert first['object']=='list' and [v['index'] for v in first['data']]==[0,1]
                assert all(len(v['embedding'])==384 and abs(sum(x*x for x in v['embedding'])-1)<1.e-5 for v in first['data'])
                assert first['usage']=={'prompt_tokens':sum(map(len,ids)),'total_tokens':sum(map(len,ids))}
                assert embed(input=ids)['data']==first['data']
                assert embed(input=ids[0])['usage']['prompt_tokens']==len(ids[0])
                encoded=embed(encoding_format='base64',dimensions=384)
                for value,target in zip(encoded['data'],first['data']):
                    actual=struct.unpack('<384f',base64.b64decode(value['embedding']))
                    assert list(actual)==target['embedding']
                with ThreadPoolExecutor(4) as pool:
                    concurrent=list(pool.map(lambda _:embed(),range(4)))
                assert all(value['data']==first['data'] for value in concurrent)
                chat=post('/v1/chat/completions',{'model':'chat','messages':[{'role':'user','content':'What is 2 + 2?'}],'max_tokens':8})
                assert chat['choices'][0]['message']['content']
                restored=embed()
                assert restored['dllm_timing']['source_tier']=='host' and restored['data']==first['data']
                chat_again=post('/v1/chat/completions',{'model':'chat','messages':[{'role':'user','content':'What is 2 + 2?'}],'max_tokens':8})
                assert chat_again['choices']==chat['choices'] and chat_again['dllm_timing']['source_tier']=='host'
                assert chat_again['dllm_timing']['weight_quantization']==args.decoder_quantization
                with urlopen(base+'/v1/models/status',timeout=180) as response:status=json.load(response)
                assert {m['id']:m['quantization'] for m in status['models']}=={'chat':args.decoder_quantization,'encoder':'none'}
                try:
                    post('/v1/models/register',{'model':'encoder','path':encoder,'quantization':'q8_0'})
                    raise AssertionError('alias quantization changed silently')
                except HTTPError as error:
                    with error:
                        if error.code != 400:
                            raise ValueError('unexpected HTTP rejection status')
                for body in ({'input':[True]}, {'input':[1]*257}, {'dimensions':128}, {'encoding_format':'int8'}, {'model':'chat'}):
                    try:embed(**body);raise AssertionError('invalid embedding request accepted')
                    except HTTPError as error:
                        with error:
                            if error.code != 400:
                                raise ValueError('unexpected HTTP rejection status')
                try:
                    post('/v1/chat/completions',{'model':'encoder','messages':[{'role':'user','content':'hi'}]})
                    raise AssertionError('encoder accepted generation')
                except HTTPError as error:
                    with error:
                        if error.code != 400:
                            raise ValueError('unexpected HTTP rejection status')
                assert embed()['data']==first['data']
                print(json.dumps({'passed':True,'device':args.device,'encoder':encoder,'decoder':decoder,
                    'text_token_equivalence':True,'float_base64_equivalence':True,'batch_order_and_usage':True,
                    'concurrent_requests':4,'encoder_decoder_switching':True,'invalid_task_recovery':True,
                    'decoder_quantization':args.decoder_quantization,'encoder_quantization':'none',
                    'sample_usage':first['usage'],'embedding_dimensions':384},indent=2))
            except Exception:
                print(logpath.read_text(),file=sys.stderr);raise
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
