"""Download only the pinned HF-format GPT-OSS-20B snapshot, without duplicates."""
import argparse
import json
from pathlib import Path
import struct

from huggingface_hub import snapshot_download


REPOSITORY='openai/gpt-oss-20b'
REVISION='6cee5e81ee83917806bbde320786a8fb61efebee'
FILES=['config.json','tokenizer.json','tokenizer_config.json','generation_config.json',
       'special_tokens_map.json','chat_template.jinja','model.safetensors.index.json','LICENSE','README.md',
       'model-00000-of-00002.safetensors','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',default='models/gpt-oss-20b')
    args=parser.parse_args()
    root=Path(snapshot_download(REPOSITORY,revision=REVISION,allow_patterns=FILES,
        local_dir=args.destination,max_workers=2,token=False)).resolve()
    config=json.loads((root/'config.json').read_text())
    if config['model_type']!='gpt_oss':raise ValueError('unexpected checkpoint architecture')
    catalog={};total=0
    for name in FILES:
        if not name.endswith('.safetensors'):continue
        path=root/name;total+=path.stat().st_size
        with path.open('rb') as source:
            count=struct.unpack('<Q',source.read(8))[0]
            if not 0<count<=32*1024**2:raise ValueError('invalid tensor header length')
            header=json.loads(source.read(count))
        for key,value in header.items():
            if key=='__metadata__':continue
            if key in catalog:raise ValueError('duplicate tensor key')
            catalog[key]={'file':name,'dtype':value['dtype'],'shape':value['shape'],
                          'bytes':value['data_offsets'][1]-value['data_offsets'][0]}
    report={'repository':REPOSITORY,'revision':REVISION,'path':str(root),'safetensor_file_bytes':total,
        'config':config,'tensors':catalog,'license':'Apache-2.0; preserved LICENSE in snapshot'}
    (root/'dllm-source.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'repository':REPOSITORY,'revision':REVISION,'path':str(root),
        'safetensor_file_bytes':total,'tensor_count':len(catalog),
        'first_layer':{k:v for k,v in catalog.items() if k.startswith('model.layers.0.')}},indent=2))


if __name__=='__main__':main()
