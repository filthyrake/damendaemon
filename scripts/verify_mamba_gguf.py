"""Prove the benchmark GGUF's dequantized tensors match damendaemon's Q8 policy."""
import argparse
import json
import torch
import gguf
from gguf.quants import dequantize
from damendaemon.models import load
from damendaemon.quantization import dequantize as restore


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('gguf')
    p.add_argument('--quantization',choices=('none','q8_0'),default='q8_0');args=p.parse_args()
    torch.set_num_threads(4)
    model=load(args.model,512,quantization=args.quantization);reader=gguf.GGUFReader(args.gguf)
    if reader.fields['general.architecture'].contents()!=model.config['model_type']:
        raise ValueError('GGUF architecture differs from the source model')
    names={'embed':'token_embd.weight','head':'output.weight','norm':'output_norm.weight'}
    suffixes={'norm':'attn_norm.weight','A':'ssm_a','D':'ssm_d','conv':'ssm_conv1d.weight',
        'conv_bias':'ssm_conv1d.bias','dt_bias':'ssm_dt.bias','in_proj':'ssm_in.weight','out_proj':'ssm_out.weight',
        'x_proj':'ssm_x.weight','dt_proj':'ssm_dt.weight','in_bias':'ssm_in.bias','out_bias':'ssm_out.bias',
        'gated_norm':'ssm_norm.weight'}
    if len({t.name for t in reader.tensors})!=len(reader.tensors):
        raise ValueError('duplicate GGUF tensor names')
    tensors={t.name:t for t in reader.tensors}
    original=dict(tensors);count=0
    for name,value in model.weights.items():
        target=names.get(name)
        if target is None:
            layer,suffix=name.split('.',1);target=f'blk.{layer}.'+suffixes[suffix]
        if name=='head' and target not in tensors and value is model.weights['embed']:
            tensor=original['token_embd.weight']
        else:tensor=tensors.pop(target)
        actual=torch.from_numpy(dequantize(tensor.data,tensor.tensor_type).copy())
        expected=restore(value,model.scales[name]) if name in model.scales else value
        if model.config['model_type']=='mamba2' and name.rsplit('.',1)[-1] in ('A','D','gated_norm'):
            actual=actual.reshape(expected.shape)
        if actual.shape!=expected.shape:raise ValueError(f'GGUF tensor shape differs: {name}')
        if not torch.equal(actual,expected):raise ValueError(f'GGUF tensor coefficients differ: {name}')
        count+=1
    if tensors:raise ValueError('unexpected GGUF tensors: '+', '.join(sorted(tensors)))
    print(json.dumps({'passed':True,'model':args.model,'gguf':args.gguf,'tensors':count,
        'all_dequantized_weights_bitwise_equal':True,'quantization':args.quantization,'input_embedding':'float32'},indent=2))


if __name__=='__main__':main()
