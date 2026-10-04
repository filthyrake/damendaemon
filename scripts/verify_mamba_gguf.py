"""Prove the benchmark GGUF's dequantized tensors match dllm's Q8 policy."""
import argparse
import json
import torch
import gguf
from gguf.quants import dequantize
from dllm.mamba import MambaWeights
from dllm.quantization import dequantize as restore


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('gguf');args=p.parse_args()
    torch.set_num_threads(4)
    model=MambaWeights(args.model,512,quantization='q8_0');reader=gguf.GGUFReader(args.gguf)
    assert reader.fields['general.architecture'].contents()=='mamba'
    names={'embed':'token_embd.weight','head':'output.weight','norm':'output_norm.weight'}
    suffixes={'norm':'attn_norm.weight','A':'ssm_a','D':'ssm_d','conv':'ssm_conv1d.weight',
        'conv_bias':'ssm_conv1d.bias','dt_bias':'ssm_dt.bias','in_proj':'ssm_in.weight','out_proj':'ssm_out.weight',
        'x_proj':'ssm_x.weight','dt_proj':'ssm_dt.weight','in_bias':'ssm_in.bias','out_bias':'ssm_out.bias'}
    tensors={t.name:t for t in reader.tensors}
    count=0
    for name,value in model.weights.items():
        target=names.get(name)
        if target is None:
            layer,suffix=name.split('.',1);target=f'blk.{layer}.'+suffixes[suffix]
        tensor=tensors.pop(target)
        actual=torch.from_numpy(dequantize(tensor.data,tensor.tensor_type).copy())
        expected=restore(value,model.scales[name]) if name in model.scales else value
        assert actual.shape==expected.shape,(name,actual.shape,expected.shape)
        assert torch.equal(actual,expected),(name,(actual-expected).abs().max().item())
        count+=1
    assert not tensors,list(tensors)
    print(json.dumps({'passed':True,'model':args.model,'gguf':args.gguf,'tensors':count,
        'all_dequantized_weights_bitwise_equal':True,'quantization':'q8_0','input_embedding':'float32'},indent=2))


if __name__=='__main__':main()
