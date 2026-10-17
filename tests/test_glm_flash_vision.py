import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from damendaemon.glm_flash_vision import FlashVision,vision_specs,audit_vision,ROOT


def fixture():
    c=dict(model_type='glm5_next_vision',depth=2,hidden_size=8,num_heads=2,in_channels=3,patch_size=2,
           temporal_patch_size=2,spatial_merge_size=2,out_hidden_size=12,intermediate_size=16,
           projection_intermediate_size=20,hidden_act='silu',attention_bias=True,attention_dropout=0.,
           rms_norm_eps=1e-5,swiglu_limit=.2)
    torch.manual_seed(108)
    w={name:torch.randn(shape)*.2 for name,shape in vision_specs(c).items()}
    for name in w:
        if name.endswith('norm1.weight') or name.endswith('norm2.weight') or '_norm.weight' in name or 'layernorm.weight' in name:w[name]=torch.ones_like(w[name])
    return c,w


def control(patches,c,w,grid,capture=None):
    """One-frame control using Conv3d/Conv2d and explicit head attention."""
    _,height,width=grid;m=c['spatial_merge_size'];d=c['hidden_size'];heads=c['num_heads'];hd=d//heads
    hidden=F.conv3d(patches.reshape(-1,c['in_channels'],c['temporal_patch_size'],c['patch_size'],c['patch_size']),
                    w['patch_embed.proj.weight'],w['patch_embed.proj.bias']).flatten(1)
    if capture is not None:capture['patch']=hidden.clone()
    coords=[(r+i,s+j) for r in range(0,height,m) for s in range(0,width,m) for i in range(m) for j in range(m)]
    def norm(x,name):return x/(x.square().mean(-1,keepdim=True)+c['rms_norm_eps']).sqrt()*w[name]
    def linear(x,name):return x@w[name+'.weight'].T+w.get(name+'.bias',0)
    def ff(x,prefix):
        gate=linear(x,prefix+'gate_proj').clamp(max=c['swiglu_limit']);up=linear(x,prefix+'up_proj').clamp(-c['swiglu_limit'],c['swiglu_limit'])
        return linear(gate.sigmoid()*gate*up,prefix+'down_proj')
    def rotate(x):
        out=torch.empty_like(x)
        for token,(row,col) in enumerate(coords):
            for j in range(hd//2):
                position=row if j<hd//4 else col;frequency=10000.**(-(j%(hd//4))/(hd//4))
                angle=torch.tensor(position*frequency);co,si=angle.cos(),angle.sin()
                out[token,:,j]=x[token,:,j]*co-x[token,:,j+hd//2]*si
                out[token,:,j+hd//2]=x[token,:,j+hd//2]*co+x[token,:,j]*si
        return out
    for i in range(c['depth']):
        p=f'blocks.{i}.';x=norm(hidden,p+'norm1.weight');qkv=linear(x,p+'attn.qkv');q,k,v=[a.reshape(-1,heads,hd) for a in qkv.chunk(3,-1)]
        q=rotate(norm(q,p+'attn.q_norm.weight'));k=rotate(norm(k,p+'attn.k_norm.weight'))
        out=torch.stack([((q[:,h]@k[:,h].T)/(hd**.5)).softmax(-1)@v[:,h] for h in range(heads)],1).reshape(-1,d)
        hidden=hidden+linear(out,p+'attn.proj');hidden=hidden+ff(norm(hidden,p+'norm2.weight'),p+'mlp.')
    hidden=norm(hidden,'post_layernorm.weight').reshape(-1,m,m,d).permute(0,3,1,2)
    merged=F.conv2d(hidden,w['downsample.weight'],w['downsample.bias']).flatten(1)
    if capture is not None:capture['downsample']=merged.clone()
    merged=linear(merged,'merger.proj');mean=merged.mean(-1,keepdim=True);variance=(merged-mean).square().mean(-1,keepdim=True)
    merged=(merged-mean)/(variance+1e-5).sqrt()*w['merger.post_projection_norm.weight']+w['merger.post_projection_norm.bias']
    merged=.5*merged*(1+torch.erf(merged/(2**.5)))
    return ff(merged,'merger.')


class FlashVisionTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)

    def test_pinned_visual_inventory_and_strict_header_contract(self):
        data=json.loads(Path('tests/fixtures/glm53_flash_contract.json').read_text());c=data['config']
        names=[ROOT+n for n in vision_specs(c['vision_config'])]
        pin=json.loads(Path('tests/fixtures/glm53_flash_vision_contract.json').read_text())
        self.assertEqual(len(names),347)
        self.assertEqual(hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest(),pin['visual_names_sha256'])
        headers={ROOT+n:dict(shape=list(shape),dtype='BF16') for n,shape in vision_specs(c['vision_config']).items()}
        self.assertEqual(audit_vision(c,names,headers)['vision_tensors'],347)
        for change in ('missing','extra','quant','shape','width'):
            changed=copy.deepcopy(c);bad=dict(headers)
            if change=='missing':bad.pop(next(iter(bad)))
            if change=='extra':bad[ROOT+'extra']={}
            if change=='quant':bad[next(iter(bad))]=dict(dtype='F8_E4M3',shape=bad[next(iter(bad))]['shape'])
            if change=='shape':bad[next(iter(bad))]=dict(dtype='BF16',shape=[1])
            if change=='width':changed['text_config']['hidden_size']=1
            with self.assertRaises(ValueError):audit_vision(changed,bad,bad)

    def test_encoder_matches_independent_convolution_attention_and_merge_control(self):
        c,w=fixture();encoder=FlashVision(c,w);patches=torch.randn(8,24)
        captured={};expected=control(patches,c,w,(1,2,4),captured);seen={};project=F.linear
        def record(x,weight,bias=None):
            result=project(x,weight,bias)
            for key,name in (('patch','patch_embed.proj.weight'),('downsample','downsample.weight')):
                if weight.data_ptr()==w[name].data_ptr():seen[key]=result.clone()
            return result
        with patch('damendaemon.glm_flash_vision.F.linear',side_effect=record):actual=encoder.forward(patches,[(1,2,4)])
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-6)
        for key in captured:torch.testing.assert_close(seen[key],captured[key],rtol=3e-5,atol=2e-6)
        self.assertIs(encoder.w['patch_embed.proj.weight'],w['patch_embed.proj.weight'])

    def test_frame_and_image_isolation_and_merge_order(self):
        c,w=fixture();encoder=FlashVision(c,w);patches=torch.randn(20,24)
        actual=encoder.forward(patches,[(2,2,4),(1,2,2)])
        expected=torch.cat([control(patches[:8],c,w,(1,2,4)),control(patches[8:16],c,w,(1,2,4)),control(patches[16:],c,w,(1,2,2))])
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-6)
        modified=patches.clone();modified[8:]=100
        torch.testing.assert_close(encoder.forward(modified,[(2,2,4),(1,2,2)])[:2],actual[:2],rtol=0,atol=0)

    def test_budget_geometry_and_nonfinite_failures_before_execution(self):
        c,w=fixture();patches=torch.randn(4,24)
        for rope in (False,[],{'rope_type':'linear'},{'rope_theta':float('nan')},{'factor':2}):
            with self.assertRaises(ValueError):vision_specs(dict(c,rope_parameters=rope))
        with self.assertRaisesRegex(ValueError,'parameter budget'):FlashVision(c,w,max_weight_bytes=1)
        for options in (dict(max_tokens=3),dict(max_workspace_bytes=1),dict(max_output_bytes=1)):
            encoder=FlashVision(c,w,**options)
            with patch('damendaemon.glm_flash_vision.F.linear',side_effect=AssertionError('projection before admission')):
                with self.assertRaises(ValueError):encoder.forward(patches,[(1,2,2)])
        encoder=FlashVision(c,w)
        for grids in ([],[(True,2,2)],[(1,3,2)],[(1,2,4)]):
            with self.assertRaises(ValueError):encoder.forward(patches,grids)
        bad=patches.clone();bad[0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):encoder.forward(bad,[(1,2,2)])
        w['patch_embed.proj.weight'].fill_(1e30)
        with self.assertRaisesRegex(ValueError,'nonfinite'):encoder.forward(torch.ones_like(patches),[(1,2,2)])
        w['patch_embed.proj.weight'].zero_();encoder.forward(patches,[(1,2,2)]) # Stateless failures do not poison future calls.
