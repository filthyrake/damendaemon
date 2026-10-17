"""Independent T5 text encoder-decoder weights and relative-position geometry."""
import math
from collections import OrderedDict
from pathlib import Path
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from .config import eos_ids, read_config, _positive_number
from .floating_parameter import require_floating


def config(path):
    c=read_config(Path(path)/'config.json')
    from .tokenizer_metadata import read_generation_config
    generation=Path(path)/'generation_config.json'
    if generation.exists():
        saved=read_generation_config(generation)
        for key in ('decoder_start_token_id','pad_token_id','eos_token_id'):
            if key in saved:c[key]=saved[key]
    if (c.get('model_type')!='t5' or c.get('architectures')!=['T5ForConditionalGeneration']
            or c.get('is_encoder_decoder') is not True or c.get('quantization_config')
            or c.get('feed_forward_proj','relu') not in ('relu','gated-gelu')):
        raise ValueError('unsupported T5 encoder-decoder configuration')
    gated=c.get('feed_forward_proj','relu')=='gated-gelu'
    activation='gelu_new' if gated else 'relu'
    if (c.get('dense_act_fn',activation)!=activation
            or type(c.get('is_gated_act',gated)) is not bool
            or c.get('is_gated_act',gated)!=gated):
        raise ValueError('unsupported T5 feed-forward activation override')
    for key in ('d_model','d_ff','d_kv','num_heads','num_layers','vocab_size'):
        if type(c.get(key)) is not int or c[key]<=0:
            raise ValueError('invalid T5 dimension: '+key)
    c.setdefault('num_decoder_layers',c['num_layers'])
    c.setdefault('relative_attention_num_buckets',32)
    c.setdefault('relative_attention_max_distance',128)
    c.setdefault('n_positions',512)
    for key in ('num_decoder_layers','relative_attention_num_buckets','relative_attention_max_distance','n_positions'):
        if type(c[key]) is not int or c[key]<=0:
            raise ValueError('invalid T5 dimension: '+key)
    buckets=c['relative_attention_num_buckets']
    if buckets<4 or buckets%4 or c['relative_attention_max_distance']<=buckets//2:
        raise ValueError('invalid T5 relative-position buckets')
    eps=c.get('layer_norm_epsilon',1.e-6)
    if not _positive_number(eps):
        raise ValueError('invalid T5 normalization epsilon')
    for key in ('decoder_start_token_id','pad_token_id'):
        if type(c.get(key)) is not int or not 0<=c[key]<c['vocab_size']:
            raise ValueError('invalid T5 special token: '+key)
    if type(c.get('tie_word_embeddings',True)) is not bool:
        raise ValueError('invalid tied embedding policy')
    if not eos_ids(c.get('eos_token_id')) or any(token>=c['vocab_size'] for token in eos_ids(c['eos_token_id'])):
        raise ValueError('T5 requires valid EOS tokens')
    return {**c,'hidden_size':c['d_model'],'max_position_embeddings':c['n_positions'],
            'layer_norm_epsilon':eps,'feed_forward_proj':c.get('feed_forward_proj','relu'),
            'tie_word_embeddings':c.get('tie_word_embeddings',True)}


def relative_buckets(query,key,buckets=32,max_distance=128,bidirectional=False):
    distance=key[None,:]-query[:,None]
    offset=0
    if bidirectional:
        buckets//=2
        offset=(distance>0).long()*buckets
        distance=distance.abs()
    else:
        distance=(-distance).clamp_min(0)
    exact=buckets//2
    large=exact+(torch.log(distance.float().clamp_min(exact)/exact)/math.log(max_distance/exact)*(buckets-exact)).long()
    return offset+torch.where(distance<exact,distance,large.clamp_max(buckets-1))


def weight_bytes(path,quantization=None):
    from .quantization import mode
    from .qwen2 import weight_files
    if mode(quantization)!='none':raise ValueError('T5 quantization is not implemented')
    config(path)
    total=0
    for file in weight_files(path):
        with safe_open(str(file),framework='pt',device='cpu') as reader:
            total+=sum(math.prod(reader.get_slice(name).get_shape())*4 for name in reader.keys())
    return total


class T5Weights:
    task='generation'
    quantization='none'
    weight_cache_status='disabled'

    def __init__(self,path,max_context,quantization=None):
        from .quantization import mode
        from .qwen2 import weight_files
        if mode(quantization)!='none':raise ValueError('T5 quantization is not implemented')
        self.path=Path(path).resolve();self.config=config(self.path);c=self.config
        if type(max_context) is not int or not 1<=max_context<=c['max_position_embeddings']:
            raise ValueError('context exceeds T5 configured limit')
        self.max_context=max_context
        self.tokenizer=Tokenizer.from_file(str(self.path/'tokenizer.json'))
        self.tokenizer.no_padding();self.tokenizer.no_truncation()
        self.eos=eos_ids(c['eos_token_id'])
        state={}
        for file in weight_files(self.path):
            with safe_open(str(file),framework='pt',device='cpu') as reader:
                for name in reader.keys():state[name]=require_floating(reader.get_tensor(name),name).float()
        def take(name,shape):
            value=state.pop(name)
            if tuple(value.shape)!=shape:raise ValueError('T5 checkpoint shape mismatch: '+name)
            return value.contiguous()
        h,inner,inter=c['d_model'],c['num_heads']*c['d_kv'],c['d_ff']
        self.weights=w={'embed':take('shared.weight',(c['vocab_size'],h))}
        for name in ('encoder.embed_tokens.weight','decoder.embed_tokens.weight'):
            if name in state and not torch.equal(take(name,w['embed'].shape),w['embed']):
                raise ValueError('T5 encoder and decoder embeddings must be shared')
        if c['tie_word_embeddings']:
            w['head']=w['embed']
            if 'lm_head.weight' in state and not torch.equal(take('lm_head.weight',w['embed'].shape),w['head']):
                raise ValueError('T5 tied output weights disagree')
        else:w['head']=take('lm_head.weight',(c['vocab_size'],h))
        # Original T5 snapshots carry this obsolete tensor. Cross-attention has
        # no relative bias; Transformers explicitly ignores the same legacy key.
        legacy='decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight'
        if legacy in state:take(legacy,(c['relative_attention_num_buckets'],c['num_heads']))
        for stack,layers in (('encoder',c['num_layers']),('decoder',c['num_decoder_layers'])):
            w[stack+'.norm']=take(stack+'.final_layer_norm.weight',(h,))
            for i in range(layers):
                source=f'{stack}.block.{i}.layer.';target=f'{stack}.{i}.'
                attn=source+'0.SelfAttention.'
                w[target+'qkv']=torch.cat([take(attn+name+'.weight',(inner,h)) for name in ('q','k','v')])
                w[target+'out']=take(attn+'o.weight',(h,inner))
                w[target+'norm1']=take(source+'0.layer_norm.weight',(h,))
                if i==0:
                    w[stack+'.relative']=take(attn+'relative_attention_bias.weight',(c['relative_attention_num_buckets'],c['num_heads'])).T.contiguous()
                if stack=='decoder':
                    cross=source+'1.EncDecAttention.'
                    w[target+'crossq']=take(cross+'q.weight',(inner,h))
                    w[target+'crosskv']=torch.cat([take(cross+name+'.weight',(inner,h)) for name in ('k','v')])
                    w[target+'crossout']=take(cross+'o.weight',(h,inner))
                    w[target+'norm2']=take(source+'1.layer_norm.weight',(h,))
                ff=source+('2' if stack=='decoder' else '1')+'.'
                w[target+'normff']=take(ff+'layer_norm.weight',(h,))
                prefix=ff+'DenseReluDense.'
                names=('wi_0','wi_1') if c['feed_forward_proj']=='gated-gelu' else ('wi',)
                w[target+'up']=torch.cat([take(prefix+name+'.weight',(inter,h)) for name in names])
                w[target+'down']=take(prefix+'wo.weight',(h,inter))
        if state:raise ValueError('unexpected T5 checkpoint tensors')
        self.bytes=sum(value.numel()*value.element_size() for value in {id(x):x for x in w.values()}.values())
        self.scales={}

    def raw_tokens(self,prompt):
        if isinstance(prompt,str):tokens=self.tokenizer.encode(prompt,add_special_tokens=True).ids
        else:tokens=list(prompt)
        if not tokens or any(type(token) is not int or not 0<=token<self.config['vocab_size'] for token in tokens):
            raise ValueError('invalid T5 encoder tokens')
        return tokens

    def chat_tokens(self,*args,**kwargs):
        raise ValueError('T5 has no chat template; use raw text/token-ID completions')


from .batch import BatchExecutor
from .device import selected_device,available_memory,empty_cache
from torch.nn import functional as F


class T5Executor:
    is_encoder_decoder=True
    requires_full_prefill=True
    supports_chunked_prefill=False
    capture_graphs=BatchExecutor.capture_graphs
    decode_batch=BatchExecutor.decode_batch

    def __init__(self,model,use_graph=True,max_batch=4,kernels=None):
        import os
        if not 1<=max_batch<=16:raise ValueError('batch capacity must be 1..16')
        self.model=model;self.c=model.config;self.context=model.max_context
        self.device=selected_device();self.max_batch=max_batch
        self.heads=self.c['num_heads'];self.dim=self.c['d_kv'];self.eps=self.c['layer_norm_epsilon']
        self.compute_dtype=self.residual_dtype=torch.float32
        selected=kernels or os.environ.get('DAMENDAEMON_KERNELS','auto')
        if selected not in ('auto','torch','triton','metal'):raise ValueError('invalid T5 kernel policy')
        self.kernels=None
        if selected!='torch':
            if self.device=='cuda':
                if selected=='metal':raise ValueError('Metal requires Apple Silicon')
                from . import kernels as native
            elif self.device=='mps':
                if selected=='triton':raise ValueError('Triton requires CUDA')
                from . import metal_kernels as native
            else:raise ValueError('T5 requires a GPU')
            self.kernels=native
        layers=self.c['num_decoder_layers']
        cache_bytes=4*layers*max_batch*self.heads*self.context*self.dim*4
        bias_bytes=2*self.heads*self.context*self.context*4
        empty_cache()
        if model.bytes+cache_bytes+bias_bytes+512*1024**2>available_memory():
            raise MemoryError('insufficient GPU memory for T5 weights and request state')
        copies={}
        self.w={}
        for name,value in model.weights.items():
            if id(value) not in copies:copies[id(value)]=value.to(self.device)
            self.w[name]=copies[id(value)]
        shape=(2,max_batch,self.heads,self.context,self.dim)
        self.kv=[torch.zeros(shape,device=self.device) for _ in range(layers)]
        self.cross_kv=[torch.zeros(shape,device=self.device) for _ in range(layers)]
        self.encoder_lengths=torch.ones(max_batch,device=self.device,dtype=torch.long)
        self.lengths=[0]*max_batch;self.batch_graphs={};self.length=0
        self.prefill_graph_limit=int(os.environ.get('DAMENDAEMON_T5_PREFILL_GRAPHS','0'))
        if not 0<=self.prefill_graph_limit<=8:raise ValueError('T5 prefill graph limit must be 0..8')
        if not use_graph or self.device!='cuda':self.prefill_graph_limit=0
        self.prefill_graphs=OrderedDict();self.prefill_graph_hits=0
        # cuBLAS retains a workspace per CUDA stream. Reuse this stream across
        # graph eviction/capture so distinct prompt shapes cannot grow that pool.
        self.prefill_warmup_stream=torch.cuda.Stream() if self.prefill_graph_limit else None
        self.arange=torch.arange(self.context,device=self.device)
        positions=torch.arange(self.context)
        self.bias={}
        for stack in ('encoder','decoder'):
            buckets=relative_buckets(positions,positions,self.c['relative_attention_num_buckets'],
                self.c['relative_attention_max_distance'],bidirectional=stack=='encoder').to(self.device)
            self.bias[stack]=self.w[stack+'.relative'][:,buckets]
        self.kernel_backend=('triton' if self.device=='cuda' else 'metal') if self.kernels else 'torch'
        if use_graph and self.device=='cuda':self.capture_graphs()

    def norm(self,x,weight):
        if self.kernels:return self.kernels.rms(x,weight,self.eps,True)
        variance=x.float().square().mean(-1,keepdim=True)
        # Finite inputs can overflow here and otherwise silently normalize to zero.
        if not torch.isfinite(variance).all():raise RuntimeError('nonfinite T5 normalization variance')
        result=(x*torch.rsqrt(variance+self.eps))*weight
        # Preserve the caller's dtype contract: the FP32 variance check must not
        # silently promote FP16/BF16 activations (and their residual stream) to FP32.
        if result.dtype!=x.dtype:result=result.to(x.dtype)
        if not torch.isfinite(result).all():raise RuntimeError('nonfinite T5 normalization output')
        return result

    def linear(self,x,weight):
        if self.kernels:
            if self.device=='mps':return self.kernels.linear(x,weight)
            # cuBLAS reuses FP32 weights across decoder rows more efficiently
            # than the small-vector kernel on the qualified A2 shapes.
            if x.shape[0]>1:return F.linear(x,weight)
            return self.kernels.linear_fp32(x,weight)
        return F.linear(x,weight)

    def feed_forward(self,x,prefix):
        hidden=self.linear(self.norm(x,self.w[prefix+'normff']),self.w[prefix+'up'])
        if self.c['feed_forward_proj']=='gated-gelu':
            gate,up=hidden.chunk(2,-1)
            hidden=F.gelu(gate,approximate='tanh')*up
        else:hidden=F.relu(hidden)
        return x+self.linear(hidden,self.w[prefix+'down'])

    def encode_batch(self,ids,lengths):
        x=F.embedding(ids,self.w['embed'])
        for layer in range(self.c['num_layers']):
            prefix=f'encoder.{layer}.'
            packed=self.linear(self.norm(x,self.w[prefix+'norm1']),self.w[prefix+'qkv'])
            qkv=packed.chunk(3,-1)
            attended=[];offset=0
            for length in lengths:
                q,k,v=[part[offset:offset+length].reshape(length,self.heads,self.dim).transpose(0,1).unsqueeze(0) for part in qkv]
                result=F.scaled_dot_product_attention(q,k,v,attn_mask=self.bias['encoder'][None,:,:length,:length],scale=1.)
                attended.append(result.transpose(1,2).reshape(length,-1));offset+=length
            x=x+self.linear(torch.cat(attended),self.w[prefix+'out'])
            x=self.feed_forward(x,prefix)
        return self.norm(x,self.w['encoder.norm'])

    def attention(self,query,ck,cv,positions,slots,relative=False):
        if self.kernels:
            return self.kernels.attention(query,ck,cv,positions,slots,scale=1.,
                relative_bias=self.w['decoder.relative'] if relative else None,
                relative_max_distance=self.c['relative_attention_max_distance'])
        k,v=ck.index_select(0,slots),cv.index_select(0,slots)
        valid=self.arange[None,:]<=positions[:,None]
        if relative:
            mask=self.bias['decoder'][:,positions,:].permute(1,0,2).unsqueeze(2)
            mask=mask.masked_fill(~valid[:,None,None,:],-float('inf'))
        else:mask=valid[:,None,None,:]
        return F.scaled_dot_product_attention(query,k,v,attn_mask=mask,scale=1.)

    @torch.inference_mode()
    def forward_batch(self,ids,positions,slots):
        count=ids.numel();x=F.embedding(ids,self.w['embed'])
        encoder_positions=self.encoder_lengths.index_select(0,slots)-1
        for layer in range(self.c['num_decoder_layers']):
            prefix=f'decoder.{layer}.'
            packed=self.linear(self.norm(x,self.w[prefix+'norm1']),self.w[prefix+'qkv'])
            q,k,v=[part.reshape(count,self.heads,self.dim) for part in packed.chunk(3,-1)]
            ck,cv=self.kv[layer].unbind(0)
            ck[slots,:,positions,:],cv[slots,:,positions,:]=k,v
            attended=self.attention(q.contiguous().unsqueeze(2),ck,cv,positions,slots,True).reshape(count,-1)
            x=x+self.linear(attended,self.w[prefix+'out'])
            query=self.linear(self.norm(x,self.w[prefix+'norm2']),self.w[prefix+'crossq']).reshape(count,self.heads,1,self.dim)
            ck,cv=self.cross_kv[layer].unbind(0)
            attended=self.attention(query,ck,cv,encoder_positions,slots).reshape(count,-1)
            x=x+self.linear(attended,self.w[prefix+'crossout'])
            x=self.feed_forward(x,prefix)
        x=self.norm(x,self.w['decoder.norm'])
        if self.c['tie_word_embeddings']:x=x*self.c['d_model']**-.5
        return self.linear(x,self.w['head'])

    @torch.inference_mode()
    def prefill_batch(self,prompts,slots):
        if (not prompts or len(prompts)!=len(slots) or len(set(slots))!=len(slots)
                or any(type(slot) is not int or not 0<=slot<self.max_batch for slot in slots)
                or any(not prompt or len(prompt)>self.context for prompt in prompts)):
            raise ValueError('invalid T5 encoder batch')
        if sum(map(len,prompts))>self.context:
            from .prefill import groups
            return torch.cat([self.prefill_batch(batch,assigned) for batch,assigned in groups(prompts,slots,self.context)])
        lengths=[len(prompt) for prompt in prompts]
        ids=torch.tensor([token for prompt in prompts for token in prompt],device=self.device)
        key=(tuple(lengths),tuple(slots))
        cached=self.prefill_graphs.get(key)
        if cached is not None:
            input_ids,_,graph,logits=cached
            input_ids.copy_(ids);graph.replay()
            self.prefill_graphs.move_to_end(key);self.prefill_graph_hits+=1
        else:
            metadata=torch.tensor([slots,lengths,[self.c['decoder_start_token_id']]*len(slots),[0]*len(slots)],device=self.device)
            if self.prefill_graph_limit:
                while len(self.prefill_graphs)>=self.prefill_graph_limit:self.prefill_graphs.popitem(last=False)
                stream=self.prefill_warmup_stream;stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):self.prefill_tensors(ids,lengths,slots,metadata)
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):logits=self.prefill_tensors(ids,lengths,slots,metadata)
                graph.replay()
                self.prefill_graphs[key]=(ids,metadata,graph,logits)
            else:logits=self.prefill_tensors(ids,lengths,slots,metadata)
        for slot in slots:self.lengths[slot]=1
        return logits

    def prefill_tensors(self,ids,lengths,slots,metadata):
        """Device-only prompt work; static shape/slot metadata makes it capturable."""
        encoded=self.encode_batch(ids,lengths)
        for layer in range(self.c['num_decoder_layers']):
            k,v=self.linear(encoded,self.w[f'decoder.{layer}.crosskv']).chunk(2,-1)
            offset=0
            for length,slot in zip(lengths,slots):
                for cache,values in zip(self.cross_kv[layer],(k,v)):
                    cache[slot,:,:length].copy_(values[offset:offset+length].reshape(length,self.heads,self.dim).transpose(0,1))
                offset+=length
        slot_ids,encoder_lengths,start_ids,positions=metadata.unbind(0)
        self.encoder_lengths.index_copy_(0,slot_ids,encoder_lengths)
        return self.forward_batch(start_ids,positions,slot_ids)

    def prefill(self,tokens,slot=0):
        logits=self.prefill_batch([tokens],[slot]);self.length=1
        return logits

    def decode(self,token):
        result=self.decode_batch([token],[0]);self.length=self.lengths[0]
        return result
