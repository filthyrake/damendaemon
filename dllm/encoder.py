"""Independent BERT sentence encoder with masked mean pooling and L2 normalization."""
from collections import OrderedDict
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from .device import selected_device, available_memory, empty_cache


def config(path):
    path = Path(path)
    c = json.loads((path/'config.json').read_text())
    if (c.get('model_type') != 'bert' or c.get('architectures') != ['BertModel']
            or c.get('is_decoder', False) or c.get('add_cross_attention', False)
            or c.get('position_embedding_type', 'absolute') != 'absolute'
            or c.get('hidden_act') != 'gelu'):
        raise ValueError('unsupported sentence encoder architecture')
    for key in ('hidden_size','intermediate_size','num_attention_heads','num_hidden_layers',
                'max_position_embeddings','vocab_size','type_vocab_size'):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError('invalid encoder dimensions')
    if c['hidden_size'] % c['num_attention_heads']:
        raise ValueError('encoder hidden size must divide into heads')
    eps = c.get('layer_norm_eps', 1.e-12)
    if type(eps) not in (int,float) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('invalid encoder normalization epsilon')
    if type(c.get('pad_token_id')) is not int or not 0 <= c['pad_token_id'] < c['vocab_size']:
        raise ValueError('encoder requires a valid padding token')
    modules = json.loads((path/'modules.json').read_text())
    kinds = ['Transformer','Pooling','Normalize']
    if (not isinstance(modules,list) or len(modules) != 3
            or any(not isinstance(m,dict) or m.get('type') != 'sentence_transformers.models.'+kind
                   for m,kind in zip(modules,kinds)) or modules[0].get('path') != ''):
        raise ValueError('supported encoder pipeline is Transformer, mean Pooling, Normalize')
    pooling_path = Path(modules[1].get('path',''))
    if pooling_path.is_absolute() or '..' in pooling_path.parts or not pooling_path.parts:
        raise ValueError('invalid pooling module path')
    pooling = json.loads((path/pooling_path/'config.json').read_text())
    expected = {'word_embedding_dimension': c['hidden_size'], 'pooling_mode_cls_token': False,
                'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False,
                'pooling_mode_mean_sqrt_len_tokens': False}
    if pooling != expected or any(type(pooling[k]) is not type(v) for k,v in expected.items()):
        raise ValueError('only declared masked mean pooling is supported')
    length = json.loads((path/'sentence_bert_config.json').read_text()).get('max_seq_length')
    if type(length) is not int or not 1 <= length <= c['max_position_embeddings']:
        raise ValueError('invalid sentence encoder length')
    return {**c, 'task':'embedding', 'layer_norm_eps':eps, 'embedding_max_length':length}


def used_tensor(name):
    return name not in ('embeddings.position_ids','embeddings.token_type_ids') and not name.startswith('pooler.')


def weight_bytes(path,quantization=None):
    from .management import quantization_mode
    if quantization_mode(quantization) != 'none':
        raise ValueError('sentence encoder quantization is not implemented')
    from .qwen2 import weight_files
    config(path)
    total = 0
    for file in weight_files(path):
        with safe_open(str(file),framework='pt',device='cpu') as reader:
            total += sum(math.prod(reader.get_slice(name).get_shape())*4 for name in reader.keys() if used_tensor(name))
    return total


class EncoderWeights:
    task = 'embedding'
    quantization = 'none'

    def __init__(self,path,max_context,quantization=None):
        from .qwen2 import weight_files
        from .quantization import mode
        if mode(quantization) != 'none':
            raise ValueError('sentence encoder quantization is not implemented')
        self.path = Path(path).resolve()
        self.config = config(self.path)
        self.max_context = min(max_context,self.config['embedding_max_length'])
        self.tokenizer = Tokenizer.from_file(str(self.path/'tokenizer.json'))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        state = {}
        for file in weight_files(self.path):
            with safe_open(str(file),framework='pt',device='cpu') as reader:
                for name in reader.keys():
                    if used_tensor(name):state[name] = reader.get_tensor(name).float()
        h,inter = self.config['hidden_size'],self.config['intermediate_size']
        def take(name,shape):
            value = state.pop(name)
            if tuple(value.shape) != tuple(shape):raise ValueError('encoder checkpoint shape mismatch')
            return value
        self.weights = w = {}
        for name,source,shape in (
            ('embed','word_embeddings',(self.config['vocab_size'],h)),
            ('position','position_embeddings',(self.config['max_position_embeddings'],h)),
            ('type','token_type_embeddings',(self.config['type_vocab_size'],h))):
            w[name] = take('embeddings.'+source+'.weight',shape)
        for suffix in ('weight','bias'):
            w['embed_norm.'+suffix] = take('embeddings.LayerNorm.'+suffix,(h,))
        for i in range(self.config['num_hidden_layers']):
            prefix = f'encoder.layer.{i}.'
            for suffix,shape in (('weight',(h,h)),('bias',(h,))):
                w[f'{i}.qkv.{suffix}'] = torch.cat([take(prefix+'attention.self.'+name+'.'+suffix,shape)
                    for name in ('query','key','value')])
            for name,source,shape in (('attn_out','attention.output.dense',(h,h)),
                ('norm1','attention.output.LayerNorm',(h,)),('fc','intermediate.dense',(inter,h)),
                ('down','output.dense',(h,inter)),('norm2','output.LayerNorm',(h,))):
                w[f'{i}.{name}.weight'] = take(prefix+source+'.weight',shape)
                w[f'{i}.{name}.bias'] = take(prefix+source+'.bias',(shape[0],))
        if state:raise ValueError('unexpected encoder checkpoint tensors')
        self.bytes = sum(t.numel()*t.element_size() for t in w.values())

    def encode(self,inputs):
        from .embeddings import options
        inputs, _, _ = options(inputs)
        tokens = [self.tokenizer.encode(value,add_special_tokens=True).ids if isinstance(value,str) else list(value)
                  for value in inputs]
        if any(not row or len(row)>self.max_context or any(t<0 or t>=self.config['vocab_size'] for t in row) for row in tokens):
            raise ValueError('embedding input exceeds model context or vocabulary')
        return tokens


class EncoderExecutor:
    def __init__(self,model,use_graph=True,**kwargs):
        self.model,self.context = model,model.max_context
        self.device = selected_device()
        self.compute_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.residual_dtype = torch.float32
        self.kernel_backend = 'torch-'+self.device+'-bert'
        self.use_graph = use_graph and self.device == 'cuda'
        self.graphs = OrderedDict()
        empty_cache()
        def dtype(name):
            return torch.float32 if name in ('embed','position','type') or 'norm' in name else self.compute_dtype
        needed = sum(t.numel()*(4 if dtype(name)==torch.float32 else 2) for name,t in model.weights.items())
        if needed+256*1024**2 > available_memory():raise MemoryError('insufficient encoder device memory')
        self.w = {name:t.to(device=self.device,dtype=dtype(name)) for name,t in model.weights.items()}

    @torch.inference_mode()
    def embed(self,tokens):
        w,c = self.w,self.model.config
        batch,length = len(tokens),max(map(len,tokens))
        # Upload the complete padded batch once instead of synchronizing once per row.
        ids = torch.tensor([values+[c['pad_token_id']]*(length-len(values)) for values in tokens],
                           device=self.device,dtype=torch.long)
        lengths = torch.tensor([len(values) for values in tokens],device=self.device)
        mask = torch.arange(length,device=self.device)[None,:] < lengths[:,None]
        # Bound retained graph allocations and avoid capturing large padded batches.
        if self.use_graph and batch <= 8:
            key = (batch,length)
            if key not in self.graphs:
                if len(self.graphs) >= 4:
                    torch.cuda.synchronize()
                    self.graphs.popitem(last=False)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._forward(ids,mask)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self._forward(ids,mask)
                self.graphs[key] = (graph,ids,mask,output)
            graph,static_ids,static_mask,output = self.graphs[key]
            static_ids.copy_(ids)
            static_mask.copy_(mask)
            graph.replay()
            self.graphs.move_to_end(key)
            # Do not expose a buffer overwritten by a later request's replay.
            return output.clone()
        return self._forward(ids,mask)

    def _forward(self,ids,mask):
        w,c = self.w,self.model.config
        batch,length = ids.shape
        def norm(value,name):
            return F.layer_norm(value,(c['hidden_size'],),w[name+'.weight'],w[name+'.bias'],c['layer_norm_eps'])
        def linear(value,name):return F.linear(value.to(self.compute_dtype),w[name+'.weight'],w[name+'.bias'])
        x = norm(F.embedding(ids,w['embed'])+w['position'][:length]+w['type'][0],'embed_norm')
        heads,dim = c['num_attention_heads'],c['hidden_size']//c['num_attention_heads']
        for i in range(c['num_hidden_layers']):
            q,k,v = [value.view(batch,length,heads,dim).transpose(1,2) for value in linear(x,f'{i}.qkv').chunk(3,-1)]
            attended = F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:])
            x = norm(x+linear(attended.transpose(1,2).reshape(batch,length,-1),f'{i}.attn_out'),f'{i}.norm1')
            x = norm(x+linear(F.gelu(linear(x,f'{i}.fc').float()),f'{i}.down'),f'{i}.norm2')
        pooled = (x.float()*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
        return F.normalize(pooled,p=2,dim=-1)
