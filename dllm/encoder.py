"""Independent BERT sentence embeddings and trained cross-encoder relevance scores."""
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
    if (c.get('model_type') != 'bert' or c.get('architectures') not in (['BertModel'],['BertForSequenceClassification'])
            or c.get('is_decoder', False) or c.get('add_cross_attention', False)
            or c.get('position_embedding_type', 'absolute') != 'absolute'
            or c.get('hidden_act') != 'gelu' or c.get('quantization_config')):
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
    if c['architectures']==['BertForSequenceClassification']:
        labels=c.get('num_labels',len(c.get('id2label',{})))
        if (type(labels) is not int or labels!=1 or c.get('problem_type') not in (None,'regression')
                or c.get('sbert_ce_default_activation_function','torch.nn.modules.linear.Identity')!='torch.nn.modules.linear.Identity'):
            raise ValueError('reranking requires one raw relevance output')
        return {**c,'task':'rerank','layer_norm_eps':eps,'embedding_max_length':c['max_position_embeddings']}
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


def used_tensor(name,task='embedding'):
    if task=='rerank':return name not in ('bert.embeddings.position_ids','bert.embeddings.token_type_ids')
    return name not in ('embeddings.position_ids','embeddings.token_type_ids') and not name.startswith('pooler.')


def weight_bytes(path,quantization=None):
    from .management import quantization_mode
    if quantization_mode(quantization) != 'none':
        raise ValueError('sentence encoder quantization is not implemented')
    from .qwen2 import weight_files
    c=config(path)
    total = 0
    for file in weight_files(path):
        with safe_open(str(file),framework='pt',device='cpu') as reader:
            total += sum(math.prod(reader.get_slice(name).get_shape())*4 for name in reader.keys() if used_tensor(name,c['task']))
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
        self.task=self.config['task']
        self.max_context = min(max_context,self.config['embedding_max_length'])
        self.tokenizer = Tokenizer.from_file(str(self.path/'tokenizer.json'))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        state = {}
        for file in weight_files(self.path):
            with safe_open(str(file),framework='pt',device='cpu') as reader:
                for name in reader.keys():
                    if used_tensor(name,self.task):
                        key=name.removeprefix('bert.') if self.task=='rerank' else name
                        if key in state:raise ValueError('duplicate encoder checkpoint tensor')
                        state[key] = reader.get_tensor(name).float()
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
        if self.task=='rerank':
            for name,shape in (('pooler.dense.weight',(h,h)),('pooler.dense.bias',(h,)),
                    ('classifier.weight',(1,h)),('classifier.bias',(1,))):
                w[name]=take(name,shape)
        if state:raise ValueError('unexpected encoder checkpoint tensors')
        self.bytes = sum(t.numel()*t.element_size() for t in w.values())

    def encode(self,inputs):
        if self.task!='embedding':raise ValueError('model supports reranking, not embeddings')
        from .embeddings import options
        inputs, _, _ = options(inputs)
        tokens = [self.tokenizer.encode(value,add_special_tokens=True).ids if isinstance(value,str) else list(value)
                  for value in inputs]
        if any(not row or len(row)>self.max_context or any(t<0 or t>=self.config['vocab_size'] for t in row) for row in tokens):
            raise ValueError('embedding input exceeds model context or vocabulary')
        return tokens


    def encode_pairs(self,query,documents):
        if self.task!='rerank':raise ValueError('model does not support reranking')
        from .reranking import options
        query,documents,_,_=options(query,documents)
        encodings=[self.tokenizer.encode(query,doc,add_special_tokens=True) for doc in documents]
        if any(not e.ids or len(e.ids)>self.max_context
                or any(t>=self.config['vocab_size'] for t in e.ids)
                or any(t>=self.config['type_vocab_size'] for t in e.type_ids) for e in encodings):
            raise ValueError('query/document pair exceeds model context or vocabulary')
        return [e.ids for e in encodings],[e.type_ids for e in encodings]


class EncoderExecutor:
    def __init__(self,model,use_graph=True,**kwargs):
        self.model,self.context = model,model.max_context
        self.device = selected_device()
        self.compute_dtype = torch.float16 if self.device == 'cuda' and model.task=='embedding' else torch.float32
        self.residual_dtype = torch.float32
        self.kernel_backend = 'torch-'+self.device+'-bert'+('-rerank' if model.task=='rerank' else '')
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
        if self.model.task!='embedding':raise ValueError('model does not support embeddings')
        return self._run(tokens)

    @torch.inference_mode()
    def rerank(self,tokens,token_types):
        if self.model.task!='rerank':raise ValueError('model does not support reranking')
        from .reranking import length_groups
        groups=length_groups(tokens)
        if groups is None:return self._run(tokens,token_types).reshape(-1)
        # Length buckets reduce padded attention and projections. Variable
        # bucket shapes bypass graph capture to avoid cold-capture churn.
        scores=[self._run([tokens[i] for i in group],[token_types[i] for i in group],
            allow_graph=False).reshape(-1) for group in groups]
        order=[i for group in groups for i in group]
        inverse=[0]*len(order)
        for offset,index in enumerate(order):inverse[index]=offset
        return torch.cat(scores).index_select(0,torch.tensor(inverse,device=self.device))

    def _run(self,tokens,token_types=None,allow_graph=True):
        w,c = self.w,self.model.config
        batch,length = len(tokens),max(map(len,tokens))
        # Upload the complete padded batch once instead of synchronizing once per row.
        ids = torch.tensor([values+[c['pad_token_id']]*(length-len(values)) for values in tokens],
                           device=self.device,dtype=torch.long)
        types=(torch.tensor([values+[0]*(length-len(values)) for values in token_types],
            device=self.device,dtype=torch.long) if token_types is not None else None)
        lengths = torch.tensor([len(values) for values in tokens],device=self.device)
        mask = torch.arange(length,device=self.device)[None,:] < lengths[:,None]
        # Bound retained graph allocations and avoid capturing large padded batches.
        if allow_graph and self.use_graph and batch <= 8:
            key = (batch,length)
            if key not in self.graphs:
                if len(self.graphs) >= 4:
                    torch.cuda.synchronize()
                    self.graphs.popitem(last=False)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._forward(ids,mask,types)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self._forward(ids,mask,types)
                self.graphs[key] = (graph,ids,mask,types,output)
            graph,static_ids,static_mask,static_types,output = self.graphs[key]
            static_ids.copy_(ids)
            static_mask.copy_(mask)
            if static_types is not None:static_types.copy_(types)
            graph.replay()
            self.graphs.move_to_end(key)
            # Do not expose a buffer overwritten by a later request's replay.
            return output.clone()
        return self._forward(ids,mask,types)

    def _forward(self,ids,mask,types=None):
        w,c = self.w,self.model.config
        batch,length = ids.shape
        def norm(value,name):
            return F.layer_norm(value,(c['hidden_size'],),w[name+'.weight'],w[name+'.bias'],c['layer_norm_eps'])
        def linear(value,name):return F.linear(value.to(self.compute_dtype),w[name+'.weight'],w[name+'.bias'])
        type_embeddings=w['type'][0] if types is None else F.embedding(types,w['type'])
        x = norm(F.embedding(ids,w['embed'])+w['position'][:length]+type_embeddings,'embed_norm')
        heads,dim = c['num_attention_heads'],c['hidden_size']//c['num_attention_heads']
        for i in range(c['num_hidden_layers']):
            q,k,v = [value.view(batch,length,heads,dim).transpose(1,2) for value in linear(x,f'{i}.qkv').chunk(3,-1)]
            attended = F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:])
            x = norm(x+linear(attended.transpose(1,2).reshape(batch,length,-1),f'{i}.attn_out'),f'{i}.norm1')
            x = norm(x+linear(F.gelu(linear(x,f'{i}.fc').float()),f'{i}.down'),f'{i}.norm2')
        if self.model.task=='rerank':
            return linear(torch.tanh(linear(x[:,0],'pooler.dense')),'classifier')
        pooled = (x.float()*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
        return F.normalize(pooled,p=2,dim=-1)
