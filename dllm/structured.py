"""Optional grammar constraints; independent per-request matching state."""
import json
import math


def response_schema(value):
    """Validate our deliberately bounded JSON-schema subset before admission."""
    if value is None or value == {'type': 'text'}:
        return None
    if not isinstance(value, dict):
        raise ValueError('response_format must be an object')
    if value == {'type': 'json_object'}:
        return json.dumps({'type': 'object', 'additionalProperties': True})
    if set(value) != {'type', 'json_schema'} or value['type'] != 'json_schema':
        raise ValueError('unsupported response format')
    options = value['json_schema']
    if (not isinstance(options, dict) or set(options) - {'name', 'description', 'strict', 'schema'}
            or not isinstance(options.get('name'), str) or not 1 <= len(options['name']) <= 64
            or options.get('strict', True) is not True
            or ('description' in options and not isinstance(options['description'], str))):
        raise ValueError('json_schema requires a name and strict schema')
    schema = options.get('schema')
    try:
        encoded = json.dumps(schema, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError('invalid JSON schema') from error
    if len(encoded.encode()) > 16384:
        raise ValueError('schema exceeds 16 KiB')
    count = 0
    def visit(node, depth=0):
        nonlocal count
        count += 1
        if depth > 16 or count > 256 or not isinstance(node, dict):
            raise ValueError('schema is too complex or not an object')
        if 'anyOf' in node:
            branches=node['anyOf']
            if (set(node)-{'anyOf','description','title'} or not isinstance(branches,list)
                    or not 1<=len(branches)<=64
                    or any(not isinstance(node[k],str) for k in ('description','title') if k in node)):
                raise ValueError('invalid anyOf schema')
            for branch in branches:visit(branch,depth+1)
            return
        kind = node.get('type')
        if kind not in ('object', 'array', 'string', 'integer', 'number', 'boolean', 'null'):
            raise ValueError('schema requires one supported type')
        allowed = {'type', 'description', 'title'}
        if kind == 'object':
            allowed |= {'properties', 'required', 'additionalProperties'}
            props = node.get('properties', {})
            required = node.get('required', [])
            if (not isinstance(props, dict) or any(not isinstance(k,str) for k in props)
                    or not isinstance(required, list) or any(not isinstance(k,str) for k in required)
                    or len(set(required)) != len(required) or not set(required) <= set(props)
                    or type(node.get('additionalProperties', False)) is not bool):
                raise ValueError('invalid object schema')
            for child in props.values():
                visit(child, depth+1)
        elif kind == 'array':
            allowed |= {'items'}
            visit(node.get('items'), depth+1)
        else:
            allowed |= {'enum', 'const'}
            values = node.get('enum', [node['const']] if 'const' in node else [])
            if not isinstance(values, list) or ('enum' in node and not values) or ('enum' in node and 'const' in node):
                raise ValueError('invalid enum or const')
            for item in values:
                valid = {'string': isinstance(item,str), 'integer': type(item) is int,
                         'number': type(item) in (int,float) and (type(item) is int or math.isfinite(item)),
                         'boolean': type(item) is bool, 'null': item is None}[kind]
                if not valid:
                    raise ValueError('enum or const does not match schema type')
        if set(node) - allowed or any(not isinstance(node[k],str) for k in ('description','title') if k in node):
            raise ValueError('unsupported schema keyword')
    visit(schema)
    return encoded


def matches_schema(value,schema):
    """Independent final-value check for the already validated bounded subset."""
    if 'anyOf' in schema:return any(matches_schema(value,branch) for branch in schema['anyOf'])
    kind=schema['type']
    valid={'object':isinstance(value,dict),'array':isinstance(value,list),
        'string':isinstance(value,str),'integer':type(value) is int,
        'number':type(value) in (int,float) and (type(value) is int or math.isfinite(value)),
        'boolean':type(value) is bool,'null':value is None}[kind]
    if not valid:return False
    if kind=='object':
        props=schema.get('properties',{})
        return (set(schema.get('required',[]))<=set(value)
            and (schema.get('additionalProperties',False) or not set(value)-set(props))
            and all(matches_schema(item,props[key]) for key,item in value.items() if key in props))
    if kind=='array':return all(matches_schema(item,schema['items']) for item in value)
    return ('const' not in schema or value==schema['const']) and ('enum' not in schema or value in schema['enum'])


def prepare_options(response_format, stops, thinking, ignore_eos=False):
    schema = response_schema(response_format)
    if schema is not None:
        if stops or thinking is True or ignore_eos:
            raise ValueError('structured output cannot use stop strings, thinking or ignore_eos')
    return schema, thinking


class Constraint:
    def __init__(self, model, schema):
        if getattr(model.tokenizer, "suppressed_token_ids", ()):
            raise ValueError("structured output is not yet qualified for suppressed GGUF tokens")
        if hasattr(model.tokenizer, "native_model"):
            raise ValueError("structured output is not yet qualified for positional SentencePiece GGUF tokenizers")
        try:
            import xgrammar as xgr
            from transformers import PreTrainedTokenizerFast
        except ImportError as error:
            raise ValueError('structured output requires dllm[structured]') from error
        self.xgr = xgr
        size = model.config['vocab_size']
        compiler = getattr(model, '_grammar_compiler', None)
        if compiler is None:
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=getattr(model.tokenizer, 'backend_tokenizer', model.tokenizer))
            info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=size,
                                                     stop_token_ids=list(model.eos))
            compiler = xgr.GrammarCompiler(info, max_threads=2, cache_limit_bytes=64*1024*1024)
            model._grammar_compiler = compiler
        # Keep property order deterministic; any_order relaxes required-key checks.
        compiled = compiler.compile_json_schema(schema, strict_mode=True, max_whitespace_cnt=8)
        self.matcher = xgr.GrammarMatcher(compiled)
        self.mask = xgr.allocate_token_bitmask(1, size)
        self.indices = None

    def apply(self, logits):
        import torch
        if not self.matcher.fill_next_token_bitmask(self.mask):
            return logits
        # Do not alter the graph-owned logits or another request's row.
        result = logits.clone()
        mask = self.mask.to(logits.device)
        if logits.device.type == 'mps':
            if self.indices is None:
                self.indices = torch.arange(logits.numel(), device=logits.device, dtype=torch.int32)
            allowed = ((mask[0, self.indices // 32] >> (self.indices % 32)) & 1).bool()
            result.masked_fill_(~allowed, -float('inf'))
        else:
            self.xgr.apply_token_bitmask_inplace(result.unsqueeze(0), mask)
        return result

    def accept(self, token):
        if not self.matcher.accept_token(token):
            raise RuntimeError('sampled token violates the response grammar')
