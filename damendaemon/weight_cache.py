"""Explicit, atomic cache of prepared dense weights; never loads pickle.

Source identity uses file metadata, not a checkpoint content digest. Local
checkpoint directories keep the full size/mtime/ctime receipt; network-mounted
sources (NFS/SMB and similar) drop ctime, which some servers jitter on reads,
while keeping device/inode/size/mtime. The cache directory itself must remain
trusted local data. Bump VERSION whenever preparation or packing semantics
change.
"""
from .config import read_config
from .network_filesystem import network_device
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

import torch
from safetensors import safe_open, SafetensorError
from safetensors.torch import save_file

VERSION = 3  # Validate source inventory, tied heads and floating dtypes before packing.
DENSE = ('qwen2', 'qwen3', 'llama', 'phi3', 'mistral')


def tensor_schema(config, quantization):
    h = config['hidden_size']
    heads = config['num_attention_heads']
    d = config.get('head_dim', h // heads)
    qkv = (heads + 2 * config['num_key_value_heads']) * d
    intermediate = config['intermediate_size']
    shapes = {'embed': (config['vocab_size'], h), 'norm': (h,),
              'head': (config['vocab_size'], h)}
    for i in range(config['num_hidden_layers']):
        shapes.update({f'{i}.qkv.weight': (qkv, h), f'{i}.out': (h, heads * d),
                       f'{i}.gate_up': (2 * intermediate, h),
                       f'{i}.down': (h, intermediate),
                       f'{i}.norm1': (h,), f'{i}.norm2': (h,)})
        if config['model_type'] == 'qwen2':
            shapes[f'{i}.qkv.bias'] = (qkv,)
        if config['model_type'] == 'qwen3':
            shapes[f'{i}.q_norm'] = shapes[f'{i}.k_norm'] = (d,)
    schema = {}
    for name, shape in shapes.items():
        if len(shape) == 2 and name != 'embed':
            rows, width = shape
            schema['w.' + name] = ((rows, width // (2 if quantization == 'q4_0' else 1)),
                                   torch.uint8 if quantization == 'q4_0' else torch.int8)
            schema['s.' + name] = ((rows, width // 32), torch.float16)
        else:
            schema['w.' + name] = (shape, torch.float16)
    return schema


class PreparedCache:
    def __init__(self, root, model_path, config, quantization, files, schema=None):
        self.root = Path(root)
        self.config = config
        self.files = [Path(model_path) / 'config.json', *files]
        index = Path(model_path) / 'model.safetensors.index.json'
        if index.exists():
            self.files.append(index)
        self.identity = self.snapshot()
        manifest = {'version': VERSION, 'config': config, 'quantization': quantization,
                    'source': self.identity}
        self.manifest = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
        self.key = hashlib.sha256(self.manifest.encode()).hexdigest()
        self.path = self.root / (self.key + '.safetensors')
        # schema=None records the prepared tensor inventory in the cache entry
        # itself (hybrid checkpoints: per-layer tensor sets follow layer_types,
        # so there is no single derivation). Dense keeps the explicit schema.
        self.schema = schema
        self.recorded = schema is None

    def snapshot(self):
        result = []
        for path in self.files:
            stat = path.stat()
            # Network filesystems may jitter ctime on reads; the strict
            # record applies to local cache sources only.
            if network_device(stat.st_dev):
                result.append([str(path.resolve()), stat.st_dev, stat.st_ino,
                               stat.st_size, stat.st_mtime_ns])
            else:
                result.append([str(path.resolve()), stat.st_dev, stat.st_ino,
                               stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
        return result

    def check_source(self):
        if self.snapshot() != self.identity:
            raise RuntimeError('checkpoint changed while preparing cached weights')

    def load(self):
        try:
            with safe_open(str(self.path), framework='pt', device='cpu') as reader:
                metadata = reader.metadata()
                if self.recorded:
                    if not metadata or metadata.get('manifest') != self.manifest:
                        return None
                    schema = self.parse_recorded(metadata.get('schema'))
                    if schema is None:
                        return None
                else:
                    if metadata != {'manifest': self.manifest}:
                        return None
                    schema = self.schema
                assert schema is not None
                if set(reader.keys()) != set(schema):
                    return None
                tensors = {}
                for name, (shape, dtype) in schema.items():
                    value = reader.get_tensor(name)
                    if tuple(value.shape) != shape or value.dtype != dtype:
                        return None
                    tensors[name] = value
        except (OSError, ValueError, RuntimeError, SafetensorError):
            return None
        self.check_source()
        return ({k[2:]: v for k, v in tensors.items() if k.startswith('w.')},
                {k[2:]: v for k, v in tensors.items() if k.startswith('s.')})

    @staticmethod
    def parse_recorded(encoded):
        if not encoded:
            return None
        try:
            recorded = json.loads(encoded)
            return {name: (tuple(shape), getattr(torch, dtype))
                    for name, (shape, dtype) in recorded.items()}
        except (ValueError, AttributeError, TypeError):
            return None

    def recorded_schema(self):
        try:
            with safe_open(str(self.path), framework='pt', device='cpu') as reader:
                metadata = reader.metadata()
            if not metadata or metadata.get('manifest') != self.manifest:
                return None
            return self.parse_recorded(metadata.get('schema'))
        except (OSError, SafetensorError):
            return None

    @staticmethod
    def matches(tensors, schema):
        return (set(tensors) == set(schema) and all(
            tuple(tensors[k].shape) == shape and tensors[k].dtype == dtype
            for k, (shape, dtype) in schema.items()))

    @staticmethod
    def encode_recorded(weights, scales):
        inventory = {}
        for prefix, group in (('w.', weights), ('s.', scales)):
            for name, tensor in group.items():
                inventory[prefix + name] = [list(tensor.shape),
                                            str(tensor.dtype).replace('torch.', '', 1)]
        return json.dumps(inventory, sort_keys=True, separators=(',', ':'))

    def save(self, weights, scales):
        self.check_source()
        tensors = {'w.' + k: v.contiguous() for k, v in weights.items()}
        tensors.update({'s.' + k: v.contiguous() for k, v in scales.items()})
        metadata = {'manifest': self.manifest}
        if self.recorded:
            # The first save records the prepared inventory; later saves for
            # the same checkpoint identity must match it, or a silently
            # different preparation would replace the entry.
            if self.path.exists():
                recorded = self.recorded_schema()
                if recorded is None or not self.matches(tensors, recorded):
                    raise ValueError('prepared weights do not match the recorded cache schema')
                schema = recorded
            else:
                metadata['schema'] = self.encode_recorded(weights, scales)
                schema = self.parse_recorded(metadata['schema'])
        else:
            schema = self.schema
        assert schema is not None
        if set(tensors) != set(schema) or any(
                tuple(tensors[k].shape) != shape or tensors[k].dtype != dtype
                for k, (shape, dtype) in schema.items()):
            raise ValueError('prepared weights do not match cache schema')
        temporary = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.damendaemon-weights-', dir=self.root)
            os.close(fd)
            save_file(tensors, temporary, metadata=metadata)
            self.check_source()
            os.replace(temporary, self.path)
        except (OSError, SafetensorError) as error:
            warnings.warn(f'prepared weight cache write skipped: {error}', RuntimeWarning)
            return False
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        return True


def configured_cache(model_path, config, quantization, files):
    root = os.environ.get('DAMENDAEMON_WEIGHT_CACHE')
    hybrid = (quantization == 'q4_0' and config.get('model_type') == 'qwen3_5_text')
    if (not root or quantization not in ('q4_0', 'q8_0')
            or (config['model_type'] not in DENSE and not hybrid)):
        return None
    # Revalidate configuration after acquiring the source identity so a config
    # edit between the caller's parse and fingerprint cannot seed the wrong key.
    from .config import decoder_config
    schema = None if hybrid else tensor_schema(config, quantization)
    cache = PreparedCache(root, model_path, config, quantization, files, schema=schema)
    if decoder_config(read_config(Path(model_path) / 'config.json')) != config:
        raise RuntimeError('checkpoint configuration changed while loading')
    from .qwen2 import weight_files
    if weight_files(model_path) != files:
        raise RuntimeError('checkpoint shard index changed while loading')
    cache.check_source()
    return cache
