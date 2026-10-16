"""Freeze pinned vocabulary-only llama.cpp outputs for byte-level GGUF specials."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from gguf import GGUFWriter
from tokenizers.pre_tokenizers import ByteLevel


def metadata(pre, kind, eos):
    return {'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': pre,
            'tokenizer.ggml.tokens': sorted(ByteLevel.alphabet()) + ['<|endoftext|>', 'ab', 'bcx', '<s>'],
            'tokenizer.ggml.token_type': [1] * 256 + [3, kind, kind, 3],
            'tokenizer.ggml.merges': ['a b'], 'tokenizer.ggml.eos_token_id': 256,
            'tokenizer.ggml.bos_token_id': 259,
            'tokenizer.ggml.add_bos_token': eos, 'tokenizer.ggml.add_eos_token': eos}


def generate(oracle):
    cases = []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'vocab.gguf'
        for pre in ('qwen2', 'llama-bpe'):
            for kind in (3, 4):
                for eos in (False, True):
                    source = metadata(pre, kind, eos)
                    writer = GGUFWriter(str(path), 'qwen2')
                    writer.add_tokenizer_model('gpt2')
                    writer.add_tokenizer_pre(pre)
                    writer.add_token_list(source['tokenizer.ggml.tokens'])
                    writer.add_token_types(source['tokenizer.ggml.token_type'])
                    writer.add_token_merges(source['tokenizer.ggml.merges'])
                    writer.add_eos_token_id(256)
                    writer.add_bos_token_id(259)
                    writer.add_bool('tokenizer.ggml.add_bos_token', eos)
                    writer.add_bool('tokenizer.ggml.add_eos_token', eos)
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.close()
                    samples = []
                    for text in ('abcx', 'ab', 'bcx', 'abcxabcx', ' abcx ', 'ababcx', 'abcxab',
                                 'a<|endoftext|>abcx', '', 'café 😀', 'a\0b', '12345'):
                        for add in (False, True):
                            output = subprocess.run([oracle, str(path), str(int(add))], input=text.encode(),
                                                    capture_output=True, check=True, timeout=10).stdout.decode()
                            result = dict(line.split(':', 1) for line in output.splitlines())
                            result['ids'] = [int(i) for i in result['ids'].split(',') if i]
                            samples.append(dict(text=text, add_special=add, expected=result))
                    cases.append(dict(pre=pre, kind=kind, eos=eos, samples=samples))
    return dict(reference='llama.cpp 836d57176dc699a726c55418e4f96b8ca628e1bf', cases=cases)


if __name__ == '__main__':
    Path('tests/fixtures/gguf-bpe-specials.json').write_text(
        json.dumps(generate(sys.argv[1]), ensure_ascii=False, indent=2) + '\n')
