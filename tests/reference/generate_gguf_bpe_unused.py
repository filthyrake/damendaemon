"""Independent vocabulary-only CPU oracle for byte-level UNUSED token semantics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from gguf import GGUFWriter
from tokenizers.pre_tokenizers import ByteLevel


def metadata(pre, unused, eos):
    tokens = sorted(ByteLevel.alphabet()) + ['<|endoftext|>', 'aa', 'ab', 'abc', '<user>', 'xy', '<s>']
    kinds = [1] * 256 + [3, 1, 1, 1, 4, 1, 3]
    for index in unused:
        kinds[index] = 5
    return {'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': pre,
            'tokenizer.ggml.tokens': tokens, 'tokenizer.ggml.token_type': kinds,
            'tokenizer.ggml.merges': ['a a', 'a b', 'ab c'],
            'tokenizer.ggml.eos_token_id': 256, 'tokenizer.ggml.bos_token_id': 262,
            'tokenizer.ggml.add_bos_token': eos, 'tokenizer.ggml.add_eos_token': eos}


def generate(oracle):
    cases = []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'vocab.gguf'
        for pre in ('qwen2', 'llama-bpe'):
            for unused in ([64], [258], [259], [260], [261], [64, 258, 259, 260, 261]):
                for eos in (False, True):
                    source = metadata(pre, unused, eos)
                    writer = GGUFWriter(str(path), 'qwen2')
                    writer.add_tokenizer_model('gpt2')
                    writer.add_tokenizer_pre(pre)
                    writer.add_token_list(source['tokenizer.ggml.tokens'])
                    writer.add_token_types(source['tokenizer.ggml.token_type'])
                    writer.add_token_merges(source['tokenizer.ggml.merges'])
                    writer.add_eos_token_id(256)
                    writer.add_bos_token_id(262)
                    writer.add_bool('tokenizer.ggml.add_bos_token', eos)
                    writer.add_bool('tokenizer.ggml.add_eos_token', eos)
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.close()
                    samples = []
                    inputs = [(text, add, None) for text in
                              ('a', 'aa', 'ab', 'abc', 'ba', ' ab xy ', 'xy', '<user>abc',
                               'a<|endoftext|>abc', '', 'café 😀', 'a\0b') for add in (False, True)]
                    inputs += [('', False, ids) for ids in ([*unused], [*unused, 259],
                               [256, *unused, 259], [64, *unused, 65],
                               [source['tokenizer.ggml.tokens'].index('Ã'), *unused,
                                source['tokenizer.ggml.tokens'].index('©')])]
                    for text, add, explicit in inputs:
                        args = [oracle, str(path), str(int(add))]
                        if explicit is not None:
                            args.append(','.join(map(str, explicit)))
                        output = subprocess.run(args, input=text.encode(), capture_output=True,
                                                check=True, timeout=10).stdout.decode()
                        result = dict(line.split(':', 1) for line in output.splitlines())
                        result['ids'] = [int(i) for i in result['ids'].split(',') if i]
                        samples.append(dict(text=text, add_special=add, explicit=explicit, expected=result))
                    cases.append(dict(pre=pre, unused=unused, eos=eos, samples=samples))
    return dict(reference='llama.cpp 836d57176dc699a726c55418e4f96b8ca628e1bf', cases=cases)


if __name__ == '__main__':
    Path('tests/fixtures/gguf-bpe-unused.json').write_text(
        json.dumps(generate(sys.argv[1]), ensure_ascii=False, indent=2) + '\n')
