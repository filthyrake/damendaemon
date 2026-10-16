"""Freeze pinned CPU vocabulary-only reference outputs for overlapping specials."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from gguf import GGUFWriter
from tests.test_gguf_sentencepiece import metadata


def generate(oracle):
    cases = []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'vocab.gguf'
        for kind in (3, 4):
            for tied in (False, True):
                for prefix in (False, True):
                    for eos in (False, True):
                        source = metadata()
                        source['tokenizer.ggml.token_type'][262] = kind
                        source['tokenizer.ggml.tokens'].append('bcx')
                        source['tokenizer.ggml.token_type'].append(kind)
                        source['tokenizer.ggml.scores'].append(0.)
                        if tied:
                            source['tokenizer.ggml.scores'][264] = -5.
                        writer = GGUFWriter(str(path), 'llama')
                        writer.add_tokenizer_model('llama')
                        writer.add_token_list(source['tokenizer.ggml.tokens'])
                        writer.add_token_types(source['tokenizer.ggml.token_type'])
                        writer.add_token_scores(source['tokenizer.ggml.scores'])
                        writer.add_bos_token_id(1)
                        writer.add_eos_token_id(2)
                        writer.add_unk_token_id(0)
                        writer.add_bool('tokenizer.ggml.add_space_prefix', prefix)
                        writer.add_bool('tokenizer.ggml.add_eos_token', eos)
                        writer.write_header_to_file()
                        writer.write_kv_data_to_file()
                        writer.close()
                        samples = []
                        for text in ('abcx', 'ab', 'bcx', 'abcxabcx', ' abcx ', 'ababcx',
                                     'abcxab', 'a</s>abcx', '<user>abcx', '', 'café 😀', 'a\0b'):
                            for add in (False, True):
                                output = subprocess.run([oracle, str(path), str(int(add))], input=text.encode(),
                                                        capture_output=True, check=True, timeout=10).stdout.decode()
                                result = dict(line.split(':', 1) for line in output.splitlines())
                                result['ids'] = [int(i) for i in result['ids'].split(',') if i]
                                samples.append(dict(text=text, add_special=add, expected=result))
                        cases.append(dict(kind=kind, tied=tied, prefix=prefix, eos=eos, samples=samples))
    return dict(reference='llama.cpp 836d57176dc699a726c55418e4f96b8ca628e1bf', cases=cases)


if __name__ == '__main__':
    Path('tests/fixtures/sentencepiece-specials.json').write_text(
        json.dumps(generate(sys.argv[1]), ensure_ascii=False, indent=2) + '\n')
