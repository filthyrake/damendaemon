"""Freeze vocabulary-only llama.cpp oracle outputs; no model weights or inference.

Usage: .venv/bin/python -m tests.reference.generate_sentencepiece_unused ORACLE
Oracle must link the CPU-only llama.cpp pin recorded in the output fixture.
"""
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
        for unused in ((259,), (260,), (262,), (263,), (265,), (259, 260, 262, 263, 265)):
            for tied in (False, True):
                for prefix in (False, True):
                    source = metadata()
                    for index in unused:
                        source['tokenizer.ggml.token_type'][index] = 5
                    source['tokenizer.ggml.add_space_prefix'] = prefix
                    if tied:
                        source['tokenizer.ggml.tokens'] += ['c', 'bc']
                        source['tokenizer.ggml.token_type'] += [1, 1]
                        source['tokenizer.ggml.scores'] += [-10., -3.]
                    writer = GGUFWriter(str(path), 'llama')
                    writer.add_tokenizer_model('llama')
                    writer.add_token_list(source['tokenizer.ggml.tokens'])
                    writer.add_token_types(source['tokenizer.ggml.token_type'])
                    writer.add_token_scores(source['tokenizer.ggml.scores'])
                    writer.add_bos_token_id(1)
                    writer.add_eos_token_id(2)
                    writer.add_unk_token_id(0)
                    writer.add_bool('tokenizer.ggml.add_space_prefix', prefix)
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.close()
                    samples = []
                    inputs = [(text, add, None) for text in
                              ('', 'a', 'ab', 'abc', 'ba', ' a', 'a b', '<user>ab', 'a</s>ab', 'café 😀', 'a\0b')
                              for add in (False, True)]
                    first = unused[0]
                    inputs += [('', False, ids) for ids in
                               ([first], [first, 264], [264, first], [1, first, 264],
                                [first, 1, 264], [100, first, 101], [first, first, 264],
                                [*unused, 264], [198, first, 172])]
                    for text, add, explicit in inputs:
                        args = [oracle, str(path), str(int(add))]
                        if explicit is not None:
                            args.append(','.join(map(str, explicit)))
                        output = subprocess.run(args, input=text.encode(), capture_output=True,
                                                check=True, timeout=10).stdout.decode()
                        result = dict(line.split(':', 1) for line in output.splitlines())
                        result['ids'] = [int(i) for i in result['ids'].split(',') if i]
                        samples.append(dict(text=text, add_special=add, explicit=explicit, expected=result))
                    cases.append(dict(unused=unused, tied=tied, prefix=prefix, samples=samples))
    return dict(reference='llama.cpp 836d57176dc699a726c55418e4f96b8ca628e1bf', cases=cases)


if __name__ == '__main__':
    target = Path('tests/fixtures/sentencepiece-unused.json')
    target.write_text(json.dumps(generate(sys.argv[1]), ensure_ascii=False, indent=2) + '\n')
