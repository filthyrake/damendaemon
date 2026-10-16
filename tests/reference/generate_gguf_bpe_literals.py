"""Independent CPU reference for literal Unicode special-token decoding."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from gguf import GGUFWriter
from tests.reference.generate_gguf_bpe_specials import metadata as base_metadata


def metadata(pre, kind, literal):
    source = base_metadata(pre, kind, False)
    source['tokenizer.ggml.tokens'][257] = literal
    source['tokenizer.ggml.tokens'].append('xy')
    source['tokenizer.ggml.token_type'].append(1)
    source['tokenizer.ggml.merges'] = ['x y']
    return source


def generate(oracle):
    cases = []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'vocab.gguf'
        for pre in ('qwen2', 'llama-bpe'):
            for kind in (3, 4):
                for literal in ('<é>', '<Ġ>', '<😀>', '<é Ġ>'):
                    source = metadata(pre, kind, literal)
                    writer = GGUFWriter(str(path), 'qwen2')
                    writer.add_tokenizer_model('gpt2')
                    writer.add_tokenizer_pre(pre)
                    writer.add_token_list(source['tokenizer.ggml.tokens'])
                    writer.add_token_types(source['tokenizer.ggml.token_type'])
                    writer.add_token_merges(source['tokenizer.ggml.merges'])
                    writer.add_eos_token_id(256)
                    writer.add_bos_token_id(259)
                    writer.add_bool('tokenizer.ggml.add_bos_token', False)
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.close()
                    inputs = [(text, None) for text in
                              (literal, literal + literal, ' ' + literal + ' xy', 'café' + literal + '😀',
                               'a<|endoftext|>' + literal, '', 'café 😀')]
                    first = source['tokenizer.ggml.tokens'].index('Ã')
                    second = source['tokenizer.ggml.tokens'].index('©')
                    inputs += [('', ids) for ids in ([first, 257, second], [first, 256, second], [257, 64, 257])]
                    samples = []
                    for text, explicit in inputs:
                        args = [oracle, str(path), '0']
                        if explicit is not None:
                            args.append(','.join(map(str, explicit)))
                        output = subprocess.run(args, input=text.encode(), capture_output=True,
                                                check=True, timeout=10).stdout.decode()
                        result = dict(line.split(':', 1) for line in output.splitlines())
                        result['ids'] = [int(i) for i in result['ids'].split(',') if i]
                        samples.append(dict(text=text, explicit=explicit, expected=result))
                    cases.append(dict(pre=pre, kind=kind, literal=literal, samples=samples))
    return dict(reference='llama.cpp 836d57176dc699a726c55418e4f96b8ca628e1bf', cases=cases)


if __name__ == '__main__':
    Path('tests/fixtures/gguf-bpe-literals.json').write_text(
        json.dumps(generate(sys.argv[1]), ensure_ascii=False, indent=2) + '\n')
