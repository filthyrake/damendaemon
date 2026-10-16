"""Independent CPU oracle for SentencePiece literal specials and prefix policy."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from gguf import GGUFWriter
from tests.test_gguf_sentencepiece import metadata as base_metadata


def metadata(literal, kind, tied, prefix):
    source = base_metadata()
    source['tokenizer.ggml.tokens'][265] = literal
    source['tokenizer.ggml.token_type'][265] = kind
    source['tokenizer.ggml.token_type'][260] = 5
    source['tokenizer.ggml.add_space_prefix'] = prefix
    if tied:
        source['tokenizer.ggml.scores'][264] = -5.
    return source


def generate(oracle):
    cases = []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'vocab.gguf'
        for literal in ('▁user', ' user', ' x▁y', 'é'):
            for kind in (3, 4):
                for tied in (False, True):
                    for prefix in (False, True):
                        source = metadata(literal, kind, tied, prefix)
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
                        inputs = [(text, None) for text in (literal, 'a' + literal + 'ab', literal + literal,
                                  ' ab ', literal + ' café', '', 'a</s>' + literal)]
                        inputs += [('', ids) for ids in ([265], [1, 265, 264], [260, 265], [265, 260, 264],
                                   [1, 260, 265, 264], [198, 265, 172], [198, 1, 172])]
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
                        cases.append(dict(literal=literal, kind=kind, tied=tied, prefix=prefix, samples=samples))
    return dict(reference='llama.cpp 836d57176dc699a726c55418e4f96b8ca628e1bf', cases=cases)


if __name__ == '__main__':
    Path('tests/fixtures/sentencepiece-literals.json').write_text(
        json.dumps(generate(sys.argv[1]), ensure_ascii=False, indent=2) + '\n')
