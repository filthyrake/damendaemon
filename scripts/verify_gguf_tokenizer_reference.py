"""Check standalone GGUF encoding and decoding against captured native API results."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from dllm.gguf_model import tokenizer
from gguf import GGUFReader
from dllm.gguf_sentencepiece import SentencePieceTokenizer


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--requests', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--hf-tokenizer', help='diagnostic fast-tokenizer control; differences do not define the GGUF gate')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    metadata = {key: value.contents() for key, value in GGUFReader(args.model).fields.items()}
    runtime, _, _ = tokenizer(metadata)
    serialized = Tokenizer.from_str(runtime.to_str())
    restored = SentencePieceTokenizer(serialized) if metadata.get('tokenizer.ggml.model') == 'llama' and metadata.get('tokenizer.ggml.add_space_prefix', True) else serialized
    requests = [json.loads(line) for line in Path(args.requests).read_text().splitlines()]
    references = [json.loads(line) for line in Path(args.reference).read_text().splitlines()]
    if len(requests) != len(references):raise ValueError('request/reference count mismatch')
    hf = Tokenizer.from_file(args.hf_tokenizer) if args.hf_tokenizer else None
    report = dict(gguf_sha256=digest(args.model), requests_sha256=digest(args.requests),
                  reference_sha256=digest(args.reference),
                  reference_revision='836d57176dc699a726c55418e4f96b8ca628e1bf',
                  reference_adapter_sha256=digest(Path(__file__).parent/'reference/gguf_tokenizer.cpp'),
                  hf_tokenizer_sha256=digest(args.hf_tokenizer) if hf else None, cases=[], passed=False)
    for request, reference in zip(requests, references):
        ids = runtime.encode(request['text'], add_special_tokens=request['add_special_tokens']).ids
        repeated = restored.encode(request['text'], add_special_tokens=request['add_special_tokens']).ids
        case = dict(**request, expected_ids=reference['ids'], actual_ids=ids,
                    encode_equal=ids == reference['ids'] == repeated, decode=[])
        if hf:
            case['hf_ids'] = hf.encode(request['text'], add_special_tokens=request['add_special_tokens']).ids
            case['hf_equal'] = case['hf_ids'] == reference['ids']
        for skip in (False, True):
            expected = reference['decoded_skip_special' if skip else 'decoded']
            actual = runtime.decode(ids, skip_special_tokens=skip)
            case['decode'].append(dict(skip_special_tokens=skip, expected=expected, actual=actual,
                equal=expected == actual == restored.decode(ids, skip_special_tokens=skip),
                unwrapped_equal=expected == serialized.decode(ids, skip_special_tokens=skip)))
        report['cases'].append(case)
    report['passed'] = all(case['encode_equal'] and all(d['equal'] for d in case['decode']) for case in report['cases'])
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], encode_checks=len(report['cases']),
        decode_checks=sum(len(c['decode']) for c in report['cases']),
        hf_mismatches=sum(not c.get('hf_equal', True) for c in report['cases']),
        unwrapped_decode_mismatches=sum(not d['unwrapped_equal'] for c in report['cases'] for d in c['decode']))))
    if not report['passed']:raise AssertionError('GGUF tokenizer reference mismatch')


if __name__ == '__main__':
    main()
