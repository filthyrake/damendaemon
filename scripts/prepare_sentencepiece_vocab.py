"""Create a vocabulary-only GGUF fixture from an identity-normalized SPM BPE model.

Validation dependencies: sentencepiece, protobuf, gguf. No model weights are
written, and this fixture must never be presented as an inference checkpoint.
"""
import argparse
from pathlib import Path

from gguf import GGUFWriter
from sentencepiece import sentencepiece_model_pb2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sentencepiece_model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    model = sentencepiece_model_pb2.ModelProto()
    model.ParseFromString(Path(args.sentencepiece_model).read_bytes())
    normalizer = model.normalizer_spec
    if (model.trainer_spec.model_type != 2 or normalizer.name != 'identity'
            or normalizer.precompiled_charsmap or normalizer.remove_extra_whitespaces
            or not normalizer.escape_whitespaces or model.trainer_spec.treat_whitespace_as_suffix):
        raise ValueError('fixture requires the Llama-style identity SPM BPE convention')
    writer = GGUFWriter(args.output, 'llama')
    writer.add_name('validation-only SentencePiece vocabulary fixture; no weights')
    writer.add_tokenizer_model('llama'); writer.add_tokenizer_pre('default')
    writer.add_token_list([piece.piece for piece in model.pieces])
    writer.add_token_scores([piece.score for piece in model.pieces])
    writer.add_token_types([piece.type for piece in model.pieces])
    writer.add_bos_token_id(model.trainer_spec.bos_id)
    writer.add_eos_token_id(model.trainer_spec.eos_id)
    writer.add_unk_token_id(model.trainer_spec.unk_id)
    writer.add_bool('tokenizer.ggml.add_space_prefix', normalizer.add_dummy_prefix)
    writer.add_bool('tokenizer.ggml.add_bos_token', True)
    writer.add_bool('tokenizer.ggml.add_eos_token', False)
    writer.write_header_to_file(); writer.write_kv_data_to_file()
    writer.write_tensors_to_file(); writer.close()


if __name__ == '__main__':
    main()
