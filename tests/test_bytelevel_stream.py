"""Incremental bytes must agree with the CPU tokenizer, including added pieces."""
from pathlib import Path
import tempfile
import unittest
from tokenizers import Tokenizer,models,decoders,AddedToken
from damendaemon.bytelevel_stream import BYTE_VALUES,ByteLevelStream


class ByteLevelStreamTests(unittest.TestCase):
    def test_ordinary_and_added_pieces_match_tokenizer_decode(self):
        pieces=('custom','aé','aé🙂','Ġcustom','<new>')
        for added in (False,True):
            vocabulary={char:i for i,char in enumerate(BYTE_VALUES)}
            if not added:vocabulary.update({piece:len(vocabulary)+i for i,piece in enumerate(pieces)})
            tokenizer=Tokenizer(models.BPE(vocab=vocabulary,merges=[]));tokenizer.decoder=decoders.ByteLevel()
            if added:tokenizer.add_tokens([AddedToken(piece,special=False) for piece in (*pieces,'Ã','©')])
            for piece in pieces:
                for sequence in ([piece],['Ã','©',piece],[piece,'Ã','©'],['Ã',piece,'©'],['Ã',piece,'Ã']):
                    with self.subTest(added=added,sequence=sequence):
                        ids=[tokenizer.token_to_id(value) for value in sequence]
                        stream=ByteLevelStream(tokenizer,())
                        actual=''.join(stream.feed(token) for token in ids)+stream.finish()
                        self.assertEqual(actual,tokenizer.decode(ids,skip_special_tokens=False))

    def test_special_control_admission_and_invalid_ids_are_preserved(self):
        tokenizer=Tokenizer(models.BPE(vocab={'a':0},merges=[]));tokenizer.decoder=decoders.ByteLevel()
        tokenizer.add_special_tokens([AddedToken('<reserved>',special=True)])
        token=tokenizer.token_to_id('<reserved>')
        with self.assertRaises(ValueError):ByteLevelStream(tokenizer,()).feed(token)
        self.assertEqual(ByteLevelStream(tokenizer,('<reserved>',)).feed(token),'<reserved>')
        with self.assertRaises(ValueError):ByteLevelStream(tokenizer,()).feed(999)

    def test_reference_stream_accepts_ordinary_added_text(self):
        from tests.test_deepseek_generation import text_fixture
        from damendaemon.deepseek_text import DeepSeekText,EOS
        from damendaemon.deepseek_generation import TokenStream
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);original=text_fixture(path)
            original.tokenizer.add_tokens([AddedToken('custom aé🙂',special=False)])
            original.tokenizer.save(str(path/'tokenizer.json'))
            text=DeepSeekText(path,vocab_size=original.tokenizer.get_vocab_size())
            stream=TokenStream(text)
            events=stream.feed(text.tokenizer.token_to_id('custom aé🙂'))
            events+=stream.feed(text.tokenizer.token_to_id(EOS))+stream.finish()
            self.assertEqual(''.join(event.get('content','') for event in events),'custom aé🙂')

    def test_transparent_nested_bytelevel_decoder(self):
        tokenizer=Tokenizer(models.BPE(vocab={char:i for i,char in enumerate(BYTE_VALUES)},merges=[]))
        tokenizer.add_tokens([AddedToken('Ġcustom',special=False)])
        for depth in (1,4):
            decoder=decoders.ByteLevel()
            for _ in range(depth):decoder=decoders.Sequence([decoder])
            tokenizer.decoder=decoder
            ids=[tokenizer.token_to_id(piece) for piece in ('Ã','©','Ġcustom','Ã')]
            stream=ByteLevelStream(tokenizer,())
            actual=''.join(stream.feed(token) for token in ids)+stream.finish()
            self.assertEqual(actual,tokenizer.decode(ids,skip_special_tokens=False))
        for decoder in (decoders.Sequence([]),decoders.Sequence([decoders.ByteLevel(),decoders.Replace('a','b')]),
                        decoders.Sequence([decoders.Sequence([decoders.ByteFallback()])])):
            tokenizer.decoder=decoder
            with self.assertRaisesRegex(ValueError,'ByteLevel'):ByteLevelStream(tokenizer,())
