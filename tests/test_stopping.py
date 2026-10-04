import unittest
from dllm.stopping import stop_sequences, text_delta


class StoppingTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(stop_sequences(None), ())
        self.assertEqual(stop_sequences('end'), ('end',))
        for value in ([], '', [''], ['x']*5, ['x'*257], 3, [False]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                stop_sequences(value)

    def test_every_split_of_unicode_stop(self):
        stop = '🦙終わり!'
        for cut in range(len(stop)+1):
            with self.subTest(cut=cut):
                first = 'hello '+stop[:cut]
                delta, emitted, stopped = text_delta(first, '', (stop,))
                self.assertEqual(delta, 'hello ')
                delta, emitted, stopped = text_delta('hello '+stop+' hidden', emitted, (stop,))
                self.assertTrue(stopped)
                self.assertEqual(delta, '')
                self.assertEqual(emitted, 'hello ')

    def test_partial_match_released_on_mismatch_or_length_limit(self):
        self.assertEqual(text_delta('a STO', '', ('STOP',)), ('a ', 'a ', False))
        self.assertEqual(text_delta('a STORK', 'a ', ('STOP',)), ('STORK', 'a STORK', False))
        self.assertEqual(text_delta('a STO', 'a ', ('STOP',), True), ('STO', 'a STO', False))

    def test_earliest_match_and_overlapping_prefixes(self):
        self.assertEqual(text_delta('abcXYZlaterEND', '', ('END','XYZ')), ('abc','abc',True))
        self.assertEqual(text_delta('abcXYZ', '', ('XYZQ','XYZ')), ('abc','abc',True))
        self.assertEqual(text_delta('abcXY', '', ('XYZQ','XYZ')), ('abc','abc',False))

    def test_partial_utf8_and_revision(self):
        self.assertEqual(text_delta('abc\ufffd', 'abc'), ('','abc',False))
        self.assertEqual(text_delta('abc\ufffd', 'abc', final=True), ('\ufffd','abc\ufffd',False))
        with self.assertRaises(RuntimeError):
            text_delta('revised', 'old')
