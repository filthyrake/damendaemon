import io
import json
import unittest
from scripts.validation_stream import consume_nonterminal_prefix


def wire(*events):
    return io.BytesIO(''.join('data: '+(event if isinstance(event,str) else json.dumps(event))+'\n\n'
                            for event in events).encode())


class CancellationPrefixTests(unittest.TestCase):
    def test_reads_only_requested_nonterminal_prefix(self):
        chunk={'choices':[{'delta':{'content':'a'},'finish_reason':None}]}
        with wire(chunk,chunk,'[DONE]') as response:
            consume_nonterminal_prefix(response)
            self.assertEqual(response.read(),b'data: [DONE]\n\n')

    def test_rejects_terminal_missing_and_invalid_events(self):
        chunk={'choices':[{'delta':{'role':'assistant'}}]}
        for event in ('[DONE]',{'choices':[{'finish_reason':'stop'}]},
                      {'choices':[{'finish_reason':'length'}]}, {'choices':[]},
                      {'choices':[None]},[],{'error':'failed'},'bad JSON'):
            with self.subTest(event=event),wire(chunk,event) as response:
                with self.assertRaises(ValueError):consume_nonterminal_prefix(response)
        with wire(chunk) as response:
            with self.assertRaises(ValueError):consume_nonterminal_prefix(response)

    def test_invalid_count_does_not_read(self):
        for count in (0,-1,True,1.5):
            with wire('[DONE]') as response:
                with self.assertRaises(ValueError):consume_nonterminal_prefix(response,count)
                self.assertEqual(response.tell(),0)
