import json
import unittest
import torch
from dllm.structured import response_schema, prepare_options
from dllm.sampling import Sampler, Sampling, sample_rows


def format_for(schema):
    return {'type':'json_schema', 'json_schema':{'name':'test','strict':True,'schema':schema}}


class StructuredTests(unittest.TestCase):
    def test_formats_and_nested_subset(self):
        self.assertIsNone(response_schema(None))
        self.assertIsNone(response_schema({'type':'text'}))
        self.assertEqual(json.loads(response_schema({'type':'json_object'}))['type'],'object')
        schema={'type':'object','properties':{'values':{'type':'array','items':{'type':'string','enum':['é','\\n']}}},
                'required':['values'],'additionalProperties':False}
        self.assertEqual(json.loads(response_schema(format_for(schema))),schema)

    def test_unsupported_constraints_and_invalid_schemas_fail_closed(self):
        for schema in ({'type':'string','pattern':'x'}, {'type':'number','minimum':0},
                       {'type':'array'}, {'type':'string','enum':[]}, {'type':'integer','const':True},
                       {'type':'object','required':['absent']}, {'type':'object','required':['a','a']},
                       {'type':'object','additionalProperties':{}}, {'$ref':'#/a'}, {'type':['string','null']}):
            with self.subTest(schema=schema),self.assertRaises(ValueError):
                response_schema(format_for(schema))
        nested={'type':'string'}
        for _ in range(18):
            nested={'type':'array','items':nested}
        with self.assertRaises(ValueError):
            response_schema(format_for(nested))
        with self.assertRaises(ValueError):
            response_schema(format_for({'type':'string','const':'x'*17000}))

    def test_incompatible_options(self):
        fmt={'type':'json_object'}
        for stops,thinking,ignore in ((('x',),None,False),((),True,False),((),None,True)):
            with self.assertRaises(ValueError):
                prepare_options(fmt,stops,thinking,ignore_eos=ignore)
        self.assertIsNone(prepare_options(fmt,(),None)[1])

    def test_mixed_batch_masks_before_greedy_and_sampling(self):
        class Constraint:
            def apply(self, logits):
                return logits.masked_fill(torch.tensor([True,False,True]),-float('inf'))
        for options in (Sampling(),Sampling(.8,top_k=1,seed=17)):
            constrained,plain=Sampler(options),Sampler(Sampling())
            constrained.constraint=Constraint()
            logits=torch.tensor([[10.,1.,0.],[10.,1.,0.]])
            self.assertEqual(sample_rows(logits,[constrained,plain],greedy=torch.tensor([0,0])),[1,0])
            self.assertEqual(logits[0,0].item(),10.)
