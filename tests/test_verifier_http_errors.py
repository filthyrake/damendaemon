"""Execute actual rejection handlers in isolation; never import GPU verifiers."""
import ast
import copy
import io
import json
from pathlib import Path
import unittest
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[1]/'scripts'
STATUS_HANDLERS={
    'verify_embeddings_service':(400,400,400), 'verify_distributed_service':(400,400),
    'verify_completions':(400,), 'verify_harmony_service':(400,),
    'verify_gpt_oss_service':(400,), 'verify_management':(400,),
    'verify_metal_quant_service':(400,), 'verify_model_service':(400,),
    'verify_paged_service':(400,), 'verify_penalties':(400,),
    'verify_partition_service':(500,400), 'verify_service':(400,),
    'verify_t5_service':(400,), 'verify_reranker_service':(400,),
    'verify_structured':(400,), 'verify_tools_service':(400,),
    'verify_vision_service':(400,),
}
DIAGNOSTIC_HANDLERS=('verify_gguf_runtime','verify_speculative_service')


def handlers(name):
    return sorted((node for node in ast.walk(ast.parse((ROOT/(name+'.py')).read_text()))
                   if isinstance(node,ast.ExceptHandler) and isinstance(node.type,ast.Name)
                   and node.type.id=='HTTPError'),key=lambda node:node.lineno)


def compile_handler(handler,optimize):
    # Keep the source handler intact, including its returns and error reads.
    template=ast.parse('def probe(failure):\n    try:\n        raise failure\n    except HTTPError as error:\n        pass\n')
    template.body[0].body[0].handlers=[copy.deepcopy(handler)]
    ast.fix_missing_locations(template)
    namespace={'HTTPError':HTTPError,'json':json,'expected_status':400}
    exec(compile(template,'<actual verifier handler>','exec',optimize=optimize),namespace)
    return namespace['probe']


class VerifierHttpErrorTests(unittest.TestCase):
    def test_expected_and_unexpected_statuses_close_responses(self):
        for name,statuses in STATUS_HANDLERS.items():
            actual=handlers(name)
            self.assertEqual(len(actual),len(statuses),name)
            for index,(handler,expected) in enumerate(zip(actual,statuses)):
                for optimize in (0,2):
                    for status in (expected,503):
                        with self.subTest(name=name,index=index,optimize=optimize,status=status):
                            body=io.BytesIO(b'{"error":"fixture"}')
                            error=HTTPError('http://fixture',status,'fixture',{},body)
                            probe=compile_handler(handler,optimize)
                            try:
                                if status==expected:probe(error)
                                else:
                                    with self.assertRaises((ValueError,AssertionError)):probe(error)
                                self.assertTrue(body.closed,'handler leaked its error response')
                            finally:error.close()

    def test_diagnostic_handlers_close_even_on_decode_failure(self):
        for name in DIAGNOSTIC_HANDLERS:
            for handler in handlers(name):
                for payload in (b'failure',b'\xff'):
                    with self.subTest(name=name,payload=payload):
                        body=io.BytesIO(payload);error=HTTPError('http://fixture',500,'fixture',{},body)
                        try:
                            with self.assertRaises((RuntimeError,UnicodeError)):
                                compile_handler(handler,0)(error)
                            self.assertTrue(body.closed,'diagnostic handler leaked its response')
                        finally:error.close()

    def test_rerank_errors_close_without_reading_response_bodies(self):
        class UnreadBody(io.BytesIO):
            def read(self,*args,**kwargs):raise AssertionError('error response body read')
        actual=handlers('compare_rerank_a2')
        self.assertEqual(len(actual),1)
        for optimize in (0,2):
            with self.subTest(optimize=optimize):
                body=UnreadBody(b'private diagnostic')
                error=HTTPError('http://fixture',503,'fixture',{},body)
                try:
                    with self.assertRaisesRegex(RuntimeError,'^HTTP 503$'):
                        compile_handler(actual[0],optimize)(error)
                    self.assertTrue(body.closed)
                finally:error.close()
