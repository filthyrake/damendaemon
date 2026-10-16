import json
from pathlib import Path
import tempfile
import unittest
from dllm.cpu_reference_chat import CpuReferenceChat
from tests.test_cpu_hybrid_checkpoint import snapshot
from tests.test_cpu_pipeline_generation import tokenizer


class CpuReferenceChatTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);snapshot(self.path);tokenizer(self.path)
        self.config=json.loads((self.path/'config.json').read_text())
        self.template=self.path/'override.jinja'
        self.template.write_text("{% for m in messages %}{{ m.content }} {% endfor %}{% if enable_thinking %}w3 {% endif %}{% if add_generation_prompt %}w4{% endif %}")

    def text(self):return CpuReferenceChat(self.path,self.config,self.template)

    def test_template_matches_independent_tokens_and_thinking_options(self):
        text=self.text();messages=[{'role':'user','content':'hello world'}]
        self.assertEqual(text.chat_tokens(messages),[1,2,4])
        self.assertEqual(text.chat_tokens(messages,enable_thinking=False),[1,2,4])
        self.assertEqual(text.chat_tokens(messages,enable_thinking=True),[1,2,3,4])
        self.assertEqual(messages,[{'role':'user','content':'hello world'}])

    def test_rejects_nontext_history_and_bounds(self):
        text=self.text()
        for messages in ([],[{'role':'tool','content':'hello'}],[{'role':'user','content':[]}],
                         [{'role':'assistant','content':'hello','reasoning_content':'world'}],
                         [{'role':'user','content':'x'*(1024**2+1)}],
                         [{'role':'user','content':''}]*257):
            with self.subTest(messages=str(messages)[:80]),self.assertRaises(ValueError):text.chat_tokens(messages)
        with self.assertRaises(ValueError):text.chat_tokens([{'role':'user','content':'hello'}],enable_thinking=1)
        self.template.write_text("{% for i in range(20) %}{{ messages[0].content }}{% endfor %}")
        with self.assertRaisesRegex(ValueError,'rendered prompt'):self.text().chat_tokens([{'role':'user','content':'x'*100000}])

    def test_identity_binds_template_and_special_tokens(self):
        original=self.text().chat_identity
        self.template.write_text('hello {{ bos_token }}')
        changed=self.text().chat_identity;self.assertNotEqual(original,changed)
        metadata=self.path/'tokenizer_config.json'
        metadata.write_text(json.dumps({'bos_token':{'content':'world'}}))
        self.assertNotEqual(changed,self.text().chat_identity)
        self.assertEqual(self.text().chat_tokens([{'role':'user','content':'hello'}]),[1,2])
        for token in (12,{},'unmapped'):
            metadata.write_text(json.dumps({'bos_token':token}))
            with self.assertRaises(ValueError):self.text()

    def test_bad_template_metadata_fails_during_preparation(self):
        for source in ('{%','x'*(128*1024+1),''):
            self.template.write_text(source)
            with self.assertRaises(ValueError):self.text()

    def test_empty_render_chunks_do_not_accumulate_objects(self):
        import weakref
        from types import SimpleNamespace
        class Chunk(str):pass
        refs=[];closed=[]
        def generate(**options):
            try:
                for index in range(20):
                    if index>=2:
                        self.assertIsNone(refs[index-2](),'renderer retains earlier empty chunks')
                    chunk=Chunk('')
                    refs.append(weakref.ref(chunk))
                    yield chunk
                yield 'hello'
            finally:closed.append(True)
        text=self.text();text.template=SimpleNamespace(generate=generate)
        self.assertEqual(text.chat_tokens([{'role':'user','content':'hello'}]),[1])
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(closed,[True])
