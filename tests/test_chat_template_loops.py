"""Bound loop work independently of emitted bytes, using tiny CPU fixtures."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from jinja2 import TemplateError
from dllm.chat_template import compile_template


class TemplateLoopTests(unittest.TestCase):
    def limit(self, count):
        return patch('dllm.chat_template.MAX_TEMPLATE_ITERATIONS', count, create=True)

    def test_silent_nested_and_filtered_loops_are_bounded(self):
        sources = (
            '{% for i in range(10) %}{% for j in range(10) %}{% set ignored = i+j %}{% endfor %}{% endfor %}',
            '{% for i in range(100) if false %}x{% endfor %}',
            '{% for i in range(10) %}{% for j in range(10) if false %}x{% endfor %}{% endfor %}',
            '{% set _loop_step = 0 %}{% for i in range(100) %}{% endfor %}',
        )
        for source in sources:
            for streamed in (False,True):
                with self.subTest(source=source,streamed=streamed),self.limit(20):
                    template=compile_template(source)
                    with self.assertRaisesRegex(TemplateError,'loop iteration'):
                        ''.join(template.generate()) if streamed else template.render()

    def test_recursive_loops_and_rejected_children_share_budget(self):
        chain=[]
        for _ in range(30):chain=[dict(children=chain,keep=True)]
        filtered=[dict(keep=True,children=[dict(keep=False,children=[]) for _ in range(30)])]
        for source,items in (
            ('{% for item in items recursive %}{{ loop(item.children) }}{% endfor %}',chain),
            ('{% for item in items if item.keep recursive %}{{ loop(item.children) }}{% endfor %}',filtered),
        ):
            with self.subTest(source=source),self.limit(20):
                with self.assertRaisesRegex(TemplateError,'loop iteration'):
                    compile_template(source).render(items=items)

    def test_loopcontrols_and_filter_context_are_unchanged(self):
        with self.limit(30):
            template=compile_template('{% for i in range(8) %}{% if i==6 %}{% break %}{% endif %}{% if i%2==0 %}{% continue %}{% endif %}{{i}}{% endfor %}')
            self.assertEqual(template.render(),'135')
            self.assertEqual(compile_template('{% for i in range(6) if i%2 %}{{loop.index}}:{{i}};{% else %}empty{% endfor %}').render(),'1:1;2:3;3:5;')
            self.assertEqual(compile_template('{% for i in [] %}x{% else %}empty{% endfor %}').render(),'empty')

    def test_interleaved_generators_and_close_do_not_share_budget(self):
        with self.limit(3):
            template=compile_template('{% for i in range(3) %}{{i}}{% endfor %}')
            first,second=template.generate(),template.generate()
            self.assertEqual(next(first),'0');self.assertEqual(next(second),'0')
            self.assertEqual(''.join(first),'12');self.assertEqual(''.join(second),'12')
            abandoned=template.generate();self.assertEqual(next(abandoned),'0');abandoned.close()
            self.assertEqual(template.render(),'012')
            exhausted=compile_template('{% for i in range(5) %}{{i}}{% endfor %}').generate()
            with self.assertRaisesRegex(TemplateError,'loop iteration'):
                list(exhausted)
            self.assertEqual(template.render(),'012')

    def test_nested_render_shares_budget_but_parallel_requests_do_not(self):
        with self.limit(5):
            inner=compile_template('{% for i in range(2) %}x{% endfor %}')
            outer=compile_template('{% for i in range(2) %}{{nested()}}{% endfor %}')
            outer.environment.globals['nested']=inner.render
            with self.assertRaisesRegex(TemplateError,'loop iteration'):
                outer.render()
            self.assertEqual(inner.render(),'xx')
        with self.limit(30),ThreadPoolExecutor(max_workers=2) as pool:
            template=compile_template('{% for i in range(30) %}x{% endfor %}')
            self.assertEqual(list(pool.map(lambda _:template.render(),range(4))),['x'*30]*4)
            generator=template.generate();self.assertEqual(next(generator),'x')
            self.assertEqual(pool.submit(next,generator).result(),'x')
            generator.close()
            self.assertEqual(template.render(),'x'*30)

    def test_exhaustion_is_request_error_before_tokenization(self):
        from dllm.qwen2 import ModelWeights
        from dllm.cpu_reference_chat import CpuReferenceChat
        for implementation in (ModelWeights.chat_tokens,CpuReferenceChat.chat_tokens):
            with self.subTest(implementation=implementation.__qualname__),self.limit(20):
                model=SimpleNamespace(config={'model_type':'qwen2','vocab_size':2},special_tokens={},
                    template=compile_template('{% for i in range(30) %}{% endfor %}hello'),tokenizer=Mock())
                with self.assertRaisesRegex(ValueError,'template rendering failed') as raised:
                    implementation(model,[dict(role='user',content='hello')])
                self.assertIsInstance(raised.exception.__cause__,TemplateError)
                model.tokenizer.encode.assert_not_called()
