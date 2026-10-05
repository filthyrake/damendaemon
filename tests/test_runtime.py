from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dllm.runtime import Runtime, BusyError


class Weights:
    def __init__(self, path, context, quantization=None):
        self.path, self.bytes = path, 100
        self.config = {"model_type": "qwen2"}


class Executor:
    def __init__(self, model, use_graph):
        if model.path == "broken":
            raise RuntimeError("load failed")
        self.model = model


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.gpu = patch("dllm.runtime.require_gpu")
        self.gpu.start()
        self.modules = patch.dict("sys.modules", {
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                synchronize=lambda: None, empty_cache=lambda: None,is_initialized=lambda:False)),
            "dllm.models": SimpleNamespace(load=Weights,is_gguf=lambda _:False,weight_bytes=lambda _,**kwargs:100,config=lambda _: {"max_position_embeddings":2048}),
            "dllm.qwen2": SimpleNamespace(ModelWeights=Weights, Executor=Executor,
                                          weight_bytes=lambda _: 100),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(self.gpu.stop)

    def test_a_b_a_host_restoration(self):
        runtime = Runtime({"a": "a", "b": "b"}, gpu_models=1)
        tiers = []
        for alias in ("a", "b", "a"):
            with runtime.session(alias) as (executor, tier, _):
                tiers.append(tier)
                self.assertEqual(executor.model.path, alias)
        self.assertEqual(tiers, ["disk", "disk", "host"])
        self.assertEqual(list(runtime.gpu), ["a"])
        self.assertEqual(list(runtime.host), ["b", "a"])

    def test_resident_selection_avoids_reload(self):
        runtime = Runtime({"a": "a", "b": "b"}, gpu_models=2)
        for alias in ("a", "b"):
            with runtime.session(alias):
                pass
        with runtime.session("a") as (_, tier, _):
            self.assertEqual(tier, "gpu")

    def test_failed_executor_is_closed_and_rebuilt_from_host(self):
        from unittest.mock import Mock
        runtime=Runtime({'a':'a'})
        with runtime.session('a') as (first,_,__):
            first.failed=True
            first.close=Mock()
        with runtime.session('a') as (replacement,tier,_):
            self.assertIsNot(replacement,first)
            self.assertIs(replacement.model,first.model)
            self.assertEqual(tier,'host')
        first.close.assert_called_once_with()

    def test_rebind_moves_residency_only_after_success_and_drops_partial_failure(self):
        from unittest.mock import Mock
        runtime=Runtime({'a':'a','b':'b'})
        with runtime.session('a') as (resident,_,__):pass
        resident.can_rebind=lambda model:True
        resident.close=Mock()
        resident.rebind=Mock(side_effect=lambda model:setattr(resident,'model',model))
        with runtime.session('b') as (value,tier,_):
            self.assertIs(value,resident);self.assertEqual(tier,'disk')
            self.assertEqual(value.model.path,'b')
        self.assertEqual(list(runtime.gpu),['b'])
        with runtime.session('a') as (value,tier,_):
            self.assertIs(value,resident);self.assertEqual(tier,'host')
            self.assertEqual(value.model.path,'a')
        resident.rebind.side_effect=RuntimeError('partial copy failure')
        with self.assertRaisesRegex(RuntimeError,'partial copy'):
            with runtime.session('b'):pass
        self.assertFalse(runtime.gpu)
        resident.close.assert_called_once_with()
        with runtime.session('b') as (value,tier,_):
            self.assertIsNot(value,resident);self.assertEqual(tier,'host')

    def test_non_rebindable_executor_is_released_before_replacement_allocation(self):
        import weakref
        references=[]
        def create(model,*args):
            if references:self.assertIsNone(references[-1]())
            executor=Executor(model,False)
            references.append(weakref.ref(executor))
            return executor
        runtime=Runtime({'a':'a','b':'b'},gpu_models=1)
        with patch('dllm.runtime.make_executor',side_effect=create):
            with runtime.session('a'):pass
            with runtime.session('b'):pass

    def test_failed_load_does_not_publish_or_leak_admission(self):
        runtime = Runtime({"bad": "broken", "a": "a"}, queue_size=1)
        with self.assertRaises(RuntimeError):
            with runtime.session("bad"):
                pass
        self.assertNotIn("bad", runtime.gpu)
        with runtime.session("a"):
            pass

    def test_exception_releases_lease(self):
        runtime = Runtime({"a": "a"}, queue_size=1)
        with self.assertRaises(ValueError):
            with runtime.session("a"):
                raise ValueError("disconnected")
        with runtime.session("a"):
            pass

    def test_queue_is_bounded(self):
        runtime = Runtime({"a": "a"}, queue_size=1)
        with runtime.session("a"):
            with self.assertRaises(BusyError):
                with runtime.session("a"):
                    pass

    def test_host_budget_checked_before_loading(self):
        runtime = Runtime({"a": "a"}, host_gib=1 / 1024**3)
        with self.assertRaises(MemoryError):
            with runtime.session("a"):
                pass
        self.assertEqual(len(runtime.host), 0)

    def test_host_eviction_removes_gpu_executor(self):
        runtime = Runtime({"a": "a", "b": "b"}, host_models=1)
        for alias in ("a", "b"):
            with runtime.session(alias):
                pass
        self.assertEqual(list(runtime.host), ["b"])
        self.assertEqual(list(runtime.gpu), ["b"])

    def test_gpu_hit_refreshes_host_lru(self):
        runtime = Runtime({"a": "a", "b": "b", "c": "c"}, gpu_models=2, host_models=2)
        for alias in ("a", "b", "a", "c"):
            with runtime.session(alias):
                pass
        self.assertEqual(set(runtime.host), {"a", "c"})
        self.assertEqual(set(runtime.gpu), {"a", "c"})


    def test_model_management_residency_and_idempotent_unload(self):
        runtime=Runtime({'a':'a'})
        self.assertEqual(runtime.manage('status')['models'][0]['residency'],'disk')
        self.assertEqual(runtime.manage('load','a')['source_tier'],'disk')
        self.assertEqual(runtime.manage('load','a')['source_tier'],'gpu')
        self.assertEqual(runtime.manage('unload','a')['models'][0]['residency'],'host')
        self.assertEqual(runtime.manage('load','a')['source_tier'],'host')
        for _ in range(2):
            self.assertEqual(runtime.manage('unload','a',release_host=True)['models'][0]['residency'],'disk')
        self.assertEqual(runtime.model_ids(),['a'])
        with self.assertRaises(ValueError):runtime.manage('unload','unknown')

    def test_empty_registry_and_atomic_registration(self):
        import json
        import tempfile
        from pathlib import Path
        from tokenizers import Tokenizer,models
        runtime=Runtime({})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text('{}')
            (root/'tokenizer_config.json').write_text('{}')
            Tokenizer(models.WordLevel({'a':0})).save(str(root/'tokenizer.json'))
            with patch('dllm.models.config',return_value={'max_position_embeddings':2048}):
                for _ in range(2):
                    self.assertEqual(runtime.manage('register','a',str(root))['models'][0]['id'],'a')
                with self.assertRaises(ValueError):runtime.manage('register','a',str(root/'other'))
                with self.assertRaises(ValueError):runtime.manage('register','bad',str(root/'missing'))
            self.assertEqual(runtime.model_ids(),['a'])
            self.assertEqual(runtime.host,{})
            self.assertEqual(runtime.gpu,{})

    def test_model_quantization_is_stable_and_used_for_reload(self):
        import tempfile
        from pathlib import Path
        from tokenizers import Tokenizer,models
        with patch.dict('os.environ',{'DLLM_QUANTIZATION':'q8_0'}):
            runtime=Runtime({'initial':'initial'})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text('{}')
            (root/'tokenizer_config.json').write_text('{}')
            Tokenizer(models.WordLevel({'a':0})).save(str(root/'tokenizer.json'))
            runtime.manage('register','plain',str(root),quantization='none')
            runtime.manage('register','packed',str(root),quantization='q8_0')
            with self.assertRaises(ValueError):
                runtime.manage('register','plain',str(root),quantization='q8_0')
            self.assertEqual(runtime.quantizations,{'initial':'q8_0','plain':'none','packed':'q8_0'})
            with patch('dllm.models.load',side_effect=Weights) as load:
                runtime.manage('load','plain')
                self.assertEqual(load.call_args.kwargs,{'quantization':'none'})
                runtime.manage('unload','plain',release_host=True)
                runtime.manage('load','plain')
                self.assertEqual(load.call_args.kwargs,{'quantization':'none'})
                runtime.manage('load','packed')
                self.assertEqual(load.call_args.kwargs,{'quantization':'q8_0'})

    def test_template_identity_atomicity_and_reload(self):
        import tempfile
        from pathlib import Path
        from tokenizers import Tokenizer, models
        from dllm.chat_template import compile_template, override_metadata
        template = 'custom {{ messages }}'
        def configure(model, tokenizer, text, config):
            model.template = compile_template(text)
        def load(path, context, quantization=None):
            model = Weights(path, context, quantization)
            model.tokenizer, model.special_tokens = object(), {'eos_token':'end'}
            model.configure_tokenizer = lambda *args: configure(model, *args)
            return model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tokenizer_config.json').write_text('{}')
            Tokenizer(models.WordLevel({'a':0})).save(str(root/'tokenizer.json'))
            runtime = Runtime({'other':'other'})
            for bad in ('{{', ''):
                with self.assertRaises(ValueError):
                    runtime.manage('register', 'new', str(root), chat_template=bad)
                self.assertNotIn('new', runtime.models)
            with patch('dllm.models.config', return_value={'model_type':'t5'}), self.assertRaises(ValueError):
                runtime.manage('register', 'new', str(root), chat_template=template)
            self.assertNotIn('new', runtime.models)
            for _ in range(2):runtime.manage('register', 'new', str(root), chat_template=template)
            for different in (None, 'another {{ messages }}'):
                with self.assertRaises(ValueError):
                    runtime.manage('register', 'new', str(root), chat_template=different)
            status = runtime.manage('status')['models'][1]
            self.assertEqual(status['chat_template'], override_metadata(template))
            with patch('dllm.models.load', side_effect=load) as loader:
                for alias, expected in (('new','disk'), ('other','disk'), ('new','host')):
                    self.assertEqual(runtime.manage('load', alias)['source_tier'], expected)
                self.assertEqual(loader.call_count, 2)
                self.assertEqual(runtime.host['new'].template.render(messages='hello'), 'custom hello')
                runtime.manage('unload', 'new', release_host=True)
                self.assertEqual(runtime.manage('load', 'new')['source_tier'], 'disk')
                self.assertEqual(loader.call_count, 3)
                self.assertEqual(runtime.host['new'].template.render(messages='hello'), 'custom hello')
            with self.assertRaises(ValueError):Runtime({}, chat_templates={'missing':template})
            configured = Runtime({'new':root}, chat_templates={'new':template})
            self.assertEqual(configured.chat_templates, {'new':template})
