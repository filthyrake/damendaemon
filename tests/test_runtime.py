from types import SimpleNamespace
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
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
    def test_model_removed_while_session_waits_is_rejected_before_activation(self):
        runtime = Runtime({'a': 'a', 'b': 'b'}, queue_size=1)
        waiting = threading.Event()
        lock = threading.Lock()
        class ObservedLock:
            def acquire(self, **kwargs):
                waiting.set()
                return lock.acquire(**kwargs)
            def release(self):
                lock.release()
        runtime.lock = ObservedLock()
        def request():
            with runtime.session('a'):
                self.fail('removed model was activated')
        lock.acquire()
        with patch.object(runtime, '_activate', wraps=runtime._activate) as activate, ThreadPoolExecutor(1) as pool:
            pending = pool.submit(request)
            try:
                self.assertTrue(waiting.wait(1))
                # The concurrent remover owns the lifetime and registry locks.
                with runtime.registry_lock:
                    del runtime.models['a']
                    del runtime.quantizations['a']
                    del runtime.chat_templates['a']
            finally:
                lock.release()
            with self.assertRaisesRegex(ValueError, 'unknown model'):
                pending.result(2)
            activate.assert_not_called()
        with runtime.session('b') as (executor, _, __):
            self.assertEqual(executor.model.path, 'b')

    def test_invalid_residency_limits_fail_before_gpu_initialization(self):
        invalid = [dict(host_gib=value) for value in
                   (float('nan'), float('inf'), -float('inf'), True, '8', None, 1e-15)]
        invalid += [{name: value} for name in ('context', 'gpu_models', 'host_models')
                    for value in (True, 1.5, 0, -1, None, '2')]
        with patch('dllm.runtime.require_gpu') as initialize:
            for options in invalid:
                with self.subTest(options=options):
                    initialize.reset_mock()
                    with self.assertRaises(ValueError):
                        Runtime({}, **options)
                    initialize.assert_not_called()

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

    def test_close_releases_all_resident_and_host_models_and_is_idempotent(self):
        from unittest.mock import Mock
        runtime = Runtime({'a':'a', 'b':'b'}, gpu_models=2)
        owners = []
        for alias in ('a', 'b'):
            with runtime.session(alias) as (executor, _, __):
                executor.close = Mock()
                owners.append(executor)
        runtime.close()
        runtime.close()
        self.assertTrue(runtime.closed)
        self.assertFalse(runtime.gpu or runtime.host)
        for executor in owners:
            executor.close.assert_called_once_with()
        with self.assertRaises(BusyError):
            with runtime.session('a'):
                pass
        with self.assertRaises(BusyError):
            runtime.manage('load', 'a')
        self.assertTrue(runtime.manage('status')['closed'])

    def test_failed_close_retains_owned_weights_and_allows_retry(self):
        from unittest.mock import Mock
        runtime = Runtime({'a':'a'})
        with runtime.session('a') as (executor, _, __):
            executor.close = Mock(side_effect=RuntimeError('device busy'))
        with self.assertRaisesRegex(RuntimeError, 'device busy'):
            runtime.close()
        self.assertTrue(runtime.closing)
        self.assertFalse(runtime.closed)
        self.assertIs(runtime.gpu['a'], executor)
        self.assertTrue(runtime.host)
        executor.close.side_effect = None
        runtime.close()
        self.assertTrue(runtime.closed)
        self.assertFalse(runtime.gpu or runtime.host)

    def test_shutdown_waits_for_lifetime_lock_and_blocks_new_admission(self):
        runtime = Runtime({'a':'a'}, queue_timeout=.001)
        with runtime.session('a'):
            with self.assertRaisesRegex(BusyError, 'active requests'):
                runtime.close()
            self.assertTrue(runtime.gpu)
            with self.assertRaises(BusyError):
                with runtime.session('a'):
                    pass
        runtime.close()
        self.assertTrue(runtime.closed)

    def test_shutdown_retries_pending_build_cleanup_before_host_release(self):
        from unittest.mock import Mock
        runtime = Runtime({'a':'a'})
        with runtime.session('a'):
            pass
        failure = Mock()
        failure.close.side_effect = RuntimeError('pending build')
        runtime.failed_builds.append(failure)
        with self.assertRaisesRegex(RuntimeError, 'pending build'):
            runtime.close()
        self.assertTrue(runtime.gpu and runtime.host)
        self.assertEqual(runtime.failed_builds, [failure])
        failure.close.side_effect = None
        runtime.close()
        self.assertFalse(runtime.failed_builds or runtime.gpu or runtime.host)

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

    def test_activation_completion_failure_is_owned_but_never_reused(self):
        from unittest.mock import Mock
        runtime = Runtime({'a':'a'})
        with patch('dllm.runtime.synchronize', side_effect=RuntimeError('completion failure')), \
                self.assertRaisesRegex(RuntimeError, 'completion failure'):
            with runtime.session('a'):
                pass
        failed = runtime.gpu['a']
        failed.close = Mock()
        self.assertTrue(runtime.manage('status')['models'][0]['executor_failed'])
        with runtime.session('a') as (replacement, tier, _):
            self.assertIsNot(replacement, failed)
            self.assertEqual(tier, 'host')
        failed.close.assert_called_once_with()
        runtime.close()

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

    def test_failed_rebind_cleanup_blocks_new_construction_until_recovered(self):
        from unittest.mock import Mock
        runtime = Runtime({'a':'a', 'b':'b'})
        with runtime.session('a') as (resident, _, __):
            resident.can_rebind = lambda model: True
            resident.rebind = Mock(side_effect=RuntimeError('partial copy'))
            resident.close = Mock(side_effect=RuntimeError('cleanup still busy'))
        with self.assertRaisesRegex(RuntimeError, 'partial copy') as failure:
            with runtime.session('b'):
                pass
        self.assertFalse(runtime.gpu)
        self.assertIs(runtime.failed_executors[0], resident)
        self.assertTrue(any('cleanup still busy' in note for note in failure.exception.__notes__))
        self.assertEqual(runtime.manage('status')['pending_executor_cleanup'], 1)
        with patch('dllm.runtime.make_executor') as construct:
            with self.assertRaisesRegex(RuntimeError, 'cleanup still busy'):
                with runtime.session('a'):
                    pass
            construct.assert_not_called()
        resident.close.side_effect = None
        with runtime.session('b') as (replacement, tier, _):
            self.assertIsNot(replacement, resident)
            self.assertEqual(tier, 'host')
        runtime.close()

    def test_shutdown_retains_failed_rebind_owner_until_cleanup_completes(self):
        from unittest.mock import Mock
        runtime = Runtime({'a':'a', 'b':'b'})
        with runtime.session('a') as (resident, _, __):
            resident.can_rebind = lambda model: True
            resident.rebind = Mock(side_effect=RuntimeError('partial copy'))
            resident.close = Mock(side_effect=RuntimeError('cleanup busy'))
        with self.assertRaises(RuntimeError):
            with runtime.session('b'):
                pass
        with self.assertRaisesRegex(RuntimeError, 'cleanup busy'):
            runtime.close()
        self.assertFalse(runtime.closed)
        self.assertTrue(runtime.host)
        self.assertIs(runtime.failed_executors[0], resident)
        resident.close.side_effect = None
        runtime.close()
        self.assertTrue(runtime.closed)
        self.assertFalse(runtime.failed_executors or runtime.host)

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

    def test_registration_rejects_non_object_tokenizer_configuration_atomically(self):
        import json
        import tempfile
        from pathlib import Path
        from tokenizers import Tokenizer,models
        runtime=Runtime({})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text('{}')
            Tokenizer(models.WordLevel({'a':0})).save(str(root/'tokenizer.json'))
            for value in (None,[],['chat_template'],'text',1,True):
                with self.subTest(value=value):
                    (root/'tokenizer_config.json').write_text(json.dumps(value))
                    with self.assertRaisesRegex(ValueError,'invalid local model snapshot'):
                        runtime.manage('register','bad',str(root))
                    self.assertEqual(runtime.model_ids(),[])
                    self.assertEqual(runtime.quantizations,{})
                    self.assertEqual(runtime.chat_templates,{})
                    self.assertEqual(runtime.host,{})
                    self.assertEqual(runtime.gpu,{})
            (root/'tokenizer_config.json').write_text('{}')
            runtime.manage('register','good',str(root))
            self.assertEqual(runtime.model_ids(),['good'])

    def test_registration_rejects_broken_checkpoint_templates_before_inventory(self):
        import json
        import tempfile
        from pathlib import Path
        from tokenizers import Tokenizer,models
        cases=[(False,value) for value in ([],{},True,1,'{{','{% if %}')]
        cases += [(True,value) for value in ('{{','{% if %}')]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text('{}')
            Tokenizer(models.WordLevel({'a':0})).save(str(root/'tokenizer.json'))
            for fallback,value in cases:
                with self.subTest(fallback=fallback,value=value):
                    runtime=Runtime({})
                    template=root/'chat_template.jinja';template.unlink(missing_ok=True)
                    (root/'tokenizer_config.json').write_text(json.dumps({} if fallback else {'chat_template':value}))
                    if fallback:template.write_text(value)
                    with patch('dllm.models.weight_bytes',return_value=100) as inventory:
                        with self.assertRaisesRegex(ValueError,'invalid local model snapshot'):
                            runtime.manage('register','bad',str(root))
                        inventory.assert_not_called()
                    self.assertEqual(runtime.model_ids(),[])
                    self.assertEqual(runtime.quantizations,{})
                    self.assertEqual(runtime.chat_templates,{})
                    self.assertEqual(runtime.host,{})
                    self.assertEqual(runtime.gpu,{})
                    runtime.close()
            for fallback,value in ((False,''),(False,'{{ messages[0]["content"] }}'),(True,'{{ messages[0]["content"] }}')):
                runtime=Runtime({});(root/'chat_template.jinja').unlink(missing_ok=True)
                (root/'tokenizer_config.json').write_text(json.dumps({} if fallback else {'chat_template':value}))
                if fallback:(root/'chat_template.jinja').write_text(value)
                runtime.manage('register','valid',str(root));self.assertEqual(runtime.model_ids(),['valid'])
                runtime.close()

    def test_registration_rejects_duplicate_tokenizer_fields_before_inventory(self):
        import tempfile
        from pathlib import Path
        from tokenizers import Tokenizer,models
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text('{}')
            Tokenizer(models.WordLevel({'a':0})).save(str(root/'tokenizer.json'))
            for raw in ('{"chat_template":"first","chat_template":"second"}',
                        '{"added_tokens_decoder":{"0":"first","0":"second"}}'):
                with self.subTest(raw=raw):
                    runtime=Runtime({})
                    (root/'tokenizer_config.json').write_text(raw)
                    with patch('dllm.models.weight_bytes',return_value=100) as inventory:
                        with self.assertRaisesRegex(ValueError,'invalid local model snapshot'):
                            runtime.manage('register','bad',str(root))
                        inventory.assert_not_called()
                    self.assertEqual(runtime.model_ids(),[])
                    runtime.close()

    def test_registration_does_not_compile_unused_chat_templates(self):
        import json
        import tempfile
        from pathlib import Path
        from tokenizers import Tokenizer,models
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text('{}')
            Tokenizer(models.WordLevel({'a':0})).save(str(root/'tokenizer.json'))
            (root/'tokenizer_config.json').write_text(json.dumps({'chat_template':[]}))
            for changes in ({'task':'embedding'},{'task':'rerank'},
                            {'model_type':'mamba'},{'model_type':'mamba2'},{'model_type':'t5'}):
                with self.subTest(changes=changes):
                    runtime=Runtime({})
                    with patch('dllm.models.config',return_value={'max_position_embeddings':2048,**changes}):
                        runtime.manage('register','valid',str(root))
                    self.assertEqual(runtime.model_ids(),['valid']);runtime.close()

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
