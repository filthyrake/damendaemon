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
                synchronize=lambda: None, empty_cache=lambda: None)),
            "dllm.models": SimpleNamespace(load=Weights,weight_bytes=lambda _,**kwargs:100,config=lambda _: {"max_position_embeddings":2048}),
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
