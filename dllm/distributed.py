"""Experimental two-rank greedy generation with final-stage token selection.

Launched under torchrun. The coordinator owns prompt input; the output worker
receives only bounded control records and activations. This is a single-model,
sequential-request command path, not the HTTP scheduler or a worker supervisor.
"""
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from .config import decoder_config
from .layer_stage import LayerStage
from .partition_policy import validate
from .qwen2 import ModelWeights
from .stage_identity import fingerprint
from .stage_protocol import Command, StageState, PREFILL, DECODE, RELEASE, STOP
from .stage_setup import StageSetup
from .stage_transport import StageChannel


class StagePair:
    def __init__(self, path, context, quantization, prepare_requests):
        self.owns_group = False
        if dist.is_initialized():raise ValueError('distributed-run requires ownership of its process group')
        if quantization not in ('none','q8_0','q4_0'):
            raise ValueError('distributed-run is qualified only for FP16, Q8 or Q4 weights')
        self.state = None
        self.channel = None
        self.stage = None
        self.model = None
        self.rank = int(os.environ.get('RANK', '-1'))
        local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if (os.environ.get('WORLD_SIZE') != '2' or self.rank not in (0, 1)
                or len(visible) not in (1, 2) or len(set(visible)) != len(visible)
                or any(not device.startswith('GPU-') for device in visible)
                or not 0 <= local_rank < len(visible)):
            raise ValueError('distributed-run requires two torchrun ranks and explicit distinct GPU UUIDs')
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(visible):
            raise ValueError('selected CUDA devices are unavailable')
        torch.cuda.set_device(local_rank)
        setup = None
        ready = False
        try:
            setup = StageSetup.from_torchrun(self.rank, session='distributed-run')
            setup.check_peer()
            path = Path(path).expanduser().resolve()
            if not path.is_dir():raise ValueError('distributed-run requires a local safetensors snapshot; use dllm pull first')
            config = decoder_config(json.loads((path/'config.json').read_text()))
            validate(SimpleNamespace(config=config, max_context=context, quantization=quantization), 12, 1)
            self.model = ModelWeights(path, context, quantization=quantization)
            validate(self.model, 12, 1)
            self.requests = prepare_requests(self.model) if self.rank == 0 else None
            setup.check_peer()
            self.identity = fingerprint(self.model)
            self.stage = LayerStage(self.model, 0 if self.rank == 0 else 12,
                                   12 if self.rank == 0 else 24, max_batch=1,
                                   stage_graphs=True, kernels='triton')
            setup.ready(); setup.wait_ready(); ready = True
            self.owns_group = True
            dist.init_process_group('nccl', timeout=timedelta(seconds=30),
                                    device_id=torch.device('cuda', local_rank))
            identity = torch.tensor(list(self.identity), dtype=torch.uint8, device='cuda')
            peers = [torch.empty_like(identity) for _ in range(2)]
            dist.all_gather(peers, identity)
            if not torch.equal(peers[0], peers[1]):raise ValueError('distributed model identity mismatch')
            self.state = StageState(1, context, 1)
            self.channel = StageChannel(self.state, self.rank)
        except BaseException as error:
            if setup is not None and not ready:
                try:setup.fail(error)
                except Exception:pass
            self.close()
            raise

    @torch.inference_mode()
    def step(self, tokens=None):
        request = None
        if self.rank == 0:
            if (not isinstance(tokens, list) or not tokens
                    or any(type(t) is not int or not 0 <= t < self.model.config['vocab_size'] for t in tokens)):
                raise ValueError('invalid distributed token input')
            prefill = self.state.lengths[0] == 0
            if (prefill and len(tokens) > self.state.context) or (not prefill and len(tokens) != 1):
                raise ValueError('invalid distributed token shape')
            request = Command(self.state.epoch, self.state.sequence, PREFILL if prefill else DECODE,
                              len(tokens), ((0, self.state.generations[0] + int(prefill), self.state.lengths[0]),))
        command = self.channel.begin(request)
        if command.operation in (RELEASE, STOP):
            self.channel.finish(command)
            return None
        try:
            prefill = command.operation == PREFILL
            if command.operation not in (PREFILL, DECODE):raise ValueError('unsupported distributed-run command')
            positions = (torch.arange(command.rows, device='cuda') if prefill else
                         torch.tensor([command.entries[0][2]], device='cuda'))
            slot = torch.zeros(1, dtype=torch.int64, device='cuda')
            if self.rank == 0:
                inputs = torch.tensor(tokens, device='cuda')
                hidden = self.stage.run_prefill(inputs, positions, 0) if prefill else self.stage.run_decode(inputs, positions, slot)
                operation = dist.P2POp(dist.isend, hidden, 1)
            else:
                hidden = torch.empty((command.rows, self.model.config['hidden_size']),
                                     dtype=self.stage.residual_dtype, device='cuda')
                operation = dist.P2POp(dist.irecv, hidden, 0)
            for work in dist.batch_isend_irecv([operation]):work.wait()
            if self.rank == 1:
                logits = self.stage.run_prefill(hidden, positions, 0) if prefill else self.stage.run_decode(hidden, positions, slot)
                result = logits.argmax(-1)
            else:
                result = torch.empty(1, dtype=torch.int64, device='cuda')
            self.channel.complete_tokens(command, result)
            return result.item() if self.rank == 0 else None
        except BaseException:
            self.state.poison()
            raise

    def serve_worker(self):
        if self.rank != 1:raise ValueError('only the output rank runs the worker loop')
        while not self.state.closed:self.step()

    def release(self):
        """Fence completed request ownership before the resident slot is reused."""
        if self.rank != 0:raise ValueError('only coordinator releases requests')
        if self.state.failed or self.state.closed or self.state.pending is not None:
            raise RuntimeError('distributed session is not ready for release')
        if not self.state.lengths[0]:return
        command = Command(self.state.epoch, self.state.sequence, RELEASE, 0,
                          ((0, self.state.generations[0], self.state.lengths[0]),))
        self.channel.finish(self.channel.begin(command))

    def stop(self):
        if self.rank != 0 or self.state is None or self.state.closed or self.state.failed:return
        if self.state.pending is not None:
            self.state.poison()
            return
        command = Command(self.state.epoch, self.state.sequence, STOP, 0, ())
        self.channel.finish(self.channel.begin(command))

    def close(self):
        if self.owns_group and dist.is_initialized():dist.destroy_process_group()
        self.owns_group = False
        self.stage = None
        self.channel = None
