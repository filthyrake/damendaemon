"""Experimental two-rank greedy generation with final-stage token selection.

Launched under torchrun. The coordinator owns prompt input; the output worker
receives only bounded control records and activations. This is a single-model,
bounded batched command path; the HTTP scheduler lives in distributed_serving.
"""
from datetime import timedelta
import json
import hashlib
import os
import struct
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from .config import decoder_config
from .layer_stage import LayerStage
from .partition_policy import validate, packed_prefill_enabled
from .qwen2 import ModelWeights
from .stage_identity import fingerprint
from .stage_protocol import Command, StageState, PREFILL, DECODE, RELEASE, STOP, PING, VERSION, FAILED_TOKEN, PREFILL_BATCH, token_rows
from .stage_setup import StageSetup
from .stage_transport import StageChannel


class StagePair:
    def __init__(self, path, context, quantization, prepare_requests, session='distributed-run', max_batch=1):
        if type(max_batch) is not int or not 1 <= max_batch <= 4:
            raise ValueError('distributed batch capacity must be 1..4')
        self.max_batch = max_batch
        self.owns_group = False
        if dist.is_initialized():raise ValueError('distributed-run requires ownership of its process group')
        if quantization not in ('none','q8_0','q4_0'):
            raise ValueError('distributed-run is qualified only for FP16, Q8 or Q4 weights')
        self.packed_prefill = packed_prefill_enabled(quantization) and max_batch > 1
        self.state = None
        self.channel = None
        self.stage = None
        self.model = None
        self.failure_tokens = None
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
            setup = StageSetup.from_torchrun(self.rank, session=session)
            setup.check_peer()
            path = Path(path).expanduser().resolve()
            if not path.is_dir():raise ValueError('distributed-run requires a local safetensors snapshot; use dllm pull first')
            config = decoder_config(json.loads((path/'config.json').read_text()))
            validate(SimpleNamespace(config=config, max_context=context, quantization=quantization), 12, max_batch)
            self.model = ModelWeights(path, context, quantization=quantization)
            validate(self.model, 12, max_batch)
            self.requests = prepare_requests(self.model) if self.rank == 0 else None
            setup.check_peer()
            self.identity = fingerprint(self.model)
            self.stage = LayerStage(self.model, 0 if self.rank == 0 else 12,
                                   12 if self.rank == 0 else 24, max_batch=max_batch,
                                   stage_graphs=True, kernels='triton')
            if self.rank == 1:
                self.failure_tokens = torch.full((max_batch,), FAILED_TOKEN, dtype=torch.int64, device='cuda')
            setup.ready(); setup.wait_ready(); ready = True
            self.owns_group = True
            dist.init_process_group('nccl', timeout=timedelta(seconds=30),
                                    device_id=torch.device('cuda', local_rank))
            agreement = self.identity + struct.pack('<qqqqq', VERSION, context, max_batch, int(self.packed_prefill),
                                                     int(self.stage.q4_linear is not None))
            identity = torch.tensor(list(hashlib.sha256(agreement).digest()), dtype=torch.uint8, device='cuda')
            peers = [torch.empty_like(identity) for _ in range(2)]
            dist.all_gather(peers, identity)
            if not torch.equal(peers[0], peers[1]):raise ValueError('distributed model or session bounds mismatch')
            self.state = StageState(1, context, max_batch)
            self.channel = StageChannel(self.state, self.rank)
        except BaseException as error:
            if setup is not None and not ready:
                try:setup.fail(error)
                except Exception:pass
            self.close()
            raise

    def _tokens(self, tokens):
        if (not isinstance(tokens, list) or not tokens
                or any(type(t) is not int or not 0 <= t < self.model.config['vocab_size'] for t in tokens)):
            raise ValueError('invalid distributed token input')

    def prefill(self, tokens, slot=0):
        if self.rank != 0:raise ValueError('only coordinator submits prompts')
        self._tokens(tokens)
        if (type(slot) is not int or not 0 <= slot < self.max_batch
                or self.state.lengths[slot] or len(tokens) > self.state.context):
            raise ValueError('invalid or occupied distributed prefill slot')
        command = Command(self.state.epoch, self.state.sequence, PREFILL, len(tokens),
                          ((slot, self.state.generations[slot] + 1, 0),))
        return self._exchange(command, tokens)[0]

    def prefill_batch(self, prompts, slots):
        if self.rank != 0:raise ValueError('only coordinator submits prompts')
        if (not isinstance(prompts, list) or not prompts or not isinstance(slots, list)
                or len(prompts) != len(slots)
                or any(type(slot) is not int or not 0 <= slot < self.max_batch for slot in slots)
                or len(set(slots)) != len(slots)):
            raise ValueError('invalid packed distributed prompts or slots')
        for prompt in prompts:self._tokens(prompt)
        if sum(map(len,prompts)) > self.state.context or any(self.state.lengths[slot] for slot in slots):
            raise ValueError('packed distributed prompts exceed budget or occupy live slots')
        if len(prompts) == 1:return [self.prefill(prompts[0],slots[0])]
        if not self.packed_prefill:raise ValueError('packed distributed prefill is disabled')
        command = Command(self.state.epoch, self.state.sequence, PREFILL_BATCH, sum(map(len,prompts)),
                          tuple((slot,self.state.generations[slot]+1,len(prompt)) for slot,prompt in zip(slots,prompts)))
        return self._exchange(command,[token for prompt in prompts for token in prompt])

    def decode(self, tokens, slots):
        if self.rank != 0:raise ValueError('only coordinator submits decode')
        self._tokens(tokens)
        if (not isinstance(slots, list) or len(slots) != len(tokens)
                or any(type(slot) is not int or not 0 <= slot < self.max_batch for slot in slots)
                or len(set(slots)) != len(slots)):
            raise ValueError('invalid distributed decode slots')
        command = Command(self.state.epoch, self.state.sequence, DECODE, len(tokens),
                          tuple((slot, self.state.generations[slot], self.state.lengths[slot]) for slot in slots))
        return self._exchange(command, tokens)

    def step(self, tokens=None):
        if self.rank == 1:return self._exchange(None, None)
        if self.state.lengths[0] == 0:return self.prefill(tokens)
        return self.decode(tokens, [0])[0]

    @torch.inference_mode()
    def _exchange(self, request, tokens):
        command = self.channel.begin(request)
        if command.operation in (RELEASE, STOP, PING):
            self.channel.finish(command)
            return None
        try:
            prefill = command.operation == PREFILL
            if command.operation not in (PREFILL, DECODE, PREFILL_BATCH):raise ValueError('unsupported distributed-run command')
            positions = slots = None
            if command.operation != PREFILL_BATCH:
                positions = (torch.arange(command.rows, device='cuda') if prefill else
                             torch.tensor([entry[2] for entry in command.entries], device='cuda'))
                slots = torch.tensor([entry[0] for entry in command.entries], dtype=torch.int64, device='cuda')
            if self.rank == 0:
                inputs = torch.tensor(tokens, device='cuda')
                hidden = self._run_stage(command, inputs, positions, slots)
                operation = dist.P2POp(dist.isend, hidden, 1)
            else:
                hidden = torch.empty((command.rows, self.model.config['hidden_size']),
                                     dtype=self.stage.residual_dtype, device='cuda')
                operation = dist.P2POp(dist.irecv, hidden, 0)
            for work in dist.batch_isend_irecv([operation]):work.wait()
            if self.rank == 1:
                result = self._output_tokens(command, hidden, positions, slots)
            else:
                result = torch.empty(token_rows(command), dtype=torch.int64, device='cuda')
            self.channel.complete_tokens(command, result)
            return self._result_tokens(result) if self.rank == 0 else None
        except BaseException:
            self.state.poison()
            raise

    def _run_stage(self, command, inputs, positions, slots):
        if command.operation == PREFILL_BATCH:
            return self.stage.run_prefill_packed(inputs,[entry[2] for entry in command.entries],
                                                [entry[0] for entry in command.entries])
        if command.operation == PREFILL:return self.stage.run_prefill(inputs,positions,command.entries[0][0])
        return self.stage.run_decode(inputs,positions,slots)

    def _output_tokens(self, command, hidden, positions, slots):
        try:
            logits = self._run_stage(command, hidden, positions, slots)
            return logits.argmax(-1)
        except Exception:
            # The coordinator is already waiting for this fixed-size result.
            # Signal an execution exception on the existing transfer,
            # without adding a collective to every successful decode. A broken
            # CUDA context/transport still requires the process-owner deadline.
            rows = token_rows(command)
            failure = self.failure_tokens[:rows]
            self.channel.complete_tokens(command, failure)
            raise

    def _result_tokens(self, result):
        tokens = result.reshape(-1).tolist()
        if FAILED_TOKEN in tokens:raise RuntimeError('distributed output stage failed')
        if any(not 0 <= token < self.model.config['vocab_size'] for token in tokens):
            raise RuntimeError('invalid distributed output token')
        return tokens

    def serve_worker(self):
        if self.rank != 1:raise ValueError('only the output rank runs the worker loop')
        while not self.state.closed:self.step()

    def release(self, slots=None):
        """Fence completed request ownership before the resident slot is reused."""
        if self.rank != 0:raise ValueError('only coordinator releases requests')
        if self.state.failed or self.state.closed or self.state.pending is not None:
            raise RuntimeError('distributed session is not ready for release')
        slots = [0] if slots is None else slots
        if (not isinstance(slots, list) or not slots
                or any(type(slot) is not int or not 0 <= slot < self.max_batch for slot in slots)
                or len(set(slots)) != len(slots)):
            raise ValueError('invalid distributed release slots')
        slots = [slot for slot in slots if self.state.lengths[slot]]
        if not slots:return
        command = Command(self.state.epoch, self.state.sequence, RELEASE, 0,
                          tuple((slot, self.state.generations[slot], self.state.lengths[slot]) for slot in slots))
        self.channel.finish(self.channel.begin(command))

    def heartbeat(self):
        """Keep an idle resident peer inside the bounded collective deadline."""
        if self.rank != 0:raise ValueError('only coordinator sends heartbeats')
        command = Command(self.state.epoch, self.state.sequence, PING, 0, ())
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
        self.failure_tokens = None
