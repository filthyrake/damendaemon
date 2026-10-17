"""Matched control-overhead experiment; final GPU returns token IDs.

Fixed workload vs validated control, off/on/on/off on identical two-worker
stages. Optional comparisons isolate completion or admission overhead. This
measures protocol cost, not HTTP throughput or RDMA capability.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import socket
import time
import torch
import torch.distributed as dist
from damendaemon.qwen2 import ModelWeights
from damendaemon.layer_stage import LayerStage
from damendaemon.partition_policy import validate
from damendaemon.stage_protocol import Command, StageState, PREFILL, DECODE
from damendaemon.stage_transport import StageChannel
from damendaemon.stage_identity import fingerprint
from damendaemon.stage_setup import StageSetup


class ReadbackChannel(StageChannel):
    """Previous coordinator readback, retained only as a matched control."""
    def _received_words(self, sent_words):
        return self.header.cpu().tolist()


def transfer(tensor, sender, rank):
    operation = dist.P2POp(dist.isend if rank == sender else dist.irecv, tensor, 1 - rank)
    for work in dist.batch_isend_irecv([operation]):work.wait()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('model'); p.add_argument('--trace', required=True); p.add_argument('--output', required=True)
    p.add_argument('--repeats', type=int, default=6)
    p.add_argument('--compare-completion', action='store_true')
    p.add_argument('--compare-admission', action='store_true')
    p.add_argument('--placement', choices=('same-host', 'cross-host'), default='same-host')
    args = p.parse_args()
    if args.repeats < 1:p.error('positive repetitions required')
    if args.compare_completion and args.compare_admission:p.error('choose one comparison')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2:p.error('two workers required')
    if rank == 0:
        Path(args.output).write_text(json.dumps(dict(passed=False, final_report=False))+'\n')
    torch.cuda.set_device(local_rank)
    setup = None; setup_finished = False
    try:
        setup = StageSetup.from_torchrun(rank, session='benchmark')
        setup.check_peer()
        model = ModelWeights(args.model, 512, quantization='q8_0'); validate(model, 12, 1)
        setup.check_peer()
        identity = fingerprint(model)
        stage = LayerStage(model, 0 if rank == 0 else 12, 12 if rank == 0 else 24,
                           max_batch=1, stage_graphs=True, kernels='triton')
        setup.ready(); setup.wait_ready(); setup_finished = True
        dist.init_process_group('nccl', timeout=timedelta(seconds=30), device_id=torch.device('cuda', local_rank))
        local = torch.tensor(list(identity), dtype=torch.uint8, device='cuda')
        peers = [torch.empty_like(local) for _ in range(2)]; dist.all_gather(peers, local)
        if not torch.equal(peers[0], peers[1]):raise ValueError('model identity mismatch')
        worker = dict(rank=rank, hostname=socket.gethostname(), device=torch.cuda.get_device_name(),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=str(torch.__version__),
            nccl=list(torch.cuda.nccl.version()), requested_transport=os.environ.get('NCCL_NET', 'auto'),
            network_interface=os.environ.get('NCCL_SOCKET_IFNAME'), model_sha256=identity.hex())
        Path(args.output + f'.worker{rank}.json').write_text(json.dumps(worker, indent=2)+'\n')
        prompt = None
        if rank == 0:
            trace = torch.load(args.trace, map_location='cpu', weights_only=True)
            if trace['context'] != 512 or trace.get('reference_quantization') != 'q8_0':
                raise ValueError('reference precision/context mismatch')
            prompt = next(prompt for event in trace['trace'] if event['kind'] == 'prefill'
                          for prompt in event['prompts'] if len(prompt) == 255)
        groups = []
        expected = None
        state = StageState(1, 512, 1); channel = StageChannel(state, rank)
        modes = ('control', 'token-ack') if args.compare_completion else ('fixed', 'control')
        if args.compare_admission:modes = ('readback', 'local-record')
        for mode in (modes[0], modes[1], modes[1], modes[0]):
            if args.compare_admission:
                channel = (ReadbackChannel if mode == 'readback' else StageChannel)(state, rank)
            control = mode != 'fixed'
            runs = []
            for iteration in range(args.repeats + 2):
                dist.barrier(); torch.cuda.synchronize()
                started = time.perf_counter(); first = None; generated = []
                for index in range(32):
                    prefill = index == 0
                    rows = 255 if prefill else 1
                    position = 0 if prefill else 254 + index
                    command = None
                    if control:
                        request = None
                        if rank == 0:
                            generation = state.generations[0] + int(prefill)
                            request = Command(state.epoch, state.sequence, PREFILL if prefill else DECODE,
                                              rows, ((0, generation, position),))
                        command = channel.begin(request)
                    positions = torch.arange(rows, device='cuda') if prefill else torch.tensor([position], device='cuda')
                    slots = torch.zeros(1, dtype=torch.int64, device='cuda')
                    if rank == 0:
                        inputs = torch.tensor(prompt if prefill else [generated[-1]], device='cuda')
                        hidden = stage.run_prefill(inputs, positions, 0) if prefill else stage.run_decode(inputs, positions, slots)
                    else:
                        hidden = torch.empty((rows, model.config['hidden_size']), dtype=stage.residual_dtype, device='cuda')
                    transfer(hidden, 0, rank)
                    if rank == 1:
                        logits = stage.run_prefill(hidden, positions, 0) if prefill else stage.run_decode(hidden, positions, slots)
                        token = logits.argmax(-1)
                    else:
                        token = torch.empty(1, dtype=torch.int64, device='cuda')
                    if mode in ('token-ack', 'readback', 'local-record'):
                        channel.complete_tokens(command, token)
                    else:
                        transfer(token, 1, rank)
                        if control:channel.finish(command)
                    if rank == 0:generated.append(token.item())
                    if prefill:first = time.perf_counter()
                torch.cuda.synchronize(); finished = time.perf_counter()
                if rank == 0:
                    if expected is None:expected = generated
                    if generated != expected:raise AssertionError('token output changed between protocol modes')
                    if iteration >= 2:
                        runs.append(dict(first_token_seconds=first-started, total_seconds=finished-started,
                            decode_tokens_per_second=31/(finished-first), tokens=generated))
            if rank == 0:groups.append(dict(mode=mode, runs=runs))
        if rank == 0:
            summaries = []
            for mode in modes:
                runs = [row for group in groups if group['mode'] == mode for row in group['runs']]
                summaries.append(dict(mode=mode, **{key:statistics.median(r[key] for r in runs)
                    for key in ('first_token_seconds','total_seconds','decode_tokens_per_second')}))
            report = dict(passed=True, final_report=True, model=args.model, model_sha256=identity.hex(), quantization='q8_0',
                context=512, prompt_tokens=255, output_tokens=32, batch=1, torch=str(torch.__version__),
                scope=f'{args.placement} two-worker protocol overhead ABBA; token-only return; not HTTP or RDMA',
                groups=groups, summaries=summaries)
            Path(args.output).write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(summaries))
    except BaseException as error:
        if setup is not None and not setup_finished:
            try:setup.fail(error)
            except Exception:pass  # Preserve the original failure if the store disappeared.
        if rank == 0:
            Path(args.output).write_text(json.dumps(dict(passed=False, final_report=True, error=repr(error)))+'\n')
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__ == '__main__':main()
