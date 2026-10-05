"""Real NCCL proof of coordinator-driven bounded stage commands.

Only rank zero reads the trusted reference trace. Rank one receives fixed int64
control records and typed activations. Full logits return solely for independent
validation; this is not a serving transport or a performance benchmark.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from dllm.qwen2 import ModelWeights
from dllm.layer_stage import LayerStage
from dllm.partition_policy import validate
from dllm.stage_protocol import Command, StageState, PREFILL, DECODE, RELEASE, STOP

from dllm.stage_transport import StageChannel
from dllm.stage_identity import fingerprint

def commands(trace, state):
    for event in trace['trace']:
        if event['kind'] == 'prefill':
            for tokens, slot, target in zip(event['prompts'], event['slots'], event['targets']):
                if state.lengths[slot]:
                    yield RELEASE, [], [(slot, state.generations[slot], state.lengths[slot])], []
                yield PREFILL, tokens, [(slot, state.generations[slot] + 1, 0)], [target]
        elif event['kind'] == 'decode':
            yield DECODE, event['tokens'], [(s, state.generations[s], state.lengths[s])
                                           for s in event['slots']], event['targets']
        else:
            raise ValueError('unknown reference event')
    yield STOP, [], [], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--trace', required=True, help='opened only by coordinator rank zero')
    parser.add_argument('--quantization', choices=('none', 'q8_0'), default='none')
    parser.add_argument('--output', required=True)
    parser.add_argument('--inject-stale-sequence', action='store_true')
    parser.add_argument('--token-completion', action='store_true')
    args = parser.parse_args()
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2:parser.error('exactly two workers required')
    torch.cuda.set_device(local_rank)
    report = dict(passed=False, final_report=False, rank=rank, checks=[], commands=0,
        device=torch.cuda.get_device_name(), quantization=args.quantization,
        scope='coordinator-driven NCCL control correctness; not serving or performance',
        reads_reference_trace=rank == 0, token_completion=args.token_completion)
    output = Path(args.output.format(rank=rank))
    output.write_text(json.dumps(report, indent=2)+'\n')
    state = StageState(1, 512)
    try:
        model = ModelWeights(args.model, 512, quantization=args.quantization)
        validate(model, 12, 4)
        identity = fingerprint(model)
        stage = LayerStage(model, 0 if rank == 0 else 12, 12 if rank == 0 else 24,
                           max_batch=4, stage_graphs=True, kernels='triton')
        dist.init_process_group('nccl', timeout=timedelta(seconds=30),
                                device_id=torch.device('cuda', local_rank))
        local = torch.tensor(list(identity), dtype=torch.uint8, device='cuda')
        peers = [torch.empty_like(local) for _ in range(2)]
        dist.all_gather(peers, local)
        if not torch.equal(peers[0], peers[1]):raise ValueError('worker model identity mismatch')
        report['model_sha256'] = identity.hex()
        channel = StageChannel(state, rank)
        source = None
        if rank == 0:
            trace = torch.load(args.trace, map_location='cpu', weights_only=True)
            if trace['context'] != 512 or trace.get('reference_quantization', 'none') != args.quantization:
                raise ValueError('reference policy mismatch')
            source = iter(commands(trace, state))
        while True:
            tokens, targets = [], []
            if rank == 0:
                operation, tokens, entries, targets = next(source)
                sequence = state.sequence
                if args.inject_stale_sequence and sequence == 3:sequence -= 1
                request = Command(state.epoch, sequence, operation, len(tokens), tuple(entries))
            else:
                request = None
            command = channel.begin(request)
            slots = [entry[0] for entry in command.entries]
            if command.operation in (PREFILL, DECODE):
                positions = (torch.arange(command.rows, device='cuda') if command.operation == PREFILL else
                             torch.tensor([e[2] for e in command.entries], device='cuda'))
                slot_tensor = torch.tensor(slots, device='cuda')
                if rank == 0:
                    inputs = torch.tensor(tokens, device='cuda')
                    hidden = (stage.run_prefill(inputs, positions, slots[0]) if command.operation == PREFILL else
                              stage.run_decode(inputs, positions, slot_tensor))
                    operation = dist.P2POp(dist.isend, hidden, 1)
                else:
                    hidden = torch.empty((command.rows, model.config['hidden_size']),
                                         dtype=stage.residual_dtype, device='cuda')
                    operation = dist.P2POp(dist.irecv, hidden, 0)
                for work in dist.batch_isend_irecv([operation]):work.wait()
                if rank == 1:
                    logits = (stage.run_prefill(hidden, positions, slots[0]) if command.operation == PREFILL else
                              stage.run_decode(hidden, positions, slot_tensor))
                    operation = dist.P2POp(dist.isend, logits, 0)
                else:
                    rows = 1 if command.operation == PREFILL else command.rows
                    logits = torch.empty((rows, model.config['vocab_size']), dtype=stage.residual_dtype, device='cuda')
                    operation = dist.P2POp(dist.irecv, logits, 1)
                for work in dist.batch_isend_irecv([operation]):work.wait()
                if rank == 0:
                    actual = logits.float().cpu()
                    for row, target in enumerate(targets):
                        target = target.squeeze(0); error = actual[row] - target
                        relative = (error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1e-6)).item()
                        absolute = error.abs().max().item()
                        match = actual[row].argmax().item() == target.argmax().item()
                        report['checks'].append(dict(operation=command.operation, slot=slots[row],
                            relative_rms=relative, max_absolute=absolute, greedy_match=match,
                            passed=relative < .01 and absolute < .5 and match))
            if args.token_completion and command.operation in (PREFILL, DECODE):
                rows = 1 if command.operation == PREFILL else command.rows
                returned = logits.argmax(-1) if rank == 1 else torch.empty(rows, dtype=torch.int64, device='cuda')
                channel.complete_tokens(command, returned)
                if rank == 0 and not torch.equal(returned, logits.argmax(-1)):
                    raise AssertionError('returned tokens differ from final-stage logits')
            else:
                channel.finish(command)
            report['commands'] += 1
            if state.closed:break
        passed = rank == 1 or bool(report['checks']) and all(c['passed'] for c in report['checks'])
        result = torch.tensor([int(passed)], device='cuda')
        dist.all_reduce(result, op=dist.ReduceOp.MIN)
        report['passed'] = bool(result.item())
        if not report['passed']:raise AssertionError('independent stage protocol reference gate failed')
    except BaseException as error:
        state.poison()
        report['error'] = repr(error)
        raise
    finally:
        report['epoch_failed'] = state.failed
        report['final_report'] = True
        output.write_text(json.dumps(report, indent=2)+'\n')
        if dist.is_initialized():dist.destroy_process_group()


if __name__ == '__main__':main()
