"""Observe layer outputs in an isolated hybrid replay; timings are not meaningful."""
import argparse
import inspect
import json
import textwrap
import types
import torch
from damendaemon import hybrid
from damendaemon.qwen2 import ModelWeights
from damendaemon.runtime import require_gpu, make_executor


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace')
    p.add_argument('layers')
    p.add_argument('--chunk', type=int, default=0)
    args = p.parse_args()
    if args.chunk < 0:
        p.error('chunk must be nonnegative')
    require_gpu()
    torch.set_num_threads(4)
    trace = torch.load(args.trace, map_location='cpu', weights_only=True)
    reference = torch.load(args.layers, map_location='cpu', weights_only=True)
    if not trace.get('complete', False) or not reference.get('complete', False):
        raise ValueError('completed focused reference required')
    if len(reference['events']) != len(trace['trace']):
        raise ValueError('reference event counts differ')
    model = ModelWeights(trace['path'], trace['context'])
    executor = make_executor(model, use_graph=False,
                             max_batch=max(max(e['slots']) for e in trace['trace']) + 1)
    source = textwrap.dedent(inspect.getsource(hybrid.HybridExecutor.forward_embeddings))
    needle = "        x = x + self.linear(activated, self.w[f'{i}.down'])"
    if source.count(needle) != 1:
        raise ValueError('runtime observation point changed')
    source = source.replace(needle, needle + '\n        self._observe_layer(i, x)')
    namespace = vars(hybrid).copy()
    exec(compile(source, '<diagnostic layer observation>', 'exec'), namespace)
    executor.forward_embeddings = types.MethodType(namespace['forward_embeddings'], executor)
    captured = {}
    executor._observe_layer = lambda index, value: captured.update(
        {index: value[:, -1].detach().float().cpu().clone()})
    comparisons, logits = [], []

    def errors(actual, expected):
        delta = actual.float().cpu() - expected
        return dict(relative_rms=(delta.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-6)).item(),
                    max_absolute=delta.abs().max().item())

    for event, expected in zip(trace['trace'], reference['events']):
        if event['source_event_index'] != expected['source_event_index'] or len(event['slots']) != 1:
            raise ValueError('focused reference ordering differs')
        captured.clear()
        slot = event['slots'][0]
        if event['kind'] == 'prefill':
            prompt = event['prompts'][0]
            if args.chunk:
                for start in range(0, len(prompt), args.chunk):
                    output = executor.prefill_chunk(prompt[start:start+args.chunk], slot, start,
                                                    final=start+args.chunk >= len(prompt))
            else:
                output = executor.prefill(prompt, slot)
        else:
            output, _ = executor.decode_batch(event['tokens'], event['slots'])
        if captured.keys() != expected['layers'].keys():
            raise ValueError('layer observations differ')
        for index, actual in captured.items():
            comparisons.append(dict(source_event_index=event['source_event_index'], layer=index,
                                    kind=model.config['layer_types'][index], **errors(actual, expected['layers'][index])))
        logits.append(dict(source_event_index=event['source_event_index'],
                           greedy_match=output.argmax().item()==event['targets'][0].argmax().item(),
                           **errors(output, event['targets'][0])))
    print(json.dumps(dict(model=trace['path'], chunk=args.chunk, layers=comparisons, logits=logits,
                          limitation='Diagnostic observations synchronize layer outputs; not a performance result.'), indent=2))


if __name__ == '__main__':
    main()
