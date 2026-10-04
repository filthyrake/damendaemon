"""A2 proof of recurrence, convolution, graph replay, and inactive-slot isolation."""
import json
import torch
from dllm.runtime import require_gpu
from dllm.hybrid_kernels import scan, conv_step, output_head


def reference(q, k, v, decay, beta, state):
    q, k = [a.float()*torch.rsqrt(a.float().square().sum(-1, keepdim=True)+1.e-6) for a in (q, k)]
    repeat = v.shape[2] // q.shape[2]
    q, k = [a.repeat_interleave(repeat, 2) for a in (q, k)]
    q = q * q.shape[-1]**-.5
    output = []
    for t in range(q.shape[1]):
        state = state * decay[:, t, :, None, None].exp()
        residual = (v[:, t].float()-(state*k[:, t, :, :, None]).sum(-2))*beta[:, t, :, None]
        state = state + k[:, t, :, :, None]*residual.unsqueeze(-2)
        output.append((state*q[:, t, :, :, None]).sum(-2))
    return torch.stack(output, 1), state


@torch.inference_mode()
def main():
    require_gpu()
    torch.manual_seed(573)
    checks = []
    for length, kh, vh, kd, vd in ((1, 2, 4, 64, 64), (7, 2, 2, 128, 128),
                                   (65, 2, 4, 128, 128), (257, 2, 2, 128, 128)):
        slots = torch.tensor([3, 1], device='cuda')
        q, k = [torch.randn(2, length, kh, kd, device='cuda', dtype=torch.float16) for _ in range(2)]
        v = torch.randn(2, length, vh, vd, device='cuda', dtype=torch.float16)
        decay = -torch.rand(2, length, vh, device='cuda')
        beta = torch.rand_like(decay)
        initial = torch.randn(4, vh, kd, vd, device='cuda')
        state = initial.clone()
        target, target_state = reference(q, k, v, decay, beta, initial[slots])
        actual = scan(q, k, v, decay, beta, state, slots)
        torch.testing.assert_close(actual.float(), target, atol=.002, rtol=.002)
        torch.testing.assert_close(state[slots], target_state, atol=1.e-5, rtol=1.e-4)
        assert torch.equal(state[[0, 2]], initial[[0, 2]])
        # Reset must ignore arbitrary prior recurrent state.
        target, target_state = reference(q, k, v, decay, beta, torch.zeros_like(initial[slots]))
        actual = scan(q, k, v, decay, beta, state, slots, reset=True)
        torch.testing.assert_close(actual.float(), target, atol=.002, rtol=.002)
        torch.testing.assert_close(state[slots], target_state, atol=1.e-5, rtol=1.e-4)
        if length > 1:
            split = length//2
            state = initial.clone()
            first = scan(q[:, :split], k[:, :split], v[:, :split], decay[:, :split], beta[:, :split], state, slots, True)
            second = scan(q[:, split:], k[:, split:], v[:, split:], decay[:, split:], beta[:, split:], state, slots)
            torch.testing.assert_close(torch.cat((first, second), 1), actual, atol=.002, rtol=.002)
        checks.append({'length': length, 'key_heads': kh, 'value_heads': vh, 'reset_and_isolation': True})
    width, channels = 4, 97
    state = torch.randn(4, channels, width, device='cuda', dtype=torch.float16)
    initial = state.clone()
    weights = torch.randn(channels, 1, width, device='cuda', dtype=torch.float16)
    x = torch.randn(2, channels, device='cuda', dtype=torch.float16)
    slots = torch.tensor([3, 1], device='cuda')
    for _ in range(3):
        conv_step(x, weights, state, slots)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = conv_step(x, weights, state, slots)
    state.copy_(initial)
    for order in ([1, 3], [3, 1]):
        slots.copy_(torch.tensor(order, device='cuda'))
        before = state.clone()
        history = torch.cat((before[slots, :, 1:], x[:, :, None]), -1)
        target = torch.nn.functional.silu((history.float()*weights.squeeze(1).float()).sum(-1))
        graph.replay()
        torch.testing.assert_close(result.float(), target, atol=.004, rtol=.002)
        torch.testing.assert_close(state[slots], history, atol=0, rtol=0)
        assert torch.equal(state[[0, 2]], before[[0, 2]])
    for rows, width, vocabulary in ((1, 1024, 129), (4, 1536, 259)):
        x = torch.randn(rows, width, device='cuda', dtype=torch.float16)
        weight = torch.randn(vocabulary, width, device='cuda', dtype=torch.float16)
        actual = output_head(x, weight)
        target = x.float() @ weight.float().T
        torch.testing.assert_close(actual, target, atol=.0002, rtol=.0002)
    print(json.dumps({'passed': True, 'fp32_output_projection': True, 'scan': checks, 'convolution_graph_reorder': True}, indent=2))


if __name__ == '__main__':
    main()
