"""Experimental grouped Q4 experts; dequantize tiles without full weight expansion."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _linear(X, W, S, ORDER, COUNTS, STARTS, TILE_ENDS, Y,
            E:tl.constexpr, TOP:tl.constexpr, N:tl.constexpr, K:tl.constexpr,
            SHARED:tl.constexpr, BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr, BE:tl.constexpr):
    tile = tl.program_id(0)
    experts = tl.arange(0,BE)
    ends = tl.load(TILE_ENDS+experts,experts<E,other=2147483647)
    expert = tl.sum((tile>=ends).to(tl.int32),0)
    if expert<E:
        previous = tl.load(TILE_ENDS+expert-1,expert>0,other=0)
        rows = (tile-previous)*BM+tl.arange(0,BM)
        count = tl.load(COUNTS+expert)
        start = tl.load(STARTS+expert)
        route = tl.load(ORDER+start+rows,rows<count,other=0)
        source = route//TOP if SHARED else route
        outputs = tl.program_id(1)*BN+tl.arange(0,BN)
        columns = tl.arange(0,BK)
        accumulator = tl.full((BM,BN),0.,tl.float32)
        for step in range(tl.cdiv(K,BK)):
            k = step*BK+columns
            x = tl.load(X+source[:,None]*K+k[None,:],
                        (rows[:,None]<count)&(k[None,:]<K),other=0).to(tl.float32)
            packed = tl.load(W+(expert*N+outputs[None,:])*(K//2)
                             +(k[:,None]//32)*16+k[:,None]%16,
                             (outputs[None,:]<N)&(k[:,None]<K),other=0).to(tl.uint32)
            values = ((packed >> ((k[:,None]%32//16)*4)) & 15).to(tl.float32)-8
            scales = tl.load(S+(expert*N+outputs[None,:])*(K//32)+k[:,None]//32,
                             (outputs[None,:]<N)&(k[:,None]<K),other=0).to(tl.float32)
            accumulator = tl.dot(x,values*scales,accumulator,input_precision='tf32x3')
        tl.store(Y+route[:,None]*N+outputs[None,:],accumulator,
                 (rows[:,None]<count)&(outputs[None,:]<N))


def linear(x, weight, scales, routes, outputs, shared_input):
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.float32 or x.shape[-1]%32:
        raise ValueError('CUDA FP32 inputs aligned to Q4 groups required')
    if (weight.shape != (routes.experts*outputs,x.shape[-1]//2) or weight.dtype != torch.uint8
            or scales.shape != (weight.shape[0],x.shape[-1]//32) or scales.dtype != torch.float16
            or not weight.is_contiguous() or not scales.is_contiguous()
            or any(value.device != x.device for value in (weight,scales,routes.order))):
        raise ValueError('invalid grouped Q4 matrix storage')
    expected_rows = routes.rows//routes.top if shared_input else routes.rows
    if x.shape[0] != expected_rows:
        raise ValueError('input rows disagree with expert routes')
    x = x.contiguous()
    result = torch.empty((routes.rows,outputs),device=x.device,dtype=x.dtype)
    _linear[(routes.tiles,tr.cdiv(outputs,64))](x,weight,scales,routes.order,routes.counts,
        routes.starts,routes.tile_ends,result,routes.experts,routes.top,outputs,x.shape[-1],
        shared_input,routes.block_rows,64,64,tr.next_power_of_2(routes.experts),num_warps=4,num_stages=2)
    return result


def install_validation_dispatch():
    """Opt-in model experiment, installed after the Q4 construction policy."""
    import hashlib
    from pathlib import Path
    from damendaemon.moe import OlmoeExecutor
    from damendaemon.grouped_moe import prepare_routes
    if torch.cuda.get_device_name() != 'NVIDIA A2':
        raise ValueError('grouped Q4 model experiment is limited to A2')
    original = OlmoeExecutor.experts
    report = dict(kind='experimental grouped Q4 expert prefill', selected_calls=0,
                  fallback_calls=0, workspace_limit_bytes=256*1024**2,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def experts(self, hidden, layer, prefill):
        c = self.c
        up, down = self.w[f'{layer}.experts_up'], self.w[f'{layer}.experts_down']
        rows = hidden.shape[0]*c['num_experts_per_tok']
        scratch = rows*max(3*c['intermediate_size'], c['intermediate_size']+c['hidden_size'])*4 + 16*1024**2
        if (not prefill or hidden.dtype != torch.float32 or up.dtype != torch.uint8
                or down.dtype != torch.uint8 or not 1 <= hidden.shape[0] <= 2048
                or c['num_experts'] > 256 or scratch > report['workspace_limit_bytes']):
            report['fallback_calls'] += 1
            return original(self, hidden, layer, prefill)
        report['selected_calls'] += 1
        probabilities, selected = self._linear(hidden,self.w[f'{layer}.router']).softmax(-1).topk(
            c['num_experts_per_tok'],dim=-1)
        if c['norm_topk_prob']:
            probabilities = probabilities/probabilities.sum(-1,keepdim=True)
        routes = prepare_routes(selected,c['num_experts'])
        packed = linear(hidden,up,self._expert_scales(up),routes,2*c['intermediate_size'],True)
        activated = self.kernels.swiglu(packed,True)
        del packed
        projected = linear(activated,down,self._expert_scales(down),routes,c['hidden_size'],False)
        del activated
        return self.expert_kernels.combine(projected,probabilities)

    OlmoeExecutor.experts = experts
    return report
