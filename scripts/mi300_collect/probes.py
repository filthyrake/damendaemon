#!/usr/bin/env python
"""MI300X capability probes for damendaemon #62 (MI300-class qual) and #719
(CUDA+ROCm fleet split). Single-GPU, bounded, evidence-oriented:

  cap sections: torch/ROCm env, device+arch props, dtype + FP8 variant
  semantics (fn vs fnuz), torch._scaled_mm support matrix, a representative
  Triton q4-dequant kernel (compile, correctness vs reference, .amdgcn ISA
  dump), optional --bench: time-boxed D2D bandwidth + bf16/fp8 GEMM.

Writes probe_results.json INCREMENTALLY so partial evidence survives a
crash or timeout. Allocates memory only on --gpu (default 0). Nothing here
changes system state. Run via collect.sh, or directly:
  python probes.py cap --gpu 0 --out <dir> [--bench]
"""
import argparse, json, os, shutil, sys, time


def _w(store, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(store, f, indent=1, default=str)
    os.replace(tmp, path)


def section(store, outpath, name, fn):
    print(f'[.] {name} ...', flush=True)
    t0 = time.time()
    try:
        res = fn()
        if not isinstance(res, dict):
            res = {'value': res}
        res['_seconds'] = round(time.time() - t0, 2)
        print(f'[+] {name} ok ({res["_seconds"]}s)', flush=True)
    except Exception as e:  # noqa: BLE001 - evidence, not control flow
        res = {'error': f'{type(e).__name__}: {e}',
               '_seconds': round(time.time() - t0, 2)}
        print(f'[!] {name} FAILED: {e}', flush=True)
    store[name] = res
    _w(store, outpath)


def rocm_env():
    keys = ('ROCM_PATH', 'HIP_PATH', 'HSA_ENABLE_SDMA', 'HSA_FORCE_FINE_GRAIN_PCIE',
            'MIOPEN_FIND_MODE', 'MIOPEN_USER_DB_PATH', 'AMD_LOG_LEVEL',
            'TRITON_CACHE_DIR', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
            'GPU_MAX_HW_QUEUES', 'NCCL_', 'RCCL_', 'TORCH_BLAS_PREFER_')
    env = {k: os.environ[k] for k in os.environ if k.startswith(keys)}
    return {'captured': env, 'count': len(env)}


def device_props(gpu):
    import torch
    n = torch.cuda.device_count()
    allp = []
    for i in range(n):
        try:
            p = torch.cuda.get_device_properties(i)
            allp.append({'index': i, 'name': p.name,
                         'gcnArchName': getattr(p, 'gcnArchName', None),
                         'total_mem_GiB': round(p.total_memory / 2**30, 1),
                         'sm_count': p.multi_processor_count,
                         'is_integrated': p.is_integrated,
                         'major': p.major, 'minor': p.minor})
        except Exception as e:  # noqa: BLE001
            allp.append({'index': i, 'error': f'{type(e).__name__}: {e}'})
    free, total = torch.cuda.mem_get_info(gpu)
    return {'count': n, 'devices': allp,
            'selected_gpu': gpu,
            'free_GiB': round(free / 2**30, 2),
            'total_GiB': round(total / 2**30, 2)}


def dtype_scan(dev):
    import torch
    results = {}
    names = ['float16', 'bfloat16', 'float32', 'float8_e4m3fn', 'float8_e5m2',
             'float8_e4m3fnuz', 'float8_e5m2fnuz', 'uint8', 'int8']
    for name in names:
        entry: dict = {'exists': hasattr(torch, name)}
        if not entry['exists']:
            results[name] = entry
            continue
        dt = getattr(torch, name)
        try:
            entry['finfo_max'] = (float(torch.finfo(dt).max)
                                   if dt.is_floating_point else None)
        except Exception as e:  # noqa: BLE001
            entry['finfo_max'] = f'{type(e).__name__}: {e}'
        try:
            a = torch.randn(64, device=dev).to(dt)
            b = (a.to(torch.float32) * 2.0).to(dt)
            _ = a.to(torch.float32) + b.to(torch.float32)
            entry['cast_roundtrip_gpu'] = 'ok'
        except Exception as e:  # noqa: BLE001
            entry['cast_roundtrip_gpu'] = f'{type(e).__name__}: {e}'
        results[name] = entry
    return results


FP8 = ('float8_e4m3fn', 'float8_e4m3fnuz', 'float8_e5m2', 'float8_e5m2fnuz')


def fp8_semantics(dev):
    """Byte-level ground truth for FP8 variants. THE discriminator between
    fn (CUDA-style, bias 7) and fnuz (AMD, bias 8): encoding of 1.0 is 0x38
    vs 0x40 and finfo max 448 vs 240 (validated on CPU torch 2.13). Records
    a small decode table straight from device memory so nothing is inferred
    from headers."""
    import torch
    out = {}
    for name in FP8:
        if not hasattr(torch, name):
            out[name] = {'exists': False}
            continue
        dt = getattr(torch, name)
        info = {'exists': True}
        try:
            one = torch.tensor([1.0], dtype=torch.float32, device=dev).to(dt)
            info['byte_of_1.0'] = '0x%02x' % int(one.view(torch.uint8).item())
            info['finfo_max'] = float(torch.finfo(dt).max)
            raw = torch.tensor(list(range(256)), dtype=torch.uint8, device=dev)
            vals = raw.view(dt).float()
            probe_bytes = [0x00, 0x38, 0x30, 0x7E, 0x7F, 0x80, 0xFF]
            info['decode_table'] = {'0x%02x' % b: float(vals[b].item())
                                    for b in probe_bytes}
            nanmask = torch.isnan(vals.cpu())
            info['nan_bytes'] = ['0x%02x' % i for i in
                                 nanmask.nonzero().flatten().tolist()]
        except Exception as e:  # noqa: BLE001
            info['error'] = f'{type(e).__name__}: {e}'
        out[name] = info
    return out


def scaled_mm_matrix(dev):
    """Does this stack do FP8 matmul, with which input dtypes/layouts?
    A plain bf16 matmul sanity anchor runs first; each fp8 cell records
    max abs error vs fp32 on a 256x256x256 case or the exact error."""
    import torch
    out = {'has_scaled_mm': hasattr(torch, '_scaled_mm')}
    if not out['has_scaled_mm']:
        return out
    M = K = N = 256
    a32 = torch.randn(M, K, device=dev)
    b32 = torch.randn(K, N, device=dev)
    ref = a32 @ b32
    bf = torch.matmul(a32.to(torch.bfloat16), b32.to(torch.bfloat16))
    out['bf16_matmul_max_err'] = (bf.float() - ref).abs().max().item()
    sa = torch.ones(M, device=dev)
    sb = torch.ones(1, device=dev)
    for aname in FP8:
        if not hasattr(torch, aname):
            continue
        for bname in FP8:
            if not hasattr(torch, bname):
                continue
            for layout in ('row', 'col'):
                key = f'{aname.split("_")[-1]}x{bname.split("_")[-1]}:{layout}'
                try:
                    A = a32.to(getattr(torch, aname))
                    B = b32.to(getattr(torch, bname))
                    if layout == 'col':
                        B = B.t().contiguous().t()
                    C = torch._scaled_mm(A, B, scale_a=sa, scale_b=sb,
                                         out_dtype=torch.bfloat16)
                    out[key] = {'ok': True,
                                'max_abs_err_vs_fp32': round(
                                    (C.float() - ref).abs().max().item(), 4)}
                except Exception as e:  # noqa: BLE001
                    out[key] = {'ok': False,
                                'err': f'{type(e).__name__}: {str(e)[:160]}'}
    return out


def triton_q4(dev, out_dir):
    """Representative damendaemon kernel shape (q4_0-style: fp16 scale +
    16 packed nibble bytes per 32 values) compiled through Triton on the
    AMD backend. Verifies numerics vs a plain-torch reference, then copies
    the compiled .amdgcn/.hsaco from the Triton cache into evidence."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _q4_dequant(S, P, Y, BLOCKS: tl.constexpr):
        b = tl.program_id(0)
        s = tl.load(S + b).to(tl.float32)
        p = tl.load(P + b * 16 + tl.arange(0, 16))
        lo = (p & 0x0F).to(tl.float32) - 8.0
        hi = (p >> 4).to(tl.float32) - 8.0
        base = b * 32
        off = base + tl.arange(0, 16)
        tl.store(Y + off, lo * s)
        tl.store(Y + off + 16, hi * s)

    res = {'triton_version': triton.__version__}
    blocks = 4096
    torch.manual_seed(0)
    scales = (torch.randn(blocks).to(torch.float16) * 0.25).to(dev)
    packed_cpu = torch.randint(0, 256, (blocks, 16), dtype=torch.uint8)
    packed = packed_cpu.to(dev)
    # plain-torch CPU reference, vectorized (same arithmetic the kernel does)
    vals = torch.empty(blocks, 32, dtype=torch.float32)
    vals[:, :16] = (packed_cpu & 0x0F).to(torch.float32) - 8.0
    vals[:, 16:] = (packed_cpu >> 4).to(torch.float32) - 8.0
    ref = (vals * scales.cpu().to(torch.float32).unsqueeze(1)).reshape(-1).to(dev)
    y = torch.empty_like(ref)
    _q4_dequant[(blocks,)](scales, packed, y, BLOCKS=blocks)
    res['max_abs_err_vs_reference'] = (y - ref).abs().max().item()
    res['pass'] = bool((y - ref).abs().max().item() < 1e-3)
    # capture compiled artifacts from the triton cache
    try:
        cache = os.environ.get('TRITON_CACHE_DIR',
                               os.path.expanduser('~/.triton/cache'))
        hits = []
        for root, _dirs, files in os.walk(cache):
            age = time.time() - os.path.getmtime(root)
            if age < 120:
                for fn in files:
                    if fn.endswith(('.amdgcn', '.hsaco', '.ttir', '.ttgir')):
                        src = os.path.join(root, fn)
                        dst = os.path.join(out_dir, 'triton_' + fn)
                        shutil.copy2(src, dst)
                        hits.append({'file': fn, 'bytes': os.path.getsize(src)})
        res['isa_artifacts'] = hits
    except Exception as e:  # noqa: BLE001
        res['isa_capture_error'] = f'{type(e).__name__}: {e}'
    return res


def benches(dev, budget_s):
    """Opt-in, time-boxed: HBM D2D bandwidth and one bf16 + one fp8 GEMM
    size. No throughput claims beyond these numbers; sized to fit free VRAM."""
    import torch
    out = {'budget_s': budget_s}
    deadline = time.monotonic() + budget_s
    free, total = torch.cuda.mem_get_info(dev)
    out['free_GiB_at_start'] = round(free / 2**30, 2)
    nbytes = min(2 * 2**30, int(free * 0.4))
    src = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    for _ in range(2):
        dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 5
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    out['d2d_copy_GBs_rw'] = round(2 * nbytes / dt / 1e9, 1)
    del src, dst
    if time.monotonic() < deadline:
        m = 8192
        A = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
        B = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
        for _ in range(3):
            A @ B
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            A @ B
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        out['bf16_gemm_8192_tflops'] = round(2 * m**3 / dt / 1e12, 1)
        del A, B
    if time.monotonic() < deadline and hasattr(torch, '_scaled_mm'):
        try:
            m = 8192
            A = torch.randn(m, m, device=dev).to(torch.float8_e4m3fnuz
                                                 if hasattr(torch,
                                                    'float8_e4m3fnuz')
                                                 else torch.float8_e4m3fn)
            B = torch.randn(m, m, device=dev).to(A.dtype).t().contiguous().t()
            sa = torch.ones(m, device=dev)
            sb = torch.ones(1, device=dev)
            for _ in range(3):
                torch._scaled_mm(A, B, scale_a=sa, scale_b=sb,
                                 out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                torch._scaled_mm(A, B, scale_a=sa, scale_b=sb,
                                 out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 10
            out['fp8_gemm_8192_dtype'] = str(A.dtype)
            out['fp8_gemm_8192_tflops'] = round(2 * m**3 / dt / 1e12, 1)
        except Exception as e:  # noqa: BLE001
            out['fp8_gemm_error'] = f'{type(e).__name__}: {str(e)[:200]}'
    torch.cuda.empty_cache()
    out['free_GiB_at_end'] = round(
        torch.cuda.mem_get_info(dev)[0] / 2**30, 2)
    return out


def main():
    p = argparse.ArgumentParser(description=((__doc__ or '') .splitlines() or ['MI300X probes'])[0])
    p.add_argument('mode', choices=['cap'],
                   help='cap = capability probes (single GPU, bounded)')
    p.add_argument('--gpu', type=int, default=0, help='GPU index to probe')
    p.add_argument('--out', default='.', help='output directory')
    p.add_argument('--bench', action='store_true',
                   help='opt-in: time-boxed bandwidth/GEMM microbenches')
    p.add_argument('--bench-budget', type=int, default=120,
                   help='wall-clock seconds allowed for benches')
    args = p.parse_args()
    import platform
    os.makedirs(args.out, exist_ok=True)
    outpath = os.path.join(args.out, 'probe_results.json')
    store = {'meta': {
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'host': platform.node(),
        'python': sys.version.split()[0],
        'argv': sys.argv[1:],
    }}
    _w(store, outpath)
    import torch
    if not torch.cuda.is_available():
        store['meta']['error'] = ('torch.cuda.is_available() is False - no AMD '
                                  'GPU visible to this torch build (check '
                                  'driver, /dev/kfd access, HSA/ROCR_VISIBLE_DEVICES)')
        _w(store, outpath)
        print('NO CUDA/HIP DEVICE VISIBLE - wrote meta.error, exiting 3',
              flush=True)
        return 3
    torch.cuda.set_device(args.gpu)
    dev = torch.device('cuda', args.gpu)
    section(store, outpath, 'rocm_env', rocm_env)
    section(store, outpath, 'device_props', lambda: device_props(args.gpu))
    section(store, outpath, 'dtype_scan', lambda: dtype_scan(dev))
    section(store, outpath, 'fp8_semantics', lambda: fp8_semantics(dev))
    section(store, outpath, 'scaled_mm_matrix', lambda: scaled_mm_matrix(dev))
    section(store, outpath, 'triton_q4', lambda: triton_q4(dev, args.out))
    if args.bench:
        section(store, outpath, 'benches',
                lambda: benches(dev, args.bench_budget))
    store['meta']['finished_utc'] = time.strftime(
        '%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    _w(store, outpath)
    print(f'PROBES DONE -> {outpath}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
