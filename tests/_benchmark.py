"""Shared CLI for measuring xattn itself; no external implementation dependency."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess

import torch
import xattn

from tests._numerics import (
    Case, HEAD_DIMS, OPERATIONS, case_seed, make_inputs, options_and_segments,
    assert_output, assert_gradients,
)
from tests.softdelta.reference import _attention_from_mask


def measure(function, warmup, iterations, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {'latency_us': statistics.median(samples), 'samples_us': samples}


def validate(operation, case, inputs, options, qi, ki, deterministic, high_precision):
    """FP64 sampled outputs and sparse-dO gradients on the actual timed inputs."""
    soft = operation.startswith('softdelta_')
    visibility = operation.removeprefix('softdelta_')
    refs = [t.detach().double().requires_grad_() for t in inputs]
    q, k, v = refs[:3]
    if case.previous:
        k, v = torch.cat((refs[-2], k), 1), torch.cat((refs[-1], v), 1)
    positions = torch.tensor(sorted({0, case.q_length-1, *range(0, case.q_length, max(1, case.q_length//16))}), device=q.device)
    qp = positions[:, None] + k.shape[1] - case.q_length
    kp = torch.arange(k.shape[1], device=q.device)[None]
    left = torch.ones_like(kp, dtype=torch.bool)
    if visibility == 'window':left = kp >= qp - case.span
    elif visibility == 'chunk':left = kp >= qp // case.span * case.span - case.span
    reads = []
    for branch in range(2 if soft else 1):
        mask = (left & (kp <= qp-branch))[None].expand(case.batch, -1, -1)
        if qi is not None:mask = mask & (qi[:, positions, None] == ki[:, None])
        qr = q[:, positions, branch::2] if soft else q[:, positions]
        part = _attention_from_mask(qr, k, v.flatten(3) if soft else v, mask, case.dim**-0.5)
        reads.append(part.reshape(case.batch, len(positions), 4, case.groups, case.value_dim//case.groups) if soft else part)
    gate = refs[3][:, positions] if soft else None
    expected = reads[0] - gate.sigmoid()*reads[1] if soft else reads[0]
    prefix = OPERATIONS[operation]
    fwd = getattr(xattn, prefix+'_fwd'); bwd = getattr(xattn, prefix+'_bwd')
    base = inputs[:4 if soft else 3]
    fwd_options = dict(options, high_precision_output=high_precision)
    if soft:fwd_options['deterministic'] = deterministic
    low, state, lse = fwd(*base, **fwd_options)
    assert_output(low[:, positions], expected, q.dtype if q.dtype != torch.float64 else inputs[0].dtype,
                  reads if soft else None, gate)
    sparse_dy = torch.zeros_like(low)
    sparse_dy[:, positions] = torch.randn_like(low[:, positions]) / expected.numel()**0.5
    saved = state if soft or high_precision else low
    grads = bwd(sparse_dy, *base, saved, lse, **options, deterministic=deterministic)[:len(inputs)]
    reference = torch.autograd.grad(expected, refs, sparse_dy[:, positions].double())
    assert_gradients(grads, reference, inputs[0].dtype)
    return fwd, bwd, base, fwd_options, low, saved, lse


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):digest.update(chunk)
    return digest.hexdigest()


def main(operations):
    parser=argparse.ArgumentParser(description='xattn forward/backward latency and throughput')
    parser.add_argument('--dtype', choices=('fp16','bf16'), nargs='+', default=['bf16'])
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--dims', type=int, nargs=2, metavar=('D','V'), default=(64,64))
    parser.add_argument('--full-matrix', action='store_true')
    parser.add_argument('--kv-heads', type=int, nargs='+', choices=(4,2,1), default=[4,2,1])
    parser.add_argument('--span', type=int, default=256)
    parser.add_argument('--cases', nargs='+', choices=('dense','segment','previous','short'), default=['dense'])
    parser.add_argument('--high-precision', action='store_true')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    if min(args.length,*args.dims,args.iterations,args.repeats)<=0 or args.span<0 or args.warmup<0:
        parser.error('invalid shape or timing count')
    if any(op.endswith('chunk') for op in operations) and args.span==0:
        parser.error('chunk span must be positive')
    if any(op.startswith('softdelta_') for op in operations) and args.dims[1]%4:
        parser.error('SoftDelta value dimension must be divisible by four')
    if 'full' in operations and 'previous' in args.cases:
        parser.error('causal_flash_attn does not expose previous KV')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(9,0):
        raise RuntimeError('SM90 GPU required')
    root=Path(__file__).resolve().parents[1]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=args.output or Path.home()/'codex_logs/xattn/benchmarks'/f'{stamp}-{operations[0]}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    dims=[(d,v) for d in HEAD_DIMS for v in HEAD_DIMS] if args.full_matrix else [tuple(args.dims)]
    records=[]
    gpu=torch.cuda.get_device_properties(0)
    extension=importlib.util.find_spec('xattn_cuda').origin
    provenance={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                'extension':extension,'extension_sha256':sha256(extension),'torch':torch.__version__,
                'cuda':torch.version.cuda,'gpu':gpu.name,'gpu_uuid':str(gpu.uuid),
                'benchmark_sha256':sha256(__file__),
                'source_sha256':{str(path.relative_to(root)):sha256(path) for path in
                                 [*sorted((root/'xattn').rglob('*.py')), *sorted((root/'tests').rglob('*.py'))]}}
    manifest=[{'operation':op,'dtype':dtype,'dim':d,'value_dim':v,'kv_heads':kv,'case':name,
               'deterministic':det,'high_precision':args.high_precision,'length':args.length,'span':args.span}
              for op in operations for dtype in args.dtype for d,v in dims for kv in args.kv_heads
              for name in args.cases for det in (False,True)]
    coverage=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def save():
        output.write_text(json.dumps({'manifest':manifest,'coverage_sha256':coverage,'provenance':provenance,
                                     'warmup':args.warmup,'iterations':args.iterations,'repeats':args.repeats,
                                     'records':records},indent=2)+'\n')
    save()
    for identity in manifest:
        op=identity['operation']; dtype={'fp16':torch.float16,'bf16':torch.bfloat16}[identity['dtype']]
        name=identity['case']; det=identity['deterministic']
        case=Case(name, 17 if name=='short' else args.length, args.length,
                  identity['dim'],identity['value_dim'], metadata='segment' if name=='segment' else 'dense',
                  previous=(args.span if op.endswith('chunk') else 65) if name=='previous' else 0,span=args.span,batch=1)
        if case.q_length>case.k_length:case=Case(**{**asdict(case),'q_length':case.k_length})
        try:
            inputs=make_inputs(op,case,dtype,identity['kv_heads'],'cuda')
            options,qi,ki=options_and_segments(op,case,inputs,'sm90')
            fwd,bwd,base,fwd_options,low,saved,lse=validate(op,case,inputs,options,qi,ki,det,args.high_precision)
            dy=torch.randn_like(low)
            forward=lambda:fwd(*base,**fwd_options)
            backward=lambda:bwd(dy,*base,saved,lse,**options,deterministic=det)
            with torch.no_grad():
                fwd_result=measure(forward,args.warmup,args.iterations,args.repeats)
                bwd_result=measure(backward,args.warmup,args.iterations,args.repeats)
            tokens=case.batch*case.q_length*4
            for result in (fwd_result,bwd_result):result['Mquery_head_tokens_per_s']=tokens/result['latency_us']
            records.append({**identity,'shape':asdict(case),'correctness':'passed_sampled_fp64',
                            'fwd':fwd_result,'bwd':bwd_result,
                            'seed':case_seed(op,case,dtype,identity['kv_heads']),
                            'input_sha256':[hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for t in inputs],
                            'dy_sha256':hashlib.sha256(dy.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()})
            print(op,name,identity['dtype'],identity['kv_heads'],'det',det,
                  'fwd_us',fwd_result['latency_us'],'bwd_us',bwd_result['latency_us'],flush=True)
        except Exception as exc:
            records.append({**identity,'error':repr(exc)})
            save()
            raise
        save()
    print(output,flush=True)
