"""Write the fused group_softdelta SM90 instances: every (D, V) bucket pair (D and V
each 32, 64, 96, 128, 160, 192 or 256; see buckets.h), forward and backward,
fp16 and bf16 (backward per logical head and in reader groups), causal and, in
the *_local files, sliding window / sliding chunk. Run from any directory;
rewrites csrc/group_softdelta/instantiations.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'instantiations'
BUCKETS = (32, 64, 96, 128, 160, 192, 256)
DTYPES = (('bf16', 'cutlass::bfloat16_t'), ('fp16', 'cutlass::half_t'))


def main():
    ROOT.mkdir(exist_ok=True)
    for old in ROOT.glob('group_softdelta_*_sm90.cu'):
        old.unlink()
    for d in BUCKETS:
        for v in BUCKETS:
            for name, element in DTYPES:
                for kind, template in (('fwd', 'fwd'), ('bwd', 'bwd'), ('bwd_groups', 'bwd'),
                                       ('fwd_local', 'fwd'), ('bwd_local', 'bwd'), ('bwd_groups_local', 'bwd')):
                    path = ROOT / f'group_softdelta_{kind}_hdim{d}_vdim{v}_{name}_sm90.cu'
                    path.write_text(f'#include "group_softdelta/{template}_template.cuh"\n\n'
                                    f'XATTN_GROUP_SOFTDELTA_{kind.upper()}_SM90_INSTANTIATE({element}, {d}, {v})\n')


if __name__ == '__main__':
    main()
