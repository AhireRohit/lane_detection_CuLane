"""CPU port of CLRerNet's CUDA lane NMS (vendors/clrernet/libs/models/layers/nms/src/nms_kernel.cu + nms.cpp).

Same interface as the compiled `nms` extension: nms(boxes, scores, overlap, top_k) -> (keep, num_to_keep, parent).
Line-by-line translation of devIoU + nms_collect, including float32 accumulation and the C int-cast rounding, so that
the kept lanes are identical (checked against the CUDA op by scripts/verify_cpu.py). Only used where no GPU exists
(the Hugging Face Space); the benchmark results all use the original CUDA op.
Proposal layout (77 = 5 + 72): [cls0, cls1, start_y, start_x, length*n_strips, 72 x-offsets].
"""
import numpy as np
import torch

N_OFFSETS = 72
N_STRIPS = N_OFFSETS - 1
F32 = np.float32


def _span(row):
    start = int(float(F32(row[2]) * F32(N_STRIPS)) + 0.5)               # (int)(a[2] * N_STRIPS - 0 + 0.5)
    t = F32(F32(start) + F32(row[4])); t = F32(t - F32(1))               # start + a[4] - 1   (float32)
    neg = 1 if F32(F32(row[4]) - F32(1)) < 0 else 0
    end = int(float(t) + 0.5 - neg)                                      # + 0.5 - ((a[4]-1) < 0), C truncation
    return start, end


def _overlaps(flat, ra, rb, sa, ea, sb, eb, thr):
    """devIoU for proposals ra and rb (row indices) of the contiguous buffer `flat` (rows of 77 floats)."""
    start = max(sa, sb)
    end = min(min(ea, eb), N_OFFSETS - 1)
    if end < start:
        return False
    # The kernel reads a[5 + i] through a raw pointer. For valid outputs 0 <= start <= end <= 71, so this stays inside
    # the proposal's 72 offsets. start < 0 only arises from out-of-range outputs (start_y < 0); the pointer then reads
    # the floats just before the offsets (below i = -5, the previous proposal), reproduced here via the flat buffer
    # (clipped at its start, where the kernel's read would be undefined).
    i = np.arange(start, end + 1) + 5
    d = np.abs(flat[np.clip(77 * ra + i, 0, None)] - flat[np.clip(77 * rb + i, 0, None)]).astype(F32)
    dist = np.cumsum(d, dtype=F32)[-1]                                   # sequential float32 sum, as the kernel loop
    return bool(dist < F32(F32(thr) * F32(end - start + 1)))


def nms(boxes, scores, overlap, top_k):
    dev = boxes.device
    idx = torch.sort(scores, 0, descending=True)[1].cpu().numpy()      # nms.cpp: scores.sort(0, true)
    b = np.ascontiguousarray(boxes.detach().to(torch.float32).cpu().numpy())
    assert b.shape[1] == 5 + N_OFFSETS, b.shape
    flat = b.reshape(-1)
    n = len(idx)
    spans = [_span(b[i]) for i in range(len(b))]
    removed = np.zeros(n, bool)
    keep, parent = [], np.zeros(len(b), np.int64)
    for i in range(n):                                                   # nms_collect: greedy over sorted order
        if removed[i]:
            continue
        ii = idx[i]; keep.append(ii)
        sa, ea = spans[ii]
        for j in range(i + 1, n):                                        # mask covers later boxes only
            jj = idx[j]
            if not removed[j]:
                sb, eb = spans[jj]
                if _overlaps(flat, ii, jj, sa, ea, sb, eb, overlap):
                    removed[j] = True; parent[jj] = len(keep)
        parent[ii] = len(keep)
        if len(keep) == top_k:
            break
    k = np.zeros(n, np.int64); k[:len(keep)] = keep
    return (torch.from_numpy(k).to(dev), torch.tensor(min(top_k, len(keep)), device=dev),
            torch.from_numpy(parent).to(dev))
