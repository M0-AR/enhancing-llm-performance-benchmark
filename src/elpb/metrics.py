"""Shared metrics: CIs, latency stats, quality floors, Pareto helpers."""
import numpy as np, time, json

def mean_ci(x, c=1.96):
    x = np.asarray(x, dtype=float)
    m = float(x.mean()) if len(x) else 0.0
    se = float(x.std(ddof=1) / (len(x) ** 0.5)) if len(x) > 1 else 0.0
    return m, m - c * se, m + c * se

def latency_stats(fn, repeats=20, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "n": len(a)}

def pareto_frontier(points):
    """points: list of (cost, quality_loss, latency). Return indices of non-dominated (min cost, min loss, min latency)."""
    idx = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j: continue
            if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2] and q != p:
                dominated = True
                break
        if not dominated:
            idx.append(i)
    return idx

def bleu_like(pred_tokens, ref_tokens):
    # tiny unigram-F1 as BLEU proxy for offline CPU (deterministic, no external dep)
    ps, rs = set(pred_tokens), set(ref_tokens)
    if not ps or not rs: return 0.0
    inter = len(ps & rs)
    prec = inter / max(1, len(ps)); rec = inter / max(1, len(rs))
    return 2 * prec * rec / max(1e-9, prec + rec)

def ter_like(pred, ref):
    return 1.0 - bleu_like(pred, ref)
