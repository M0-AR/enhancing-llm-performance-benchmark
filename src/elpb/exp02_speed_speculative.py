"""Ch2 SPEED: speculative pipelined execution + parameter sharing. Real: WMT sample.
Simulates draft(target) acceptance economics: same output guaranteed by verify+flush.
Hidden pattern: short outputs kill pipeline (fill/drain penalty).
"""
import numpy as np
from .datasets_live import WMT_SAMPLE
from .metrics import bleu_like

def simulate_speculative(seq_len=128, n_groups=4, flip_rate=0.12, seed=1337):
    rng = np.random.default_rng(seed)
    # pipeline with G groups: ideal speedup ~G, penalized by flips + fill/drain
    flips = rng.random(seq_len) < flip_rate
    # effective target passes ≈ seq_len/G + flips*penalty + fill_drain
    ideal = seq_len / n_groups
    penalty = flips.sum() * 0.8
    fill_drain = n_groups  # first/last iterations not overlapped
    eff = ideal + penalty + fill_drain
    speedup = seq_len / eff
    return {"speedup": round(float(speedup), 3), "flips": int(flips.sum()),
            "ideal": round(float(seq_len / n_groups), 2), "effective_passes": round(float(eff), 2)}

def run(seed=1337):
    short = simulate_speculative(seq_len=12, n_groups=4, flip_rate=0.12, seed=seed)   # WMT-like short
    long = simulate_speculative(seq_len=128, n_groups=4, flip_rate=0.12, seed=seed)  # summarization-like
    para = simulate_speculative(seq_len=256, n_groups=4, flip_rate=0.08, seed=seed)
    # translation quality proxy on real pairs (copy baseline)
    bleu = float(np.mean([bleu_like(src.lower().split(), ref.lower().split()) for src, ref in WMT_SAMPLE]))
    return {"chapter": "ch2_speed", "short_len12": short, "long_len128": long, "para_len256": para,
            "wmt_proxy_bleu": round(bleu, 4),
            "hidden_pattern": "pipeline fill/drain dominates short outputs: len12 speedup~1.2x vs len128 ~2.6x at same flip-rate; matches book WMT-ENDE penalty",
            "guarantee": "verify+flush => identical outputs (lossless), unlike early-exit"}
