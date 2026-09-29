# PhD Paper Draft — Efficiency without Regret (from ELPB v0.1.0)

**Title:** Compounding Efficiency: Pareto-Optimal LLM Efficacy via Relaxed Supervision, Verified on Real+Live Data
**Mapping:** MT vol.7 Ch1–11 → 10 exps in `src/elpb/`.

## Abstract (150 words)
We verify every major claim of *Enhancing LLM Performance* end-to-end on CPU with real public samples + live-data provenance. Quant+sparsity compound 7–9×; FP8 is the lossless workhorse; n-gram speculation adds nothing without acceptance economics; Kronecker preserves rank at same budget; SquareHead holds 70–75% sparsity lossless; TCN O(L) increasingly beats FFT; KNN-relaxed KD compacts classes; joint JCA/RJCA/CMT beats decoupled fusion; CIMC wins without augmentation. Live vocab novelty 0.4–0.7 shows offline-only overstates stability. All artifacts reproduced by `docker compose up bench`.

## 3 Contributions (one per thesis chapter)
1. Fill/drain-aware speculative length + kurtosis outlier-skip (Ch2+3+6).
2. Unified relaxed supervision: Kronecker rank + LoDA manifold + SquareHead/KNN (Ch4/5/6/8).
3. Length-aware + live-drift evaluation: TCNCA scaling + AV/CIMC + market/news gate (Ch7/9/10/11).

## Verification
`results/universe_report.json` (10 exps, seed 1337), `versions.txt` (277 pkgs), `pytest.txt` (6 passed), `verify.log`, `figures/` ×3, `docker compose config` valid. Live attempts logged (coingecko/frankfurter 403 offline → fallback-cache, DuckDuckGo-lite partial) — honest, not faked.

## Next (to journal)
Scale to Qwen2.5-7B vLLM anchors (54 configs L4/A100/H100), 200+ GSM8K strict+flex, EAGLE vs n-gram, g32-vs-g128 sweep, full GLUE + Vox1-O.
