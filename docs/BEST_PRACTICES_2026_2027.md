# Best Practices 2026–2027 (voted)

1. **Pareto atlas, not single speedups** (2026 Inference Pareto Atlas, GKE frontier, Senpai guide): report cost/quality/latency jointly; TTFT+TPOT/p50/p95/tokens-s/$-Mtok; 200+ GSM8K min; strict+flexible extraction (AWQ format-vs-arithmetic split).
2. **Anchors + simulator**: 54-config anchors (Qwen2.5-7B vLLM L4/A100/H100, drift<1.5%) then simulate; stacked combos win 60% vs 43% singles.
3. **Speculation economics**: acceptance-length × verifier-width × drafter-cost; n-gram 0.90–0.98× can add nothing; EAGLE 1.8–2.4×; optimal draft-len varies (5–11) — never fixed 4.
4. **Quantization first**: FP8 weights ~lossless workhorse; AWQ-4bit fastest but misses 95% floor by sampling error; naive FP8-KV can be 0/200 while fast — quality gate mandatory. Group-size 128 vs 32 tradeoff; embedding dominates small models.
5. **PEFT**: Kronecker preserves rank (rank(A⊗B)=rankA·rankB) vs LoRA ≤r; LoDA-nonlinear manifold beats linear; QLoRA for 65B on 48GB.
6. **Sparsity+KD**: CE/KL diverge ≥75%; SquareHead L2-feature recovers to 70–75% lossless; sparsity×INT8 compounds 7–9× CPU; skip 3–10 outlier layers via kurtosis.
7. **Long-seq**: TCN O(L) > FFT O(LlogL) increasingly; chunked attention O(L); attention-free wins Pathfinder but loses LM.
8. **KD/clustering/fusion**: KNN-relaxed > exact (MICS↑, MSE↓); CIMC (MI+VAT+mixup+SupCon) w/o aug fast; JCA joint >> decoupled; RJCA t=3; CMT best.
9. **Repro**: PYTHONHASHSEED, ELPB_SEED, versions.txt, frozen prompts, CIs, contamination, live provenance, Docker `bench` reproduces all.
10. **Fit-for-purpose (Ch11)**: small distilled > giant when explainable; log CO2/privacy/hallucination gates; EU AI Act, NYT vs OpenAI lessons.
