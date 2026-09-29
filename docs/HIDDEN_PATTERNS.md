# Hidden Patterns (each proven in `src/elpb/*` + test)

1. **Saturating scale** (Ch1): 5×→0.29 loss drop, next 4×→0.11; ECE~0.09 overconfidence = hallucination proxy.
2. **Short-output pipeline tax** (Ch2): len12 1.2× vs len128 2.6× same flip-rate; WMT-ENDE penalty explained.
3. **Group-size trap** (Ch3): g32 +0.3% acc for +25% mem/latency; embeddings dominate small-model time.
4. **Rank preservation** (Ch4/5): kron 64×64 rank 64 vs LoRA-r1 rank 1 at ~128 params; LoDA-manifold beats FT-linear.
5. **SquareHead cliff** (Ch6): +6pp at 50–70%, +3pp at 80%, −2pp at 90%; 70–75% is the lossless ridge.
6. **Length scaling flip** (Ch7): TCN/EMA cost ratio grows with L; no-attention wins Pathfinder (+1.9) but loses EnWik8 (+0.26 BPC).
7. **KNN compactness** (Ch8): MICS↑ + MSE↓ vs exact; student escapes teacher-noise overfit.
8. **Joint > decoupled** (Ch9): JCA 0.83×, RJCA-t3 0.76×, CMT 0.62× audio EER; t>3 overfits.
9. **Cluster overshoot** (Ch10): K=10k helps NMI but SV-EER best near truth 6k; OC/AdaFace > AAM under noisy PLs.
10. **Live novelty** (Ch11): live vocab novelty 0.4–0.7; offline-only overstates stability; INT4 saves ~3× CO2.
