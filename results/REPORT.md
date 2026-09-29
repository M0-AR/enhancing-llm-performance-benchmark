# ELPB Universe Report
seed=1337 elapsed=11.3s

## ch1_fundamentals
- Hidden: log-linear gains saturate: last 4x scale gives <40% of first 5x gain; ECE~0.09 shows systematic overconfidence (hallucination proxy)

## ch2_speed
- Hidden: pipeline fill/drain dominates short outputs: len12 speedup~1.2x vs len128 ~2.6x at same flip-rate; matches book WMT-ENDE penalty

## ch3_int4_cpu
- Hidden: group-size trap: g32 gains ~0.3% acc but costs ~25% mem/latency vs g128; embedding table dominates small-model runtime (Amdahl), matching book Sec.2.5.3
- Book: 20-80ms next-token 6-20B on Xeon; 1.6x next-token / 8.4x first-token vs GGML (book Table 3.4/3.5)

## ch4_ch5_krona_loda
- Hidden: Kronecker preserves rank (64 vs LoRA<=1) at same budget (~128 params): explains book +0.4 GLUE over LoRA and LoDA+ beating FTWq,Wv
- Book: KronA 86.14 / KronABres 86.57 vs LoRA 85.73 GLUE; LoDA(+) beats LoRA+FT on E2E/DART

## ch6_sparse
- Hidden: SquareHead wins 50-80% (+3-7pp over CE), collapses at 90%: matches book 70-75% lossless, 80% -5pp; sparsity+INT8 compound ~7-9x CPU
- Book: MPT 70% lossless 2.6x FP32 / 7.5x INT8; Whisper/T5 2-2.7x; SquareHead > CE/Standard-KD

## ch7_tcnca
- Hidden: gap widens with L: TCN O(L) beats FFT O(LlogL) increasingly; attention-free TCN wins Pathfinder-like but loses EnWik8 (-0.26 BPC): attention needed for LM
- Book: TCNCA 1.01 BPC > MEGA 1.02; LRA +0.8% avg; 1.22-1.42x fwd speedup

## ch8_class_kd
- Hidden: KNN relaxation lowers MSE and raises intra-class compactness (MICS): student stops overfitting teacher noise, focuses on class manifold
- Book: KNN-KD beats MetaDistil/ReptileDistil on MRPC/MNLI/RTE; HateBERT6 keeps 98-99% with 50% params

## ch9_av_fusion
- Hidden: joint intra+inter (JCA) >> decoupled CA/SA; recursion t=3 optimal (t>3 overfits); CMT best (-38% vs audio): heterogeneity reduced by joint rep
- Book: CMT 1.7 EER vs audio 2.5 / visual 3.8 Vox1-O; RJCA t=3 best

## ch10_cimc
- Hidden: CIMC wins without explicit augmentation (fast, low-mem); overshooting K (10k vs 6k truth) helps NMI but SV-EER best near truth; OCSoftmax/AdaFace > AAM under noisy PLs
- Book: CIMC 63.9% ACC w/o aug / 72.5% w/ aug vs AHC 60.2%; SV 3.0 EER SOTA

## ch11_live_gate
- Hidden: live novelty 0.4-0.7 proves offline-only eval overstates stability; INT4+sparsity is the fit-for-purpose lever (3x CO2 cut) when quality floor holds

Pareto demo frontier idx: [0, 1, 2]