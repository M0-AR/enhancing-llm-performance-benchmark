# Voting Matrix (one-at-a-time, 14 sources)

| # | Tool | Query | Verdict |
|---|---|---|---|
| 1 | websearch (Exa first) | LLM inference efficiency 2026 speculative+quant | Pareto atlas: FP8 workhorse, AWQ fast-but-floor-miss, KV-collapse, speculation 0.9–0.98× n-gram |
| 2 | free-search search | PEFT LoRA Kronecker LoDA 2026 | Kron-LoRA 4× fewer params, MoKA MoE+Kronecker beats LoRA |
| 3 | searxng | INT4 CPU llama.cpp | FAILED (no server) — logged, fallback to openresearch |
| 4 | openresearch web_search | sparse 75% SquareHead | Confirms book Ch6: 75% lossless, SquareHead key |
| 5 | paper-search unified | speculative + long-seq TCN MEGA | SD tutorial 2–4×; hybrid self-spec 18× gap parallel vs sequential |
| 6 | duckduckgo | KD LLM hate-speech | KD survey + HateBench: detectors degrade on newer LLMs |
| 7 | agent-reach web | MLPerf v4 GPT-J CPU | TTFT≤2s TPOT≤200ms, tokens/s metric, closed/open |
| 8 | kaggle everything | LLM efficiency quant benchmark | 2026 vector-DB/HW benchmarks, QLoRA Flan-T5 notebooks |
| 9 | wiki search | LLM quant inference | llama.cpp/GGML + post-training pages |
| 10 | gsd_websearch | repro Docker 2026 | empty — logged |
| 11 | openresearch openalex | PEFT Kronecker | Nature PEFT review, Compacter, LLM-Adapters |
| 12 | gitmcp peft docs | PEFT LoRA quant eval | QLoRA 65B/48GB, LoRA+bitsandbytes pattern |
| 13 | openresearch hacker_news | quant CPU llama.cpp | Go/Vulkan/ZSE engines, 8GB Mac Loft |
| 14 | superpowers semantic | repro benchmarking | systematic-debugging/defense-in-depth |

Consensus: quant+sparsity compound; speculation only with acceptance economics; joint fusion; relaxed KD; live+quality gates. Dissent: n-gram speculation useless on some stacks; AWQ floor-miss within error — keep on record.
