# Zero → Hero (learn the book by running)

0. `pip install -r requirements.txt && pytest -q` — green first.
1. `python scripts/fetch_live.py` — see live vs cached provenance.
2. `python scripts/run_all.py --out results/universe_report.json` — 10 exps, ~10s CPU.
3. Open `results/REPORT.md` + `results/figures/` — match each figure to book tables (3.4/3.5, 4.x GLUE, 5.x E2E, 6.x GSM8K, 7.x EnWik8/LRA).
4. `python scripts/verify_repro.py` — seed + artifacts + live gate.
5. `docker compose up bench` — share exactly what you ran.
6. Pick one hidden pattern, tighten its test, open a PR — that's your PhD Chapter draft.
