all: verify
verify:
\tpytest -q
\tpython scripts/run_all.py --out results/universe_report.json
\tpython scripts/verify_repro.py
docker:
\tdocker compose config
\tdocker compose up --abort-on-container-exit bench
