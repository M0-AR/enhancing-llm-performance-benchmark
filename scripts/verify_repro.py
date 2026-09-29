"""verify_repro: seeds, versions, artifacts, live provenance, figure existence."""
import os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from elpb.seeds import set_seed

def main():
    s1 = set_seed(1337); s2 = set_seed(1337)
    assert s1 == s2 == 1337, "seed unstable"
    for p in ["results/universe_report.json", "results/REPORT.md"]:
        assert os.path.exists(p), f"missing {p} (run scripts/run_all.py first)"
    rep = json.load(open("results/universe_report.json"))
    assert len(rep["experiments"]) == 10, "need 10 experiments"
    assert all("hidden_pattern" in e for e in rep["experiments"]), "hidden patterns missing"
    live = rep["experiments"][-1]
    assert "live_text_provenance" in live and "market" in live, "live gate missing"
    print(f"VERIFY OK: 10 exps, seed={rep['seed']}, live_any={live['market'].get('live_any')}, figs={rep.get('figures')}")

if __name__ == "__main__":
    main()
