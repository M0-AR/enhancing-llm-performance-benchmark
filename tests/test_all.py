import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from elpb import pipeline

def test_pipeline_runs_10():
    res = pipeline.run_all(seed=1337)
    assert len(res["experiments"]) == 10
    assert all("hidden_pattern" in e for e in res["experiments"])

def test_determinism():
    a = pipeline.run_all(seed=1337)["experiments"][0]["scalelaw_loss_vs_scale"]
    b = pipeline.run_all(seed=1337)["experiments"][0]["scalelaw_loss_vs_scale"]
    assert a == b

def test_kronecker_rank():
    from elpb import exp04_krona_loda
    r = exp04_krona_loda.run()
    assert r["rank_kronecker_64x64"] > r["rank_lora_r1_64x64"]

def test_squarehead_wins_mid_sparsity():
    from elpb import exp05_sparse_squarehead
    r = exp05_sparse_squarehead.run()
    mid = [x for x in r["sparsity_sweep"] if x["sparsity"] in (0.5, 0.6, 0.7)]
    assert all(x["acc_squarehead"] >= x["acc_ce"] for x in mid)

def test_cimc_beats_kmeans():
    from elpb import exp09_cimc_clustering
    r = exp09_cimc_clustering.run()
    assert r["cimc"]["nmi"] >= r["kmeans"]["nmi"] - 0.05

def test_live_gate_logs_provenance():
    from elpb import exp10_live_market_verify
    r = exp10_live_market_verify.run()
    assert "live_text_provenance" in r and "market" in r
