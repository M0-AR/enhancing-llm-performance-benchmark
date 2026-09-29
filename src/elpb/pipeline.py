"""Pipeline: run all 10 exps deterministically, emit Pareto + figures + report."""
import os, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .seeds import set_seed, versions_manifest
from . import (exp01_fundamentals_scaling, exp02_speed_speculative, exp03_int4_cpu,
               exp04_krona_loda, exp05_sparse_squarehead, exp06_tcnca_longseq,
               exp07_class_kd, exp08_av_fusion, exp09_cimc_clustering, exp10_live_market_verify)
from .metrics import pareto_frontier

EXPS = [exp01_fundamentals_scaling, exp02_speed_speculative, exp03_int4_cpu, exp04_krona_loda,
        exp05_sparse_squarehead, exp06_tcnca_longseq, exp07_class_kd, exp08_av_fusion,
        exp09_cimc_clustering, exp10_live_market_verify]

def run_all(seed=1337):
    seed = set_seed(seed)
    t0 = time.time()
    results = {"seed": seed, "versions": versions_manifest(), "experiments": []}
    for e in EXPS:
        r = e.run(seed=seed)
        results["experiments"].append(r)
    # Pareto over (latency-proxy, quality-loss-proxy): build from exps
    pts = [(2.6, 0.02, 2.6), (1.8, 0.03, 1.8), (1.2, 0.06, 1.0)]  # illustrative, documented
    results["pareto_demo_idx"] = pareto_frontier(pts)
    results["elapsed_s"] = round(time.time() - t0, 2)
    return results

def save_figures(res, figdir):
    os.makedirs(figdir, exist_ok=True)
    # Fig1: scaling law
    sc = res["experiments"][0]["scalelaw_loss_vs_scale"]
    xs = [s for s, _ in sc]; ys = [l for _, l in sc]
    plt.figure(); plt.plot(xs, ys, marker="o"); plt.xscale("log"); plt.xlabel("scale"); plt.ylabel("loss proxy")
    plt.title("Ch1: scale vs loss (saturating)"); plt.savefig(os.path.join(figdir, "fig01_scaling.png"), dpi=120); plt.close()
    # Fig2: speculative speedup short vs long
    sp = res["experiments"][1]
    plt.figure(); plt.bar(["len12", "len128", "len256"], [sp["short_len12"]["speedup"], sp["long_len128"]["speedup"], sp["para_len256"]["speedup"]])
    plt.ylabel("speedup"); plt.title("Ch2: pipeline fill/drain penalty"); plt.savefig(os.path.join(figdir, "fig02_speed.png"), dpi=120); plt.close()
    # Fig5: sparsity sweep
    sw = res["experiments"][4]["sparsity_sweep"]
    plt.figure(); plt.plot([r["sparsity"] for r in sw], [r["acc_ce"] for r in sw], marker="o", label="CE")
    plt.plot([r["sparsity"] for r in sw], [r["acc_squarehead"] for r in sw], marker="s", label="SquareHead")
    plt.xlabel("sparsity"); plt.ylabel("acc"); plt.legend(); plt.title("Ch6: SquareHead recovery")
    plt.savefig(os.path.join(figdir, "fig05_sparse.png"), dpi=120); plt.close()
    return ["fig01_scaling.png", "fig02_speed.png", "fig05_sparse.png"]
