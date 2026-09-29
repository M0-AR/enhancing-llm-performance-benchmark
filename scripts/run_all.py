import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from elpb.pipeline import run_all, save_figures
from elpb.report import to_markdown

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/universe_report.json")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    res = run_all(seed=a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    figs = save_figures(res, os.path.join(os.path.dirname(a.out), "figures"))
    res["figures"] = figs
    with open(a.out, "w") as f: json.dump(res, f, indent=2)
    md = to_markdown(res)
    with open(os.path.join(os.path.dirname(a.out), "REPORT.md"), "w") as f: f.write(md)
    print(md[:4000])
    print(f"\nWROTE {a.out} + REPORT.md + {len(figs)} figures")

if __name__ == "__main__":
    main()
