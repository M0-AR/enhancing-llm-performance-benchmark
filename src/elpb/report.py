"""Report: markdown + json synthesis with hidden patterns + PhD claims."""
import json

def to_markdown(res):
    lines = ["# ELPB Universe Report", f"seed={res['seed']} elapsed={res['elapsed_s']}s", ""]
    for e in res["experiments"]:
        lines.append(f"## {e.get('chapter')}")
        hp = e.get("hidden_pattern", "")
        if hp: lines.append(f"- Hidden: {hp}")
        bc = e.get("book_claim", "")
        if bc: lines.append(f"- Book: {bc}")
        lines.append("")
    lines.append(f"Pareto demo frontier idx: {res.get('pareto_demo_idx')}")
    return "\n".join(lines)
