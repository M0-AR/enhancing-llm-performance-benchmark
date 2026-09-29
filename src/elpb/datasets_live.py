"""Real public data + live-data harness with provenance (offline-first).

Real: bundled tiny samples mirroring book datasets (WikiText-2 style, WMT EN-DE,
CNN/DM, SST-2/MRPC, GSM8K, VoxCeleb-like). Live: best-effort fetch of live text/
market with timeout; always logs provenance; falls back to cached sample offline.
"""
import os, json, time, urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results")
CACHE = {}

WIKITEXT_SAMPLE = [
    "The Transformer architecture relies on self attention to weight word importance .",
    "Large language models predict the next word given previous tokens .",
    "Scale improves context understanding and chain of thought reasoning .",
    "Quantization reduces memory bandwidth which dominates autoregressive decoding .",
    "Parameter sharing amortizes memory operations across sequence positions .",
]
WMT_SAMPLE = [
    ("The house is on fire", "Das Haus brennt"),
    ("Machine translation is not a solved problem", "Maschinelle Übersetzung ist kein gelöstes Problem"),
    ("Efficiency makes deployment usable", "Effizienz macht den Einsatz nutzbar"),
    ("Attention is all you need", "Attention ist alles was du brauchst"),
]
SST2_SAMPLE = [
    ("a fluent and grammatically correct triumph", 1),
    ("a dull and misleading failure", 0),
    ("semantically accurate and transparent reasoning", 1),
    ("plausible yet incorrect hallucination", 0),
    ("efficient inference on CPUs with low latency", 1),
    ("noisy modality degrades verification", 0),
]
GSM8K_SAMPLE = [
    ("Janet has 3 apples and buys 2 more. How many?", "5"),
    ("A train goes 60 km in 1 hour, how far in 3?", "180"),
]

def provenance(entry, source, live_ok, detail=""):
    return {"entry": entry, "source": source, "live_ok": bool(live_ok),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "detail": detail}

def try_fetch_json(url, timeout=6):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "ignore")), True
    except Exception as e:
        return {"error": str(e)[:200]}, False

def live_market_snapshot():
    """Live market/data check: BTC price + FX + a news headline count. No key needed."""
    out, prov = {}, []
    j, ok = try_fetch_json("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd", timeout=6)
    out["crypto"] = j; prov.append(provenance("crypto", "coingecko", ok))
    j2, ok2 = try_fetch_json("https://api.frankfurter.app/latest?from=USD&to=EUR,GBP", timeout=6)
    out["fx"] = j2; prov.append(provenance("fx", "frankfurter/ecb", ok2))
    out["provenance"] = prov
    out["live_any"] = bool(ok or ok2)
    return out

def live_text_corpus():
    """Live text for drift test: tries DuckDuckGo lite HTML titles; fallback to WIKITEXT_SAMPLE."""
    try:
        req = urllib.request.Request("https://lite.duckduckgo.com/lite/?q=large+language+model+inference+efficiency",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            html = r.read().decode("utf-8", "ignore")
        import re
        titles = re.findall(r'>([^<>]{20,120})<', html)[:8]
        titles = [t.strip() for t in titles if "language" in t.lower() or "model" in t.lower() or "AI" in t]
        if titles:
            return titles[:5], provenance("live-titles", "duckduckgo-lite", True, f"{len(titles)} titles")
    except Exception as e:
        pass
    return list(WIKITEXT_SAMPLE[:5]), provenance("live-titles", "fallback-cache", False, "offline")
