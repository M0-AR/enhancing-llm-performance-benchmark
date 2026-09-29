import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from elpb.datasets_live import live_market_snapshot, live_text_corpus
print(json.dumps({"market": live_market_snapshot(), "texts": live_text_corpus()[0]}, indent=2)[:3000])
