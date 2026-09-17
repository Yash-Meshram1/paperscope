"""Run from repository root: python scripts/evaluate.py"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
from datetime import datetime, timezone
import hashlib
import json
import uuid
from paperscope.engine import Config, load_engine
from paperscope.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    config = Config.from_env()
    engine = load_engine(config)
    try:
        runs = {}
        for name in ('single_paper', 'multi_paper'):
            raw = (config.root / f'data/evaluation/{name}.json').read_bytes()
            fixture = json.loads(raw)
            print(f"Evaluating {name}: {len(fixture['questions'])} questions", flush=True)
            def retrieve(query, top_k):
                results = engine.retrieve(query, top_k=top_k)
                print(f"  {len(results)} passages: {query}", flush=True)
                return results
            runs[name] = {**evaluate(fixture['questions'], retrieve, args.k),
                          'fixture_sha256':hashlib.sha256(raw).hexdigest(),
                          'fixture_provenance':fixture['provenance']}
        report = {'timestamp':datetime.now(timezone.utc).isoformat(),
                  'entry_point':'Engine.retrieve -> smart_retrieve',
                  'provenance':engine.provenance, 'runs':runs}
        directory = config.root / 'data/evaluation/runs'
        directory.mkdir(exist_ok=True)
        path = directory / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.json"
        with path.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(path)
    finally:
        engine.close()


if __name__ == '__main__': main()
