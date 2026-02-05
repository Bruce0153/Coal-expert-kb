from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/artifacts/curated_pairs.jsonl")
    ap.add_argument("--outdir", default="data/artifacts")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.inp)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for line in inp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        # expect instruction/input/output
        if not all(k in obj for k in ("instruction", "input", "output")):
            continue
        rows.append(obj)

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    val = rows[:n_val]
    train = rows[n_val:]

    (outdir / "lora_train.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in train) + "\n",
        encoding="utf-8",
    )
    (outdir / "lora_val.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in val) + "\n",
        encoding="utf-8",
    )
    print({"train": len(train), "val": len(val)})


if __name__ == "__main__":
    main()
