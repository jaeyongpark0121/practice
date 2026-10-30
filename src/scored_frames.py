"""List the scored video frames of every Track 1 episode from the public sample submission.

Row ids are ``t1_<episode>_<frame>_<array>_<point>``; frame 999999 marks per-episode arrays.

    python src/scored_frames.py --sample kit/data/track_1_sample_submission.parquet --out work/scored_frames.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

EPISODE_FRAME = 999999


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=Path, default=Path("kit/data/track_1_sample_submission.parquet"))
    parser.add_argument("--out", type=Path, default=Path("work/scored_frames.json"))
    args = parser.parse_args()

    frames = defaultdict(set)
    for row_id in pq.read_table(args.sample, columns=["row_id"]).column("row_id").to_pylist():
        _, episode, frame, _, _ = row_id.split("_")
        if int(frame) != EPISODE_FRAME:
            frames[int(episode)].add(int(frame))
    scored = {episode: sorted(found) for episode, found in sorted(frames.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scored))
    total = sum(len(f) for f in scored.values())
    print(f"wrote {args.out}: {len(scored)} episodes, {total} scored frames")


if __name__ == "__main__":
    main()
