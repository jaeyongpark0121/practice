"""Write placeholder Track 1 episode files to rehearse packing and the uploader's --dry-run.

Every episode gets the MHR rest pose, a static 20 cm cube one metre to the side, and identity
rotation. The files only exercise the submission path; never upload them as a real result.

    python src/make_placeholder.py --data data/track_1 --out out/placeholder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/track_1"))
    parser.add_argument("--out", type=Path, default=Path("out/placeholder"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cube = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    episodes = [json.loads(line) for line in (args.data / "meta/episodes.jsonl").read_text().splitlines() if line]
    for episode in episodes:
        index, length = episode["episode_index"], episode["length"]
        np.savez(
            args.out / f"episode_{index:06d}.npz",
            pose=np.zeros((length, 136), np.float32),
            scales=np.zeros(68, np.float32),
            shape=np.zeros(45, np.float32),
            object_rotation=np.broadcast_to(np.eye(3), (length, 3, 3)).copy(),
            object_translation=np.tile([1.0, 0.0, 0.0], (length, 1)),
            object_scale=np.float64(1.0),
        )
        cube.export(args.out / f"episode_{index:06d}_object.ply")
    print(f"wrote {len(episodes)} placeholder episodes to {args.out}")


if __name__ == "__main__":
    main()
