"""Draw an episode's submitted object mesh and body vertices over video frames, as a visual check.

Projects ``out/<run>/episode_XXXXXX_object.ply`` at each frame's translation (green) and the
camera-frame body vertices from run_human.py (blue) with the episode's camera matrix, for a few
evenly spaced processed frames, into one contact sheet. Run inside ~/v2d/.venv-obj:

    python src/render_check.py --episode 14
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh

from run_human import iter_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/track_1"))
    parser.add_argument("--human", type=Path, default=Path("work/human"))
    parser.add_argument("--run", type=Path, default=Path("out/v0"))
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    name = f"episode_{args.episode:06d}"
    human = np.load(args.human / f"{name}.npz")
    episode = np.load(args.run / f"{name}.npz")
    mesh = trimesh.load(args.run / f"{name}_object.ply")
    frames, K = human["frames"], human["intrinsics"]
    picks = sorted({int(frames[i]) for i in np.linspace(0, len(frames) - 1, args.count).astype(int)})
    video = args.data / f"videos/chunk-000/observation.images.exo_camera/{name}.mp4"
    tiles = []
    for index, rgb in iter_frames(video, picks):
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        body = human["verts_cam"][int(np.flatnonzero(frames == index)[0])][::8]
        scene = mesh.vertices @ episode["object_rotation"][index].T * float(episode["object_scale"]) + episode["object_translation"][index]
        for points, colour in ((body, (255, 120, 0)), (scene, (0, 255, 0))):
            uvw = points @ K.T
            for u, v in (uvw[:, :2] / uvw[:, 2:]).astype(int):
                cv2.circle(image, (u, v), 2, colour, -1)
        cv2.putText(image, f"frame {index}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        tiles.append(cv2.resize(image, None, fx=0.4, fy=0.4))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)])
    out = args.run / "check" / f"{name}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
