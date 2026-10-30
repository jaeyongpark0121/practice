"""Smooth one episode's camera-frame body vertices over time and fit one MHR identity to them.

Reads ``work/human/episode_XXXXXX.npz`` from run_human.py, smooths each contiguous run of
processed frames with a Gaussian along time, and converts the result to MHR parameters with the
kit's ``mesh_to_mhr_params.convert`` (one shape and scale per episode, as the submission requires).
Reports vertex acceleration before and after smoothing. Run inside the v2d-ubuntu env:

    python src/fit_body_identity.py --episode 14
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def runs(frames: np.ndarray) -> list[slice]:
    """Maximal runs of consecutive frame numbers, as slices into the frame axis."""
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    edges = [0, *breaks.tolist(), len(frames)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def smooth(verts: np.ndarray, frames: np.ndarray, sigma: float) -> np.ndarray:
    out = verts.copy()
    for part in runs(frames):
        out[part] = gaussian_filter1d(verts[part], sigma, axis=0, mode="nearest")
    return out


def acceleration_cm(verts: np.ndarray, frames: np.ndarray, scored: np.ndarray) -> float:
    """Mean second-difference length of the vertices, in cm per frame squared, at scored frames."""
    middle = np.flatnonzero((frames[1:-1] - frames[:-2] == 1) & (frames[2:] - frames[1:-1] == 1)) + 1
    middle = middle[np.isin(frames[middle], scored)]
    second = verts[middle + 1] - 2 * verts[middle] + verts[middle - 1]
    return float(np.linalg.norm(second, axis=-1).mean() * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing width in frames")
    parser.add_argument("--human", type=Path, default=Path("work/human"))
    parser.add_argument("--mhr-model", type=Path, default=Path.home() / "v2d/checkpoints/mhr/mhr_model.pt")
    parser.add_argument("--kit", type=Path, default=Path("kit"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.kit / "tools/track1"))
    from mesh_to_mhr_params import convert

    data = np.load(args.human / f"episode_{args.episode:06d}.npz")
    frames, scored, raw = data["frames"], data["scored"], data["verts_cam"]
    smoothed = smooth(raw, frames, args.sigma)
    print(f"vertex acceleration at scored frames: raw {acceleration_cm(raw, frames, scored):.3f} cm, "
          f"smoothed {acceleration_cm(smoothed, frames, scored):.3f} cm")

    started = time.time()
    fitted = convert(smoothed, str(args.mhr_model), precision="float32", model_batch=64,
                     log=lambda *line, **options: None)
    report = json.loads(fitted["report"]) if isinstance(fitted["report"], str) else fitted["report"]
    print(f"identity fit: {(time.time() - started) / len(frames):.2f} s/frame, "
          f"vertex error vs smoothed input {fitted['per_frame_vertex_error_mm'].mean():.1f} mm mean, "
          f"{fitted['per_frame_vertex_error_mm'].max():.1f} mm max")

    out = args.human / f"episode_{args.episode:06d}_mhr.npz"
    np.savez(out, frames=frames, scored=scored, pose=fitted["pose"], scales=fitted["scales"],
             shape=fitted["shape"], vertex_error_mm=fitted["per_frame_vertex_error_mm"],
             report=json.dumps(report, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
