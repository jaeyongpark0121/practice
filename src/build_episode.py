"""Merge one episode's body fit and object track into the submission's episode files.

Reads ``work/human/episode_XXXXXX_mhr.npz`` (fit_body_identity.py) and
``work/object/episode_XXXXXX.npz`` (run_object.py). The object is a rigid mesh that only
translates (identity rotation, scale 1). Each frame's MoGe points go onto the body's scale with
that frame's body depth ratio, which also cancels MoGe's per-frame scale wobble. A frame's object
position is the median of its points no deeper than the object itself behind its nearest points,
so background seen through a thin mask (a pan held edge-on) is left out. The mesh comes from the
frame with the largest mask and its neighbours, each centred on its frame's position; the visible
surface is mirrored about its far side along the viewing ray to close the unseen back, then
voxelised into a closed surface. Translation is the per-frame position, interpolated over frames
without one, median-filtered against single-frame spikes and Gaussian-smoothed within each
stretch. Video frames outside the processed set copy the nearest processed frame.
Run inside ~/v2d/.venv-obj:

    python src/build_episode.py --episode 14
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage

from fit_body_identity import acceleration_cm, runs, smooth


def fill_nearest(frames: np.ndarray, values: np.ndarray, length: int) -> np.ndarray:
    """One row per video frame, each copied from the nearest processed frame."""
    return values[np.abs(np.arange(length)[:, None] - frames[None]).argmin(axis=1)]


def interpolate_missing(frames: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linearly fill NaN rows along the frame axis; ends hold the nearest valid row."""
    ok = np.isfinite(values).all(axis=1)
    if not ok.any():
        raise SystemExit("the object has no points in any frame")
    return np.stack([np.interp(frames, frames[ok], values[ok, k]) for k in range(values.shape[1])], axis=1)


def object_centres(points: np.ndarray, offsets: np.ndarray, extent: float) -> np.ndarray:
    """Median of each frame's points at most ``extent`` deeper than its nearest tenth; NaN when a
    frame has too few points."""
    out = np.full((len(offsets) - 1, 3), np.nan)
    for i in range(len(out)):
        frame = points[offsets[i]:offsets[i + 1]]
        if len(frame) >= 20:
            out[i] = np.median(frame[frame[:, 2] <= np.percentile(frame[:, 2], 10) + extent], axis=0)
    return out


HANDS = (range(21, 42), range(42, 63))  # MHR70 right and left hand keypoints (wrists at 41, 62)


def hand_positions(verts: np.ndarray, keypoints: np.ndarray, intrinsics: np.ndarray,
                   margin: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Per frame and hand: the 2D centre of its keypoints [N, 2, 2] and its depth [N, 2], the 20th
    percentile depth of body vertices projecting into the keypoints' box (NaN if too few)."""
    uvw = verts @ intrinsics.T
    uv = uvw[..., :2] / uvw[..., 2:]
    centres, depths = np.zeros((len(verts), 2, 2)), np.full((len(verts), 2), np.nan)
    for h, index in enumerate(HANDS):
        points = keypoints[:, list(index)]
        low, high = points.min(axis=1) - margin, points.max(axis=1) + margin
        centres[:, h] = points.mean(axis=1)
        for i in range(len(verts)):
            inside = np.all((uv[i] >= low[i]) & (uv[i] <= high[i]), axis=1)
            if inside.sum() >= 10:
                depths[i, h] = np.percentile(verts[i, inside, 2], 20)
    return centres, depths


def pull_to_hands(centre: np.ndarray, hands: tuple[np.ndarray, np.ndarray], intrinsics: np.ndarray,
                  extent: float, reach: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Move an object position onto its own viewing ray at hand depth when it lies within
    ``reach`` metres of a hand in the image but more than ``extent`` behind it: then the mask's
    depth is background seen past a thin, held object. With both hands in reach it takes their
    mean depth. Returns the positions and which moved."""
    hand_uv, hand_z = hands
    uvw = centre @ intrinsics.T
    uv = uvw[:, :2] / uvw[:, 2:]
    near = np.linalg.norm(uv[:, None] - hand_uv, axis=2) < intrinsics[0, 0] * reach / hand_z
    count = near.sum(axis=1)
    z = np.where(near, hand_z, 0).sum(axis=1) / np.maximum(count, 1)
    moved = (count > 0) & np.isfinite(centre[:, 2]) & (centre[:, 2] > z + extent)
    out = centre.copy()
    out[moved] = centre[moved] * (z[moved] / centre[moved, 2])[:, None]
    return out, moved


def despike(values: np.ndarray, frames: np.ndarray, size: int = 5) -> np.ndarray:
    """Median filter along time within each stretch of consecutive frames."""
    out = values.copy()
    for part in runs(frames):
        out[part] = ndimage.median_filter(values[part], size=(size, 1), mode="nearest")
    return out


def object_mesh(cloud: np.ndarray, view: np.ndarray, voxels: int = 40) -> trimesh.Trimesh:
    """Close a visible-surface point cloud (object frame) into a mesh: drop depth outliers, mirror
    about the far side along ``view``, voxelise, and run marching cubes."""
    along = cloud @ view
    spread = 1.4826 * np.median(np.abs(along - np.median(along)))
    cloud = cloud[np.abs(along - np.median(along)) <= 4 * spread + 0.02]
    along = cloud @ view
    far = np.percentile(along, 90)
    full = np.concatenate([cloud, cloud - 2 * (along - far)[:, None] * view])
    pitch = float(np.clip(np.ptp(full, axis=0).max() / voxels, 0.004, 0.03))
    low = full.min(axis=0) - 2 * pitch
    index = np.floor((full - low) / pitch).astype(int)
    matrix = np.zeros(index.max(axis=0) + 3, bool)
    matrix[tuple(index.T)] = True
    matrix = ndimage.binary_fill_holes(ndimage.binary_closing(ndimage.binary_dilation(matrix), iterations=2))
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(matrix, pitch=pitch)
    mesh.apply_translation(low)
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/track_1"))
    parser.add_argument("--human", type=Path, default=Path("work/human"))
    parser.add_argument("--object", type=Path, default=Path("work/object"))
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing of translation, in frames")
    parser.add_argument("--around", type=int, default=2, help="neighbouring frames merged into the mesh")
    parser.add_argument("--out", type=Path, default=Path("out/v0"))
    args = parser.parse_args()

    name = f"episode_{args.episode:06d}"
    lengths = {e["episode_index"]: e["length"] for e in map(json.loads, (args.data / "meta/episodes.jsonl").read_text().splitlines())}
    length = lengths[args.episode]
    body = np.load(args.human / f"{name}_mhr.npz")
    track = np.load(args.object / f"{name}.npz")
    frames, scored = track["frames"], body["scored"]
    if not np.array_equal(frames, body["frames"]):
        raise SystemExit("body and object were run on different frames")

    ratio = np.where(np.isfinite(track["ratio"]), track["ratio"], np.nanmedian(track["ratio"]))
    points, offsets, area = track["points"], track["point_offsets"], track["area"]
    best = int(np.argmax(area))
    depth = points[offsets[best]:offsets[best + 1], 2]
    extent = np.percentile(depth, 90) - np.percentile(depth, 10) + 0.1
    centre = object_centres(points, offsets, extent) * ratio[:, None]
    raw = np.load(args.human / f"{name}.npz")
    hands = hand_positions(raw["verts_cam"], raw["keypoints_2d"], raw["intrinsics"])
    centre, moved = pull_to_hands(centre, hands, raw["intrinsics"], extent * np.median(ratio))
    filled = interpolate_missing(frames, centre)
    translation = smooth(despike(filled, frames), frames, args.sigma)

    picks = [i for i in range(best - args.around, best + args.around + 1)
             if 0 <= i < len(frames) and offsets[i + 1] > offsets[i] and np.isfinite(centre[i]).all()]
    cloud = np.concatenate([points[offsets[i]:offsets[i + 1]] * ratio[i] - centre[i] for i in picks])
    mesh = object_mesh(cloud, centre[best] / np.linalg.norm(centre[best]))

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"{name}.npz",
             pose=fill_nearest(frames, body["pose"], length).astype(np.float32),
             scales=body["scales"], shape=body["shape"],
             object_rotation=np.broadcast_to(np.eye(3), (length, 3, 3)).copy(),
             object_translation=fill_nearest(frames, translation, length),
             object_scale=np.float64(1.0))
    mesh.export(args.out / f"{name}_object.ply")

    seen = np.isfinite(centre).all(axis=1)
    print(f"{name}: body depth ratio {np.median(ratio):.3f}, object points on {seen.mean():.0%} of frames, "
          f"{moved.sum()} moved to hand depth, "
          f"mesh from frames {frames[picks].tolist()}: {np.round(mesh.extents * 100).astype(int).tolist()} cm, "
          f"{len(mesh.faces)} faces, watertight {mesh.is_watertight}")
    print(f"object translation acceleration at scored frames: raw {acceleration_cm(filled[:, None], frames, scored):.3f} cm, "
          f"smoothed {acceleration_cm(translation[:, None], frames, scored):.3f} cm")
    print(f"wrote {args.out / name}.npz and _object.ply ({length} frames)")


if __name__ == "__main__":
    main()
