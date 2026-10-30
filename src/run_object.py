"""Find, track and back-project the episode's object over the processed frames of one Track 1 episode.

Reads the processed frames, person boxes, camera matrix and camera-frame body vertices written by
run_human.py. Per contiguous stretch of frames, Grounding DINO boxes the object from the episode's
text prompt (preferring boxes near the person), SAM 2.1 tracks that box as a mask, and MoGe-2 gives
a metric point map per frame with the same field of view. The body's SAM 3D Body depth over its
MoGe depth is stored per frame, so the object points can be put on the body's scale later. One
model is loaded at a time. Run inside ~/v2d/.venv-obj:

    python src/run_object.py --episode 14
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from run_human import iter_frames, release

GDINO = "IDEA-Research/grounding-dino-base"
SAM2 = "facebook/sam2.1-hiera-large"
MOGE = "Ruicheng/moge-2-vitl-normal"


def stretches(frames: list[int]) -> list[list[int]]:
    """Maximal runs of consecutive frame numbers."""
    out = [[frames[0]]]
    for frame in frames[1:]:
        if frame == out[-1][-1] + 1:
            out[-1].append(frame)
        else:
            out.append([frame])
    return out


def pick_box(boxes: np.ndarray, scores: np.ndarray, person: np.ndarray, shape: tuple) -> tuple | None:
    """The best detection touching the person's box grown by half its size, else the best overall.
    Boxes covering over half the image (the whole room) are ignored."""
    area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    keep = area < 0.5 * shape[0] * shape[1]
    boxes, scores = boxes[keep], scores[keep]
    if not len(boxes):
        return None
    grow = (person[2:] - person[:2]) / 2
    low, high = person[:2] - grow, person[2:] + grow
    near = np.all((boxes[:, :2] < high) & (boxes[:, 2:] > low), axis=1)
    pool = np.flatnonzero(near) if near.any() else np.arange(len(boxes))
    best = pool[scores[pool].argmax()]
    return boxes[best], float(scores[best])


def find_object(video: Path, parts: list[list[int]], person: dict[int, np.ndarray], prompt: str,
                tries: int = 10) -> dict[int, tuple[int, np.ndarray, float]]:
    """Per stretch, keyed by its first frame: the earliest of its first ``tries`` frames where
    Grounding DINO finds the prompt, as (frame, box, score)."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(GDINO)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).cuda().eval()
    opening = {frame: part[0] for part in parts for frame in part[:tries]}
    found = {}
    with torch.no_grad():
        for index, rgb in iter_frames(video, sorted(opening)):
            if opening[index] in found:
                continue
            inputs = processor(images=rgb, text=prompt, return_tensors="pt").to("cuda")
            result = processor.post_process_grounded_object_detection(
                model(**inputs), inputs.input_ids, threshold=0.25, text_threshold=0.2,
                target_sizes=[rgb.shape[:2]])[0]
            if len(result["boxes"]):
                pick = pick_box(result["boxes"].cpu().numpy(), result["scores"].cpu().numpy(), person[index], rgb.shape)
                if pick is not None:
                    found[opening[index]] = (index, *pick)
    return found


def track_object(video: Path, frames: list[int], starts: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Bit-packed SAM 2.1 masks per frame, tracked forward from each stretch's detection frame
    with a fresh streaming session. Frames before a stretch's detection get no mask."""
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    processor = Sam2VideoProcessor.from_pretrained(SAM2)
    model = Sam2VideoModel.from_pretrained(SAM2, dtype=torch.bfloat16).cuda().eval()
    masks, session = {}, None
    with torch.no_grad():
        for index, rgb in iter_frames(video, frames):
            if index in starts:
                session = processor.init_video_session(inference_device="cuda", dtype=torch.bfloat16)
            elif index - 1 not in masks:  # a gap, or a stretch whose detection is still ahead
                session = None
                continue
            inputs = processor(images=rgb, device="cuda", return_tensors="pt")
            if index in starts:
                processor.add_inputs_to_inference_session(
                    inference_session=session, frame_idx=0, obj_ids=1,
                    input_boxes=[[starts[index].tolist()]], original_size=inputs.original_sizes[0])
            output = model(inference_session=session, frame=inputs.pixel_values[0].to(torch.bfloat16))
            mask = processor.post_process_masks([output.pred_masks], original_sizes=inputs.original_sizes,
                                                binarize=True)[0]
            masks[index] = np.packbits(mask.cpu().numpy().reshape(-1, *rgb.shape[:2])[0])
    return masks


def body_depth_ratio(verts: np.ndarray, intrinsics: np.ndarray, depth: np.ndarray, cell: int = 8) -> float:
    """Median of SAM 3D Body depth over MoGe depth at the body's front-most vertex per image cell."""
    uvw = verts @ intrinsics.T
    u, v = (uvw[:, :2] / uvw[:, 2:]).round().astype(int).T
    h, w = depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], verts[inside, 2]
    order = np.argsort(z)
    _, first = np.unique(((v // cell) * w + u // cell)[order], return_index=True)
    front = order[first]
    moge = depth[v[front], u[front]]
    ok = np.isfinite(moge) & (moge > 0)
    return float(np.median(z[front][ok] / moge[ok])) if ok.sum() >= 50 else np.nan


def measure_points(video: Path, frames: list[int], masks: dict[int, np.ndarray], verts: np.ndarray,
                   intrinsics: np.ndarray, debug: dict[int, Path], sample: int = 2000) -> dict[str, np.ndarray]:
    """MoGe-2 point maps per frame: the body depth ratio, the object mask's area and box, the median
    of its (eroded) points, and up to ``sample`` of those points, all in raw MoGe metres."""
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(MOGE).cuda().eval()
    rng = np.random.default_rng(0)
    kernel = np.ones((5, 5), np.uint8)
    out = {key: [] for key in ("ratio", "area", "mask_box", "centre", "points")}
    with torch.no_grad():
        for i, (index, rgb) in enumerate(iter_frames(video, frames)):
            h, w = rgb.shape[:2]
            fov_x = float(np.degrees(2 * np.arctan(w / (2 * intrinsics[0, 0]))))
            result = model.infer(torch.from_numpy(rgb).cuda().permute(2, 0, 1).float() / 255, fov_x=fov_x)
            cloud, depth = result["points"].cpu().numpy(), result["depth"].cpu().numpy()
            out["ratio"].append(body_depth_ratio(verts[i], intrinsics, depth))
            mask = (np.unpackbits(masks[index])[:h * w].reshape(h, w).astype(bool) if index in masks
                    else np.zeros((h, w), bool))
            ys, xs = np.nonzero(mask)
            out["area"].append(len(xs))
            out["mask_box"].append([xs.min(), ys.min(), xs.max(), ys.max()] if len(xs) else [-1] * 4)
            inner = cv2.erode(mask.astype(np.uint8), kernel).astype(bool) & np.isfinite(depth)
            points = cloud[inner]
            out["centre"].append(np.median(points, axis=0) if len(points) >= 20 else np.full(3, np.nan))
            if len(points) > sample:
                points = points[rng.choice(len(points), sample, replace=False)]
            out["points"].append(points.astype(np.float32))
            if index in debug:
                overlay = rgb.copy()
                overlay[mask] = (0.5 * overlay[mask] + [127, 0, 0]).astype(np.uint8)
                cv2.imwrite(str(debug[index]), cv2.resize(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), None, fx=0.5, fy=0.5))
    points = out.pop("points")
    return {**{key: np.array(value) for key, value in out.items()},
            "points": np.concatenate(points), "point_offsets": np.cumsum([0] + [len(p) for p in points])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/track_1"))
    parser.add_argument("--human", type=Path, default=Path("work/human"))
    parser.add_argument("--out", type=Path, default=Path("work/object"))
    args = parser.parse_args()

    video = args.data / f"videos/chunk-000/observation.images.exo_camera/episode_{args.episode:06d}.mp4"
    meta = {e["episode_index"]: e for e in map(json.loads, (args.data / "meta/episodes_metadata.jsonl").read_text().splitlines())}
    prompt = meta[args.episode]["object_prompt"].replace("_", " ").strip().lower().rstrip(".") + "."
    human = np.load(args.human / f"episode_{args.episode:06d}.npz")
    frames = human["frames"].tolist()
    parts = stretches(frames)
    print(f"episode {args.episode}: '{prompt}', {len(frames)} frames in {len(parts)} stretches")

    timings = {}
    started = time.time()
    found = find_object(video, parts, dict(zip(frames, human["boxes"])), prompt)
    release()
    timings["find"] = time.time() - started
    print(f"object found in {len(found)} of {len(parts)} stretches, scores "
          f"{[round(score, 2) for _, _, score in found.values()]}")

    started = time.time()
    masks = track_object(video, frames, {index: box for index, box, _ in found.values()})
    release()
    timings["track"] = time.time() - started

    debug_dir = args.out / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, 4).astype(int)]
    debug = {index: debug_dir / f"episode_{args.episode:06d}_frame{index:05d}.jpg" for index in picks}
    started = time.time()
    measured = measure_points(video, frames, masks, human["verts_cam"], human["intrinsics"], debug)
    release()
    timings["points"] = time.time() - started

    out = args.out / f"episode_{args.episode:06d}.npz"
    detections = np.array([[start, index, *box, score] for start, (index, box, score) in sorted(found.items())]).reshape(-1, 7)
    np.savez(out, frames=np.array(frames), prompt=prompt, detections=detections, **measured)
    seen = measured["area"] > 0
    print(f"seconds per frame: { {stage: round(s / len(frames), 2) for stage, s in timings.items()} }, "
          f"total {sum(timings.values()) / 60:.1f} min")
    print(f"mask on {seen.mean():.0%} of frames, median area {np.median(measured['area'][seen]) if seen.any() else 0:.0f} px; "
          f"body depth ratio median {np.nanmedian(measured['ratio']):.3f}, "
          f"10-90% {np.nanpercentile(measured['ratio'], 10):.3f}-{np.nanpercentile(measured['ratio'], 90):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
