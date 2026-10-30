"""Run SAM 3D Body over the scored stretches of one Track 1 episode.

Frames are streamed from the video once per stage, and each stage loads a single model, so a
long episode fits inside the WSL memory cap: a torchvision person detector for the box, MoGe-2
for one camera matrix per (static) camera, then SAM 3D Body. Vertices are saved in the OpenCV
camera frame in metres (``pred_vertices + pred_cam_t``). Run inside the v2d-ubuntu env:

    python src/run_human.py --episode 14
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def frames_to_process(scored: list[int], length: int, pad: int) -> list[int]:
    """Every frame within ``pad`` of a scored frame, so smoothing has context at stretch ends."""
    wanted = {f + d for f in scored for d in range(-pad, pad + 1)}
    return sorted(f for f in wanted if 0 <= f < length)


def iter_frames(video: Path, indices: list[int]):
    """Yield ``(index, rgb)`` for the sorted indices, decoding sequentially so they are exact."""
    capture, wanted, index = cv2.VideoCapture(str(video)), set(indices), 0
    last = indices[-1]
    while index <= last:
        ok, bgr = capture.read()
        if not ok:
            raise SystemExit(f"{video} ended at frame {index}, before {last}")
        if index in wanted:
            yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` [N, 4] with every box in ``b`` [M, 4], as [N, M]."""
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    overlap = np.prod(np.clip(high - low, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return overlap / (area(a)[:, None] + area(b)[None] - overlap)


def follow_one_person(frames: list[int], candidates: list[tuple[np.ndarray, np.ndarray]],
                      switch_cost: float = 20.0) -> np.ndarray:
    """One box per frame by Viterbi. Each frame rewards its top-scoring person (1 point, plus the
    score to break ties) and charges ``switch_cost`` for jumping to a box that barely overlaps the
    previous pick in the next video frame, so the top-scoring person wins except for runs of fewer
    than ``2 * switch_cost`` frames where a bystander (camera operator, seated people) edges ahead.
    Moving within a stretch costs nothing, and each new stretch after a gap is chosen afresh."""
    def reward(scores: np.ndarray) -> np.ndarray:
        return (scores == scores.max()) * (scores.max() > 0) + 0.1 * scores

    totals, back = reward(candidates[0][1]), []
    for (before, (previous, _)), (frame, (boxes, scores)) in zip(zip(frames[:-1], candidates[:-1]),
                                                                  zip(frames[1:], candidates[1:])):
        step = totals[:, None] - switch_cost * (box_iou(previous, boxes) < 0.3) * (frame - before == 1)
        back.append(step.argmax(axis=0))
        totals = step.max(axis=0) + reward(scores)
    pick = [int(totals.argmax())]
    for pointers in reversed(back):
        pick.append(int(pointers[pick[-1]]))
    pick.reverse()
    return np.stack([boxes[i] for (boxes, _), i in zip(candidates, pick)])


def detect_person_candidates(video: Path, indices: list[int], min_score: float = 0.2,
                             keep: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    """COCO person boxes and scores per frame, best first. A frame with no candidate keeps the
    previous frame's boxes at zero score."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").cuda().eval()
    candidates = []
    with torch.no_grad():
        for _, rgb in iter_frames(video, indices):
            found = model([torch.from_numpy(rgb).permute(2, 0, 1).float().cuda() / 255])[0]
            people = (found["labels"] == 1) & (found["scores"] >= min_score)  # sorted by score
            boxes, scores = found["boxes"][people][:keep].cpu().numpy(), found["scores"][people][:keep].cpu().numpy()
            if len(boxes) == 0:
                if not candidates:
                    raise SystemExit("no person in the first processed frame")
                boxes, scores = candidates[-1][0], np.zeros(len(candidates[-1][0]), np.float32)
            candidates.append((boxes, scores))
    return candidates


def detect_person_boxes(video: Path, indices: list[int]) -> np.ndarray:
    return follow_one_person(indices, detect_person_candidates(video, indices))


def estimate_intrinsics(video: Path, indices: list[int], samples: int = 5) -> np.ndarray:
    """One 3x3 K for the static camera: the median of MoGe-2 on evenly spaced frames."""
    from tools.build_fov_estimator import load_moge, run_moge

    picks = [indices[i] for i in np.linspace(0, len(indices) - 1, samples).astype(int)]
    model = load_moge("cuda").eval()
    with torch.no_grad():
        per_frame = np.concatenate([run_moge(model, rgb, "cuda") for _, rgb in iter_frames(video, sorted(set(picks)))])
    return np.median(per_frame, axis=0)


def run_body(video: Path, indices: list[int], boxes: np.ndarray, intrinsics: np.ndarray,
             checkpoint: Path) -> dict[str, np.ndarray]:
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, config = load_sam_3d_body(str(checkpoint / "model.ckpt"), device="cuda",
                                     mhr_path=str(checkpoint / "assets/mhr_model.pt"))
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=config,
                                   human_detector=None, human_segmentor=None, fov_estimator=None)
    cam_int = torch.from_numpy(intrinsics[None]).float()
    collected = {key: [] for key in ("verts_cam", "mhr_model_params", "shape_params", "keypoints_2d")}
    for (_, rgb), box in zip(iter_frames(video, indices), boxes):
        with contextlib.redirect_stdout(io.StringIO()):  # the estimator prints on every call
            output = estimator.process_one_image(rgb, bboxes=box[None], cam_int=cam_int)[0]
        collected["verts_cam"].append((output["pred_vertices"] + output["pred_cam_t"]).astype(np.float32))
        collected["mhr_model_params"].append(output["mhr_model_params"])
        collected["shape_params"].append(output["shape_params"])
        collected["keypoints_2d"].append(output["pred_keypoints_2d"])
    return {key: np.stack(value) for key, value in collected.items()}


def projection_iou(verts_cam: np.ndarray, intrinsics: np.ndarray, boxes: np.ndarray) -> float:
    """Mean IoU between the detector box and the box of the projected vertices."""
    uv = verts_cam @ intrinsics.T
    uv = uv[..., :2] / uv[..., 2:]
    projected = np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1)
    low, high = np.maximum(projected[:, :2], boxes[:, :2]), np.minimum(projected[:, 2:], boxes[:, 2:])
    overlap = np.prod(np.clip(high - low, 0, None), axis=1)
    area = lambda b: np.prod(b[:, 2:] - b[:, :2], axis=1)
    return float(np.mean(overlap / (area(projected) + area(boxes) - overlap)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--pad", type=int, default=5, help="extra frames around each scored stretch")
    parser.add_argument("--data", type=Path, default=Path("data/track_1"))
    parser.add_argument("--scored", type=Path, default=Path("work/scored_frames.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "v2d/checkpoints/sam-3d-body-vith")
    parser.add_argument("--sam3d-repo", type=Path, default=Path.home() / "v2d/sam-3d-body")
    parser.add_argument("--out", type=Path, default=Path("work/human"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.sam3d_repo))
    video = args.data / f"videos/chunk-000/observation.images.exo_camera/episode_{args.episode:06d}.mp4"
    lengths = {e["episode_index"]: e["length"] for e in map(json.loads, (args.data / "meta/episodes.jsonl").read_text().splitlines())}
    scored = json.loads(args.scored.read_text())[str(args.episode)]
    indices = frames_to_process(scored, lengths[args.episode], args.pad)
    print(f"episode {args.episode}: {len(scored)} scored frames, processing {len(indices)}")

    timings = {}
    started = time.time()
    boxes = detect_person_boxes(video, indices)
    release()
    timings["boxes"] = time.time() - started

    started = time.time()
    intrinsics = estimate_intrinsics(video, indices)
    release()
    timings["intrinsics"] = time.time() - started

    started = time.time()
    body = run_body(video, indices, boxes, intrinsics, args.checkpoint)
    timings["body"] = time.time() - started
    release()

    out = args.out / f"episode_{args.episode:06d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, frames=np.array(indices), scored=np.array(scored), boxes=boxes, intrinsics=intrinsics, **body)
    per_frame = {stage: round(seconds / len(indices), 2) for stage, seconds in timings.items()}
    print(f"seconds per frame: {per_frame}, total {sum(timings.values()) / 60:.1f} min")
    print(f"K = {np.round(intrinsics, 1).tolist()}")
    print(f"projected-vs-detected box IoU: {projection_iou(body['verts_cam'], intrinsics, boxes):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
