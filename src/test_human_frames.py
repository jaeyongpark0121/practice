"""Quick check of SAM 3D Body on a few frames of one Track 1 video.

Loads one model at a time to stay inside the WSL memory cap: a light torchvision person
detector for the box, MoGe-2 for the camera intrinsics, then SAM 3D Body. Writes overlay
images and the raw per-frame outputs. Run inside the v2d-ubuntu env from the project root:

    python src/test_human_frames.py --episode 14 --frames 90 110 130 150 170
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def read_frames(video: Path, indices: list[int]) -> list[np.ndarray]:
    """BGR frames at the given indices, decoded sequentially so the indices are exact."""
    capture, wanted, frames = cv2.VideoCapture(str(video)), set(indices), {}
    index = 0
    while len(frames) < len(wanted):
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = bgr
        index += 1
    capture.release()
    return [frames[i] for i in indices]


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def report(stage: str, started: float, count: int) -> None:
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[{stage}] {(time.time() - started) / count:.2f} s/frame, GPU peak {peak:.2f} GiB")
    torch.cuda.reset_peak_memory_stats()


def detect_person_boxes(frames_rgb: list[np.ndarray]) -> np.ndarray:
    """One [x0, y0, x1, y1] box per frame: the highest-scoring COCO person."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").cuda().eval()
    boxes = []
    with torch.no_grad():
        for rgb in frames_rgb:
            image = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda() / 255
            found = model([image])[0]
            people = found["labels"] == 1
            if not people.any():
                raise SystemExit("no person detected in a test frame")
            best = found["scores"].masked_fill(~people, -1).argmax()
            boxes.append(found["boxes"][best].cpu().numpy())
    return np.stack(boxes)


def estimate_intrinsics(frames_rgb: list[np.ndarray]) -> np.ndarray:
    """One 3x3 K for the static camera: the median of MoGe-2's per-frame estimates."""
    from tools.build_fov_estimator import load_moge, run_moge

    model = load_moge("cuda").eval()
    with torch.no_grad():
        per_frame = np.concatenate([run_moge(model, rgb, "cuda") for rgb in frames_rgb])
    return np.median(per_frame, axis=0)


def run_body(frames_rgb, boxes, intrinsics, checkpoint: Path):
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, config = load_sam_3d_body(str(checkpoint / "model.ckpt"), device="cuda",
                                     mhr_path=str(checkpoint / "assets/mhr_model.pt"))
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=config,
                                   human_detector=None, human_segmentor=None, fov_estimator=None)
    cam_int = torch.from_numpy(intrinsics[None]).float()
    outputs = [estimator.process_one_image(rgb, bboxes=box[None], cam_int=cam_int)[0]
               for rgb, box in zip(frames_rgb, boxes)]
    return outputs, estimator.faces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", type=int, default=14)
    parser.add_argument("--frames", type=int, nargs="+", default=[90, 110, 130, 150, 170])
    parser.add_argument("--data", type=Path, default=Path("data/track_1"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path.home() / "v2d/checkpoints/sam-3d-body-vith")
    parser.add_argument("--sam3d-repo", type=Path, default=Path.home() / "v2d/sam-3d-body")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(args.sam3d_repo))
    out = args.out or Path(f"work/test_human_ep{args.episode:02d}")
    out.mkdir(parents=True, exist_ok=True)
    video = (args.data / "videos/chunk-000/observation.images.exo_camera"
             / f"episode_{args.episode:06d}.mp4")

    frames_bgr = read_frames(video, args.frames)
    frames_rgb = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in frames_bgr]

    started = time.time()
    boxes = detect_person_boxes(frames_rgb)
    release()
    report("person boxes", started, len(frames_rgb))

    started = time.time()
    intrinsics = estimate_intrinsics(frames_rgb)
    release()
    report("intrinsics", started, len(frames_rgb))
    print("K =", np.round(intrinsics, 1).tolist())

    started = time.time()
    outputs, faces = run_body(frames_rgb, boxes, intrinsics, args.checkpoint)
    report("SAM 3D Body", started, len(frames_rgb))

    from tools.vis_utils import visualize_sample_together

    for frame, bgr, output in zip(args.frames, frames_bgr, outputs):
        overlay = visualize_sample_together(bgr, [output], faces)
        cv2.imwrite(str(out / f"frame_{frame:06d}.jpg"), overlay.astype(np.uint8))
    keys = ("pred_vertices", "pred_cam_t", "focal_length", "mhr_model_params", "shape_params",
            "scale_params", "pred_keypoints_2d")
    np.savez(out / "outputs.npz", frames=np.array(args.frames), boxes=boxes, intrinsics=intrinsics,
             **{key: np.stack([np.asarray(o[key]) for o in outputs]) for key in keys})
    print(json.dumps({key: list(np.shape(outputs[0][key])) for key in keys}))
    print(f"wrote {len(outputs)} overlays and outputs.npz to {out}")


if __name__ == "__main__":
    main()
