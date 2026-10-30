# V2D Challenge Track 1 — team T0121

Monocular third-person video → MHR human and rigid object per episode, run on one laptop
(RTX 4060 Laptop 8 GB, WSL2 Ubuntu 22.04). Data, weights, the official kit and outputs are not
in this repository.

## Human (`~/v2d/.venv`: SAM 3D Body, MoGe-2, torchvision)

1. `src/scored_frames.py` lists each episode's scored frames from the sample submission.
2. `src/run_human.py` processes every scored frame ±5: Faster R-CNN person boxes linked over time
   (Viterbi, so brief jumps to bystanders are ignored), one camera matrix from MoGe-2, and
   SAM 3D Body vertices in the OpenCV camera frame.
3. `src/fit_body_identity.py` Gaussian-smooths the vertices within each stretch and fits MHR
   pose, scales and shape with the kit's `mesh_to_mhr_params.convert`.

## Object (`~/v2d/.venv-obj`: Grounding DINO, SAM 2.1, MoGe-2)

4. `src/run_object.py` boxes the object from the episode's text prompt with Grounding DINO near
   the person, tracks it with SAM 2.1, and back-projects the mask with MoGe-2. It also stores the
   per-frame SAM 3D Body / MoGe depth ratio.
5. `src/build_episode.py` puts the object on the body's scale and models it as translation only.
   It uses the nearest-depth median of the points, and hand depth when a held thin object shows
   background. It closes the visible surface into a voxel mesh and writes the kit's episode NPZ
   and `_object.ply`.
6. `src/render_check.py` draws the posed mesh and body over frames for a visual check.

Batches: `src/run_all_human.sh`, `src/run_all_object.sh`, launched through
`src/run_all_human_overnight.ps1` (keeps Windows awake).
