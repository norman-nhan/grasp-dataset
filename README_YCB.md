# YCB-affordance grasp-candidate dataset pipeline

This guide generates OpenArm parallel-jaw grasp annotations from the exported
YCB human-grasp-affordance point clouds. GraspNet proposes candidates from
multiple occlusion-aware views; the complete metric cloud then checks gripper
occupancy, approach clearance, contacts, antipodality, and friction cones.

The point-wise affordance labels are proximity-derived from annotated MANO hand
meshes. They are human-grasp pseudo-contact labels, not measured force or robot
grasp-success labels.

## 1. Enter the project

```bash
cd /ros2_ws/not-ros-pkg/dataset-generation
source .venv/bin/activate
```

The expected exported input is:

```text
data/ycb_affordance_data/<object-id>/
├── affordance.npz
├── affordance.ply
└── metadata.json
```

YCB metadata marks these point coordinates as metres. Do not pass
`--longest-extent-m` for normal YCB exports.

If the affordance files have not been exported, run the exporter from
`/ros2_ws/not-ros-pkg/ycb_affordance_dataset` in its own environment:

```bash
python -u export_human_grasp_affordance.py
```

It writes `outputs/human_grasp_affordance/<object-id>/`. Either copy/link that
directory to `dataset-generation/data/ycb_affordance_data`, or pass each
`affordance.npz` explicitly with `--npz`.

## 2. Optional: prepare textured MuJoCo models

The official downloader is configured to fetch only the selected object IDs and
their Google-16k meshes:

```bash
source .venv/bin/activate
python download_ycb_dataset.py
```

Convert downloaded meshes to standalone MuJoCo models:

```bash
python convert_ycb_to_mujoco.py \
  --input assets/ycb-affordance \
  --output assets/ycb-affordance-mujoco \
  --downsampling-ratio 1.0
```

`1.0` keeps the full mesh as CoACD input. A smaller value simplifies only the
collision-decomposition input; it never downsamples the visual mesh. Each
converted object contains `model.xml`, the textured visual assets, convex
collision meshes, and `metadata.json`.

Inspect a model with:

```bash
python -m mujoco.viewer \
  --mjcf assets/ycb-affordance-mujoco/025_mug/model.xml
```

The converted model is useful for visualization and later simulation. The
point-cloud evaluator in section 4 reads `affordance.npz`, not `model.xml`.

## 3. Generate multi-view GraspNet candidates

Activate the CUDA GraspNet environment:

```bash
source /ros2_ws/.venv/graspnet_env/bin/activate
```

First run one object:

```bash
OBJECT=025_mug
OUT=outputs/graspnet_multiview/ycb/$OBJECT
mkdir -p "$OUT"

python generate_multiview_graspnet_grasp.py \
  --data-root data/ycb_affordance_data \
  --object "$OBJECT" \
  --views 8 \
  --max-candidates 200 \
  --output-json "$OUT/candidates.json" \
  --output "$OUT/candidates.png" \
  --views-output "$OUT/visible_views.png"
```

YCB metric coordinates are detected from `metadata.json`; the saved annotation
should report `metric_scale.method` as `source_metric_coordinates` with
`scale_to_m: 1.0`.

The generator samples random camera directions, uses perspective point-splat
z-buffer visibility to form partial clouds, runs GraspNet on each partial view,
combines all proposals, applies NMS, checks the surviving proposals against the
complete cloud, and saves up to `--max-candidates` candidates.

This is visibility-based sampling rather than distance cropping: a point hidden
behind a nearer projected surface is excluded from that camera view.

## 4. Check collision, contacts, antipodality, and friction cones

Return to the project environment:

```bash
source .venv/bin/activate

python eval_grasp.py \
  --grasps "$OUT/candidates.json" \
  --friction 0.8 \
  --output-json "$OUT/evaluated.json" \
  --output "$OUT/evaluated.png"
```

For each candidate, evaluation is ordered as follows:

1. Check full-cloud occupancy in the left finger, right finger, and palm.
2. Check the straight approach sweep against the complete cloud.
3. Locate left/right contact bands.
4. Check opposing normals and the Coulomb friction cones.

A collision failure is rejected before antipodal evaluation and records
`antipodal_evaluated: false`. Accepted rows are under `antipodal.rows`; their
indices in the original GraspNet array are under `antipodal.source_indices`.

View the output summary without opening the PNG:

```bash
python - <<'PY'
import json
from pathlib import Path

result = json.loads(Path("outputs/graspnet_multiview/ycb/025_mug/evaluated.json").read_text())
print(result["antipodal"]["summary"])
PY
```

## 5. Run candidate generation for every exported YCB object

This loop skips objects that already have `candidates.json` and logs only
objects whose generation command failed:

```bash
source /ros2_ws/.venv/graspnet_env/bin/activate

ROOT=data/ycb_affordance_data
OUT_ROOT=outputs/graspnet_multiview/ycb
FAILED="$OUT_ROOT/failed_objects.txt"
mkdir -p "$OUT_ROOT"
: > "$FAILED"

find "$ROOT" -mindepth 2 -maxdepth 2 -name affordance.npz -print0 |
while IFS= read -r -d '' NPZ; do
  OBJECT=$(basename "$(dirname "$NPZ")")
  OUT="$OUT_ROOT/$OBJECT"
  if [ -f "$OUT/candidates.json" ]; then
    echo "Skipping $OBJECT"
    continue
  fi
  mkdir -p "$OUT"
  python generate_multiview_graspnet_grasp.py \
    --npz "$NPZ" \
    --views 8 \
    --max-candidates 200 \
    --output-json "$OUT/candidates.json" \
    --output "$OUT/candidates.png" \
    --views-output "$OUT/visible_views.png" || echo "$OBJECT" >> "$FAILED"
done
```

Then evaluate every successful annotation:

```bash
source .venv/bin/activate

find outputs/graspnet_multiview/ycb -name candidates.json -print0 |
while IFS= read -r -d '' GRASPS; do
  OUT=$(dirname "$GRASPS")
  [ -f "$OUT/evaluated.json" ] && continue
  python eval_grasp.py \
    --grasps "$GRASPS" \
    --output-json "$OUT/evaluated.json" \
    --output "$OUT/evaluated.png"
done
```

## Output interpretation

- `candidates.json`: merged GraspNet Nx17 proposals in the YCB object frame.
- `evaluated.json`: original proposals plus collision, contact, antipodal, and
  friction-cone diagnostics.
- `antipodal.rows`: candidates accepted by this geometric point-cloud test.
- `*.png`: inspection images only; JSON is the machine-readable annotation.
- Antipodal acceptance is not physical success. It does not establish arm IK,
  reachability, gravity stability, dynamic behavior, or real execution success.
