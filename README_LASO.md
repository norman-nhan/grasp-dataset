# LASO grasp-candidate dataset pipeline

This guide generates OpenArm parallel-jaw grasp annotations from LASO human
grasp-affordance point clouds. GraspNet proposes candidates from multiple
occlusion-aware views. The complete cloud, reconstructed mesh, and actual
OpenArm gripper geometry are then used for geometric validation.

LASO does not contain measured physical scale, mass, material, shell thickness,
or watertight object meshes. The metric dimensions and physics used here are
explicit simulation assumptions, not ground truth.

## 1. Enter the project

Run preprocessing and evaluation in the project environment:

```bash
cd /ros2_ws/not-ros-pkg/dataset-generation
source .venv/bin/activate
```

The expected source layout is:

```text
data/laso_grasp_only_affordance/<class>/<object-id>/
├── affordance.npz
├── affordance.ply
└── metadata.json
```

If these files have not been exported yet, run the LASO reviewer exporter from
`/ros2_ws/not-ros-pkg/laso-reviewer`:

```bash
source ~/.venv/laso_env/bin/activate
python export_human_grasp_affordance.py \
  --dataset data/LASO_dataset \
  --output-dir ../dataset-generation/data/laso_grasp_only_affordance
```

The exporter skips shapes without a LASO `grasp` annotation instead of treating
them as negative examples.

## 2. Prepare reconstructed and collision geometry

Reconstruct all point clouds:

```bash
source .venv/bin/activate
python reconstruct_mesh.py \
  --all \
  --data-root data/laso_grasp_only_affordance
```

This writes `reconstructed_mesh.ply` beside each `affordance.npz`. Reconstruction
can lose thin structures or close holes, so inspect important objects before
treating the result as reliable collision geometry.

Create MuJoCo-compatible visual OBJ files:

```bash
python convert_ply2obj.py \
  --data-root data/laso_grasp_only_affordance
```

Create convex collision components:

```bash
python coacd_decompose.py \
  --all \
  --data-root data/laso_grasp_only_affordance
```

The output for each successfully decomposed object is
`collision_meshes/part_*.obj`. To retry only entries in a failure log, use the
retry cells at the bottom of `coacd_decompose.ipynb`.

Create standalone metric MuJoCo models:

```bash
python convert_laso_to_mujoco.py \
  --data-root data/laso_grasp_only_affordance
```

This reads class-level longest-dimension assumptions from
`laso_physical_dims_assumed.json` and applies one uniform transform to the
visual and collision meshes. It writes the following:

```text
<class>/<object-id>/mujoco/
├── model.xml
├── metadata.json
├── visual.obj
└── collision_*.obj
```

The default mass is an assumed `0.1 kg`; the default friction tuple is
`1.2 0.01 0.001`. The per-object `mujoco/metadata.json` records every
assumption and the source-to-metric transform.

## 3. Generate multi-view GraspNet candidates

Official GraspNet inference requires the CUDA environment:

```bash
source /ros2_ws/.venv/graspnet_env/bin/activate
```

First test one object:

```bash
OBJECT=mug/1bc5d303ff4d6e7e1113901b72a68e7c
OUT=outputs/graspnet_multiview/laso/$OBJECT
mkdir -p "$OUT"

python generate_multiview_graspnet_grasp.py \
  --data-root data/laso_grasp_only_affordance \
  --object "$OBJECT" \
  --views 8 \
  --max-candidates 200 \
  --output-json "$OUT/candidates.json" \
  --output "$OUT/candidates.png" \
  --views-output "$OUT/visible_views.png"
```

Without `--transform` or `--longest-extent-m`, the generator identifies the
LASO class and uses `laso_physical_dims_assumed.json`. The scale is:

```text
assumed class longest dimension [m] / source-cloud longest dimension
```

That scalar scales all three axes equally. Override it only when you have a
better physical measurement:

```bash
python generate_multiview_graspnet_grasp.py \
  --data-root data/laso_grasp_only_affordance \
  --object "$OBJECT" \
  --longest-extent-m 0.10 \
  --output-json "$OUT/candidates.json" \
  --output "$OUT/candidates.png" \
  --views-output "$OUT/visible_views.png"
```

The visibility stage is a perspective point-splat z-buffer, not a distance
crop. Each view produces a partial cloud, candidates from all views are merged,
NMS is applied, and the complete metric cloud is used for the generator's
model-free collision filter.

## 4. Check full-mesh approach collision

Return to the project environment:

```bash
source .venv/bin/activate

python eval_grasp_mujoco.py \
  --grasps "$OUT/candidates.json" \
  --mesh "data/laso_grasp_only_affordance/$OBJECT/reconstructed_mesh.ply" \
  --collision-dir "data/laso_grasp_only_affordance/$OBJECT/collision_meshes" \
  --output-json "$OUT/collision.json"
```

This places the actual OpenArm gripper at every GraspNet pose and checks the
complete straight approach against the precomputed convex object components.
It writes collision-free rows and original candidate indices under
`mujoco_collision.rows` and `mujoco_collision.source_indices`.

Inspect a candidate interactively with:

```bash
python eval_grasp_mujoco.py \
  --grasps "$OUT/candidates.json" \
  --mesh "data/laso_grasp_only_affordance/$OBJECT/reconstructed_mesh.ply" \
  --collision-dir "data/laso_grasp_only_affordance/$OBJECT/collision_meshes" \
  --view-grasp 0
```

## 5. Check contacts, antipodality, and friction cones

```bash
python eval_grasp.py \
  --grasps "$OUT/collision.json" \
  --friction 0.8 \
  --output-json "$OUT/evaluated.json" \
  --output "$OUT/evaluated.png"
```

`eval_grasp.py` first performs its full-cloud finger, palm, and approach-sweep
checks. Only candidates passing those checks reach contact-pair and Coulomb
friction-cone evaluation. Results are stored under `antipodal`.

The evaluators preserve the original `graspnet.rows` and attach independent
diagnostics. Therefore, the strict final LASO set is the intersection:

```text
set(mujoco_collision.source_indices) ∩ set(antipodal.source_indices)
```

Do not interpret antipodal acceptance as physical grasp success. Arm IK,
reachability, gravity stability, object dynamics, and execution uncertainty are
separate validation stages.

## 6. Run candidate generation for every LASO object

The following loop is resumable because it skips an existing `candidates.json`:

```bash
source /ros2_ws/.venv/graspnet_env/bin/activate

ROOT=data/laso_grasp_only_affordance
OUT_ROOT=outputs/graspnet_multiview/laso
FAILED="$OUT_ROOT/failed_objects.txt"
mkdir -p "$OUT_ROOT"
: > "$FAILED"

find "$ROOT" -name affordance.npz -print0 |
while IFS= read -r -d '' NPZ; do
  REL=${NPZ#"$ROOT"/}
  OBJECT=${REL%/affordance.npz}
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

Run both evaluators for every successful annotation:

```bash
source .venv/bin/activate

ROOT=data/laso_grasp_only_affordance
OUT_ROOT=outputs/graspnet_multiview/laso
EVAL_FAILED="$OUT_ROOT/evaluation_failed_objects.txt"
: > "$EVAL_FAILED"

find "$OUT_ROOT" -name candidates.json -print0 |
while IFS= read -r -d '' GRASPS; do
  REL=${GRASPS#"$OUT_ROOT"/}
  OBJECT=${REL%/candidates.json}
  OUT=$(dirname "$GRASPS")
  [ -f "$OUT/evaluated.json" ] && continue

  python eval_grasp_mujoco.py \
    --grasps "$GRASPS" \
    --mesh "$ROOT/$OBJECT/reconstructed_mesh.ply" \
    --collision-dir "$ROOT/$OBJECT/collision_meshes" \
    --output-json "$OUT/collision.json" && \
  python eval_grasp.py \
    --grasps "$OUT/collision.json" \
    --output-json "$OUT/evaluated.json" \
    --output "$OUT/evaluated.png" || echo "$OBJECT" >> "$EVAL_FAILED"
done
```

Keep generation and evaluation failure logs separate: “no GraspNet
candidates,” “invalid collision mesh,” and “no accepted antipodal grasp”
describe different failure stages.

## Output interpretation

- `candidates.json`: merged GraspNet Nx17 proposals in the metric object frame.
- `collision.json`: original candidates plus MuJoCo approach-collision results.
- `evaluated.json`: collision diagnostics plus full-cloud contact, antipodal,
  and friction-cone results.
- `*.png`: inspection images only; JSON is the machine-readable annotation.
- LASO affordance values describe human grasp regions. They do not certify that
  an OpenArm grasp is reachable or physically successful.
