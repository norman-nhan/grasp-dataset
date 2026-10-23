# OpenArm offline grasp dataset generator

Standalone geometry sampling → gripper-only MuJoCo trials → independent LASO labels → resumable JSON datasets. New runs use a minimal environment: the gripper, one target object, and an optional support plane. No ROS, MoveIt, IK, neural inference, or full-arm simulation.

## Download the official ycb dataset mesh and texture
Before run below program, change output dir and object you want to download, etc. inside `download_ycb_dataset.py` first.
```bash
python download_ycb_dataset.py
```

## Create model.xml for each ycb dataset object
```bash
python convert_ycb_to_mujoco.py
```


## Simple OpenArm gripper scene

The `openarm-scene` command opens a minimal MuJoCo scene containing the real
OpenArm gripper geometry, a support plane, and one previously converted YCB
`model.xml` from `assets/ycb-affordance-mujoco/`. List the available converted
objects, then open one by ID:

```bash
cd /ros2_ws/not-ros-pkg/dataset-generation
source .venv/bin/activate
openarm-scene --list
openarm-scene --object 025_mug
```

An explicit converted model can also be selected:

```bash
openarm-scene \
  --model assets/ycb-affordance-mujoco/025_mug/model.xml
```

For a headless smoke test or saved preview:

```bash
MUJOCO_GL=egl openarm-scene --object 025_mug \
  --render outputs/simple-scenes/025_mug.png \
  --save-xml outputs/simple-scenes/025_mug.xml
```

The scene imports the converted model's full textured visual mesh, object mass,
free joint, and all CoACD collision components without regenerating geometry.
It then adds the corrected OpenArm gripper model. `--save-xml` writes this merged
OpenArm-plus-object scene; the source object `model.xml` remains unchanged.

## Convert downloaded YCB meshes to MuJoCo

The official YCB downloader extracts Google-16k assets under
`assets/ycb-affordance/<object>/google_16k/`. Convert every downloaded object
into a directly loadable MuJoCo model with:

```bash
cd /ros2_ws/not-ros-pkg/dataset-generation
source .venv/bin/activate
python scripts/convert_ycb_to_mujoco.py
```

Convert only selected objects or replace an earlier conversion:

```bash
python scripts/convert_ycb_to_mujoco.py \
  --objects 006_mustard_bottle 025_mug 032_knife
python scripts/convert_ycb_to_mujoco.py --overwrite
```

Downsample the geometry used for collision decomposition while retaining the
full-resolution textured visual mesh:

```bash
python scripts/convert_ycb_to_mujoco.py \
  --downsampling-ratio 0.25 \
  --overwrite
```

The ratio is the target fraction of input faces: `1.0` disables simplification,
`0.5` targets half, and `0.25` targets one quarter. It must be greater than zero
and at most one. Because the collision input and generated convex components
change, rerunning an existing object with another ratio requires `--overwrite`.

Open any generated model in the native MuJoCo viewer:

```bash
python -m mujoco.viewer \
  --mjcf assets/ycb-affordance-mujoco/025_mug/model.xml
```

Outputs are written to `assets/ycb-affordance-mujoco/<object>/`. Every object
directory is portable and contains `model.xml`, the textured visual mesh,
texture, convex collision meshes, and `metadata.json`. A batch `manifest.json`
records converted, existing, and failed objects. Each XML includes a free body,
floor, lighting, camera, texture and physical collision geometry, and is loaded
with MuJoCo as part of conversion validation.

The default 0.1 kg object mass is an explicit simulation assumption. Change it
with `--mass`; it is not a measured YCB property. CoACD collision components
approximate concave objects rather than replacing them with one convex hull.

## Docker quick start

After entering the existing container with `./docker/run.sh`, change into the project:

```bash
cd /ros2_ws/not-ros-pkg/dataset-generation
source .venv/bin/activate
openarm-dataset full --num-candidates 24 --output outputs/example.json
```

The environment is `/ros2_ws/not-ros-pkg/dataset-generation/.venv` inside Docker (the same mounted directory as the host project). It was rebuilt with container-compatible Python 3.12. A virtual environment created on the host is not portable into this container. To recreate it inside Docker:

```bash
cd /ros2_ws/not-ros-pkg/dataset-generation
UV_CACHE_DIR=/tmp/openarm-dataset-uv-cache uv sync --python 3.12 --extra dev --frozen
```

The cache override avoids the existing permissions issue in `/ros2_ws/.cache/uv`. The `.python-version` file pins Python 3.12 for this project.

## Installation

From this directory:
+
```bash
uv sync --extra dev
source .venv/bin/activate
```

The environment is local to `not-ros-pkg/dataset-generation/.venv`; dependencies are pinned in `uv.lock`. `uv run openarm-dataset ...` also works without activation. Host Python 3.12 was used for validation. Within the repository Docker bind mount, run `uv sync` there to create an environment for that interpreter/platform.

## Repository integration and model sources

- `src/openarm_mujoco_bringup/config/scene_*.yaml` defines target geometry, physical properties, and environment.
- For legacy scene inputs, the corresponding `config/scene_*/scene_*.xml` is the existing **URDF-corrected** MuJoCo model. Standalone object YAML files (`standalone: true`) load their own mesh and collision components, and extract only the gripper from the repository cube model; they require no generated scene for that object. `--model` can select another generated XML. A missing or stale model is an error; this tool does not run the ROS-dependent scene builder.
- `assets/openarm_mujoco_v1/meshes/{collision,visual}/gripper/` supplies the actual palm and finger meshes. Known Docker package asset paths are resolved against the local checkout.
- The right-hand subtree, sibling finger transforms, inertials, joint limits, tendon, actuator parameters, finger equality, exclusions, optional scene fingertip pads, and object collision geometry come from the generated XML. The arm is omitted. Source tables, pedestals, cameras, and other objects are omitted by default. `--environment source` explicitly reproduces old scene-based experiments. The source XML supplies gripper geometry/control; it is never simulated as a full robot.
- The checked-in model has one `split_right` tendon position actuator (`kp=2000`, force range ±40 N, control range 0–0.044 m) and two slide joints. These are extracted, not independently configured. The baseline raw MJCF has different frames/actuation and must not replace the corrected scene XML.
- `not-ros-pkg/mesh-grasp-annotator/export_laso.py` and `not-ros-pkg/laso-reviewer/server.py` establish the annotation formats. The existing transform utility imports ROS messages, so numerical rigid transforms here use SciPy instead of importing ROS or copying its quaternion implementation.

Sampling uses transformed visual triangles; MuJoCo uses the scene's existing collision components. MuJoCo takes a convex hull per collision mesh. The bottle uses simplified collision primitives. These representations can differ enough to reject a visually plausible grasp.

## Corrected approach-accessibility checks

New physical runs use evaluator `approach-contact-v2`. Contact-pair antipodality does not imply that a gripper can enter the final pose. The approach is validated independently, with t+he original +Z convention preserved:

1. Compute a backoff that puts the **whole open gripper** behind the target's collision-bounds supporting plane along the selected approach axis, with an exterior margin. A fixed 6 cm minimum alone is insufficient for a cup cavity.
2. Validate the complete straight translation using signed distances between every relevant gripper/obstacle collision-geometry pair. Conservative advancement takes at most 80% of the current distance budget; it never imposes a minimum step across a small gap. Insufficient clearance or a check limit causes rejection.
3. Permit shallow, localized finger contacts only within the terminal approach window; other entry contacts are forbidden. During closing/lift/hold, intended contacts must stay near the declared contact patch on the corresponding finger. Unexpected contacts abort execution before subsequent phases.
4. Limit approach speed when backoff grows. Use scratch `MjData` for geometric queries: screening does not teleport or reset the active object's state.

Defaults: maximum backoff 0.5 m; exterior margin 10 mm; approach clearance 0.2 mm; maximum advancement step 5 mm; terminal window 1 mm with at most 0.05 mm steps and 0.1 mm allowed localized penetration; contact-patch radius 25 mm; maximum approach speed 0.1 m/s. Override these `Physics` fields with `--physics`: `max_pregrasp_distance`, `exterior_clearance`, `approach_clearance`, `approach_max_step`, `approach_max_checks`, `terminal_contact_distance`, `terminal_contact_step`, `contact_patch_radius`, `contact_penetration_tolerance`, `max_approach_speed`.

The fixed-orientation geometric check operates on MuJoCo's convex components, not exact arbitrary-mesh continuous collision detection. Terminal contact handling uses fine finite steps and tolerances, which must be smaller than features you need to resolve. The method certifies this local straight entry against the target and modeled obstacles; it does not search curved paths or prove full-arm reachability. A gripper can legitimately finish inside a cavity if it fits through the opening along a validated path.

`full` and `evaluate` automatically add a **nominal** `approach_screen` and `approach_valid` to each candidate; every perturbed trial also gets its own approach report. Reports include the exterior pregrasp, backoff, query count, minimum sampled clearance, blocking geoms/distance when available, settings fingerprint, and rejection reasons. Accessibility is independent of physical success and LASO labels. Rejected candidates are retained with unmeasured physical labels; a runtime abort after execution begins is a measured failure with `failure_reasons` and `runtime_violation`. Slip/lift metrics can be null if the trial aborted before they were measured.

Existing datasets can receive nominal approach labels without changing historical trial results:

```bash
openarm-dataset screen --output outputs/ycb-varied/approach/032_knife.json
openarm-dataset visualize --output outputs/ycb-varied/approach/032_knife.json --filter accessible --all
```
+
Use `--filter inaccessible` to inspect rejected paths. The selected entry path is cyan when clear, orange when rejected, with a red marker at the blocked position when available; the blue sphere is the exterior pregrasp. The terminal prints the approach reasons. Raw `--all` continues to include negative poses.

Historical physical trials use different acceptance rules, so they remain unchanged and cannot be resumed into the new evaluator. Generate a new output path for corrected physical results. For example, rerun a batch with an explicit fresh directory:

```bash
python scripts/generate_variations.py --output-dir outputs/ycb-varied-v2
MUJOCO_GL=egl python scripts/test_ycb.py --output-dir outputs/ycb-v2 --render
```

These commands start new evaluations; the historical batch directories cannot be extended under the new rules. The viewer identifies historical labels explicitly. All 1,920 records in `outputs/ycb-varied/` have now been screened nominally; this does not retroactively make their old physical labels v2 results. No nominal mug entry was found among the current 384 candidates.

Fresh corrected results are in `outputs/approach-v2/` (96 bottle candidates and 384 mug candidates, one unperturbed trial each):

```bash
openarm-dataset visualize --output outputs/approach-v2/006_mustard_bottle.json --filter accessible --all
openarm-dataset visualize --output outputs/approach-v2/025_mug.json --filter inaccessible --all
```

## Generate and evaluate

### All bottle, knife and mug examples

`scripts/generate_object_examples.py` catalogs all matching local LASO shapes
and five YCB scans (mustard bottle, bleach cleanser, Windex bottle, knife, mug).
The YCB Windex scan is incomplete in the official distribution. LASO currently
h+as 734 distinct matching shapes: 328 bottles, 255 knives and 151 mugs across
train/validation/test. With 384 candidates per object, the full batch targets
739 objects and up to 283,776 candidates. Candidate generation may find fewer
pairs on some geometries. One physical trial is requested for each candidate;
inaccessible approaches remain unmeasured geometry rejections.

Inside Docker, in this project directory:

```bash
python scripts/generate_object_examples.py --status
```

The batch is stored under `outputs/bottle-knife-mug/`: `manifest.json` tracks
each object's preparation/evaluation/rendering state, errors, counts and paths;
`logs/` contains per-object logs; `ycb/` and `laso/<split>/<class>/` contain JSON
datasets and PNG previews. Traces are compressed external files. Preview poses
prefer a physical success, otherwise an accessible candidate, otherwise a
rejected candidate; green lines in an overview do not imply all poses succeeded.

To start or resume after the current runner exits:

```bash
python scripts/generate_object_examples.py --workers 4
```

Completed objects are skipped, interrupted trials resume, and failed jobs are
retried with the same settings. A process lock rejects simultaneous runners on
the same output. Changing candidate count, split selection or seed requires a
new output directory. For a smaller separate validation-only batch:

``+`bash
python scripts/generate_object_examples.py --splits val \
  --output-dir outputs/bottle-knife-mug-val --workers 4
```

LASO uses `prepare_laso.py`: an assumed longest extent of 180 mm for bottles,
220 mm for knives, 85 mm for mugs, and 0.1 kg mass for each. A voxel shell around
the normalized source points supplies approximate material thickness; up to
three explicitly recorded radii are tried for connectivity/source coverage,
without using physical success or semantic masks. Disconnected or invalid
reconstructions are recorded as preparation failures, not physical failures.
Unlike the single mug example below, arbitrary reconstructed mugs have no
universal cavity guarantee. Sizes, topology, thickness and mass require review
before using these as physical ground truth. LASO masks use the exact recorded
similarity transform; YCB LASO labels remain null because no registration is
available. No semantic filter biases geometric sampling.

84 of the matching LASO objects have no `grasp` affordance mask. Those objects
still receive physical evaluation, with null overlap labels and an explicit
`missing_grasp_affordance` annotation status. The batch does not substitute a
different affordance silently. Enclosed voxel air pockets are filled as part of
the solid-material assumption; source points buried in that material are
counted in provenance rather than mistaken for omitted geometry.

Example inspection after the relevant object has completed:

```bash
openarm-dataset visualize \
  --output outputs/bottle-knife-mug/ycb/006_mustard_bottle.json --filter success
openarm-dataset visualize \
  -+-output outputs/bottle-knife-mug/laso/val/bottle/48202d357e79315e45891653421dc140.json --all
```

### Smaller LASO mug example

`scripts/prepare_laso_mug.py` uses validation mug
`34ae0b61b0d8aaf2d7b20fded0142d7a` from the trusted local LASO pickle data.
LASO supplies normalized 2048-point clouds and affordance masks, **not original
meshes or measured dimensions**. This example assumes an 85 mm longest source
extent and 0.1 kg mass. It reconstructs a smoothed point-distance shell with a
2.5 mm material radius and 0.75 mm voxel pitch, then decomposes that shell for
MuJoCo. These are explicit geometric/physical assumptions; this tests the
reconstructed object, not the original physical mug. `SOURCE.json` records the
transform, dimensions, source coverage and collision settings. No whole LASO
scene is imported. The preparation rejects disconnected/non-watertight surfaces
and checks a central cavity probe before and after decomposition.

Inside Docker, from this project directory:

```bash
.venv/bin/uv sync --extra dev --extra ycb --extra laso
source .venv/bin/activate
python scripts/prepare_laso_mug.py
openarm-dataset full \
  --scene assets/laso/34ae0b61b0d8aaf2d7b20fded0142d7a-shell/object.yaml \
  --support none --num-candidates 384 --approaches 12 \
  --trace-storage external \
  --annotation ../laso-reviewer/data/LASO_dataset \
  --shape-id 34ae0b61b0d8aaf2d7b20fded0142d7a --split val \
  --annotation-transform assets/laso/34ae0b61b0d8aaf2d7b20fded0142d7a-shell/annotation_to_object.json \
  --+output outputs/laso-mug/grasps.json
openarm-dataset visualize --output outputs/laso-mug/grasps.json --filter accessible --all
openarm-dataset visualize --output outputs/laso-mug/grasps.json --filter success
```

Preparation requires a fresh directory; evaluation requires a fresh output path
unless resuming the same run. `--support none` uses the existing temporary
gravity-compensated fixture until closure so the example tests gripper/object
accessibility without a table. The normal exterior approach and contact rules
remain enabled. The registered LASO `grasp` mask is used for keep-free overlap
labels only; it does not bias geometric sampling or decide physical success.

The completed example at `outputs/laso-mug/grasps.json` contains 384 candidates:
102 have a clear exterior approach, 43 succeed in simulation, 59 are measured
failures and 282 are rejected before execution. All 384 have LASO overlap labels;
39 successful grasps also meet the configured keep-free threshold (at most 20%
of the LASO grasp-region points within the 10 mm gripper clearance). This is a
region-overlap heuristic, not a human hand feasibility test. Candidate 25 is one
successful example with zero overlap; `outputs/laso-mug/success-25.png` shows it.
The reconstructed body spans about 65 mm, total width including its handle is
89 mm, and height is 73 mm. The body fits the roughly 82.5 mm gripper opening.

Defaults select the repository's 4 cm cube:

```bash
uv run openarm-dataset full --num-candidates 24 --num-trials 1 --seed 42 --output outputs/box.json
uv run openarm-dataset inspect --output outputs/box.json
uv run openarm-dataset visualize --output outputs/box.json --grasp-id 1
```

Separ+ate stages and resume:

```bash
uv run openarm-dataset candidates --scene ../../src/openarm_mujoco_bringup/config/scene_ycb_bottle.yaml --object ycb_mustard_bottle --num-candidates 512 --output outputs/bottle.json
uv run openarm-dataset evaluate --output outputs/bottle.json --num-trials 1
uv run openarm-dataset evaluate --output outputs/bottle.json --num-trials 3
```

`evaluate` always skips saved trial IDs. `full --resume` resumes an existing file. Data is atomically replaced after every trial; interruption can repeat at most the unfinished trial. Scene content, referenced asset bytes, object triangles, numerical MuJoCo version, and simulation configuration identify the run. Changed scenes or physics require a new output. Increasing trial count is allowed. Keep the saved scene/model paths accessible when resuming or viewing on another computer.

Sampling flags include `--surface-points`, `--neighbors`, `--pairs-per-point`, `--approaches`, `--min-width`, `--max-width`, `--friction`, and `--angular-tolerance-deg`. Contact search uses a KD-tree and bounds the neighbors examined per point. It returns up to the requested candidate count; the CLI reports the actual count. Multiple angles are generated per pair. The CLI now retains at most one pair per surface point by default so successive groups of approach angles cover different surface locations. `--pairs-per-point` controls this limit; stored older datasets keep their existing candidates.

The Python wrapper is equivalent:

```bash
.venv/bin/python generate_grasp_dataset.py full --num-candidates 24 --output outputs/example.json
```

## Minimal YCB experiments

Run these commands **inside Docker**, from `/ros2_ws/not-ros-pkg/dataset-generation`:

```bash
source .venv/bin/activate
UV_CACHE_DIR=/tmp/openarm-dataset-uv-cache uv sync --extra dev --extra ycb
python+ scripts/prepare_ycb.py
MUJOCO_GL=egl python scripts/test_ycb.py --num-candidates 96 --render
```

This prepares and evaluates official Google 16k scans for `006_mustard_bottle`, `025_mug`, `032_knife`, and `037_scissors`. Files are downloaded from the [official YCB distribution](https://ycb-benchmarks.s3.amazonaws.com/index.html), licensed CC BY 4.0. Source URLs, archive SHA256, decomposition parameters, and seed are recorded in each object's `SOURCE.json` and YAML. Downloaded/prepared assets and generated results are ignored by Git; rerun the scripts to reproduce them.

CoACD supplies convex collision components (rather than filling the mug with a single hull). Inertia is inferred from volume-weighted collision components. The default test mass is an explicit **0.1 kg simulation assumption for each object**, not a measured YCB mass; `prepare_ycb.py --mass` sets it for a fresh preparation, and the YAML records it. Prepared configurations are reused if present. Changing mass or decomposition means creating a new configuration and output dataset.

The test script uses `--environment minimal --support none`: only the target, actual OpenArm gripper, and a mocap control body exist. Gravity compensation temporarily supports the target during settling/approach/closure, then is disabled before lift and hold. No object pose is teleported during the trial. This is a free-space fixture-release stability test, **not a pickup from a table**. Gravity, actuator/contact dynamics, lift, slip and sustained bilateral contact still determine success. Trials record `temporary_fixture` and `evaluation_kind: free_space_stability`. For a single plain support plane use `--support plane`; no original scene furniture is imported.

One object independently:

```bash
openarm-dataset full --scene assets/ycb/032_knife/object.yaml --support none --num-candidates 96 --output outputs/knife-free.json
```

YCB scans contain no LASO masks. The YCB tests intentionally leave LASO labels null; LASO labels from a different shape cannot be assumed to apply.

## Larger runs with variation

Inside the activated Docker project directory:

```bash
python +scripts/generate_variations.py
```

The default batch creates two separate experiment sets for each of the four prepared YCB objects under `outputs/ycb-varied/`:

- `approach/`: 384 candidates × 5 trials. Samples 12,000 surface points, 12 approach angles per pair, and retains one pair per point. Initial object orientation has 30° per-axis Gaussian rotation-vector noise; approach has 2 mm translation and 5° per-axis rotation noise.
- `loading/`: 96 candidates × 4 trials. The candidate set is the first 96 from the same deterministic sampling run. Initial orientation and approach are fixed; gravity direction varies by a 20° per-axis Gaussian rotation vector after closing.
- Both sets independently sample friction multipliers from 0.6–1.4 and mass multipliers from 0.7–1.3 (0.07–0.13 kg with the current assumed 0.1 kg base masses). Seed is 20261016. Candidate openings down to 1 mm are included to retain thin-feature pinches. Minimal free-space fixture release is used throughout.

There are 1,920 grasp records and 9,216 attempted trials in the default batch. Geometrically rejected trials remain unmeasured, with null success. The approach set combines initial-pose accessibility and changed loading; the loading set isolates loading variation after closure. These sets are stored separately and should not be pooled as if they measured identical conditions.

The script runs two independent simulator processes at a time and checkpoints each trial. Run the same command to resume; completed trial IDs are skipped. `--workers`, `--num-candidates`, `--num-trials`, `--loading-candidates`, `--loading-trials`, `--seed` and `--output-dir` configure a new batch. Use a new output directory when changing sampling or variation settings; existing datasets keep their recorded settings and candidates.

`manifest.json` records commands and completed-job summaries; `logs/` contains per-object experiment output. Detailed traces are compressed into sibling `.traces/` directories instead of being rewritten into the main JSON after every trial. Keep each JSON and its trace directory together when moving or sharing a dataset. Existing inline-trace datasets remain supported. The Python helper `grasp_dataset.storage.load_trace(dataset_path, trial)` reads either format and verifies external trace checksums.

```bash
openarm-dataset visualize --output outputs/ycb-varied/approach/032_knife.json --all
openarm-dataset visualize --output outputs/ycb-varied/loading/032_knife.json --filter success
```

For a single custom run, add `--trace-storage external` to `openarm-dataset full`; resume inherits that storage setting. Variation configurations are in `examples/ycb_appr+oach_variation.json` and `examples/ycb_loading_variation.json`.

## Browse all saved candidates

```bash
openarm-dataset visualize --output outputs/example.json --all
openarm-dataset visualize --output outputs/ycb/032_knife.json --all
openarm-dataset visualize --output outputs/ycb/032_knife.json --filter success
```

N or Right selects the next candidate; P or Left selects the previous; A toggles all candidate overlays. Selection wraps. The terminal prints the selected grasp ID, physical outcome, evaluation status, and overlap. Old saved datasets remain viewable with their original source environment, so their tables can still appear; new runs default to minimal environments.

For a PNG overview without a GUI:

```bash
MUJOCO_GL=egl openarm-dataset visualize --output outputs/ycb/032_knife.json --all --render outputs/knife-all.png
```

## Frames and geometry

All lengths are metres. `grasp_pose_object` is **object_T_tcp**, a right-handed 4×4 transform. World execution computes `world_T_object @ object_T_tcp` from the **current simulation pose after settling**. Quaternions use `[w,x,y,z]`.

The TCP axes match the repository hand: Y is the closing axis and +Z points from palm toward fingertips. The corrected scene TCP is at hand Z=0.0835 m; the tool reads this transform from the model. Opposing inner-face locations and distal contact depth come from actual finger collision mesh bounds (including pads when present). A mesh-derived distal contact region is a sampling heuristic, not a calibrated tactile contact model. The midpoint is offset accordingly; it is not assumed to be the TCP origin. The cube model's usable mesh-derived maximum gap is approximately 0.08248 m.

Visual mesh parts are concatenated after scale and mesh-to-object transforms, preserving their local placement. Nonfinite geometry is rejected; degenerate triangles are removed with a w+arning and counted in metadata. Normals are repaired per component. Watertight status is recorded; non-watertight meshes may have ambiguous outward normals. No affordance filtering happens during sampling.

## Physics and robustness configuration

Pass a JSON file with any `Physics` fields using `--physics config.json`. `--num-trials` and `--seed` set the trial count and seed for a new evaluation. On resume the saved seed/settings are reused unless explicitly supplying compatible settings. Example:

```json
{
  "orientation_deg": 5,
  "approach_noise_m": 0.001,
  "approach_noise_deg": 2,
  "friction_low": 0.8,
  "friction_high": 1.2,
  "mass_low": 0.9,
  "mass_high": 1.1,
  "max_slip": 0.015,
  "max_rotation_deg": 25,
  "min_lift": 0.04
}
```

Object and approach orientation perturbations are independent zero-mean Gaussian rotation vectors, with the specified per-axis degree standard deviation. Translation noise is Gaussian in metres. Friction and mass multipliers are uniform over the specified intervals. Friction scales both gripper and target geoms so the source model's contact priority cannot hide the variation; each saved contact includes effective solver friction. Mass changes scale the source inertia while preserving COM. Individual seeds derive from `(seed, grasp_id, trial_id)`.

Two experiment types are deliberately distinct:

- **Accessibility and execution:** `orientation_deg` changes the initial object orientation before settling. Approach noise, table obstruction, settling, geometry rejection, and ph+ysical stability all influence execution. `accessibility_rate` measures the fraction reaching a physical trial; `execution_success_rate` measures successful grasps divided by all requested completed/rejected trials. This is gripper accessibility only, not arm reachability.
- **Loading stability:** use `loading_tilt_deg` with zero initial-orientation and approach noise. The same object-relative grasp is closed first; the gravity vector is then rotated by a sampled Gaussian rotation vector during lift/hold. This changes load direction without conflating it with initial accessibility. Combining loading tilt with orientation/approach noise is rejected. Gravity magnitude remains unchanged. Very large tilts can make a world-Z lift inappropriate; this is an experimental load test, not a model of physically rotating a human-held object.

Each trial resets qpos, velocities, activation/control state, gravity, mass, inertia, and friction. The object settles with the hand away. The full approach from an exterior pregrasp is screened using signed-distance advancement; only localized terminal finger contacts are allowed. Initial placement happens before the active trial. During approach, close, lift, and hold, only the mocap target moves; the free hand follows a weld and fingers close through the source actuator. Neither object nor hand qpos is teleported during these phases.

Default phases: settle 0.8 s, approach 0.6 s, close/stabilize 0.8 s, lift 0.08 m over 1 s, hold 0.8 s, timestep 0.002 s. Success requires at least 0.04 m object lift throughout hold, ≤0.015 m relative translation slip, ≤25° relative rotation slip, bilateral force-bearing finger contacts, no support contact during hold, and no unwanted palm/environment contacts during execution. Contacts and all success criteria are checked every integration step; exported traces are sampled every 0.02 s. Lift trajectories hold orientation fixed.

Contact wrenches use `mj_contactForce`: the first three values are force and the last three torque in the contact frame. Frame rows are world axes; `frame.T @ force[:3]` gives force on geom2 in world coordinates. Geometry names/IDs, contact position, effective friction, finger association, joint positions, and actuator forces are saved.

## LASO labeling

Labels require an explicit registration; no automatic mesh-to-LASO alignment is assumed. Source options:

- Existing `laso-point-cloud-export/v1` `.laso.json` with a sibling metric PLY and per-affordance `point_indices`.
- A **trusted local** LASO directory containing `objects_<split>.pkl` (shape-ID → point array) and `anno_<split>.pkl` (records with `shape_id`, `affordance`, `mask`, and class metadata). Pickle loading executes code; only use your trusted local dataset.

Create a JSON 4×4 similarity matrix mapping annotation coordinates directly into the target object's metric local frame. Raw LASO coordinates need their original scale and centering as well as mesh registration; exported PLY coordinates have already been metricized by `export_laso.py`. An identity matrix is valid only when the coordinates already align.

```bash
u+v run openarm-dataset label --output outputs/bottle.json --annotation /path/to/object.laso.json --annotation-transform /path/to/annotation_to_object.json --affordance grasp --clearance 0.01 --overlap-threshold 0.2
```

For pickle directories add `--shape-id ID --split val`. `full` can also accept annotation arguments; absent an annotation, LASO labels stay null with an explicit message.

Overlap is the fraction of human-affordance points inside or within the clearance of any actual gripper convex collision component, placed at the candidate's nominal contact width. It includes fingers and palm, handles interior points, and ranges from zero to one. `handover_suitable = overlap <= threshold` is an **affordance-clearance label**, independent of physical success. It does not assert that a failed grasp can be handed over. Overlapping candidates are always retained. Registered points and labeling settings are saved for reproducibility and visualization.

## Dataset schema

One portable JSON document, schema `openarm-grasp-dataset/v1`:

- `metadata`: generator version, scene/model paths and fingerprint, full target config (mesh IDs/scales, mass/friction), mesh validation, actual gripper limits and TCP offset, sampling seed/settings, physics settings, MuJoCo version, frame conventions, and optional annotation source/transform/points/threshold.
- `grasps[]`: object ID, grasp ID, object_T_tcp, required width, contact points/normals, antipodal score, geometry/collision status, rejection reasons, evaluation status, physical success/rate/slip, accessibility/execution rates, overlap, suitability, and `trial_results`.
- `trial_results[]`: trial seed and perturbations, experiment kind, actual mass/friction, initial settled object pose, status/reasons, physical success, lift/slip/rotation metrics, final joint state, and time traces with poses, joint positions, actuator forces, and contact wrenches.

`physical_success` means **any measured trial succeeded**. `physical_success_rate` is successes divided by physically evaluated trials. Use `execution_success_rate` for successes divided by all attempted configurations, including geometry rejections. Geometrically rejected trials have `success: null`; a candidate with no measured trial has null physical labels. `collision_valid` means at least one trial passed initial geometric screening, not that every perturbation was clear. `handover_suitable` remains independent. The Python API is `grasp_dataset.storage.load/save`.

## Inspection and tests

The native MuJoCo viewer shows the object, actual selected gripper geometry, cyan intended contacts, and yellow LASO points. Select a candidate with `--grasp-id`; N/Right and P/Left br+owse every saved candidate. `--all` overlays all contact segments and approach stems around the object; A toggles that overlay. Green indicates physical success, red a measured failure, gray a geometry rejection, blue pending evaluation, and magenta excessive affordance overlap. `--filter success|failure|rejected|pending` restricts the candidates. It is a paused pose inspector, not a replay. Overview lines represent poses; the selected candidate shows the actual collision geometry. It needs a desktop/OpenGL context; generation and tests work headlessly. Full meshes retain source materials, and collision geoms carry result colors.

```bash
uv run pytest -q
```

Tests cover multi-part mesh scaling, round-trip transforms, orthonormal grasp frames, antipodal conditions, opening constraints, intended contacts versus penetration, physical successes and measured failures, reset consistency, LASO interior/clearance overlap, independent negative labels, serialization, deterministic generation, robustness, and real scene objects. See `VALIDATION.md` for measured outcomes.

## Limitations

No full-arm reachability guarantee; no automatic LASO registration, human hand model, arm planning, learned model, or automatic LASO-to-YCB label transfer. Sampling currently supports boxes and triangle mesh targets. For legacy scene inputs, the generated XML must match YAML. Standalone YAML inputs need only target meshes and collision components. Gripper motion has an ideal 6-DoF mocap controller; results depend on weld compliance, contact solver, source collision approximations, and simulated friction. Contact depth is mesh-derived. Candidate diversity and coverage need tuning for thin or heavily obstructed objects. JSON with contact traces favors transparency over storage throughput; large production datasets may need a chunked format. Resume is single-writer. The native viewer requires interactive desktop validation on the user's display.
+
