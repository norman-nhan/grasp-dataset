# Approach-accessibility correction — 2026-10-16

Implemented evaluator `approach-contact-v2` with `exterior-sweep-v1` geometric screening. The original +Z approach-axis convention is unchanged. This section supersedes the acceptance semantics of the historical endpoint-only trials below; those physical trial records remain intact.

## Checks and regressions

- Adaptive backoff places the entire open gripper beyond the target collision bounds along the approach axis, rather than assuming 6 cm is exterior space.
- Signed-distance conservative advancement checks the complete fixed-orientation translation. Terminal localized contact handling uses fine finite steps and explicit tolerances; this is not an exact general mesh CCD or a curved-path planner.
- Scratch simulation data isolates geometric queries from live trial state.
- Early finger-object contact is forbidden. Closing/lift/hold contacts must be on the corresponding finger near its intended object patch. Unexpected contacts abort subsequent phases and produce measured failures; pre-execution path rejections have null physical labels.
- Longer backoffs use an approach-speed limit.

**28 tests passed in Docker with `MUJOCO_GL=egl`**. Hollow-cup tests show that an empty final pose and old 6 cm pregrasp can both be collision-free while entry from below crosses the closed bottom. The checker rejects that path, including a 20-micrometre-bottom variant, and accepts entry through the open mouth. Tests also cover state preservation, early contact policy, closing aborts, historical-label preservation, mixed-evaluator resume prevention, real cube holds and real YCB bottle success.

The formerly successful source-scene bottle candidate 345 now fails the swept clearance check even though its old two endpoint checks both pass. It is explicitly retained as a regression case; its previous success did not establish an accessible entry.

## Existing dataset classification

Added nominal `approach_screen` and `approach_valid` fields to all 1,920 records in `outputs/ycb-varied/`, verifying hashes of historical physical trials and labels before/after each dataset update. Nominal clear paths:

- Approach set (384 candidates each): bottle 30, mug 0, knife 172, scissors 17.
- Loading subset (96 each): bottle 11, mug 0, knife 35, scissors 4.

`manifest.json` stores these separate nominal-screen counts. They do not change historical physical outcomes or imply v2 re-evaluation. New physical results require a new output path; combining old acceptance rules with the corrected evaluator is blocked.

## Fresh corrected physical validation

Fresh datasets under `outputs/approach-v2/`, using the same candidate-generation settings/seed as the larger run and one unperturbed trial per candidate:

- `006_mustard_bottle.json`: 96 candidates, 11 nominally accessible; 2 physical successes, 9 measured failures, 85 pre-execution rejections.
- `025_mug.json`: 384 candidates, all rejected before execution; physical labels remain null. No accessible mug entry or successful mug grasp has been demonstrated for this candidate set.

Both have rendered overviews. The bottle view selects physically successful candidate 43 among accessible entries. The mug view displays rejected orientations and a blocked-path marker. Both images were visually inspected. Approach arrows now show direction explicitly; blue marks the exterior pregrasp, cyan a clear path, orange a rejected path and red the blocking position when available.

Use `--filter accessible` to view clear nominal entries, `--filter inaccessible` for rejected entries, and raw `--all` for all saved negatives. The viewer warns when physical labels predate this evaluator. A gripper may finish inside a cavity if its complete geometry can actually enter through the opening; there is no blanket inside-cup rejection rule.

# Larger YCB variation run — 2026-10-16

Completed inside Docker with `python scripts/generate_variations.py`. Saved in `outputs/ycb-varied/` with per-job logs and a machine-readable `manifest.json`.

- **1,536 candidate poses**: 384 per YCB object, sampled from 12,000 surface points with 12 approach angles and one retained pair per surface point; seed 20261016; openings down to 1 mm.
- **Approach experiments**: five trials per pose, 7,680 total attempts. Initial orientation varies with 30° per-axis Gaussian rotation-vector noise; approach varies by 2 mm and 5° per-axis Gaussian noise. Friction multipliers range from 0.6–1.4; mass multipliers range from 0.7–1.3.
- **Loading experiments**: the first 96 poses per object are reused for four additional trials each, 1,536 total attempts. Initial pose/approach are fixed; gravity direction varies with 20° per-axis Gaussian rotation-vector noise after closure. Friction and mass vary over the same ranges.
- Combined storage: **1,920 grasp records and 9,216 trial attempts** across the separate experiment sets. Base masses are the documented 0.1 kg assumptions, so varied masses are 0.07–0.13 kg. Free-space fixture release is used; no original scene furniture is present.

Approach results (successful measured trials / all attempts): bottle 8/1,920, mug 0/1,920, knife 71/1,920, scissors 33/1,920. The number of candidates with at least one success is respectively 7, 0, 52 and 20.

Loading results (successful measured trials / all attempts): bottle 2/384, mug 0/384, knife 15/384, scissors 10/384. Candidates with at least one success: 2, 0, 8 and 6.

Across both sets, 1,834 trials reached physical execution: 139 succeeded and 1,695 failed. The remaining 7,382 attempts were rejected geometrically and retain null physical-success labels. A successful mug grasp still has not been demonstrated. These rates describe this sampler and the configured perturbations; they do not establish hardware performance.

Detailed traces are stored as independently compressed `.traces/*.json.gz` files, referenced by each trial with SHA256 and sample count. The main JSON is atomically checkpointed after each trial. Keep the JSON and its sibling trace directory together. Resume of a completed 1,920-trial dataset was verified to repeat no trials and leave the JSON byte-for-byte unchanged.

**21 tests passed** after the trace storage change. Final audit checks trial counts/IDs, seeds, opening/pose frames, perturbation ranges, physical-success thresholds, matching loading subsets, and every external trace checksum. Each experiment has an all-candidate PNG overview beside its JSON. Native inspection:

```bash
openarm-dataset visualize --output outputs/ycb-varied/approach/032_knife.json --all
openarm-dataset visualize --output outputs/ycb-varied/loading/032_knife.json --filter success
```

# Minimal-environment YCB validation — 2026-10-16

Latest changes were evaluated **inside the existing `openarm-grasping` Docker container**, using its rebuilt project-local Python 3.12 environment. No source tables, pedestals, full robot, cameras, or nearby objects were imported into these runs.

## Official YCB object tests

Prepared the official YCB Google 16k scans with `scripts/prepare_ycb.py`, then ran:

```bash
MUJOCO_GL=egl python scripts/test_ycb.py --num-candidates 96 --render
```

Final spatially diverse sample, seed 42, one trial per candidate:

- `006_mustard_bottle`: 1 physical success, 15 measured failures, 80 geometry rejections.
- `025_mug`: 0 physical successes, 1 measured failure, 95 geometry rejections.
- `032_knife`: 4 physical successes, 56 measured failures, 36 geometry rejections.
- `037_scissors`: 3 physical successes, 6 measured failures, 87 geometry rejections.

Results and all-candidate PNG overviews are saved in `outputs/ycb/`. Rejections have null physical labels and remain in the dataset. The source downloads and collision components are under `assets/ycb/`; both directories are ignored generated artifacts. The original exploratory run is retained separately in `outputs/ycb-initial/`.

These are free-space fixture-release tests (`support: none`): target gravity compensation is enabled during settle, approach and close, then **disabled before lift and hold**. Object poses are not teleported during active trials. Success requires actual contact dynamics, lift, limited slip/rotation and sustained bilateral contact. They establish intrinsic grasp performance under this fixture assumption, not pickup from a table or arm reachability. All four configured masses are explicitly assumed to be 0.1 kg; they are not measured YCB masses. Inertia comes from volume-weighted convex components.

CoACD generated 3 bottle, 32 mug, 2 knife and 11 scissors collision components. Archive hashes, provenance, decomposition settings and seed are saved per object and in dataset metadata. The mug's extents are approximately 0.117 × 0.093 × 0.081 m; the gripper maximum gap is about 0.08248 m. Wrapping around its body is not generally feasible. A separate 192-candidate test allowing widths down to 1 mm yielded 0 successes, 6 measured failures and 186 geometry rejections (`outputs/ycb-mug-thin.json`). A successful mug handle/rim grasp has **not** been demonstrated.

YCB has no LASO labels in these scans. No LASO annotation was transferred between different shapes; affordance labels remain null.

## Visualization and tests

**20 tests passed** in Docker with `MUJOCO_GL=egl`, including the existing geometry/physics/reset/resume tests, standalone mesh loading without an object scene XML, omission of source scenery, fixture release and reset, spatial pair diversity, candidate outcome filters, and a real all-candidate EGL PNG render. Cube, YCB mug and scissors overviews were visually inspected. Interactive N/P/arrow navigation is implemented with MuJoCo's native key callback; desktop keyboard interaction still needs user verification.

Use `visualize --all` to overlay all saved contact/approach poses. N/Right and P/Left browse the selected candidate's full gripper geometry; A toggles the overlays. Filter by outcome with `--filter`. Rendering saved original cube data was verified as well; historical datasets retain their original environment to preserve their meaning.

The CLI now defaults to one retained pair per sampled surface point, spreading candidate groups across the mesh. Older datasets retain their stored candidate poses and remain viewable/resumable.

---

# Earlier scene-based validation — 2026-10-16

Environment: project-local uv `.venv`, CPython 3.12.3, MuJoCo 3.16.0, NumPy 2.5.4, SciPy 1.18.1, trimesh 5.1.1. Exact dependency resolution is in `uv.lock`.

## Physical results

Commands were run headlessly from the repository root using the project's `.venv/bin/python generate_grasp_dataset.py` wrapper. The commands below are equivalent from the project directory:

```bash
uv run openarm-dataset full --num-candidates 24 --seed 42 --output outputs/validation-box.json
uv run openarm-dataset full --scene ../../src/openarm_mujoco_bringup/config/scene_ycb_bottle.yaml --num-candidates 346 --seed 42 --output outputs/validation-bottle.json
```

Final implementation, one trial per candidate:

- Box: 24 candidates; 7 physically successful, 1 measured physical failure, 16 geometric rejections with null physical labels.
- Actual YCB mustard bottle mesh and scene: 346 candidates; 1 physically successful (ID 345), 1 measured physical failure (ID 344), 344 geometric rejections. The mesh has two degenerate triangles; they are reported and removed for sampling.
- An exploratory 512-candidate bottle run found 1 success, 3 measured failures, and 508 rejections. Most candidates intersect the source collision proxies. Increasing candidate count alone does not guarantee good coverage.
- An exploratory 128-candidate real knife run retained all candidates but rejected all before physical execution. Contacts showed table/pedestal/gripper obstruction. Knife physical pickup has **not** been demonstrated.

The final box and bottle JSON files are in ignored `outputs/`, including contact traces and separate evaluation statuses. They contain null LASO labels because no real annotation registration was supplied. A successful box hold was rendered headlessly with EGL and visually inspected: the actual gripper held the cube above the table. The native interactive viewer has not been exercised on the user's desktop.

## Annotation validation

The existing exported LASO knife annotation `not-ros-pkg/mesh-grasp-annotator/exports/knife_1460eded8006b10139c78a1e40e247f3.laso.json` was loaded with its real PLY and masks: 305 grasp-region points. No identity transform was asserted to align that cloud with the simulation knife. Real-object handover-suitability labels remain unverified until an explicit annotation-to-object registration is supplied.

Synthetic, explicitly aligned LASO fixtures verify both source formats, metric transforms, interior/clearance overlap, and preservation of overlapping grasps. A physical box success is independently labeled unsuitable when its human region is occupied by the gripper. The CLI labeling test confirms that labeling retains candidates and does not alter physical trial records.

## Tests

```bash
uv run pytest -q
```

16 tests cover mesh scaling and multi-part loading; rigid transforms; antipodal conditions; rotation matrices, TCP offset and width limits; deterministic candidate sampling; reset state consistency; geometric rejection and allowed intended contacts; actual box success and failure; contact forces; actual bottle pickup; LASO export and pickle masks; overlap and independent labels; serialization; perturbation reproducibility; loading-gravity variation; resume/extension without repeating completed trials; stale-scene detection; cross-process fingerprint stability; and preservation of physical results during CLI labeling.

Expected warning: two degenerate faces removed from the real bottle mesh. This is a mesh validation report, not an ignored simulation failure.

## Practical limits

### All-category batch (2026-10-17, ongoing)

`scripts/generate_object_examples.py` catalogs 739 jobs: 5 YCB scans and 734
distinct LASO bottle/knife/mug shapes across val/test/train. Each requests 384
candidates and one nominal physical trial, with 12 sampled approach directions.
The batch is running in `openarm-grasping` independently of the chat process;
`outputs/bottle-knife-mug/manifest.json` is the live source of status. This is
not a claim that all LASO jobs have completed or can be reconstructed.

The five YCB jobs completed with 1,920 candidates: 238 nominally accessible,
11 physical successes, 227 measured failures and 1,682 geometry rejections.
YCB meshes retain their native scale; no LASO annotation registration is
invented. The official Windex scan is incomplete, as noted in the manifest.

LASO preparation uses assumed class dimensions and reconstructed solids.
84 source shapes lack a `grasp` mask (73 bottles, 4 knives, 7 mugs); these retain
null keep-free labels while physics can still be evaluated. Preparation errors
are recorded separately from physical failures. Source points already buried
inside assumed solid material are not mistaken for omitted points; the buried
count is recorded. Large visual meshes use OBJ to avoid MuJoCo's 200,000-face
STL decoder limit while retaining triangles. Reconstruction connectivity and
outside-source coverage remain checked, and disconnected sources can fail.

Final regression run: 32 passed. New tests cover catalog deduplication and
class/split scope, plan/resume/status manifests, and distinguishing buried
source points from omitted points. Live resume retained completed YCB outputs
and resumed LASO physical trials from saved checkpoints. The full test log is
`outputs/object-batch-tests.log`. See the live manifest and per-object logs for
actual LASO completion and failures; the full batch is not yet validated.

The first completed LASO batch object, validation bottle
`48202d357e79315e45891653421dc140`, has 384 candidates, 184 accessible approaches,
77 physical successes, 107 measured failures and 200 geometry rejections. All
384 have registered LASO overlap labels; 75 physical successes satisfy the
configured keep-free threshold. Its generated PNG was visually inspected.
This establishes live generalized LASO preparation/evaluation/label/render
execution, not correctness of every other reconstructed object.

### LASO mug reconstruction and evaluation (2026-10-16)

Evaluated LASO validation mug `34ae0b61b0d8aaf2d7b20fded0142d7a` in the existing
`openarm-grasping` Docker container using the project `.venv`. LASO only supplies
normalized points/masks, so the 85 mm source extent, 0.1 kg mass and 2.5 mm
point-shell material radius are assumptions. The resulting surface is one
watertight component, with 65.23 x 89.25 x 73.49 mm bounds. Maximum distance from
a source point to the reconstructed surface is 3.80 mm. CoACD produced 64
collision components. This is an approximate reconstructed LASO object, not an
original mesh with measured physical properties.

A vertical ray at object x=0, y=15 mm passes through the mouth and first hits
the surface bottom at z=-30.92 mm, and collision geometry at z=-30.59 mm. A ray
along X at y=-30 mm, z=0 passes through the handle hole with no intersections
in either representation. These are spot checks, not a proof of global mesh
equivalence. The synthetic reconstruction regression independently verifies
an open mouth, closed bottom and source-point coverage.

`outputs/laso-mug/grasps.json`: 384 candidates with 12 approach orientations,
unchanged approach/contact thresholds, no support plane and an initial
gravity-compensated fixture until closure. Results: 102 nominally accessible,
43 physical successes, 59 measured failures, 282 geometry rejections with
unmeasured physical outcomes. Trials include close/lift/hold. All candidates
are labeled using the source LASO grasp mask and the exact stored similarity
transform; 39 physical successes meet the default overlap threshold. Sampling
does not use that mask. Candidate 25 has physical success and zero overlap.

EGL preview `outputs/laso-mug/success-25.png` was visually inspected. Final
regression run: **29 passed**. The 51 warnings comprise two expected bottle
mesh-cleanup warnings and 49 third-party scikit-image/NumPy deprecation warnings.
The full log is `outputs/laso-mug/tests.log`.

These results establish gripper-only simulation behavior. They do not establish full OpenArm reachability, real hardware success, calibrated friction accuracy, human receiving-hand safety, or general grasp coverage. The default sampler is an initial geometry baseline, and collision proxies strongly affect real-object acceptance. No full-arm or ROS code was changed.
