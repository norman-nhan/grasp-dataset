# Visualizations
- `visualize_surface_normal.py`

# Mesh Reconstruction
- `reconstruct_mesh.py`

# Collision-Aware Approximate Convex Decomposition (CoACD)
For collision checking in Mujoco, non-convex mesh must be decomposed into multiple convex meshes.
- `coacd_decompose.py`

# Build mjcf
- `convert_laso_to_mujoco.py`: create `model.xml` for each object in laso root
- `convert_ycb_to_mujoco.py`: create `model.xml`  for each object in ycb affordance root

# Grasp Candidates Generation
- `generate_graspnet_grasp.py`: using GraspNet to generate grasp candidates. Also doesn't make sense if we don't split it to multi-view.
- `generate_graspnet_grasp_all.py`: using GraspNet to generate grasp candidates for all object under data root.
- `generate_analytical_grasp.py`: not really good because the grasp candidates doesn't make any sense.
- `generate_multiview_graspnet_grasp.py`: MAIN, split full cloud into multi-view partial cloud and combine grasp candidates from different views.

# Evaluate generated grasp candidates
- `eval_grasp.py`: filter grasp candidates based on antipodal and friction cones.
- `eval_grasp_mujoco.py`: check grasp candidates collision in mujoco.

# Utils
- `convert_ply2obj.py`: convert ply to obj because mujoco doesn't support ply.
- `download_ycb_dataset.py`: useful to download the official ycb dataset.
- `build_laso_object_types.py`: create `laso_object_types.json` to know which object id is belong to which object category.

# Data structure
```
mujoco/ # Contains collision meshes, visual mesh, model.xml
affordance.npz
affordance.ply
metadata.json
```