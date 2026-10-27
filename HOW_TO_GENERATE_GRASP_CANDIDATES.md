**Dataset**: YCB-Affordance + LASO

1. Run `export_human_grasp_affordance.py` to get `affordance.npz`, `affordance.ply` and `metadata.json`. Then prepare data structure like below.
```
data/ or data/somedir/
    |--object-id/
        |--affordance.npz # contains points [N,3], normals [N,3], affordance_score[N]
        |--affordance.ply 
        |--metadata.json
```
2. Reconstruct mesh from point cloud for LASO using MeshLab.
```
python reconstruct_mesh.py --all
```
3. Decompose non-convex mesh using CoACD
```
python coacd_decompose.py --all
```
