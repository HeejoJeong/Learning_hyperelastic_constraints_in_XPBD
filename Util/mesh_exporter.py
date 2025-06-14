import torch
import numpy as np
from pathlib import Path


def orient_faces_outward(verts: torch.Tensor, faces: torch.Tensor):

    center = verts.mean(0)
    v0, v1, v2 = (verts[faces[:, i]] for i in range(3))
    fn = torch.cross(v1 - v0, v2 - v0)
    fc = (v0 + v1 + v2) / 3

    flip = (fn * (center - fc)).sum(-1) > 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces

def compute_vertex_normals(verts, faces):
    v0, v1, v2 = (verts[faces[:, i]] for i in range(3))
    fn = torch.cross(v1 - v0, v2 - v0)
    area = fn.norm(dim=-1, keepdim=True) + 1e-12
    fn = fn / area

    n_vert = torch.zeros_like(verts)
    for i in range(3):
        n_vert.index_add_(0, faces[:, i], fn * area)
    return torch.nn.functional.normalize(n_vert, dim=-1)

def export_obj_sequence_with_normals(x_traj, faces_np, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)

    faces = torch.from_numpy(faces_np).long().to(x_traj.device)   # (m,3)

    for t in range(x_traj.shape[0]):
        verts = x_traj[t]                                         # (n,3)

        faces_fix = orient_faces_outward(verts, faces.clone())

        n_vert = compute_vertex_normals(verts, faces_fix)
        faces_obj = faces_fix + 1

        with open(out_dir / f"frame_{t:04d}.obj", "w") as f:
            f.write("# frame {:04d}\n".format(t))
            f.write("s 1\n")                                      # smoothing on
            for v in verts.cpu():  f.write(f"v  {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for n in n_vert.cpu(): f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for tri in faces_obj.cpu():
                f.write(f"f {tri[0]}//{tri[0]} {tri[1]}//{tri[1]} {tri[2]}//{tri[2]}\n")
        print(f"[✓] Saved frame_{t:04d}.obj")
