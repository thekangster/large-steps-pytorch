"""
https://github.com/DoeringChristian
"""

import trimesh
import numpy as np
import mitsuba as mi
import drjit as dr
import largesteps
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mi.set_variant("llvm_ad_rgb")

def display(render):
    plt.axis("off")
    plt.imshow(mi.util.convert_to_bitmap(render));
    plt.show()

def trimesh2mitsuba(mesh: trimesh.Trimesh) -> mi.Mesh:
    vertices: np.ndarray = np.array(mesh.vertices)
    indices: np.ndarray = np.array(mesh.faces)
    mesh = mi.Mesh(
        "trimesh",
        vertex_count=vertices.shape[0],
        face_count=indices.shape[0],
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )

    params = mi.traverse(mesh)
    params["vertex_positions"] = dr.ravel(mi.Point3f(vertices))
    params["faces"] = dr.ravel(mi.Vector3u(indices))
    params.update()
    return mesh


def mesh_loss_normal(normals: mi.Float, faces: mi.UInt) -> mi.Float:
    faces = dr.unravel(mi.Vector3u, faces)
    ...


def mesh_loss_lap(positions: mi.Float, faces: mi.UInt) -> mi.Vector3f:
    faces = dr.unravel(mi.Vector3u, faces)

    num_vertices = len(positions) // 3
    dst = dr.zeros(mi.Point3f, num_vertices)
    # print(f"{num_vertices=}")

    v1 = dr.gather(mi.Point3f, positions, faces.x)
    v2 = dr.gather(mi.Point3f, positions, faces.y)
    v3 = dr.gather(mi.Point3f, positions, faces.z)

    n12 = dr.normalize(v2 - v1)
    n13 = dr.normalize(v3 - v1)
    n23 = dr.normalize(v3 - v2)

    # Edge 1-2:
    w = dr.cot(dr.acos(dr.dot(n23, n13)))
    dr.scatter_reduce(dr.ReduceOp.Add, dst, w * v2, faces.x)
    dr.scatter_reduce(dr.ReduceOp.Add, dst, w * v1, faces.y)
    dr.scatter_reduce(dr.ReduceOp.Add, dst, -w * v1, faces.x)
    dr.scatter_reduce(dr.ReduceOp.Add, dst, -w * v2, faces.y)

    # Edge 2-3:
    w = dr.cot(dr.acos(dr.dot(n12, n13)))
    dr.scatter_reduce(dr.ReduceOp.Add, dst, w * v3, faces.y)
    dr.scatter_reduce(dr.ReduceOp.Add, dst, w * v2, faces.z)
    dr.scatter_reduce(dr.ReduceOp.Add, dst, -w * v2, faces.y)
    dr.scatter_reduce(dr.ReduceOp.Add, dst, -w * v3, faces.z)

    # Edge 3-1:
    w = dr.cot(dr.acos(dr.dot(-n12, n23)))
    dr.scatter_reduce(dr.ReduceOp.Add, dst, w * v1, faces.z)
    dr.scatter_reduce(dr.ReduceOp.Add, dst, w * v3, faces.x)
    dr.scatter_reduce(dr.ReduceOp.Add, dst, -w * v3, faces.z)
    dr.scatter_reduce(dr.ReduceOp.Add, dst, -w * v1, faces.x)

    return dr.sum(dr.sqr(dst))


def compute_matrix(positions: mi.Float, faces: mi.UInt, lambda_: float) -> torch.Tensor:
    positions = mi.TensorXf(positions, shape=(len(positions) // 3, 3))
    faces = mi.TensorXf(faces, shape=(len(faces) // 3, 3))

    positions = positions.torch()
    faces = faces.torch()

    from largesteps.geometry import compute_matrix

    M = compute_matrix(positions, faces, lambda_)

    return M


def to_differential(M: torch.Tensor, v: mi.Float) -> mi.Float:
    v = mi.TensorXf(v, shape=(len(v) // 3, 3))

    @dr.wrap_ad(source="drjit", target="torch")
    def to_differential_internal(v: torch.Tensor):
        from largesteps.parameterize import to_differential

        return to_differential(M, v)

    return to_differential_internal(v).array


def from_differential(M: torch.Tensor, u: mi.Float, method="Cholesky") -> mi.Float:
    u = mi.TensorXf(u, shape=(len(u) // 3, 3))

    @dr.wrap_ad(source="drjit", target="torch")
    def to_differential_internal(u: torch.Tensor):
        from largesteps.parameterize import from_differential

        return from_differential(M, u, method)

    return to_differential_internal(u).array


if __name__ == "__main__":
    mesh = mi.load_dict(
        {
            "type": "ply",
            "filename": "scenes/suzanne/meshes/target.ply",
            "face_normals": True,
        }
    )

    params = mi.traverse(mesh)

    positions = params["vertex_positions"]
    faces = params["faces"]

    M = compute_matrix(positions, faces, lambda_=10)
    print(f"{M=}")

    u = to_differential(M, positions)

    print(f"{type(u)=}")
    print(f"{u=}")

    v = from_differential(M, u)
