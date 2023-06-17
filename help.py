import trimesh
import numpy as np
import mitsuba as mi
import drjit as dr
import largesteps
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))

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

def tensor_to_point3f(T):
    to_vector = T.tolist()
    x = dr.zeros(mi.Float, len(to_vector))
    y = dr.zeros(mi.Float, len(to_vector))
    z = dr.zeros(mi.Float, len(to_vector))

    for i, vec in enumerate(to_vector):
        x[i] = vec[0]
        y[i] = vec[1]
        z[i] = vec[2]

    return mi.Point3f(x, y, z)


def mi_compute_face_normals(verts: mi.Float, faces: mi.UInt) -> torch.Tensor:
    verts = mi.TensorXf(verts, shape=(len(verts) // 3, 3))
    faces = mi.TensorXf(faces, shape=(len(faces) // 3, 3))

    verts = verts.torch()
    faces = faces.torch()

    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)

    v = [verts.index_select(1, fi[0]),
                 verts.index_select(1, fi[1]),
                 verts.index_select(1, fi[2])]

    c = torch.cross(v[1] - v[0], v[2] - v[0])
    n = c / torch.norm(c, dim=0)
    return n#tensor_to_point3f(n)

def tensor_to_mifloat(T: torch.Tensor) -> mi.Float:
    n = len(T)*3
    to_vector = T.tolist()
    x = dr.zeros(mi.Float, n)

    i = 0
    for vec in to_vector:
        x[i] = vec[0]
        x[i+1] = vec[1]
        x[i+2] = vec[2]
        i += 3

    return mi.Float(x)


def mi_compute_vertex_normals(verts: mi.Float, faces: mi.UInt, face_normals: torch.Tensor) -> mi.Float:
    #print(f"{verts=}")
    #print(f"{type(verts)=}")
    #print(f"{len(verts)=}")
    verts = mi.TensorXf(verts, shape=(len(verts) // 3, 3))
    faces = mi.TensorXf(faces, shape=(len(faces) // 3, 3))
    #print(f"{verts=}")

    verts = verts.torch()
    faces = faces.torch()
    #print(f"{verts=}")
    #print(f"{tensor_to_mifloat(verts)=}")

    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [verts.index_select(1, fi[0]),
             verts.index_select(1, fi[1]),
             verts.index_select(1, fi[2])]

    for i in range(3):
        d0 = v[(i + 1) % 3] - v[i]
        d0 = d0 / torch.norm(d0)
        d1 = v[(i + 2) % 3] - v[i]
        d1 = d1 / torch.norm(d1)
        d = torch.sum(d0*d1, 0)
        face_angle = safe_acos(torch.sum(d0*d1, 0))
        nn =  face_normals * face_angle
        for j in range(3):
            normals[j].index_add_(0, fi[i], nn[j])
    return tensor_to_mifloat((normals / torch.norm(normals, dim=0)).transpose(0, 1))


def display(render):
    plt.axis("off")
    plt.imshow(mi.util.convert_to_bitmap(render));
    plt.show()


def compute_matrix(positions: mi.Float, faces: mi.UInt, lambda_: float) -> torch.Tensor:
    positions = mi.TensorXf(positions, shape=(len(positions) // 3, 3))
    faces = mi.TensorXf(faces, shape=(len(faces) // 3, 3))

    positions = positions.torch()
    faces = faces.torch()

    from largesteps.geometry import compute_matrix

    M = compute_matrix(positions, faces, lambda_)

    return M


"""https://github.com/DoeringChristian"""
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
