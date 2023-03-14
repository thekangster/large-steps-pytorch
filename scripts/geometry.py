import torch
import mitsuba as mi
import drjit as dr

def remove_duplicates(v, f):
    """
    Generate a mesh representation with no duplicates and
    return it along with the mapping to the original mesh layout.
    """

    unique_verts, inverse = torch.unique(v, dim=0, return_inverse=True)
    new_faces = inverse[f.long()]
    return unique_verts, new_faces, inverse

def average_edge_length(verts, faces):
    """
    Compute the average length of all edges in a given mesh.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    return (A + B + C).sum() / faces.shape[0] / 3

def massmatrix_voronoi(verts, faces):
    """
    Compute the area of the Voronoi cell around each vertex in the mesh.
    http://www.alecjacobson.com/weblog/?p=863

    params
    ------

    v: vertex positions
    f: triangle indices
    """
    # Compute edge lengths
    l0 = (verts[faces[:,1]] - verts[faces[:,2]]).norm(dim=1)
    l1 = (verts[faces[:,2]] - verts[faces[:,0]]).norm(dim=1)
    l2 = (verts[faces[:,0]] - verts[faces[:,1]]).norm(dim=1)
    l = torch.stack((l0, l1, l2), dim=1)

    # Compute cosines of the corners of the triangles
    cos0 = (l[:,1].square() + l[:,2].square() - l[:,0].square())/(2*l[:,1]*l[:,2])
    cos1 = (l[:,2].square() + l[:,0].square() - l[:,1].square())/(2*l[:,2]*l[:,0])
    cos2 = (l[:,0].square() + l[:,1].square() - l[:,2].square())/(2*l[:,0]*l[:,1])
    cosines = torch.stack((cos0, cos1, cos2), dim=1)

    # Convert to barycentric coordinates
    barycentric = cosines * l
    barycentric = barycentric / torch.sum(barycentric, dim=1)[..., None]

    # Compute areas of the faces using Heron's formula
    areas = 0.25 * ((l0+l1+l2)*(l0+l1-l2)*(l0-l1+l2)*(-l0+l1+l2)).sqrt()

    # Compute the areas of the sub triangles
    tri_areas = areas[..., None] * barycentric

    # Compute the area of the quad
    cell0 = 0.5 * (tri_areas[:,1] + tri_areas[:, 2])
    cell1 = 0.5 * (tri_areas[:,2] + tri_areas[:, 0])
    cell2 = 0.5 * (tri_areas[:,0] + tri_areas[:, 1])
    cells = torch.stack((cell0, cell1, cell2), dim=1)

    # Different formulation for obtuse triangles
    # See http://www.alecjacobson.com/weblog/?p=874
    cells[:,0] = torch.where(cosines[:,0]<0, 0.5*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,0]<0, 0.25*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,0]<0, 0.25*areas, cells[:,2])

    cells[:,0] = torch.where(cosines[:,1]<0, 0.25*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,1]<0, 0.5*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,1]<0, 0.25*areas, cells[:,2])

    cells[:,0] = torch.where(cosines[:,2]<0, 0.25*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,2]<0, 0.25*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,2]<0, 0.5*areas, cells[:,2])

    # Sum the quad areas to get the voronoi cell
    return torch.zeros_like(verts).scatter_add_(0, faces, cells).sum(dim=1)

def compute_face_normals(verts, faces):
    """
    Compute per-face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)

    v = [verts.index_select(1, fi[0]),
                 verts.index_select(1, fi[1]),
                 verts.index_select(1, fi[2])]

    c = torch.cross(v[1] - v[0], v[2] - v[0])
    n = c / torch.norm(c, dim=0)
    return n


def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))

def compute_vertex_normals(verts, faces, face_normals):
    """
    Compute per-vertex normals from face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    face_normals : torch.Tensor
        Per-face normals
    """
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
    return (normals / torch.norm(normals, dim=0)).transpose(0, 1)

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
