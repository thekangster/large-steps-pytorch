import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from mitsuba.scalar_rgb import Transform4f as T
import numpy as np
import trimesh
from tqdm import trange

mi.set_variant('llvm_ad_rgb')
import util

suzanne = mi.load_dict(
    {
        "type": "ply",
        "filename": "scenes/suzanne/meshes/target.ply",
        "face_normals": True,
    }
)

source = mi.load_dict(
    {
        "type": "ply",
        "filename": "scenes/suzanne/meshes/source.ply",
        "face_normals": True,
    }
)

scene_dict = {
    "type": "scene",
    "integrator": {
        "type": "path",
    },
    "sensor": {
        "type": "perspective",
        "to_world": T.look_at(origin=[0, 2, 7], target=[0, 0, 0], up=[0, 0, -1]),
        #origin=(0, 2, 2), target=(0, 0, 0), up=(0, 0, -1)),
        "film": {
            "type": "hdrfilm",
        },
    },
    # "mesh": util.trimesh2mitsuba(trimesh.creation.icosphere()),
    # "mesh": suzanne,
    "floor": {
        "type": "rectangle",
        "to_world": T.translate([0, 0, -1]).scale(10),
    },
    "light": {
        "type": "point",
        "position": [0, 2, 5],
        "intensity": {
            "type": "spectrum",
            "value": 10.0,
        },
    },
}

scene_ref = mi.load_dict({**scene_dict, **{"mesh": suzanne}})

ref = mi.render(scene_ref, spp=128)
#util.display(ref)

scene = mi.load_dict({**scene_dict, **{"mesh": source}})
"""
scene = mi.load_dict(
    {
        **scene_dict,
        **{
            "mesh": util.trimesh2mitsuba(
                trimesh.creation.icosphere(subdivisions=4, radius=0.5)
            )
        },
    }
)
"""
src = mi.render(scene, spp=128)
#util.display(src)

params = mi.traverse(scene)
print(params)
# params.keep("mesh.vertex_positions")

positions = params["mesh.vertex_positions"]
faces = params["mesh.faces"]
print("vertex_normals\n")
print(params["mesh.vertex_normals"])


step_size = 3e-2 # Step size
lambda_ = 19 # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_*L)

M = util.compute_matrix(positions, faces, lambda_=lambda_)
u = util.to_differential(M, positions)

opt = mi.ad.Adam(lr=0.01)
opt["u"] = u
print(f"{opt=}")
# params.update()
# params.update(opt)

from scripts.geometry import mi_compute_vertex_normals, mi_compute_face_normals
positions = params["mesh.vertex_positions"]
faces = params["mesh.faces"]
fn = mi_compute_face_normals(positions, faces)
n = mi_compute_vertex_normals(positions, faces, fn)
print(f"{fn=}")
print(f"{n=}")

steps = 200
for it in trange(steps):
    positions = params["mesh.vertex_positions"]
    faces = params["mesh.faces"]

    u = opt["u"]
    v = util.from_differential(M, u)
    params["mesh.vertex_positions"] = v

    fn = mi_compute_face_normals(positions, faces)
    n = mi_compute_vertex_normals(positions, faces, fn)
    params["mesh.vertex_normals"] = n

    params.update()

    img = mi.render(scene, params, spp=1)
    #mi.util.write_bitmap(f"out/{i}.jpg", img)

    loss = dr.mean(dr.sqr(img - ref))
    dr.backward(loss)
    opt.step()

img = mi.render(scene, params, spp=128)
util.display(img)

