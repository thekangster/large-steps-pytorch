import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from mitsuba.scalar_rgb import Transform4f as T
import numpy as np
import trimesh
from tqdm import trange

mi.set_variant('llvm_ad_rgb')
import help 
import util

suzanne = mi.load_dict(
    {
        "type": "ply",
        #"filename": "scenes/suzanne/meshes/target.ply",
        "filename": "./our_scenes/capybara.ply",
        "face_normals": True,
        "to_world" : mi.ScalarTransform4f.rotate([0,0,1], angle = 180),
    }
)

'''
source = mi.load_dict(
    {
        "type": "ply",
        "filename": "scenes/suzanne/meshes/source.ply",
        "face_normals": True,
    }
)
'''

scene_dict = {
    "type": "scene",
    "integrator": {"type": "path"},
    "light": {
        "type": "point",
        "position": [0.0, -1.0, 7.0],
        "intensity": {
            "type": "spectrum",
            "value": 15.0,
            }
        },
    "sensor": {
        "type": "perspective",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[5, 5, 3], target=[0, 0, -1], up=[0, 0, -1]
            ),
        },
}

scene_dictold = {
    "type": "scene",
    "integrator": {
        "type": "path",
    },
    "sensor": {
        "type": "perspective",
        "to_world": T.look_at(origin=[5, 5, 5], target=[0, 0, 0], up=[0, 0, 1]),
        #origin=(0, 2, 2), target=(0, 0, 0), up=(0, 0, -1)),
        "film": {
            "type": "hdrfilm",
            "width": 1024,
            "height": 1024,
        },
    },
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

scene_ref = mi.load_dict({**scene_dictold, **{"mesh": suzanne}})
ref = mi.render(scene_ref, spp=1)
help.display(ref)

#scene = mi.load_dict({**scene_dict, **{"mesh": source}})
scene = mi.load_dict(
    {
        **scene_dictold,
        **{
            "mesh": util.trimesh2mitsuba(
                trimesh.creation.icosphere(subdivisions=4, radius=0.5)
            )
        },
    }
)
src = mi.render(scene, spp=1)
help.display(src)

params = mi.traverse(scene)
print(params)
# params.keep("mesh.vertex_positions")

positions = params["mesh.vertex_positions"]
faces = params["mesh.faces"]
print("vertex_normals\n")
print(params["mesh.vertex_normals"])


step_size = 3e-2 # Step size
lambda_ = 19 # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_*L)

M = help.compute_matrix(positions, faces, lambda_=lambda_)
u = help.to_differential(M, positions)

opt = mi.ad.Adam(lr=0.01, uniform=True)
opt["u"] = u
print(f"{opt=}")

"""
from scripts.geometry import mi_compute_vertex_normals, mi_compute_face_normals
positions = params["mesh.vertex_positions"]
faces = params["mesh.faces"]
fn = mi_compute_face_normals(positions, faces)
n = mi_compute_vertex_normals(positions, faces, fn)
print(f"{fn=}")
print(f"{n=}")
"""

steps = 300
for it in trange(steps):

    u = opt["u"]
    v = help.from_differential(M, u)
    params["mesh.vertex_positions"] = v

    positions = params["mesh.vertex_positions"]
    faces = params["mesh.faces"]

    face_normals = help.mi_compute_face_normals(positions, faces)
    n = help.mi_compute_vertex_normals(positions, faces, face_normals)
    params["mesh.vertex_normals"] = n

    params.update()

    img = mi.render(scene, params, spp=1)
    loss = dr.mean(dr.abs(img - ref))
    dr.backward(loss)

    opt.step()

img = mi.render(scene, params, spp=1)
help.display(img)

