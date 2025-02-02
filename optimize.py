import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from mitsuba.scalar_rgb import Transform4f as T
import numpy as np
import trimesh
from tqdm import trange

mi.set_variant("cuda_ad_rgb")
import help
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
    "integrator": {"type": "path"},
    "light": {
        "type": "point",
        #"position": [0.0, -1.0, 7.0],
        "position": [0, 2, 5],
        "intensity": {
            "type": "spectrum",
            "value": 15.0,
            }
        },
    "sensor": {
        "type": "perspective",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0, 2, 7], target=[0, 0, 0], up=[0, 0, -1]
            #origin=[2, 6, 2], target=[0, 0, 0], up=[0, 0, -1]
            ),
        "film": {
            "type": "hdrfilm",
        },
    },
}

scene_dictold = {
    "type": "scene",
    "integrator": {
        "type": "path",
    },
    "sensor": {
        "type": "perspective",
        "to_world": T.look_at(origin=(0, 2, 2), target=(0, 0, 0), up=(0, 0, 1)),
        "film": {
            "type": "hdrfilm",
            "width": 1024,
            "height": 1024,
        },
    },
    # "mesh": util.trimesh2mitsuba(trimesh.creation.icosphere()),
    # "mesh": suzanne,
    "light": {
        "type": "point",
        "position": [1, 2, 2],
        "intensity": {
            "type": "spectrum",
            "value": 10.0,
        },
    },
}
#scene_dictold = {
#    "type": "scene",
#    "integrator": {
#        "type": "path",
#    },
#    "sensor": {
#        "type": "perspective",
#        "to_world": T.look_at(origin=[0, 2, 7], target=[0, 0, 0], up=[0, 0, -1]),
#        #origin=(0, 2, 2), target=(0, 0, 0), up=(0, 0, -1)),
#        "film": {
#            "type": "hdrfilm",
#        },
#    },
#    "floor": {
#        "type": "rectangle",
#        "to_world": T.translate([0, 0, -1]).scale(10),
#    },
#    "light": {
#        "type": "point",
#        "position": [0, 2, 5],
#        "intensity": {
#            "type": "spectrum",
#            "value": 10.0,
#        },
#    },
#}

scene_ref = mi.load_dict({**scene_dict, **{"mesh": suzanne}})
ref = mi.render(scene_ref, spp=128)
help.display(ref)

scene = mi.load_dict({**scene_dict, **{"mesh": source}})
src = mi.render(scene, spp=128)
#help.display(src)

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

opt = mi.ad.Adam(lr=0.01)
opt["u"] = u
print(f"{opt=}")

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

img = mi.render(scene, params, spp=128)
help.display(img)

