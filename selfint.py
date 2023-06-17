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
        "to_world": T.scale(0.25),
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

scene = {
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
    "floor": {
        "type": "rectangle",
        "to_world": T.translate([0, 0, -1]).scale(10),
    },
    "light": {
        "type": "point",
        "position": [1, 2, 2],
        "intensity": {
            "type": "spectrum",
            "value": 10.0,
        },
    },
}

scene_ref = mi.load_dict({**scene, **{"mesh": suzanne}})

ref = mi.render(scene_ref, spp=128)

# scene = mi.load_dict(
#     {
#         **scene,
#         **{
#             "mesh": help.trimesh2mitsuba(
#                 trimesh.creation.icosphere(subdivisions=4, radius=0.5)
#             )
#         },
#     }
# )
scene = mi.load_dict({**scene, **{"mesh": source}})

params = mi.traverse(scene)
print(params)
params.keep("mesh.vertex_positions")

opt = mi.ad.Adam(lr=0.003, params=params)
print(f"{opt=}")

params.update(opt)

for i in trange(50):
    img = mi.render(scene, params, spp=1)
    loss = dr.mean(dr.sqr(img - ref))
    dr.backward(loss)
    opt.step()
    params.update(opt)

img = mi.render(scene, params, spp=128)
help.display(img)

