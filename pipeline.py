'''
import mitsuba as mi
import drjit as dr

import trimesh
import xatlas

mesh = trimesh.load_mesh("~/xatlas-stuff/00190663.obj")
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
xatlas.export("output.obj", mesh.vertices[vmapping], indices, uvs)

mi.set_variant('llvm_ad_rgb')
uvmesh = mi.Mesh(
    "uvmesh", 
    len(vmapping), 
    len(uvs), 
    has_vertex_normals=False, 
    has_vertex_texcoords=False,
)

uvmesh = mi.load_dict({
    "type": "obj",
    "filename": "output.obj",
    "face_normals": False,
    "to_world": mi.ScalarTransform4f.rotate([0, 0, 1], angle=10),
})
print(uvmesh)
img = mi.render(uvmesh)

plt.axis("off")
plt.imshow(mi.util.convert_to_bitmap(img));
plt.show()
'''

# imports
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import os

from mitsubasteps.geometry import compute_mitsuba_scene
from scripts.load_xml import load_scene
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform

mi.set_variant('llvm_ad_rgb')
filepath = os.path.join(os.getcwd(), "scenes", "suzanne", "suzanne.xml")

scene_params = mi.load_file(filepath)#, res=128)#, integrator='prb')
image_ref = mi.render(scene_params, spp=512)

# get largesteps scene params
scene_large = load_scene(filepath)
# Load reference shape
v_ref = scene_large["mesh-target"]["vertices"]
n_ref = scene_large["mesh-target"]["normals"]
f_ref = scene_large["mesh-target"]["faces"]
# Load source shape
v = scene_large["mesh-source"]["vertices"]
f = scene_large["mesh-source"]["faces"]

img = mi.render(compute_mitsuba_scene(v_ref, n_ref, f_ref))

plt.axis("off")
plt.imshow(mi.util.convert_to_bitmap(img));
plt.show()

#mesh.write_ply("mymesh.ply")

