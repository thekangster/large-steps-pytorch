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
from tqdm import trange
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import os
from mitsubasteps.help import compute_mitsuba_scene
from scripts.load_xml import load_scene

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

# render reference
ref_scene = compute_mitsuba_scene(v_ref, n_ref, f_ref)
ref_imgs = mi.render(ref_scene)
# plt.axis("off")
# plt.imshow(mi.util.convert_to_bitmap(ref_imgs));
# plt.show()

#ref = mi.traverse(ref_scene)

from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix

# Number of optimization steps
steps = 1000
# Step size
step_size = 3e-2
# Hyperparameter lambda of our method, used to compute the matrix (I + lambda_*L)
lambda_ = 19
# Compute the system matrix
M = compute_matrix(v, f, lambda_)
# Parameterize
u = to_differential(M, v)

# Optimization
from largesteps.optimize import AdamUniform
u.requires_grad = True
opt = AdamUniform([u], step_size)

import torch.nn as nn
loss_fn = nn.L1Loss()

def scale_independent_loss(image, ref):
    """Brightness-independent L2 loss function."""
    scaled_image = image / dr.mean(dr.detach(image))
    scaled_ref = ref / dr.mean(ref)
    return dr.mean(dr.sqr(scaled_image - scaled_ref))

from scripts.geometry import compute_vertex_normals, compute_face_normals
for it in trange(steps):
    print("starting the loop")

    # Get cartesian coordinates for parameterization
    v = from_differential(M, u, 'Cholesky')

    # Recompute vertex normals
    face_normals = compute_face_normals(v, f)
    n = compute_vertex_normals(v, f, face_normals)

    # Render images
    opt_imgs = mi.render(compute_mitsuba_scene(v, n, f))
    #to_optimize = mi.traverse(opt_imgs)
    #print(to_optimize)

    loss = scale_independent_loss(opt_imgs, ref_imgs)
    loss.backward()

    # Compute L1 image loss
    #loss = (opt_imgs - ref_imgs).abs().mean()
    #loss = (to_optimize - ref)
    #print(type(loss))
    #print(loss)

    # Backpropagate
    opt.zero_grad()
    loss.backward()
    
    # Update parameters
    opt.step()

#img = mi.render(compute_mitsuba_scene(v_ref, n_ref, f_ref))
img = mi.render(compute_mitsuba_scene(v, n, f))

plt.axis("off")
plt.imshow(mi.util.convert_to_bitmap(img));
plt.show()

#mesh.write_ply("mymesh.ply")

