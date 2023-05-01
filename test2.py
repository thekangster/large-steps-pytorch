# imports
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import os

from scripts.load_xml import load_scene
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform

filepath = os.path.join(os.getcwd(), "scenes", "suzanne", "suzanne.xml")

mi.set_variant('llvm_ad_rgb')

scene_params = mi.load_file(filepath)#, res=128)#, integrator='prb')
image_ref = mi.render(scene_params, spp=512)
'''
plt.axis("off")
plt.imshow(mi.util.convert_to_bitmap(image_ref));
plt.show()
'''
# get largesteps scene params
scene_large = load_scene(filepath)
# Load reference shape
v_ref = scene_large["mesh-target"]["vertices"]
n_ref = scene_large["mesh-target"]["normals"]
f_ref = scene_large["mesh-target"]["faces"]
# Load source shape
v = scene_large["mesh-source"]["vertices"]
f = scene_large["mesh-source"]["faces"]

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

v_ = tensor_to_point3f(v_ref)
n_ = tensor_to_point3f(n_ref)
f_ = tensor_to_point3f(f_ref)

refmesh = mi.Mesh(
    "refmesh", 
    len(v_ref), 
    #len(v_ref)-1,
    len(f_ref),
    has_vertex_normals=True, 
    has_vertex_texcoords=False,
)

mesh_params = mi.traverse(refmesh)
mesh_params['vertex_positions'] = dr.ravel(v_)
mesh_params['vertex_normals'] = dr.ravel(n_)
mesh_params['faces'] = dr.ravel(f_)

scene = mi.load_dict({
    "type": "scene",
    "integrator": {"type": "path"},
    "light": {
        'type': 'point',
        'position': [0.0, -1.0, 7.0],
        'intensity': {
            'type': 'spectrum',
            'value': 15.0,
        }
    },
    "sensor": {
        "type": "perspective",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0, 2, 7], target=[0, 0, 0], up=[0, 0, -1]
        ),
    },
    "refmesh": refmesh,
})
ref_imgs = mi.render(scene)
plt.axis("off")
plt.imshow(mi.util.convert_to_bitmap(ref_imgs));
plt.show()

from tqdm import trange
from largesteps.optimize import AdamUniform
from scripts.geometry import compute_vertex_normals, compute_face_normals

steps = 1000 # Number of optimization steps
step_size = 3e-2 # Step size
lambda_ = 19 # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_*L)

# Compute the system matrix
M = compute_matrix(v, f, lambda_)
# Parameterize
u = to_differential(M, v)
print("printing u")
print(u.size())

u.requires_grad = True
opt = AdamUniform([u], step_size)

face_norms = tensor_to_point3f(f)

for it in trange(steps):
# Get cartesian coordinates for parameterization
    v = from_differential(M, u, 'Cholesky')

# Recompute vertex normals
    face_normals = compute_face_normals(v, f)
    n = compute_vertex_normals(v, f, face_normals)

    vertex_pos = tensor_to_point3f(v)
    vertex_norms = tensor_to_point3f(n)

# create mesh
    mesh = mi.Mesh(
        "mymesh", 
        len(v), 
        #len(v_ref)-1,
        len(f),
        has_vertex_normals=True, 
        has_vertex_texcoords=False,
    )

    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertex_pos)
    mesh_params['vertex_normals'] = dr.ravel(vertex_norms)
    mesh_params['faces'] = dr.ravel(face_norms)

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path"},
        "light": {
            'type': 'point',
            'position': [0.0, -1.0, 7.0],
            'intensity': {
                'type': 'spectrum',
                'value': 15.0,
            }
        },
        "sensor": {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 2, 7], target=[0, 0, 0], up=[0, 0, -1]
            ),
        },
        "mymesh": mesh,
    })

    opt_imgs = mi.render(scene)

# Compute L1 image loss
    loss = (opt_imgs - ref_imgs).abs().mean()

# Backpropagate
    opt.zero_grad()
    loss.backward()
    
    # Update parameters
    opt.step() 

'''
plt.axis("off")
plt.imshow(mi.util.convert_to_bitmap(opt_imgs));
plt.show()
'''
#mesh.write_ply("mymesh.ply")

