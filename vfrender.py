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

# tensor -> point3f ??
vref = v_ref.tolist()
x = dr.zeros(mi.Float, len(v_ref))
y = dr.zeros(mi.Float, len(v_ref))
z = dr.zeros(mi.Float, len(v_ref))
idx = 0
for i in vref:
    x[idx] = i[0]
    y[idx] = i[1]
    z[idx] = i[2]
    idx += 1
vertex_pos = mi.Point3f(x, y, z)

# generate face indices ??
N = len(v_ref)
index = dr.arange(mi.UInt32, N - 1)
face_indices = mi.Vector3u(N - 1, (index + 1) % (N - 2), index % (N - 2))
#print("len(face_indices) = ", len(face_indices))

# generate face normals ??
nref = n_ref.tolist()
x2 = dr.zeros(mi.Float, len(v_ref))
y2 = dr.zeros(mi.Float, len(v_ref))
z2 = dr.zeros(mi.Float, len(v_ref))
idx = 0
for i in vref:
    x2[idx] = i[0]
    y2[idx] = i[1]
    z2[idx] = i[2]
    idx += 1
face_norms = mi.Point3f(x2, y2, z2)

# create mesh
mesh = mi.Mesh(
    "mymesh", 
    len(v_ref), 
    #len(f_ref), 
    len(v_ref)-1,
    has_vertex_normals=False, 
    has_vertex_texcoords=False,
)

mesh_params = mi.traverse(mesh)
print(mesh_params)
mesh_params['vertex_positions'] = dr.ravel(vertex_pos)
mesh_params['faces'] = dr.ravel(face_indices)
#mesh_params['vertex_normals'] = dr.ravel(face_norms)
print(mesh_params.update())

scene = mi.load_dict({
    "type": "scene",
    "integrator": {"type": "path"},
    "light": {"type": "constant"},
    "sensor": {
        "type": "perspective",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0, -1, 10], target=[0, 0, 0], up=[0, 0, 1]
        ),
    },
    "mymesh": mesh,
})

img = mi.render(scene)

plt.axis("off")
plt.imshow(mi.util.convert_to_bitmap(img));
plt.show()

mesh.write_ply("mymesh.ply")

