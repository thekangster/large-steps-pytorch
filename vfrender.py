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

# tensor -> point3f
vertex_pos = tensor_to_point3f(v_ref)
#vertex_pos = tensor_to_point3f(v)

# generate face indices ??
'''
N = len(v_ref)
index = dr.arange(mi.UInt32, N - 1)
face_indices = mi.Vector3u(N - 1, (index + 1) % (N - 2), index % (N - 2))
'''

# generate face normals ??
face_norms = tensor_to_point3f(f_ref)
#face_norms = tensor_to_point3f(f)

vertex_norms = tensor_to_point3f(n_ref)

# create mesh
mesh = mi.Mesh(
    "mymesh", 
    len(v_ref), 
    #len(v_ref)-1,
    len(f_ref),
    has_vertex_normals=True, 
    has_vertex_texcoords=False,
)

mesh_params = mi.traverse(mesh)
#print(mesh_params)
mesh_params['vertex_positions'] = dr.ravel(vertex_pos)
mesh_params['faces'] = dr.ravel(face_norms)
mesh_params['vertex_normals'] = dr.ravel(vertex_norms)
print(mesh_params.update())

'''
'type': 'point',
'position': [0.0, -1.0, 7.0],
'intensity': {
    'type': 'spectrum',
    'value': 15.0,
}
'''
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

img = mi.render(scene)

plt.axis("off")
plt.imshow(mi.util.convert_to_bitmap(img));
plt.show()

#mesh.write_ply("mymesh.ply")

