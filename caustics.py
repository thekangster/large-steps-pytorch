import os
from os.path import realpath, join

import drjit as dr
import mitsuba as mi

#mi.set_variant('llvm_ad_rgb')
mi.set_variant('cuda_ad_rgb')

SCENE_DIR = realpath('cscenes/')

CONFIGS = {
    'wave': {
        'emitter': 'gray',
        'reference': join(SCENE_DIR, 'references/wave-1024.jpg'),
    },
    'sunday': {
        'emitter': 'bayer',
        'reference': join(SCENE_DIR, 'references/sunday-512.jpg'),
    },
}

# Pick one of the available configs
config_name = 'sunday'
# config_name = 'wave'

config = CONFIGS[config_name]
print('[i] Reference image selected:', config['reference'])
mi.Bitmap(config['reference'])

if 'PYTEST_CURRENT_TEST' not in os.environ:
    config.update({
        'render_resolution': (128, 128),
        'heightmap_resolution': (512, 512),
        'n_upsampling_steps': 4,
        'spp': 32,
        'max_iterations': 1000,
        'learning_rate': 3e-5,
    })
else:
    # IGNORE THIS: When running under pytest, adjust parameters to reduce computation time
    config.update({
        'render_resolution': (64, 64),
        'heightmap_resolution': (128, 128),
        'n_upsampling_steps': 0,
        'spp': 8,
        'max_iterations': 25,
        'learning_rate': 3e-5,
    })
    
    
output_dir = realpath(join('.', 'outputs', config_name))
os.makedirs(output_dir, exist_ok=True)
print('[i] Results will be saved to:', output_dir)

def create_flat_lens_mesh(resolution):
    print("hello world")
    # Generate UV coordinates
    U, V = dr.meshgrid(
        dr.linspace(mi.Float, 0, 1, resolution[0]),
        dr.linspace(mi.Float, 0, 1, resolution[1]),
        indexing='ij'
    )
    texcoords = mi.Vector2f(U, V)
    
    # Generate vertex coordinates
    X = 2.0 * (U - 0.5)
    Y = 2.0 * (V - 0.5)
    vertices = mi.Vector3f(X, Y, 0.0)

    # Create two triangles per grid cell
    faces_x, faces_y, faces_z = [], [], []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            v00 = i * resolution[1] + j
            v01 = v00 + 1
            v10 = (i + 1) * resolution[1] + j
            v11 = v10 + 1
            faces_x.extend([v00, v01])
            faces_y.extend([v10, v10])
            faces_z.extend([v01, v11])

    # Assemble face buffer 
    faces = mi.Vector3u(faces_x, faces_y, faces_z)

    # Instantiate the mesh object
    mesh = mi.Mesh("lens-mesh", resolution[0] * resolution[1], len(faces_x), has_vertex_texcoords=True)
    
    # Set its buffers
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['vertex_texcoords'] = dr.ravel(texcoords)
    mesh_params['faces'] = dr.ravel(faces)
    mesh_params.update()

    return mesh

lens_res = config.get('lens_res', config['heightmap_resolution'])
lens_fname = join(output_dir, 'lens_{}_{}.ply'.format(*lens_res))

#if not os.path.isfile(lens_fname):
m = create_flat_lens_mesh(lens_res)
m.write_ply(lens_fname)
print('[+] Wrote lens mesh ({}x{} tesselation) file to: {}'.format(*lens_res, lens_fname))

