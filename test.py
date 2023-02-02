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

# use traverse to pick the parameter to optimize
params = mi.traverse(scene_params)
key = 'red.reflectance.value'
#key = 'Camera.fov.value'

param_ref = mi.Color3f(params[key])
#param_ref = mi.ChainTransform3d(params[key])

# set to another blue then update the scene
params[key] = mi.Color3f(0.01, 0.2, 0.9)
#params[key] = mi.ChainTransform3d(60)
params.update()

image_init = mi.render(scene_params, spp=128)

# preview image
#plt.axis('off')
#plt.imshow(mi.util.convert_to_bitmap(image_init))
#plt.show()

# get largesteps scene params
scene_large = load_scene(filepath)

# Load reference shape
v_ref = scene_large["mesh-target"]["vertices"]
n_ref = scene_large["mesh-target"]["normals"]
f_ref = scene_large["mesh-target"]["faces"]

# Load source shape
v = scene_large["mesh-source"]["vertices"]
f = scene_large["mesh-source"]["faces"]


#### PARAMETERIZING

#steps = 1000 # Number of optimization steps
#step_size = 3e-2 # Step size
lambda_ = 19 # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_*L)

# Compute the system matrix
M = compute_matrix(v, f, lambda_)

# Parameterize
u = to_differential(M, v)

from scripts.geometry import compute_vertex_normals, compute_face_normals

opt = mi.ad.Adam(lr=0.1)
opt[key] = params[key]
params.update(opt)

# mean square error, or L_2 error
# average of the squares of errors between current image and reference
def mse(image):
    return dr.mean(dr.sqr(image - image_ref))

iteration_count = 50
errors = []
for it in range(iteration_count):

    """
    # Get cartesian coordinates for parameterization
    v = from_differential(M, u, 'Cholesky')

    # Recompute vertex normals
    face_normals = compute_face_normals(v, f)
    n = compute_vertex_normals(v, f, face_normals)
    """


    # Perform a (noisy) differentiable rendering of the scene
    image = mi.render(scene_params, params, spp=4)

    # Evaluate the objective function from the current rendered image
    loss = mse(image)

    # Backpropogate through the rendering process
    dr.backward(loss)

    # Optimizer: take a gradient descent step
    opt.step()

    # Post-process the optimized parameters to ensure legal color values.
    opt[key] = dr.clamp(opt[key], 0.0, 1.0)

    # Update the scene state to the new optimized values
    params.update(opt)

    # Track the difference betrween the current color and the true value
    err_ref = dr.sum(dr.sqr(param_ref - params[key]))
    print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
    errors.append(err_ref)
print('\nOptimization complete.')

image_final = mi.render(scene_params, spp=128)

plt.axis('off')
plt.imshow(mi.util.convert_to_bitmap(image_final))
plt.show()

