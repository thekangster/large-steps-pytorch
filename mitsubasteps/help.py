import mitsuba as mi
import drjit as dr

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

def compute_mitsuba_scene(v, n, f):
    vertex_pos = tensor_to_point3f(v)
    normals = tensor_to_point3f(n)
    face_norms = tensor_to_point3f(f)

    mesh = mi.Mesh(
        "mymesh", 
        len(v), 
        len(f), 
        has_vertex_normals=True, 
        has_vertex_texcoords=False,
    )

    mesh_params = mi.traverse(mesh)
    #print(mesh_params)
    mesh_params['vertex_positions'] = dr.ravel(vertex_pos)
    mesh_params['faces'] = dr.ravel(face_norms)
    mesh_params['vertex_normals'] = dr.ravel(face_norms)
    mesh_params.update()
    #print(mesh_params.update())

    return mi.load_dict({
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
                origin=[0, 2, 7], target=[0, 0, 0], up=[0, 0, -1]
                ),
            },
        "mymesh": mesh,
    })

