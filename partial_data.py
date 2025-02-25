import numpy as np 
import json
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plyfile import PlyData, PlyElement


output_path = "/app/DeepSDF/data/cropped/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

def read_sdf_full_to_partial_data(filename):
    npz = np.load(filename)
    pos_tensor = npz["pos"]
    neg_tensor = npz["neg"]
    print("Positive values:", pos_tensor)
    print("Negative values:", neg_tensor)

    return [pos_tensor, neg_tensor]

read_sdf_full_to_partial_data("/app/DeepSDF/data/SdfSamples/ShapeNetV2/02828884/4647754b7c5b31cccb2a965e75be701c.npz")

def get_full_paths(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        print(data)
    
    full_paths = []
    for category, ids in data["ShapeNetV2"].items():
        for id in ids:
            full_path = os.path.join("ShapeNetV2", category, id)
            merged_file_path= os.path.join("/app/DeepSDF/data/SdfSamples", full_path )
            merged_file_path = merged_file_path + ".npz"
            full_paths.append(merged_file_path)
    
    return full_paths

json_file_path = "/app/DeepSDF/examples/splits/sv2_combined_small_test_each.json"
full_paths = get_full_paths(json_file_path)

def generate_bounding_box(bound=0.80):
    x_min = np.random.uniform(-bound, 0)
    x_max = np.random.uniform(0, bound)
    y_min = np.random.uniform(-bound, 0)
    y_max = np.random.uniform(0, bound)
    z_min = np.random.uniform(-bound, 0)
    z_max = np.random.uniform(0, bound)
    
    return {
        "x": (x_min, x_max),
        "y": (y_min, y_max),
        "z": (z_min, z_max)
    }


import matplotlib.pyplot as plt

def visualize_bounding_box(bbox):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = bbox["x"]
    y = bbox["y"]
    z = bbox["z"]
    
    vertices = [
        [x[0], y[0], z[0]],
        [x[1], y[0], z[0]],
        [x[1], y[1], z[0]],
        [x[0], y[1], z[0]],
        [x[0], y[0], z[1]],
        [x[1], y[0], z[1]],
        [x[1], y[1], z[1]],
        [x[0], y[1], z[1]]
    ]
    
    edges = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    
    poly3d = Poly3DCollection(edges, alpha=.25, linewidths=1, edgecolors='r')
    ax.add_collection3d(poly3d)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()


# bounding_box = generate_bounding_box()
# print("Generated bounding box:", bounding_box)
# visualize_bounding_box(bounding_box)

for path in full_paths:
    print(path)
    pos_val, neg_val =  read_sdf_full_to_partial_data(path)
    # print("Positive values:", pos_val)
    bounding_box = generate_bounding_box()

    # Generating the second bounding box would be a good idea to filter irregularly

    # filter could get larger 

    x_min, x_max = bounding_box["x"]
    y_min, y_max = bounding_box["y"]
    z_min, z_max = bounding_box["z"]

    within_bbox = (neg_val[:, 0] >= x_min) & (neg_val[:, 0] <= x_max) & \
                  (neg_val[:, 1] >= y_min) & (neg_val[:, 1] <= y_max) & \
                  (neg_val[:, 2] >= z_min) & (neg_val[:, 2] <= z_max)

    initial_dim = neg_val.shape[0]
    print("Shape of neg_val before filtering:", neg_val.shape)
    # print("Number of points within bounding box:", (within_bbox.shape[0]))
    neg_val = neg_val[~within_bbox]
    final_dim = neg_val.shape[0]
    print("Shape of neg_val after filtering:", neg_val.shape)
    diff = initial_dim - final_dim
    print("Number of points removed:", diff)

    output_file_path = os.path.join(output_path, os.path.basename(path))
    np.savez(output_file_path, pos=pos_val, neg=neg_val)
    print(f"Saved filtered data to {output_file_path}")

    def save_as_ply(points, filename):
        vertices = [(point[0], point[1], point[2]) for point in points]
        vertex_element = PlyElement.describe(np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
        PlyData([vertex_element]).write(filename)

    ply_output_path = os.path.join(output_path, os.path.basename(path).replace('.npz', '.ply'))
    save_as_ply(neg_val[:, :3], ply_output_path)
    print(f"Saved PLY file to {ply_output_path}")


    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.scatter(neg_val[:, 0], neg_val[:, 1], neg_val[:, 2], c='b', marker='o', s=1)
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    # plt.show()

    

    