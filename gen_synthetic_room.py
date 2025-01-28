import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_model_sdf(sdf_points, sdf_values, scale, y_angle, translation):
    """
    Transform SDF points and values from model space to room space.

    Args:
        sdf_points: (N, 3) array of point positions in model space
        sdf_values: (N,) array of SDF values
        scale: float, scaling factor
        y_angle: float, rotation angle around y-axis in degrees
        translation: (3,) array, translation vector

    Returns:
        transformed_points: (N, 3) array of transformed point positions
        transformed_values: (N,) array of transformed SDF values
    """
    # Scale points and SDF values
    scaled_points = sdf_points * scale
    scaled_values = sdf_values * scale

    # Create rotation matrix for y-axis rotation
    rot = R.from_euler('y', y_angle, degrees=True)
    rotated_points = rot.apply(scaled_points)

    # Apply translation
    transformed_points = rotated_points + translation

    return transformed_points, scaled_values


def combine_model_sdfs(model_data_list, query_points):
    """
    Combine SDF values from multiple transformed models at query points.

    Args:
        model_data_list: list of dictionaries containing:
            - sdf_points: (N, 3) array of SDF point positions
            - sdf_values: (N,) array of SDF values
            - scale: float, scaling factor
            - y_angle: float, rotation angle
            - translation: (3,) array, translation vector
        query_points: (M, 3) array of points to query SDF values for

    Returns:
        combined_sdf: (M,) array of combined SDF values
    """
    from scipy.interpolate import LinearNDInterpolator

    combined_sdf = np.full(len(query_points), np.inf)

    for model_data in model_data_list:
        # Transform model SDF points and values
        transformed_points, transformed_values = transform_model_sdf(
            model_data['sdf_points'],
            model_data['sdf_values'],
            model_data['scale'],
            model_data['y_angle'],
            model_data['translation']
        )

        # Create interpolator for this model's SDF
        interpolator = LinearNDInterpolator(transformed_points, transformed_values, fill_value=np.inf)

        # Interpolate SDF values at query points
        model_sdf = interpolator(query_points)

        # Combine with previous models using min operation
        combined_sdf = np.minimum(combined_sdf, model_sdf)

    return combined_sdf


def create_room_sdf_grid(item_dict, model_sdf_dict, resolution=32):
    """
    Create a regular grid of SDF values for an entire room.

    Args:
        item_dict: dictionary from the room dataset containing object placements
        model_sdf_dict: dictionary mapping model IDs to their SDF data
        resolution: number of points along each axis

    Returns:
        grid_points: (resolution^3, 3) array of query point positions
        sdf_values: (resolution^3,) array of SDF values
    """
    # Create regular grid
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    z = np.linspace(-0.5, 0.5, resolution)

    grid_points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    grid_points = grid_points.reshape(-1, 3)

    # Prepare model data list
    model_data_list = []
    for obj_idx, obj_path in enumerate(item_dict['objects']):
        model_id = obj_path.split('/')[-1]

        if model_id not in model_sdf_dict:
            continue

        bbox = item_dict['bboxes'][obj_idx]
        translation = bbox[0]  # Using lower corner of bbox

        model_data_list.append({
            'sdf_points': model_sdf_dict[model_id]['points'],
            'sdf_values': model_sdf_dict[model_id]['sdf'],
            'scale': item_dict['scales'][obj_idx],
            'y_angle': item_dict['y_angles'][obj_idx],
            'translation': translation
        })

    # Combine SDFs from all models
    sdf_values = combine_model_sdfs(model_data_list, grid_points)

    return grid_points, sdf_values


# Example usage:
"""
# Load room arrangement data
item_dict = np.load('path/to/item_dict.npz')

# Load pre-computed model SDFs
model_sdf_dict = {
    'model_id': {
        'points': sdf_points,
        'sdf': sdf_values
    },
    ...
}

# Create SDF grid for the room
grid_points, sdf_values = create_room_sdf_grid(item_dict, model_sdf_dict)
"""

if __name__ == '__main__':
