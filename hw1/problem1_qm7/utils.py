import numpy as np

def rotate_3d_coordinates(coordinates, x_degrees, y_degrees, z_degrees):
    """
    Rotates a set of 3D coordinates around the x, y, and z axes by specified angles.
    
    Args:
    coordinates (np.array): An N x 3 array of 3D coordinates.
    x_degrees (float): The rotation angle around the x-axis in degrees.
    y_degrees (float): The rotation angle around the y-axis in degrees.
    z_degrees (float): The rotation angle around the z-axis in degrees.
    
    Returns:
    np.array: The rotated coordinates: An N x 3 array of 3D coordinates.
    """
    x = np.deg2rad(x_degrees)
    y = np.deg2rad(y_degrees)
    z = np.deg2rad(z_degrees)

    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])

    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],  [-np.sin(y), 0, np.cos(y)]])  

    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return coordinates @ R.T #R rotates column vectors but coordinates has them on a row, so we transpose R (now rotates row vectors)
