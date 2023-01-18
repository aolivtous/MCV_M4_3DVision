import math
import random
import sys
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import map_coordinates


def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)


def get_transformed_pixels_coords(I, H, shift=None):
    ys, xs = np.indices(I.shape[:2]).astype("float64")
    if shift is not None:
        ys += shift[1]
        xs += shift[0]
    ones = np.ones(I.shape[:2])
    coords = np.stack((xs, ys, ones), axis=2)
    coords_H = (H @ coords.reshape(-1, 3).T).T
    coords_H /= coords_H[:, 2, np.newaxis]
    cart_H = coords_H[:, :2]

    return cart_H.reshape((*I.shape[:2], 2))


def apply_H_fixed_image_size(I, H, corners):
    h, w = I.shape[:2]  # when we convert to np.array it swaps

    # corners
    c1 = np.array([1, 1, 1])
    c2 = np.array([w, 1, 1])
    c3 = np.array([1, h, 1])
    c4 = np.array([w, h, 1])

    # transformed corners
    Hc1 = H @ c1
    Hc2 = H @ c2
    Hc3 = H @ c3
    Hc4 = H @ c4
    Hc1 = Hc1 / Hc1[2]
    Hc2 = Hc2 / Hc2[2]
    Hc3 = Hc3 / Hc3[2]
    Hc4 = Hc4 / Hc4[2]

    xmin = corners[0]
    xmax = corners[1]
    ymin = corners[2]
    ymax = corners[3]

    size_x = ceil(xmax - xmin + 1)
    size_y = ceil(ymax - ymin + 1)

    # transform image
    H_inv = np.linalg.inv(H)

    out = np.zeros((size_y, size_x, 3))
    shift = (xmin, ymin)
    interpolation_coords = get_transformed_pixels_coords(out, H_inv, shift=shift)
    interpolation_coords[:, :, [0, 1]] = interpolation_coords[:, :, [1, 0]]
    interpolation_coords = np.swapaxes(np.swapaxes(interpolation_coords, 0, 2), 1, 2)

    out[:, :, 0] = map_coordinates(I[:, :, 0], interpolation_coords)
    out[:, :, 1] = map_coordinates(I[:, :, 1], interpolation_coords)
    out[:, :, 2] = map_coordinates(I[:, :, 2], interpolation_coords)

    return out.astype("uint8")


def Normalise_last_coord(x):
    xn = x / x[2, :]

    return xn


def Normalize_points(points):  # function added to normalize the points
    """
    Compute a similarity transformation T ( translation & scaling ) such that the centroid of the
    new points is (0,0) and their mean distance of points to the origin is sqrt(2)
    """

    centroids = np.mean(points, axis=1)
    centroid_x = centroids[0]
    centroid_y = centroids[1]

    # compute the mean distance of points to the origin
    mean_distance = np.mean(np.sqrt((points[0] - centroid_x) ** 2 + (points[1] - centroid_y) ** 2))

    # compute the scaling factor
    s = np.sqrt(2) / mean_distance

    # compute the translation
    tx = -s * centroid_x
    ty = -s * centroid_y

    # compute the transformation matrix
    T = np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]])

    # apply the transformation
    points_norm = T @ points

    return points_norm, T


def DLT_homography(points1, points2):
    # ToDo: complete this code .......

    # normalization of x
    points1, T1 = Normalize_points(points1)

    # normalization of x'
    points2, T2 = Normalize_points(points2)

    # Apply DLT to the correspondences

    A_list = []

    for point1, point2 in zip(points1.T, points2.T):
        x, y, w = point1
        x = x / w
        y = y / w

        x_p, y_p, w_p = point2
        x_p = x_p / w_p
        y_p = y_p / w_p

        Ai1 = np.array([-x, -y, -1, 0, 0, 0, x * x_p, y * x_p, x_p])
        Ai2 = np.array([0, 0, 0, -x, -y, -1, x * y_p, y * y_p, y_p])

        A_list.append(Ai1)
        A_list.append(Ai2)

    A = np.array(A_list)

    # Solve Ah = 0
    U, D, Vt = np.linalg.svd(A)

    # take the las col of V --> last row of V_transpose and reshape it to 3x3 matrix
    H_tilde = np.reshape(Vt[-1], (3, 3))

    # Denormalization
    T2_inv = np.linalg.inv(T2)
    H = T2_inv @ H_tilde @ T1

    # normalize
    H = H / H[-1, -1]

    return H


def Inliers(H, points1, points2, th):
    "compute the inliers for the homography H, given the correspondences points1 and points2 and the distance threshold th"

    # Check that H is invertible
    if abs(math.log(np.linalg.cond(H))) > 15:
        idx = np.empty(1)
        return idx

    # ToDo: complete this code .......

    # compute the transformed points
    points1_transformed = H @ points1
    points2_transformed = np.linalg.inv(H) @ points2

    # normalize the points
    points1_norm = Normalise_last_coord(points1)
    points2_norm = Normalise_last_coord(points2)
    points1_transformed_norm = Normalise_last_coord(points1_transformed)
    points2_transformed_norm = Normalise_last_coord(points2_transformed)

    # compute the distance between the transformed points and the original points --> use the geometric distance d or the distance d orhtogonal
    d_orth_sq = np.sum(np.square(points2_norm - points1_transformed_norm), axis=0) + np.sum(
        np.square(points2_transformed_norm - points1_norm), axis=0
    )

    # compute the inliers
    idx = np.where(d_orth_sq < th**2)[0]

    return idx


def Ransac_DLT_homography(points1, points2, th, max_it):
    Ncoords, Npts = points1.shape

    it = 0
    best_inliers = np.empty(1)

    while it < max_it:
        indices = random.sample(range(1, Npts), 4)
        H = DLT_homography(points1[:, indices], points2[:, indices])
        inliers = Inliers(H, points1, points2, th)

        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers

        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0] / Npts
        pNoOutliers = 1 - fracinliers**4
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)  # avoid division by -Inf
        pNoOutliers = min(1 - eps, pNoOutliers)  # avoid division by 0
        p = 0.99
        max_it = math.log(1 - p) / math.log(pNoOutliers)

        it += 1

    # compute H from all the inliers
    H = DLT_homography(points1[:, best_inliers], points2[:, best_inliers])
    inliers = best_inliers

    return H, inliers


def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o


def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:, :3]) @ np.array([x[0], x[1], 1])
    return v


def plot_camera(P, w, h, fig, legend):
    o = optical_center(P)
    scale = 200
    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale

    x = np.array(
        [
            p1[0],
            p2[0],
            o[0],
            p3[0],
            p2[0],
            p3[0],
            p4[0],
            p1[0],
            o[0],
            p4[0],
            o[0],
            (p1[0] + p2[0]) / 2,
        ]
    )
    y = np.array(
        [
            p1[1],
            p2[1],
            o[1],
            p3[1],
            p2[1],
            p3[1],
            p4[1],
            p1[1],
            o[1],
            p4[1],
            o[1],
            (p1[1] + p2[1]) / 2,
        ]
    )
    z = np.array(
        [
            p1[2],
            p2[2],
            o[2],
            p3[2],
            p2[2],
            p3[2],
            p4[2],
            p1[2],
            o[2],
            p4[2],
            o[2],
            (p1[2] + p2[2]) / 2,
        ]
    )

    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode="lines", name=legend))

    return


def plot_image_origin(w, h, fig, legend):
    p1 = np.array([0, 0, 0])
    p2 = np.array([w, 0, 0])
    p3 = np.array([w, h, 0])
    p4 = np.array([0, h, 0])

    x = np.array([p1[0], p2[0], p3[0], p4[0], p1[0]])
    y = np.array([p1[1], p2[1], p3[1], p4[1], p1[1]])
    z = np.array([p1[2], p2[2], p3[2], p4[2], p1[2]])

    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode="lines", name=legend))

    return
