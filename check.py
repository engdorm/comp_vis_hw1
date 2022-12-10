from typing import Tuple

import numpy as np
import PIL
import matplotlib.pyplot as plt
import scipy.io
from numpy.linalg import svd


# # First Part Let's read and show Images
# src_img = plt.imread("src.jpg")
# dst_img = plt.imread("dst.jpg")
# # plt.figure(1)
# # plt.imshow(src_img), plt.title("Src Image")
# # plt.figure(2)
# # plt.imshow(dst_img), plt.title("Dst Image"), plt.show()
#
# # Second part let's read point matching
# matches = scipy.io.loadmat('matches_perfect')
# match_p_src = matches['match_p_src']
# match_p_dst = matches['match_p_dst']
#
# # # Plot images with Corresponding point
# color_list = ['b', 'g', 'r', 'y', 'w', 'k', 'm', 'b', 'g', 'r', 'y', 'w', 'k', 'm', 'b', 'g', 'r', 'y', 'w', 'k']
# fig, (ax1, ax2) = plt.subplots(1, 2)
# plt.figure(figsize=(8, 8), dpi=80)
# fig.suptitle('Perfect Matches', fontsize=16)
# ax1.set_title('Src_img')
# ax1.imshow(src_img)
# ax1.scatter(match_p_src[0], match_p_src[1], c=color_list)
#
# plt.figure(figsize=(8, 8), dpi=80)
# ax2.set_title('Dst_img')
# ax2.imshow(dst_img)
# ax2.scatter(match_p_dst[0], match_p_dst[1], c=color_list)
# plt.show()
# ##################################################################################
# # Read point that not matching:
# matches = scipy.io.loadmat('matches')
# match_Np_src = matches['match_p_src']
# match_Np_dst = matches['match_p_dst']
#
# color_list = 3 * ['b', 'g', 'r', 'y', 'w', 'k', 'm'] + ['y', 'w', 'k', 'm']
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('Matches', fontsize=16)
# plt.figure(figsize=(8, 8), dpi=80)
# ax1.set_title('Src_img')
# ax1.imshow(src_img)
# ax1.scatter(match_Np_src[0], match_Np_src[1], c=color_list)
#
# plt.figure(figsize=(8, 8), dpi=80)
# ax2.set_title('Dst_img')
# ax2.imshow(dst_img)
# ax2.scatter(match_Np_dst[0], match_Np_dst[1], c=color_list)
# plt.show()
#############################################################################################
def compute_homography_naive(match_p_src: np.ndarray,
                             match_p_dst: np.ndarray) -> np.ndarray:
    """Compute a Homography in the Naive approach, using SVD decomposition.

    Args:
        match_p_src: 2xN points from the source image.
        match_p_dst: 2xN points from the destination image.

    Returns:
        Homography from source to destination, 3x3 numpy array.
    """
    # return homography
    """INSERT YOUR CODE HERE"""
    A = []
    for i in range(match_p_src.shape[1]):
        u_src, v_src = np.float64(match_p_src[0, i]), np.float64(match_p_src[1, i])
        u_tag, v_tag = np.float64(match_p_dst[0, i]), np.float64(match_p_dst[1, i])
        A.append([u_src, v_src, 1, 0, 0, 0, -u_tag*u_src, -u_tag*v_src, -u_tag])
        A.append([0, 0, 0, u_src, v_src, 1, -v_tag * u_src, -v_tag * v_src, -v_tag])
    A = np.asarray(A, dtype=np.float64)
    u, s, vh = svd(A)
    return vh[-1].reshape(3, 3)

def compute_forward_homography_slow(
        homography: np.ndarray,
        src_image: np.ndarray,
        dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
    """Compute a Forward-Homography in the Naive approach, using loops.

    Iterate over the rows and columns of the source image, and compute
    the corresponding point in the destination image using the
    projective homography. Place each pixel value from the source image
    to its corresponding location in the destination image.
    Don't forget to round the pixel locations computed using the
    homography.

    Args:
        homography: 3x3 Projective Homography matrix.
        src_image: HxWx3 source image.
        dst_image_shape: tuple of length 3 indicating the destination
        image height, width and color dimensions.

    Returns:
        The forward homography of the source image to its destination.
    """
    # return new_image
    new_img = np.zeros(dst_image_shape, dtype=np.uint8)
    for y in range(src_image.shape[0]):
        for x in range(src_image.shape[1]):
            new_pos = homography @ np.array([x, y, 1]).T # new_pos = H *X'
            dst_x, dst_y = int(new_pos[0]/new_pos[2]), int(new_pos[1]/new_pos[2])
            if src_image.shape[1] >= dst_x >= 0 and src_image.shape[0] >= dst_y >= 0:
                new_img[dst_y, dst_x] = src_image[y, x]
    return new_img


def compute_forward_homography_fast(
        homography: np.ndarray,
        src_image: np.ndarray,
        dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
    """Compute a Forward-Homography in a fast approach, WITHOUT loops.

    (1) Create a meshgrid of columns and rows.
    (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
    in homogeneous coordinates.
    (3) Transform the source homogeneous coordinates to the target
    homogeneous coordinates with a simple matrix multiplication and
    apply the normalization you've seen in class.
    (4) Convert the coordinates into integer values and clip them
    according to the destination image size.
    (5) Plant the pixels from the source image to the target image according
    to the coordinates you found.

    Args:
        homography: 3x3 Projective Homography matrix.
        src_image: HxWx3 source image.
        dst_image_shape: tuple of length 3 indicating the destination.
        image height, width and color dimensions.

    Returns:
        The forward homography of the source image to its destination.
    """
    # return new_image
    """INSERT YOUR CODE HERE"""
    new_image = np.zeros(shape=dst_image_shape, dtype=np.uint8)
    hw_length = src_image.shape[0] * src_image.shape[1]
    x = np.linspace(0, src_image.shape[0] - 1, src_image.shape[0]).astype(int)
    y = np.linspace(0, src_image.shape[1] - 1, src_image.shape[1]).astype(int)
    yy, xx = np.meshgrid(y, x)
    yy = yy.reshape((1, hw_length))
    xx = xx.reshape((1, hw_length))
    ones = np.ones(shape=(1, hw_length))
    X = np.concatenate((yy, xx, ones), axis=0)
    Y = homography @ X
    Y /= Y[-1]
    Y_norm = Y[0:2]
    Y_norm = Y_norm.round().astype(int)

    mask = (Y_norm[1] >= 0) & (Y_norm[1] < dst_image_shape[0]) & (Y_norm[0] >= 0) & (Y_norm[0] < dst_image_shape[1])
    new_image[Y_norm[1, mask], Y_norm[0, mask]] = src_image[xx[0, mask], yy[0, mask]]
    return new_image
def test_homography(homography: np.ndarray,
                    match_p_src: np.ndarray,
                    match_p_dst: np.ndarray,
                    max_err: float) -> tuple[float, float]:
    """Calculate the quality of the projective transformation model.

    Args:
        homography: 3x3 Projective Homography matrix.
        match_p_src: 2xN points from the source image.
        match_p_dst: 2xN points from the destination image.
        max_err: A scalar that represents the maximum distance (in
        pixels) between the mapped src point to its corresponding dst
        point, in order to be considered as valid inlier.

    Returns:
        A tuple containing the following metrics to quantify the
        homography performance:
        fit_percent: The probability (between 0 and 1) validly mapped src
        points (inliers).
        dist_mse: Mean square error of the distances between validly
        mapped src points, to their corresponding dst points (only for
        inliers). In edge case where the number of inliers is zero,
        return dist_mse = 10 ** 9.
    """
    # return fit_percent, dist_mse
    """INSERT YOUR CODE HERE"""
    one_arr = np.ones((1, match_p_dst.shape[1]))
    X = np.concatenate((match_p_src, one_arr), axis=0)
    dst_p_est = homography  @ X
    dst_p_est /= dst_p_est[-1]
    # Now we're going back to regular coordinate
    dst_p_est = dst_p_est[0:2]
    # Calc error dist
    err_arr = np.sum((dst_p_est - match_p_dst) ** 2, axis=0)
    dist_arr = np.sqrt(err_arr)
    fit_percent = np.sum(dist_arr < max_err) / len(dist_arr)
    if fit_percent == 0.0:
        return fit_percent, 10 ** 9
    inliers_indx = np.where(dist_arr < max_err)
    mse_calc = np.mean(err_arr[inliers_indx])
    return fit_percent, mse_calc


def meet_the_model_points(homography: np.ndarray,
                          match_p_src: np.ndarray,
                          match_p_dst: np.ndarray,
                          max_err: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return which matching points meet the homography.

    Loop through the matching points, and return the matching points from
    both images that are inliers for the given homography.

    Args:
        homography: 3x3 Projective Homography matrix.
        match_p_src: 2xN points from the source image.
        match_p_dst: 2xN points from the destination image.
        max_err: A scalar that represents the maximum distance (in
        pixels) between the mapped src point to its corresponding dst
        point, in order to be considered as valid inlier.
    Returns:
        A tuple containing two numpy nd-arrays, containing the matching
        points which meet the model (the homography). The first entry in
        the tuple is the matching points from the source image. That is a
        nd-array of size 2xD (D=the number of points which meet the model).
        The second entry is the matching points form the destination
        image (shape 2xD; D as above).
    """
    # return mp_src_meets_model, mp_dst_meets_model
    one_arr = np.ones((1, match_p_dst.shape[1]))
    X = np.concatenate((match_p_src, one_arr), axis=0)
    dst_p_est = homography @ X
    dst_p_est /= dst_p_est[-1]
    # Now we're going back to regular coordinate
    dst_p_est = dst_p_est[0:2]
    # Calc error dist
    err_arr = np.sum((dst_p_est - match_p_dst) ** 2, axis=0)
    dist_arr = np.sqrt(err_arr)
    inliers_indx = np.where(dist_arr < max_err)
    return match_p_src[:, inliers_indx[0]], match_p_dst[:, inliers_indx[0]]



def compute_homography(
                       match_p_src: np.ndarray,
                       match_p_dst: np.ndarray,
                       inliers_percent: float,
                       max_err: float) -> np.ndarray:
    """Compute homography coefficients using RANSAC to overcome outliers.
    Args:
        match_p_src: 2xN points from the source image.
        match_p_dst: 2xN points from the destination image.
        inliers_percent: The expected probability (between 0 and 1) of
        correct match points from the entire list of match points.
        max_err: A scalar that represents the maximum distance (in
        pixels) between the mapped src point to its corresponding dst
        point, in order to be considered as valid inlier.
    Returns:
        homography: Projective transformation matrix from src to dst.
    """
    # use class notations:
    w = inliers_percent
    # t = max_err
    # p = parameter determining the probability of the algorithm to
    # succeed
    p = 0.99
    # the minimal probability of points which meets with the model
    d = 0.5
    # number of points sufficient to compute the model
    n = 4
    # number of RANSAC iterations (+1 to avoid the case where w=1)
    k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
    best_err = 10 ** 9 # Worst error initialize step
    best_homography = None
    for _ in range(k):
        rand_points = np.random.randint(low=0, high=match_p_dst.shape[1], size=[4])
        homography = compute_homography_naive(match_p_src[:, rand_points], match_p_dst[:, rand_points])
        in_src, in_dst = meet_the_model_points(homography, match_p_src, match_p_dst, max_err)
        if (in_src.shape[1] / match_p_src.shape[1]) > d:
            homography_fixed = compute_homography_naive(in_src, in_dst)
            _, err_tmp = test_homography(homography_fixed, match_p_src, match_p_dst, max_err)
            if err_tmp < best_err:
                best_err = err_tmp
                best_homography = homography_fixed

    return best_homography



# First Part Let's read and show Images
src_img = plt.imread("src.jpg")
dst_img = plt.imread("dst.jpg")

# Second part let's read point matching
matches = scipy.io.loadmat('matches.mat')
match_p_src = matches['match_p_src']
match_p_dst = matches['match_p_dst']

ransac_homography = compute_homography(match_p_src, match_p_dst, inliers_percent=0.8, max_err=25)
wrong_homography = compute_homography_naive(match_p_src, match_p_dst)


transformed_image_wrong = compute_forward_homography_fast(homography=wrong_homography, src_image=src_img ,dst_image_shape=dst_img.shape)
transformed_image_ransac = compute_forward_homography_fast(homography=ransac_homography, src_image=src_img ,dst_image_shape=dst_img.shape)


plt.imshow(transformed_image_wrong)
plt.title('Forward Panorama imperfect matches')
plt.show()

plt.imshow(transformed_image_ransac)
plt.title('Forward Panorama imperfect matches after RANSAC')
plt.show()