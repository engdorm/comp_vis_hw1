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
    img_out = np.zeros(dst_image_shape, dtype=np.uint8)
    x = np.linspace(0, src_image.shape[1], src_image.shape[1] - 1)
    y = np.linspace(0, src_image.shape[0], src_image.shape[0] - 1)
    yy, xx = np.meshgrid(x, y)
    print("Hello")

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
    dst_p_est = H  @ X
    dst_p_est /= dst_p_est[-1]
    # Now we're going back to regular coordinate
    dst_p_est = dst_p_est[0:2]
    # Calc error dist
    err_arr = np.sum((dst_p_est - match_p_dst) ** 2, axis=0)
    dist_arr = np.sqrt(err_arr)
    fit_percent = np.sum(dist_arr < max_err) / len(dist_arr)
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



def meet_the_model_points(homography, match_p_src, match_p_dst, max_err):
    pass



# First Part Let's read and show Images
src_img = plt.imread("src.jpg")
dst_img = plt.imread("dst.jpg")

# Second part let's read point matching
matches = scipy.io.loadmat('matches_perfect.mat')
match_p_src = matches['match_p_src']
match_p_dst = matches['match_p_dst']

H = compute_homography_naive(match_p_src, match_p_dst)
fit_percent, mse_calc = test_homography(homography=H, match_p_src=match_p_src, match_p_dst=match_p_dst, max_err=3)
match_p_srcl, match_p_dstl = meet_the_model_points(homography=H, match_p_src=match_p_src, match_p_dst=match_p_dst, max_err=3)
print(f"fit precent = {fit_percent}    mse = {mse_calc}")
print(f"match_p_src, match_p_ds = {match_p_srcl, match_p_dstl}")