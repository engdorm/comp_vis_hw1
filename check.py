import numpy as np
import PIL
import matplotlib.pyplot as plt
import scipy.io
from numpy.linalg import svd


# First Part Let's read and show Images
src_img = plt.imread("src.jpg")
dst_img = plt.imread("dst.jpg")
# plt.figure(1)
# plt.imshow(src_img), plt.title("Src Image")
# plt.figure(2)
# plt.imshow(dst_img), plt.title("Dst Image"), plt.show()

# Second part let's read point matching
matches = scipy.io.loadmat('matches_perfect')
match_p_src = matches['match_p_src']
match_p_dst = matches['match_p_dst']

# # Plot images with Corresponding point
color_list = ['b', 'g', 'r', 'y', 'w', 'k', 'm', 'b', 'g', 'r', 'y', 'w', 'k', 'm', 'b', 'g', 'r', 'y', 'w', 'k']
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.figure(figsize=(8, 8), dpi=80)
fig.suptitle('Perfect Matches', fontsize=16)
ax1.set_title('Src_img')
ax1.imshow(src_img)
ax1.scatter(match_p_src[0], match_p_src[1], c=color_list)

plt.figure(figsize=(8, 8), dpi=80)
ax2.set_title('Dst_img')
ax2.imshow(dst_img)
ax2.scatter(match_p_dst[0], match_p_dst[1], c=color_list)
plt.show()
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



H = compute_homography_naive(match_p_src, match_p_dst)
out_img = compute_forward_homography_slow(H, src_img, dst_img.shape)
plt.imshow(out_img), plt.title("forward transformation"), plt.show()