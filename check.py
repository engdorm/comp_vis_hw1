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

H = compute_homography_naive(match_p_src, match_p_dst)
print(f"H = {H}")