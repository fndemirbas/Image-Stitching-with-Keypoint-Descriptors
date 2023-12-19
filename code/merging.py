import numpy as np

#Merging by Transformation
def warpPerspective(image_left, image_right, homography_matrix):
    dst = np.ndarray(shape=(image_left.shape[0], image_left.shape[1] + image_right.shape[1],  3), dtype=image_left.dtype)
    dst[0:image_left.shape[0], 0:image_left.shape[1]] = image_left

    for column in range(image_right.shape[1]):
        for row in range(image_right.shape[0]):
            right_image_coordinates = np.array([column, row, 1])
            right_image_coordinates = right_image_coordinates[:,None]
            right_image_coordinates_main = np.matmul(homography_matrix, right_image_coordinates)
            right_image_coordinates_main = right_image_coordinates_main / right_image_coordinates_main[2]
            right_image_coordinates_main = np.round(right_image_coordinates_main).astype(int)
            if(right_image_coordinates_main[0][0] > 0 and right_image_coordinates_main[0][0] < dst.shape[1] and right_image_coordinates_main[1][0] > 0 and right_image_coordinates_main[1][0] < dst.shape[0]):
                dst[right_image_coordinates_main[1][0]][right_image_coordinates_main[0][0]] = image_right[row][column]


    cols = np.where(~dst.any(axis=0))[0]
    if(len(cols) != 0):
        start_col_no_black = cols[0]
        dst = dst[:,:start_col_no_black]

    return dst
