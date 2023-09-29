"""Projective Homography and Panorama Solution."""
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata

import random

PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])

class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
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
        A = 0
        for idx, (src_point, dst_point) in enumerate(zip(match_p_src.T,match_p_dst.T)):
            src_point_3coord = np.concatenate((src_point,[1]))
            u_tag = dst_point[0]
            v_tag = dst_point[1]
            first_row = np.concatenate((src_point_3coord,[0,0,0],-1*u_tag*src_point_3coord),axis=0)
            second_row = np.concatenate(([0,0,0],src_point_3coord,-1*v_tag*src_point_3coord),axis=0)
            matching_point_matrix = np.array((first_row,second_row))
            #first iter
            if isinstance(A,int):
                A = matching_point_matrix
            else:
                A = np.concatenate((A,matching_point_matrix),axis=0)
            # if idx >= 7:
            #     break
        #x = np.linalg.lstsq(A, np.zeros(A.shape[0]))
        u, s, v = svd(A)
        last_singular_vector = v[-1]
        H = last_singular_vector.reshape(3,3)
        H_normalized = H/H[-1][-1]
        return H_normalized

    @staticmethod
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
        """INSERT YOUR CODE HERE"""
        projective_image = np.zeros(dst_image_shape, dtype='uint8')
        H_target = dst_image_shape[0]
        W_target = dst_image_shape[1]
        for u in range(src_image.shape[1]):
            for v in range(src_image.shape[0]):
                new_u, new_v , const_1 = homography.dot(np.array([u,v,1]))
                new_u_rounded = round(new_u/const_1)
                new_v_rounded = round(new_v/const_1)
                if new_u_rounded >= W_target or new_u_rounded < 0 or  new_v_rounded >= H_target or new_v_rounded < 0:
                    continue
                else:
                    projective_image[new_v_rounded,new_u_rounded] = src_image[v][u]
        return projective_image
        pass

    @staticmethod
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
        projective_image = np.zeros(dst_image_shape, dtype='uint8')
        h_src = src_image.shape[0]
        w_src = src_image.shape[1]
        h_target = dst_image_shape[0]
        w_target = dst_image_shape[1]
        x = np.linspace(0,w_src-1,w_src)
        y = np.linspace(0,h_src-1,h_src)
        xv, yv = np.meshgrid(x,y)
        full_pixels_indices_matrix = np.array([xv.flatten(), yv.flatten(), np.ones_like(yv.flatten())])
        transformed_indices = np.matmul(homography, full_pixels_indices_matrix)
        transformed_indices_normlized = transformed_indices / transformed_indices[-1]
        transformed_indices_normlized_as_int = np.array([np.round_(transformed_indices_normlized)],dtype='int32')
        transformed_indices_normlized_as_int = np.squeeze(transformed_indices_normlized_as_int)
        mask_v = np.bitwise_and(transformed_indices_normlized_as_int[1] >= 0,
                                transformed_indices_normlized_as_int[1] <= h_target-1)
        mask_u = np.bitwise_and(transformed_indices_normlized_as_int[0] >= 0,
                                transformed_indices_normlized_as_int[0] <= w_target-1)

        valid_indices_in_source = np.bitwise_and(mask_v,mask_u)
        src_image_mask = valid_indices_in_source.reshape(h_src, w_src)

        valid_indicies_in_target = transformed_indices_normlized_as_int[:, valid_indices_in_source]
        u_valid = valid_indicies_in_target[0]
        v_valid = valid_indicies_in_target[1]
        projective_image[v_valid,u_valid] = src_image[src_image_mask]

        return projective_image

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
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
        curr_distance = 0
        num_of_inliers = 0
        inliers_distance_list = []
        for idx, (src_point, dst_point) in enumerate(zip(match_p_src.T, match_p_dst.T)):
            src_point_3coord = np.concatenate((src_point, [1]))
            u_tag = dst_point[0]
            v_tag = dst_point[1]
            u_target, v_target, const_target = homography.dot(src_point_3coord)
            u_target_rounded = round(u_target / const_target)
            v_target_rounded = round(v_target / const_target)
            curr_distance = np.abs(u_target_rounded - u_tag) + np.abs(v_target_rounded - v_tag)
            if curr_distance <= max_err:
                num_of_inliers+=1
                inliers_distance_list.append(curr_distance)
        dist_mse = np.square(inliers_distance_list).mean()
        fit_percent = num_of_inliers / match_p_src.shape[-1]
        return fit_percent, dist_mse

    @staticmethod
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
        """INSERT YOUR CODE HERE"""
        mp_src_meets_model = []
        mp_dst_meets_model = []
        curr_distance = 0
        num_of_inliers = 0
        inliers_distance_list = []
        inlier_idx = 0
        for idx, (src_point, dst_point) in enumerate(zip(match_p_src.T, match_p_dst.T)):
            src_point_3coord = np.concatenate((src_point, [1]))
            u_tag = dst_point[0]
            v_tag = dst_point[1]
            u_target, v_target, const_target = homography.dot(src_point_3coord)
            u_target_rounded = round(u_target / const_target)
            v_target_rounded = round(v_target / const_target)
            curr_distance = np.abs(u_target_rounded - u_tag) + np.abs(v_target_rounded - v_tag)
            if curr_distance <= max_err:
                mp_src_meets_model.append(src_point)
                mp_dst_meets_model.append(dst_point)
                inlier_idx+=1
        mp_src_meets_model = np.array(mp_src_meets_model).T
        mp_dst_meets_model = np.array(mp_dst_meets_model).T
        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
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
        t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography
        """INSERT YOUR CODE HERE"""
        # index list of the points
        index_list = list(range(match_p_src.shape[-1]))
        homography_model = np.zeros((3,3))
        prev_lowest_mse = t**2 # mse shuold be lower than this if not it's a bug and we will see bad results
        for i in range(k):
            random_indexes = random.sample(index_list, n)
            match_p_src_selected = match_p_src[:,random_indexes]
            match_p_dst_selected = match_p_dst[:,random_indexes]
            homography = self.compute_homography_naive(match_p_src_selected, match_p_dst_selected)
            fit_percent, dist_mse = self.test_homography(homography, match_p_src, match_p_dst, t)
            mp_src_meets_model, mp_dst_meets_model = self.meet_the_model_points(homography, match_p_src, match_p_dst,t)
            if fit_percent >= d:
                homography_based_on_all_inliers = self.compute_homography_naive(mp_src_meets_model, mp_dst_meets_model)
                fit_percent, dist_mse = self.test_homography(homography, mp_src_meets_model, mp_dst_meets_model, t)
                if dist_mse <= prev_lowest_mse:
                    prev_lowest_mse = dist_mse
                    homography_model = homography_based_on_all_inliers
        return homography_model

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        """INSERT YOUR CODE HERE"""
        h_dst = dst_image_shape[0]
        w_dst = dst_image_shape[1]
        h_src = src_image.shape[0]
        w_src = src_image.shape[1]
        x = np.linspace(0,w_dst-1,w_dst)
        y = np.linspace(0,h_dst-1,h_dst)
        xv, yv = np.meshgrid(x,y)
        full_pixels_indices_matrix = np.array([xv.flatten(), yv.flatten(), np.ones_like(yv.flatten())])
        transformed_indices = np.matmul(backward_projective_homography, full_pixels_indices_matrix)
        transformed_indices_normlized = transformed_indices / transformed_indices[-1]
        transformed_indices_normlized = np.squeeze(transformed_indices_normlized)
        mask_v = np.bitwise_and(transformed_indices_normlized[1] >= 0,
                                transformed_indices_normlized[1] <= h_src-1)
        mask_u = np.bitwise_and(transformed_indices_normlized[0] >= 0,
                                transformed_indices_normlized[0] <= w_src-1)

        valid_indices_compare_to_source_mask = np.bitwise_and(mask_v,mask_u)
        dst_image_mask = valid_indices_compare_to_source_mask.reshape(h_dst, w_dst)

        valid_indices_compare_to_source = transformed_indices_normlized[:, valid_indices_compare_to_source_mask]
        u_valid = valid_indices_compare_to_source[0]
        v_valid = valid_indices_compare_to_source[1]

        x_src = np.linspace(0,w_src-1,w_src)
        y_src = np.linspace(0,h_src-1,h_src)
        xv_src, yv_src = np.meshgrid(x_src,y_src)
        red_grid = griddata(np.array([xv_src.flatten(), yv_src.flatten()]).T, src_image[:,:,0].flatten(), (u_valid, v_valid), method='cubic')
        green_grid = griddata(np.array([xv_src.flatten(), yv_src.flatten()]).T, src_image[:,:,1].flatten(), (u_valid, v_valid), method='cubic')
        blue_grid = griddata(np.array([xv_src.flatten(), yv_src.flatten()]).T, src_image[:,:,2].flatten(), (u_valid, v_valid), method='cubic')

        src_img_backwards = np.zeros(dst_image_shape, dtype='uint8')
        src_img_backwards[dst_image_mask] = np.array([red_grid, green_grid, blue_grid],dtype='uint8').T
        # src_img_backwards = np.nan_to_num(src_img_backwards)
        return src_img_backwards


        pass #src_image[3,889,0]*(1-u_valid[0])*(1-v_valid[0]) + src_image[3,890,0]*u_valid[0]*(1-v_valid[0]) + src_image[4,889,0]*(1-u_valid[0]*v_valid[0] + src_image[4,890,0]*u_valid[0]*v_valid[0]

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        translation_matrix = np.array([[1, 0, -pad_left],[0, 1, -pad_up],[0, 0, 1]])
        backward_homography_plus_translation = np.matmul(backward_homography,translation_matrix)
        backward_homography_plus_translation_normalized = backward_homography_plus_translation/backward_homography_plus_translation.flatten()[-1]
        return backward_homography_plus_translation_normalized

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""
        forward_homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        panorama_rows_num, panorama_cols_num, pad_struct = self.find_panorama_shape(src_image, dst_image, forward_homography)
        backward_homography = np.linalg.inv(forward_homography)
        backward_homography = backward_homography/backward_homography.flatten()[-1]
        backward_homography_with_translation = self.add_translation_to_backward_homography(backward_homography,pad_struct.pad_left ,pad_struct.pad_up)
        src_img_backward_mapped = self.compute_backward_mapping(backward_homography_with_translation, src_image, (panorama_rows_num, panorama_cols_num,3))
        empty_panorama = np.zeros((panorama_rows_num,panorama_cols_num,3), dtype='uint8')

        dst_image_ending_rows_pixels = dst_image.shape[0]
        dst_image_starting_cols_pixels = panorama_cols_num - dst_image.shape[1]

        backwards_src_img_ending_rows_pixels = src_img_backward_mapped.shape[0]
        backwards_src_img_ending_cols_pixels_in_backward_img = pad_struct.pad_left
        # backwards_src_img_starting_cols_pixels_in_backward_img = max(backwards_src_img_ending_cols_pixels_in_backward_img - dst_image_starting_cols_pixels,0)
        # dst_image_starting_cols_pixels_for_panorama = dst_image_starting_cols_pixels - (backwards_src_img_ending_cols_pixels_in_backward_img - backwards_src_img_starting_cols_pixels_in_backward_img)
        if pad_struct.pad_left < pad_struct.pad_right:
            empty_panorama[pad_struct.pad_up:dst_image_ending_rows_pixels + pad_struct.pad_up,
            :dst_image.shape[1]] = dst_image
            empty_panorama[:backwards_src_img_ending_rows_pixels,
            dst_image.shape[1]:] = src_img_backward_mapped[:,dst_image.shape[1]:]
        else:
            empty_panorama[pad_struct.pad_up:dst_image_ending_rows_pixels+pad_struct.pad_up,dst_image_starting_cols_pixels:] = dst_image
            #empty_panorama[:backwards_src_img_ending_rows_pixels,dst_image_starting_cols_pixels_for_panorama:dst_image_starting_cols_pixels] = src_img_backward_mapped[:,backwards_src_img_starting_cols_pixels_in_backward_img:backwards_src_img_ending_cols_pixels_in_backward_img]
            empty_panorama[:backwards_src_img_ending_rows_pixels,:dst_image_starting_cols_pixels] = src_img_backward_mapped[:,:dst_image_starting_cols_pixels]
        img_panorama = empty_panorama
        return np.clip(img_panorama, 0, 255).astype(np.uint8)
