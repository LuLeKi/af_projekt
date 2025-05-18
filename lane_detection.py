import numpy as np
import time as t
from scipy.signal import convolve2d
from scipy.ndimage import label
import matplotlib.pyplot as plt



class LaneDetection:
    """
    Detects lane lines in a state image using edge detection and connected components.
    """

    debug_image = None
    THRESHOLDING_POINT = 50

    def __init__(self):
        pass

    def align_to_wrapper(self, detected_pts: np.ndarray) -> np.ndarray:
        """
        Transforms detected image coordinates (v, u) to a world coordinate system (x, y).

        Args:
            detected_pts: (N, 2) array of (v, u) image coordinates [0..95].

        Returns:
            (N, 2) array of (x, y) points in the wrapper's coordinate frame (origin bottom-left).
        """
        vs = detected_pts[:, 0].astype(float)
        us = detected_pts[:, 1].astype(float)

        xs = vs[::-1] 
        ys = us[::-1] 

        return np.stack((xs, ys), axis=1)


    def normalize_floats(self, matrix):
        """
        Normalizes a float matrix to the 0-255 range.

        Args:
            matrix: Input NumPy array.

        Returns:
            NumPy array normalized to [0, 255].
        """
        matrix = 255 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        return matrix 

    def detect(self, state_image):
        """
        Detects lane lines in the input state image.

        Processes the image using grayscale conversion, Prewitt edge detection,
        thresholding, connected components analysis, and point interpolation
        to find left and right lane points.

        Args:
            state_image: The input image array.

        Returns:
            A tuple containing two NumPy arrays: (left_lane_points, right_lane_points),
            each in the wrapper's (x, y) coordinate frame. Empty arrays if no points found.
        """
        # turn image to grayscale
        self.debug_image = state_image.copy()
        gray_img = np.dot(state_image[...,:3], [0.299, 0.587, 0.114])
        gray_normalized = self.normalize_floats(gray_img) 

        # now convolve the image
        # use prewitt
        # vertical filter
        kx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]], dtype=np.float32) 

        #horizontal filter 
        ky = np.array([[ 1,  1,  1],
               [ 0,  0,  0],
               [-1, -1, -1]], dtype=np.float32)  

        cx = convolve2d(gray_normalized, kx, mode="same", boundary="symm")
        cy = convolve2d(gray_normalized, ky, mode="same", boundary="symm")

        grad = np.sqrt(cx**2 + cy**2)
        grad = self.normalize_floats(grad)

        #remove car from image
        carx = 48 
        cary = 64 
        carh = 13 
        grad[cary:cary + carh, carx - 3:carx + 3] = 0 
        #remove stats bottom from image
        grad[cary + carh:] = 0

        # thresholding
        grad = (grad > self.THRESHOLDING_POINT) * grad * (255 / grad)

        # distinguish between left and right lane
        # by finding the nearest lane and.
        # there are situations where there are three or four lanes found -> chicane
        labeled_mask, num_features = label(grad > 0, structure=np.ones((3, 3)))

        min_dists = []
        for i in range(1, num_features + 1):
            mask = labeled_mask == i
            coords = np.argwhere(mask)
            car_pos = np.array([cary + 6, carx])
            dists = np.linalg.norm(coords - car_pos, axis=1)
            min_dist = np.min(dists)
            min_dists.append([i, min_dist])

        min_dists.sort(key=lambda x: x[1])

        if len(min_dists) >= 1:
            left_mask = labeled_mask == min_dists[0][0]
        else:
            left_mask = np.empty((0, 2))

        if len(min_dists) >= 2:
            right_mask = labeled_mask == min_dists[1][0]
        else:
            right_mask = np.empty((0, 2))

        #return an array of coordinates where the values are non-zero for every lane
        left_lane_points = np.argwhere(left_mask > 0)
        right_lane_points = np.argwhere(right_mask > 0)

        # thin out lane to one pixel by removing points with duplicate y
        unique_y = {}
        for y, x in left_lane_points:
            if y not in unique_y or abs(x - unique_y[y]) > 1:
                unique_y[y] = x

        left_lane_points = np.array([[x, y] for y, x in unique_y.items()])
        if left_lane_points.size == 0:
            left_lane_points = np.empty((0, 2))

        unique_y = {}
        for y, x in right_lane_points:
            if y not in unique_y or abs(x - unique_y[y]) > 1:
                unique_y[y] = x

        right_lane_points = np.array([[x, y] for y, x in unique_y.items()])
        if right_lane_points.size == 0:
            right_lane_points = np.empty((0, 2)) 

        # interpolating over both lanes to generate a smooth lane
        if left_lane_points.size > 0:
            left_lane_points = left_lane_points[left_lane_points[:, 1].argsort()]
            new_y = np.linspace(left_lane_points[:, 1].min(), left_lane_points[:, 1].max(), 250)
            left_lane_x = np.interp(new_y, left_lane_points[:, 1], left_lane_points[:, 0])
            left_lane_points = np.stack((left_lane_x, new_y), axis=1)

        if right_lane_points.size > 0:
            right_lane_points = right_lane_points[right_lane_points[:, 1].argsort()]
            new_y = np.linspace(right_lane_points[:, 1].min(), right_lane_points[:, 1].max(), 250)
            right_lane_x = np.interp(new_y, right_lane_points[:, 1], right_lane_points[:, 0])
            right_lane_points = np.stack((right_lane_x, new_y), axis=1)

        left_lane_points = left_lane_points[np.argsort(left_lane_points[:, 1])]
        right_lane_points = right_lane_points[np.argsort(right_lane_points[:, 1])]

 
        return self.align_to_wrapper(left_lane_points), self.align_to_wrapper(right_lane_points)
