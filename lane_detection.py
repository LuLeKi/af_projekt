import numpy as np
import cv2
import time as t
from scipy.signal import convolve2d
from scipy.ndimage import label
import matplotlib.pyplot as plt



class LaneDetection:

    debug_image = None
    left_lane = None
    right_lane = None
    detected_lane_grad = None
    # for randomize optim calculte histogram and then 
    # improve contrast
    THRESHOLDING_POINT = 50

    def __init__(self):
        pass

    def align_to_wrapper(self, detected_pts: np.ndarray) -> np.ndarray:
        """
        detected_pts: (N,2) array of (v,row, u,col) in [0..95] state‐pixel coords
        returns:      (N,2) array of (x, y) in same frame as
                    _get_lane_boundary_groundtruth  i.e. origin bottom‐left.
        """
        vs = detected_pts[:, 0].astype(float)
        us = detected_pts[:, 1].astype(float)

        xs = vs[::-1] 
        ys = us[::-1] 

        return np.stack((xs, ys), axis=1)


    def normalize_floats(self, matrix):
        matrix = 255 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        return matrix

    def binary_dilation(self, img, structure=None):
        img = img > 0
        if structure is None:
            structure = np.ones((3, 3), dtype=bool)

        # Perform convolution
        convolved = convolve2d(img.view(np.uint8), structure.view(np.uint8), mode='same', boundary='fill', fillvalue=0)

        # Any nonzero result means at least one neighbor was active
        return convolved > 0

    def grow_region(self, img, start, structure=None):
        bool_image = img > 0
        seed = np.zeros_like(bool_image, dtype=bool)
        seed[start] = True

        prev = None
        curr = seed.copy()

        while prev is None or not np.array_equal(prev, curr):
            prev = curr.copy()
            curr = self.binary_dilation(curr, structure) & bool_image
 
        return curr.astype(np.uint8) * 255

    def grow(self, img, start):
        mask = img > 0
        visited = np.zeros_like(mask, dtype=bool)
        stack = [start]
        height, width = mask.shape

        while stack:
            y, x = stack.pop()
            if not (0 <= y < height and 0 <= x < width):
                continue
            if visited[y, x] or not mask[y, x]:
                continue

            visited[y, x] = True
            # add four pints around current coordinate to stack
            # to then check if they are in the mask(image)
            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])

        return visited.astype(np.uint8) * 255

    def detect(self, state_image):
        # turn image to grayscale
        self.debug_image = state_image.copy()
        gray_img = np.dot(state_image[...,:3], [0.299, 0.587, 0.114])
        gray_normalized = self.normalize_floats(gray_img) 

        # now convolve the image
        # i want to use prewitt
        kx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]], dtype=np.float32)  # vertical filter

        ky = np.array([[ 1,  1,  1],
               [ 0,  0,  0],
               [-1, -1, -1]], dtype=np.float32)  
        # print(kx.shape, ky.shape)

        cx = convolve2d(gray_normalized, kx, mode="same", boundary="symm")
        cy = convolve2d(gray_normalized, ky, mode="same", boundary="symm")

        grad = np.sqrt(cx**2 + cy**2)
        # print(grad)
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
        self.detected_lane_grad = [(y, x) for x, y in zip(*np.where(grad > 0))] 

        labeled_mask, num_features = label(grad > 0, structure=np.ones((3, 3)))
        print(num_features)
        min_dists = []
        for i in range(1, num_features + 1):
            mask = labeled_mask == i
            coords = np.argwhere(mask)
            car_pos = np.array([cary + 6, carx])
            dists = np.linalg.norm(coords - car_pos, axis=1)
            min_dist = np.min(dists)
            min_dists.append([i, min_dist])

        min_dists.sort(key=lambda x: x[1])
        print(min_dists)

        if len(min_dists) >= 1:
            left_mask = labeled_mask == min_dists[0][0]
        else:
            left_mask = np.empty((0, 2))

        if len(min_dists) >= 2:
            right_mask = labeled_mask == min_dists[1][0]
        else:
            right_mask = np.empty((0, 2))

        self.left_lane = left_mask
        self.right_lane = right_mask

        #return an array of coordinates where the values are non-zero for every lane
        left_lane_points = np.argwhere(left_mask > 0)
        right_lane_points = np.argwhere(right_mask > 0)

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
            
        return self.align_to_wrapper(left_lane_points), self.align_to_wrapper(right_lane_points)
