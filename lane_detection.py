import numpy as np
import time as t
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt, maximum_filter, binary_closing, binary_erosion, label
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
import collections


class LaneDetection:

    debug_image = None
    left_lane = None
    right_lane = None
    detected_lane_grad = None
    test = None
    l = None 
    r = None
    # for randomize optim calculte histogram and then 
    # improve contrast
    THRESHOLDING_POINT = 50

    def __init__(self):
        pass


    def get_neighbors(self, y, x, height, width):
        """Return 8-connected neighbor coordinates within bounds."""
        neighbors = [
            (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1),
            (y - 1, x - 1), (y - 1, x + 1), (y + 1, x - 1), (y + 1, x + 1)
        ]
        return [(ny, nx) for ny, nx in neighbors if 0 <= ny < height and 0 <= nx < width]

    def count_neighbors(self, skeleton, y, x):
        """Count how many nonzero neighbors a pixel has."""
        return sum(skeleton[ny, nx] > 0 for ny, nx in self.get_neighbors(y, x, *skeleton.shape))
    
    def morphological_medial_axis(self, binary: np.ndarray, structure: np.ndarray = None) -> np.ndarray:
            """
            Compute the medial‐axis skeleton of a binary mask by repeated
            morphological erosion & opening:
                S = ⋃ₖ ( A⊖kB  –  (A⊖kB)⊕B )

            Args:
                binary    : 2D uint8 mask (0 or >0) where >0 is foreground.
                structure : 2D boolean array, default 3×3 square.

            Returns:
                2D uint8 skeleton mask (0 or 255).
            """
            if structure is None:
                structure = np.ones((3,3), bool)

            # make boolean image
            A = (binary > 0)
            skel = np.zeros_like(A)      # accumulator for the skeleton
            eroded = A.copy()

            # helper lambdas
            se_sum = int(structure.sum())
            erode = lambda img: convolve2d(img.astype(int),
                                        structure.astype(int),
                                        mode='same',
                                        boundary='fill', fillvalue=0) >= se_sum
            dilate = lambda img: convolve2d(img.astype(int),
                                            structure.astype(int),
                                            mode='same',
                                            boundary='fill', fillvalue=0) > 0

            # iterate until nothing left of A
            while eroded.any():
                # 1) erode by one
                eroded_next = erode(eroded)

                # 2) opening = dilation of that erosion
                opened = dilate(eroded_next)

                # 3) peel = current eroded – its opening
                peel = eroded & (~opened)

                # accumulate
                skel |= peel

                # move to next iteration
                eroded = eroded_next

            return (skel.astype(np.uint8) * 255)

    def trace_skeleton_path(self, skeleton):
        skeleton = (skeleton > 0).astype(np.uint8)
        coords = np.argwhere(skeleton > 0)
        if coords.size == 0:
            return np.empty((0, 2))

        endpoints = [(y, x) for y, x in coords if self.count_neighbors(skeleton, y, x) == 1]
        start = endpoints[0] if endpoints else tuple(coords[0])

        path = [start]
        visited = {start}
        current = start
        previous = None  # Keep track of the previous point

        while True:
            neighbors = self.get_neighbors(*current, *skeleton.shape)
            unvisited_neighbors = [pt for pt in neighbors if skeleton[pt] > 0 and pt not in visited]

            if not unvisited_neighbors:
                break

            next_point = None
            min_angle = np.inf

            if previous:
                prev_vector = np.array(current) - np.array(previous)

                for neighbor in unvisited_neighbors:
                    neighbor_vector = np.array(neighbor) - np.array(current)
                    # Calculate angle between vectors (using dot product)
                    dot_product = np.dot(prev_vector, neighbor_vector)
                    norm_prev = np.linalg.norm(prev_vector)
                    norm_neighbor = np.linalg.norm(neighbor_vector)
                    print(norm_prev)

                    if norm_prev > 1e-6 and norm_neighbor > 1e-6:
                        cos_angle = dot_product / (norm_prev * norm_neighbor)
                        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip for safety

                        if angle < min_angle:
                            min_angle = angle
                            next_point = neighbor
            else:
                next_point = unvisited_neighbors[0]  # If no previous, just pick one

            if next_point:
                visited.add(next_point)
                path.append(next_point)
                previous = current
                current = next_point
            else:
                break  # No suitable next point found

        return np.array([[y, x] for y, x in path]) 
    
    def fill_holes_in_lane(self, lane: np.ndarray):
        if lane is None: return lane
        lane = (lane > 0)  # bool mask
        # close 1-pixel gaps in horizontal runs
        lane = binary_closing(lane, structure=np.ones((10,10),bool))
        # close 1-pixel gaps in vertical runs
        #lane = binary_closing(lane, structure=np.ones((12,5),bool))
        # back to uint8
        lane = (lane.astype(np.uint8) * 255)

        return lane
    

    def medial_axis_skeleton(self, binary: np.ndarray) -> np.ndarray:
            """
            Pure NumPy/SciPy medial-axis skeletonization:
            1) distance transform of the binary mask
            2) keep only local maxima (the ridge of the distance map)
            """
            mask = (binary > 0)
            # 1) Euclidean distance to background
            dt = distance_transform_edt(mask)
            # 2) local maxima filter: a pixel is part of the medial axis
            #    if it's > 0 and is the maximum in its 3×3 neighborhood
            neighborhood_max = maximum_filter(dt, size=5, mode='constant', cval=0)
            ridge = (dt > 0) & (dt == neighborhood_max)
            return (ridge.astype(np.uint8) * 255) 

    def align_to_wrapper(self, detected_pts: np.ndarray) -> np.ndarray:
        """
        detected_pts: (N,2) array of (v,row, u,col) in [0..95] state‐pixel coords
        returns:      (N,2) array of (x, y) in same frame as
                    _get_lane_boundary_groundtruth  i.e. origin bottom‐left.
        """
        vs = detected_pts[:, 1].astype(float)
        us = detected_pts[:, 0].astype(float)

        xs = vs[::-1] 
        ys = us[::-1] 

        return np.stack((xs, ys), axis=1)
    
    def spline_lane(self, path: np.ndarray, num_points: int = 200, smooth: float = 0) -> np.ndarray:
        """
        Fit a parametric spline through the ordered 2D points in `path` and sample it.

        Args:
            path:     (N,2) array of [x, y] centerline points, in traversal order.
            num_points: how many points to sample on the smooth curve.
            smooth:     spline smoothing factor (0 = interpolate exactly).

        Returns:
            (num_points,2) array of [x_smooth, y_smooth].
        """
        if path.shape[0] < 2:
            return path.copy()

        # 1) Parameterize by cumulative distance along the path
        deltas = np.diff(path, axis=0)
        dist  = np.hypot(deltas[:, 0], deltas[:, 1])
        u     = np.concatenate([[0], np.cumsum(dist)])
        u     /= u[-1]  # normalize to [0,1]

        # 2) Fit spline of degree k (<=3) through x(u) and y(u) *separately*
        k = min(3, path.shape[0] - 1)
        tck_x = splrep(u, path[:, 0], s=smooth, k=k)
        tck_y = splrep(u, path[:, 1], s=smooth, k=k)

        # 3) Sample it uniformly in parameter space
        u_fine = np.linspace(0, 1, num_points)
        x_fine = splev(u_fine, tck_x)
        y_fine = splev(u_fine, tck_y)

        return np.vstack([x_fine, y_fine]).T 


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
    
    def collapse_to_one_per_row(self, skel_mask: np.ndarray) -> np.ndarray:
        """
        Given a binary skeleton mask (0/255), keep exactly one pixel in each row:
        choose the column nearest the mean x of that row.
        """
        # Find all row,col coords
        rows, cols = np.where(skel_mask > 0)
        if rows.size == 0:
            return skel_mask  # empty

        # Group columns by row
        unique = {}
        for y, x in zip(rows, cols):
            unique.setdefault(y, []).append(x)

        # Build a new mask with exactly one pixel per row
        thinned = np.zeros_like(skel_mask, dtype=np.uint8)
        for y, xs in unique.items():
            # pick the average (or median) column
            x = int(np.round(np.mean(xs)))
            thinned[y, x] = 255

        return thinned
    
    def _find_segments(self, sorted_cols: list[int]) -> list[list[int]]:
        """
        Helper function to find contiguous segments in a sorted list of column indices.
        Example: [0, 1, 3, 4, 4, 6] -> [[0, 1], [3, 4, 4], [6]] - Wait, example should be unique sorted.
        Example: [0, 1, 3, 4, 6] -> [[0, 1], [3, 4], [6]]

        Args:
            sorted_cols: A list of column indices, sorted in ascending order.

        Returns:
            A list of segments, where each segment is a list of contiguous column indices.
        """
        if not sorted_cols:
            return []
        
        # Ensure cols are unique and sorted, as duplicates or unsorted could break logic.
        # The main function sorts unique_cols_by_row[r] so this should be fine.
        
        segments = []
        if not sorted_cols:
            return segments
        
        current_segment = [sorted_cols[0]]
        for i in range(1, len(sorted_cols)):
            if sorted_cols[i] == sorted_cols[i-1] + 1:
                current_segment.append(sorted_cols[i])
            else:
                segments.append(current_segment)
                current_segment = [sorted_cols[i]]
        segments.append(current_segment) # Add the last segment
        return segments

    def collapse_to_one_per_row_a(self, skel_mask: np.ndarray) -> np.ndarray:
        """
        Given a binary skeleton mask (0/255), keep exactly one pixel in each row.
        This version is improved to handle U-turns by selecting a pixel from
        a relevant segment, maintaining continuity with the previous row where possible.

        Args:
            skel_mask: A 2D numpy array representing the binary skeleton mask.
                       Foreground pixels are > 0 (typically 255), background is 0.

        Returns:
            A new 2D numpy array of the same shape and dtype as skel_mask,
            containing the thinned skeleton with exactly one pixel per row that
            originally contained foreground pixels.
        """
        # Find all row,col coordinates of foreground pixels
        rows, cols = np.where(skel_mask > 0)
        if rows.size == 0:
            return np.zeros_like(skel_mask, dtype=np.uint8) # Return empty mask if skeleton is empty

        # Group column indices by row index
        unique_cols_by_row = {}
        for r, c in zip(rows, cols):
            unique_cols_by_row.setdefault(r, []).append(c)

        # Prepare the output mask
        thinned_mask = np.zeros_like(skel_mask, dtype=np.uint8)
        
        last_selected_x = None # Stores the x-coordinate chosen in the previously processed row

        # Process rows in ascending order (e.g., top to bottom) for consistent path following
        sorted_row_indices = sorted(unique_cols_by_row.keys())

        for r in sorted_row_indices:
            # Get and sort column indices for the current row
            # Sorting is crucial for the _find_segments logic
            xs_for_row = sorted(list(set(unique_cols_by_row[r]))) # Use set to ensure unique cols then sort

            if not xs_for_row:
                # This case should ideally not be reached if r is from unique_cols_by_row.keys()
                # and unique_cols_by_row[r] was populated.
                continue

            # Find contiguous segments of foreground pixels in the current row
            segments = self._find_segments(xs_for_row)

            if not segments:
                # This case should also ideally not be reached if xs_for_row is not empty.
                continue
            
            chosen_segment_coords = None
            if len(segments) == 1:
                # If there's only one segment, choose it
                chosen_segment_coords = segments[0]
            else:
                # Multiple segments found (e.g., in a U-turn)
                segment_means = [np.mean(seg) for seg in segments]

                if last_selected_x is None:
                    # This is the first row being processed, or continuity was broken.
                    # Choose the segment whose mean is closest to the mean of all x-coordinates in *this* row.
                    # This provides a stable starting point if multiple segments exist initially.
                    mean_of_all_xs_in_row = np.mean(xs_for_row)
                    
                    best_segment_idx = 0
                    min_dist = float('inf')
                    for i, seg_mean in enumerate(segment_means):
                        dist = abs(seg_mean - mean_of_all_xs_in_row)
                        if dist < min_dist:
                            min_dist = dist
                            best_segment_idx = i
                        # In case of a tie in distance, the first segment encountered with that min distance will be chosen.
                    chosen_segment_coords = segments[best_segment_idx]
                else:
                    # There was a selected x in the previous row.
                    # Choose the segment whose mean is closest to last_selected_x to maintain path continuity.
                    best_segment_idx = 0
                    min_dist = float('inf')
                    for i, seg_mean in enumerate(segment_means):
                        dist = abs(seg_mean - last_selected_x)
                        if dist < min_dist:
                            min_dist = dist
                            best_segment_idx = i
                    chosen_segment_coords = segments[best_segment_idx]
            
            # From the chosen segment, pick a representative x-coordinate (its mean)
            selected_x_float = np.mean(chosen_segment_coords)
            selected_x_int = int(np.round(selected_x_float))
            
            # Ensure the selected coordinate is within the mask bounds (primarily for safety,
            # as it's derived from existing column indices).
            selected_x_int = max(0, min(skel_mask.shape[1] - 1, selected_x_int))

            thinned_mask[r, selected_x_int] = 255
            last_selected_x = selected_x_int # Update for the next row's decision process

        return thinned_mask

    def interpolate_between_points(self, points, steps_per_segment=10):
        """
        Linearly interpolate between each consecutive point pair.
        points: (N, 2) array of [y, x] or [x, y] coords
        steps_per_segment: how many interpolated points per segment
        """
        if len(points) < 2:
            return points

        interpolated = []

        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]

            # Independent variable: steps between 0 and 1
            t = np.linspace(0, 1, steps_per_segment, endpoint=False)

            # Interpolate x and y separately
            x_interp = np.interp(t, [0, 1], [p0[0], p1[0]])
            y_interp = np.interp(t, [0, 1], [p0[1], p1[1]])

            interp_pts = np.stack((y_interp, x_interp), axis=1)
            interpolated.append(interp_pts)

        interpolated.append([points[-1]])  # Add the last point
        return np.vstack(interpolated) 

    def binary_erosion(self, img: np.ndarray, structure=None, iterations: int = 1) -> np.ndarray:
        """
        Performs binary erosion on a binary mask.
        """
        if img is None: return None
        img = (img > 0) # Convert to boolean mask
        if structure is None:
            structure = np.ones((3, 3), dtype=bool) # Default 3x3 square structuring element

        eroded_mask = binary_erosion(img, structure=structure, iterations=iterations)

        return (eroded_mask.astype(np.uint8) * 255)


    def grow(self, img, start):
        visited = np.zeros_like(img, dtype=bool)
        if start is None:  # Add this check
            return visited.astype(np.uint8) * 255  # Return an empty mask
        stack = [start]
        height, width = img.shape

        neighbors = lambda y, x: [
            (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1),
            (y - 1, x - 1), (y - 1, x + 1), (y + 1, x - 1), (y + 1, x + 1)
        ]

        while stack:
            y, x = stack.pop()
            if not (0 <= y < height and 0 <= x < width):
                continue
            if visited[y, x] or not img[y, x]:
                continue

            visited[y, x] = True
            for ny, nx in neighbors(y, x):
                stack.append((ny, nx))

        return visited.astype(np.uint8) * 255

    import collections

    def find_shortest_path(self,node_list_input):
        # Ensure the input is a NumPy array with integer type
        # Use astype(int) to avoid potential issues with float/object types
        nodes_np = np.array(node_list_input, dtype=int)
        
        if nodes_np.size == 0:
            return None # No nodes

        # Get start and end nodes as 1D NumPy arrays initially
        start_node_np = nodes_np[0]
        end_node_np = nodes_np[-1]
        
        # *** Convert nodes to tuples for use in BFS data structures ***
        # Tuples are hashable and work correctly in sets and dictionaries
        start_node_tuple = tuple(start_node_np)
        end_node_tuple = tuple(end_node_np)
        
        # Create a set of tuples for efficient checking if a point is a valid node
        valid_nodes_set = set(tuple(row) for row in nodes_np)
        
        # Verify start and end nodes are actually in our set (should be if from list)
        if start_node_tuple not in valid_nodes_set or end_node_tuple not in valid_nodes_set:
            # This case indicates an issue with node setup, should not happen if from list
            return "Error setting up start/end nodes."

        # BFS Initialization using tuples
        queue = collections.deque([start_node_tuple])
        visited = {start_node_tuple} # Use set for efficient lookup
        predecessor = {start_node_tuple: None} # Use dictionary to store path

        # Define possible movements (8 directions)
        # This can be done with standard Python lists/tuples
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while queue:
            current_node_tuple = queue.popleft() # Get a tuple node

            # *** This comparison is now tuple == tuple, which is correct ***
            if current_node_tuple == end_node_tuple:
                # Path found, reconstruct it using the predecessor dictionary
                path = []
                node = end_node_tuple
                while node is not None:
                    path.append(list(node)) # Convert tuple back to list for output
                    node = predecessor[node]
                path.reverse()
                return path
            
            x, y = current_node_tuple # Unpack tuple coordinates
            
            for dx, dy in moves:
                # Calculate neighbor coordinates - these will be numbers (int/numpy.int)
                neighbor_coords = (x + dx, y + dy)
                
                # *** Convert neighbor coordinates to a tuple for checking against the set ***
                neighbor_node_tuple = tuple(neighbor_coords)
                
                # *** These checks are tuple in set and tuple not in set, which works efficiently ***
                if neighbor_node_tuple in valid_nodes_set and neighbor_node_tuple not in visited:
                    visited.add(neighbor_node_tuple)
                    predecessor[neighbor_node_tuple] = current_node_tuple
                    queue.append(neighbor_node_tuple)
                    
        # If loop finishes and end_node wasn't reached
        return "No path found" 

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
            
        
        left_mask = self.binary_dilation(left_mask, structure=np.ones((3,3), dtype=bool))
        right_mask = self.binary_dilation(right_mask, structure=np.ones((3,3), dtype=bool))
        print(np.argwhere(left_mask > 0))

        left_mask = self.medial_axis_skeleton(left_mask)
        right_mask = self.medial_axis_skeleton(right_mask)

        left_mask = self.fill_holes_in_lane(left_mask)
        right_mask = self.fill_holes_in_lane(right_mask)

        print(np.argwhere(left_mask > 0))


        l = np.argwhere(left_mask > 0)
        r = np.argwhere(right_mask > 0)


        if len(l) == 0: 
            print("here")
            l = np.empty((0, 2))
        if len(r) == 0: r = np.empty((0, 2))

        return self.align_to_wrapper(l), self.align_to_wrapper(r)

        left_lane_mask_original = left_mask.copy()
        right_lane_mask_original = right_mask.copy()

        # Fill small holes in the selected lane masks (Optional, but can help skeletonization)
        # You can adjust the structure_size parameter here to fill larger holes
        fill_structure_size = 5 # Example size, tune this

        left_lane_mask_filled = self.fill_holes_in_lane(left_mask)
        right_lane_mask_filled = self.fill_holes_in_lane(right_mask)

        # --- Get masks of ONLY the filled holes ---
        # Convert to boolean masks for accurate difference
        left_holes_mask = (left_lane_mask_filled > 0) & ~(left_lane_mask_original > 0)
        right_holes_mask = (right_lane_mask_filled > 0) & ~(right_lane_mask_original > 0) 
        self.l = np.argwhere(left_holes_mask > 0)
        self.r = np.argwhere(right_holes_mask > 0) 
        print(self.l, self.r)
        left_lane = self.trace_skeleton_path(left_lane_mask_filled)
        right_lane = self.trace_skeleton_path(right_lane_mask_filled)

        #left_mask = self.binary_dilation(left_mask, structure=np.ones((2, 2), dtype=bool,))
        #right_mask = self.binary_dilation(right_mask, structure=np.ones((2, 2), dtype=bool,))


        #left_lane = np.argwhere(left_mask > 0) 
        #right_lane = np.argwhere(right_mask > 0)


        # Find component pixel indices
        #components = [np.argwhere(labeled_mask == i) for i in range(1, num_features + 1)]

        # Sort by mean x (horizontal) to assign left and right
        #components.sort(key=lambda c: np.mean(c[:, 1]))  # x = column

#        left_lane = components[0] if len(components) > 0 else np.empty((0, 2))
 #       right_lane = components[1] if len(components) > 1 else np.empty((0, 2))

        print(left_lane)


        return self.align_to_wrapper(left_lane), self.align_to_wrapper(right_lane)


        grad = self.binary_dilation(grad, structure=np.ones((3, 1), dtype=bool))
        self.detected_lane_grad = [(y, x) for x, y in zip(*np.where(grad > 0))] 

        edge_points = np.argwhere(grad > 0)

        # Filter for points within vertical window around car y-position
        window_mask = (edge_points[:, 0] >= cary - carh) & (edge_points[:, 0] <= cary + carh + 4)
        edges = edge_points[window_mask]

        # Split edges into left and right of car x-position
        left_edges = edges[edges[:, 1] < carx]
        right_edges = edges[edges[:, 1] > carx]

        # Initialize defaults
        left_p, right_p = None, None
        car_center = np.array([cary, carx])
        search_radius = 20 # Adjust this radius as needed

        # Find edges within the search radius
        distances_to_car = np.linalg.norm(edge_points - car_center, axis=1)
        within_radius_mask = distances_to_car <= search_radius
        edges_within_radius = edge_points[within_radius_mask]

        if edges_within_radius.size > 0:
            left_candidates = edges_within_radius[edges_within_radius[:, 1] < carx]
            right_candidates = edges_within_radius[edges_within_radius[:, 1] > carx]

            if left_candidates.size > 0:
                # Find the leftmost candidate (smallest x) as the left seed
                left_p = left_candidates[np.argmin(left_candidates[:, 1])]

            if right_candidates.size > 0:
                # Find the rightmost candidate (largest x) as the right seed
                right_p = right_candidates[np.argmax(right_candidates[:, 1])]

        elif edge_points.size > 0:
            # Fallback if no edges within radius, consider all edges
            left_candidates = edge_points[edge_points[:, 1] < carx]
            right_candidates = edge_points[edge_points[:, 1] > carx]

            if left_candidates.size > 0:
                left_p = left_candidates[np.argmin(left_candidates[:, 1])]
            if right_candidates.size > 0:
                right_p = right_candidates[np.argmax(right_candidates[:, 1])]


        left_lane = self.grow(grad, left_p)
        self.test = [(y, x) for x, y in np.argwhere(left_lane > 0)]
        right_lane = self.grow(grad, right_p)


        left_lane = self.medial_axis_skeleton(left_lane)
        right_lane = self.medial_axis_skeleton(right_lane)
        #print(left_lane)
        left_lane = np.argwhere(left_lane > 0)
        right_lane = np.argwhere(right_lane > 0)

#        left_lane = self.trace_skeleton_path(left_lane)
 #       print("T",left_lane)
  #      right_lane = self.trace_skeleton_path(right_lane)
        



        self.left_lane = left_lane
        self.right_lane = right_lane

     
        if left_lane.size == 0:
            left_lane = np.empty((0, 2))

        if right_lane.size == 0:
            right_lane = np.empty((0, 2)) 

            
        return self.align_to_wrapper(left_lane), self.align_to_wrapper(right_lane)
