import numpy as np
from scipy.optimize import leastsq
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
from sklearn.preprocessing import StandardScaler
from numpy.polynomial.polynomial import Polynomial

class ExtractorTools:
    """
    A collection of static methods for processing and analyzing segmentation masks and anatomical features, 
    such as fitting circles to points, filtering iris points, and clustering masks.
    """

    @staticmethod
    def circle_residuals(params, points):
        """
        Computes the residuals between the points and a fitted circle defined by the given parameters.

        Parameters:
            params (tuple): A tuple containing the circle parameters (h, k, r), where:
                - h (float): The x-coordinate of the circle's center.
                - k (float): The y-coordinate of the circle's center.
                - r (float): The radius of the circle.
            points (np.array): An array of points (x, y) to calculate the residuals for.

        Returns:
            np.array: The residuals (distance from the points to the circle) for each point.
        """
        h, k, r = params
        x, y = points.T
        return (x - h) ** 2 + (y - k) ** 2 - r ** 2
    
    @staticmethod
    def fit_circle(points):
        """
        Fits a circle to a given set of points using least squares optimization.

        Parameters:
            points (np.array): An array of points (x, y) to fit a circle to.

        Returns:
            tuple: The optimized circle parameters (h, k, r), where:
                - h (float): The x-coordinate of the circle's center.
                - k (float): The y-coordinate of the circle's center.
                - r (float): The radius of the circle.
        """
        # Initial guess: centroid of points as circle center & average distance to center as radius
        x, y = np.mean(points, axis=0)
        r = np.mean(np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2))
        params, _ = leastsq(ExtractorTools.circle_residuals, [x, y, r], args=(points,))
        return params  # Returns circle parameters: h, k, r

    @staticmethod
    def get_border_points(mask):
        """
        Extracts the topmost and bottommost points from each column of a binary mask.

        Parameters:
            mask (np.array): A binary mask where the object of interest is marked as 1.

        Returns:
            tuple: Two lists containing the top and bottom points for each column in the mask.
        """
        top_points = []
        bottom_points = []
        
        for x in range(mask.shape[1]):
            column = mask[:, x]
            
            # Get the top and bottom non-zero points
            non_zeros = np.where(column == 1)[0]
            
            if len(non_zeros) > 0:
                top_points.append((x, non_zeros[0]))
                bottom_points.append((x, non_zeros[-1]))

        return top_points, bottom_points
        
    @staticmethod
    def get_lateral_iris_circle(iris_mask, sclera_mask, dim):
        """
        Determines the inscribed circle in the lateral part of the iris, using border points from the iris mask.

        Parameters:
            iris_mask (np.array): Binary mask of the iris.
            sclera_mask (np.array): Binary mask of the sclera.
            dim (tuple): Dimensions of the image (width, height).

        Returns:
            tuple: The center and diameter of the inscribed circle.
        """
        iris_top, iris_bottom = ExtractorTools.get_border_points(iris_mask)
        combined_iris = np.vstack((iris_top, iris_bottom))
        center, diameter = ExtractorTools.iris_circle_inscribed(combined_iris)
        return center, diameter

    @staticmethod
    def iris_circle_inscribed(points):
        """
        Finds the largest inscribed circle (circle with maximum diameter) within a set of points.

        Parameters:
            points (np.array): An array of points (x, y) to determine the inscribed circle.

        Returns:
            tuple: The center and diameter of the inscribed circle.
        """
        max_distance = 0
        point1, point2 = None, None
        
        # Iterate over all pairs of points to find the maximum distance (diameter)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                if distance > max_distance:
                    max_distance = distance
                    point1, point2 = points[i], points[j]
                    
        # Calculate the midpoint of the diameter as the circle's center
        midpoint = [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]
        return midpoint, max_distance

    @staticmethod
    def find_closest_if_no_match(candidates, eyebrow_mask, reference_point, orientation):
        """
        Finds the closest point in the eyebrow mask to the reference point if no exact match is found.

        Parameters:
            candidates (list): A list of candidate points on the same x-coordinate.
            eyebrow_mask (np.array): Binary mask of the eyebrow.
            reference_point (tuple): The reference point (x, y) to compare against.
            orientation (str): The orientation to search ('inferior' or 'superior').

        Returns:
            tuple: The closest point found, either from candidates or by searching the entire eyebrow mask.
        """
        # Check if there are direct matches (candidates) on the same x-coordinate
        if len(candidates) > 0:
            if orientation == 'inferior':
                # Find the most inferior (maximum y-value) among the candidates
                selected_point = candidates[np.argmax(candidates)]
            elif orientation == 'superior':
                # Find the most superior (minimum y-value) among the candidates
                selected_point = candidates[np.argmin(candidates)]
            return (reference_point[0], selected_point)
        else:
            # No direct matches, find the closest point in the specified orientation
            # Compute distances from the reference point to all points in the eyebrow mask
            distances = np.sqrt((eyebrow_mask[:, 0] - reference_point[1]) ** 2 + (eyebrow_mask[:, 1] - reference_point[0]) ** 2)
            min_distance_idx = distances.argmin()
            target_point = eyebrow_mask[min_distance_idx]
            return (target_point[0], target_point[1])

    @staticmethod
    def cluster_masks(points, img_shape):
        """
        Clusters points in a binary mask and returns the largest connected component.

        Parameters:
            points (np.array): An array of points (x, y) in the mask.
            img_shape (tuple): The shape of the image (height, width).

        Returns:
            np.array: A binary mask of the largest connected component.
        """
        points = np.array(points)

        # Label connected components
        labeled, num_features = label(points)
        
        # Find the label with the maximum count (excluding background which is label 0)
        if num_features > 0:
            largest_label = np.bincount(labeled.ravel())[1:].argmax() + 1

            # Create a mask for only this largest component
            largest_cluster_mask = (labeled == largest_label).astype(np.uint8)
            return largest_cluster_mask
        else:
            return np.array([])  # Return an empty array if no features are found
      
    @staticmethod
    def get_all_clusters(detections_dict, crop_img_rgb):
        """
        Merges and clusters multiple anatomical masks into a final set of masks for processing.

        Parameters:
            detections_dict (dict): Dictionary containing various detected masks.
            crop_img_rgb (PIL.Image): The cropped image in RGB format.

        Returns:
            tuple: A tuple containing the final masks for the right and left sclera, iris, and brow.
        """
        # Merge right sclera and caruncle masks
        if detections_dict['right_caruncle'].any():
            merged_right_sclera_caruncle = detections_dict['right_sclera_orig'] | detections_dict['right_caruncle']
            detections_dict['right_sclera'] = merged_right_sclera_caruncle

        # Merge left sclera and caruncle masks        
        if detections_dict['left_caruncle'].any():
            merged_left_sclera_caruncle = detections_dict['left_sclera_orig'] | detections_dict['left_caruncle']
            detections_dict['left_sclera'] = merged_left_sclera_caruncle

        # Further merge the sclera with the iris masks
        detections_dict['right_sclera'] = detections_dict['right_sclera'] | detections_dict['right_iris']
        detections_dict['left_sclera'] = detections_dict['left_sclera'] | detections_dict['left_iris']

        # Cluster and extract the final masks for each anatomical feature
        right_sclera_mask = ExtractorTools.cluster_masks(detections_dict['right_sclera'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))
        left_sclera_mask = ExtractorTools.cluster_masks(detections_dict['left_sclera'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))
        
        right_iris_mask = ExtractorTools.cluster_masks(detections_dict['right_iris'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))
        left_iris_mask = ExtractorTools.cluster_masks(detections_dict['left_iris'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))
        
        right_brow_mask = ExtractorTools.cluster_masks(detections_dict['right_brow'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))
        left_brow_mask = ExtractorTools.cluster_masks(detections_dict['left_brow'], (crop_img_rgb.size[1], crop_img_rgb.size[0]))    

        return right_sclera_mask, left_sclera_mask, right_iris_mask, left_iris_mask, right_brow_mask, left_brow_mask


class PupilIris:
    """
    A class for extracting key features of the iris, such as the center, superior and inferior points, 
    and the diameter of the iris.
    """

    def __init__(self):
        pass
    
    def get_iris_info(self, iris_mask, sclera_mask, img_dim):  
        """
        Extracts the iris center, superior and inferior points, and iris diameter from the given mask.

        Parameters:
            iris_mask (np.array): Binary mask of the iris.
            sclera_mask (np.array): Binary mask of the sclera.
            img_dim (tuple): Dimensions of the image (width, height).

        Returns:
            tuple: A tuple containing:
                - centroid (np.array): The coordinates of the iris center.
                - diameter (float): The diameter of the iris.
                - superior (np.array): The coordinates of the superior point of the iris.
                - inferior (np.array): The coordinates of the inferior point of the iris.
        """
        centroid, diameter = ExtractorTools.get_lateral_iris_circle(iris_mask, sclera_mask, img_dim)
           
        centroid = [round(centroid[0], 0), round(centroid[1], 0)]
        
        superior = [centroid[0], centroid[1] - (diameter / 2)]
        inferior = [centroid[0], centroid[1] + (diameter / 2)]

        return np.array(centroid), diameter, np.array(superior), np.array(inferior)

class Sclera:
    """
    A class for extracting and analyzing key features of the sclera, such as splitting border points into 
    upper and lower sets, fitting polynomials to these points, and extracting key scleral points.
    """

    def __init__(self):
        pass
 
    @staticmethod
    def split_upper_lower(border_points, medial_canthus, lateral_canthus):
        """
        Splits border points into upper and lower sets relative to a line connecting the medial and lateral canthus.

        Parameters:
            border_points (np.array): Array of border points (x, y) from the sclera mask.
            medial_canthus (tuple): Coordinates of the medial canthus (x, y).
            lateral_canthus (tuple): Coordinates of the lateral canthus (x, y).

        Returns:
            tuple: Two arrays containing the upper and lower border points.
        """
        upper_points = []
        lower_points = []
        
        # Calculate the slope of the line connecting the medial and lateral canthus
        med_lat_line_slope = (lateral_canthus[1] - medial_canthus[1]) / (lateral_canthus[0] - medial_canthus[0])
        
        # Split points into upper and lower based on their position relative to the line
        for point in border_points:
            y_line = medial_canthus[1] + med_lat_line_slope * (point[0] - medial_canthus[0])
            if point[1] < y_line:
                upper_points.append(point)
            else:
                lower_points.append(point)
        
        return np.array(upper_points), np.array(lower_points)
   
    @staticmethod
    def fit_polynomial(points, degree=4):
        """
        Fits a polynomial of a given degree to the provided points.

        Parameters:
            points (np.array): Array of points (x, y) to fit a polynomial to.
            degree (int, optional): The degree of the polynomial to fit. Default is 4.

        Returns:
            tuple: A tuple containing:
                - coefficients (np.array): The coefficients of the fitted polynomial.
                - p (Polynomial): The fitted polynomial object.
        """
        if len(points) == 0:
            return None
        
        x = points[:, 0]
        y = points[:, 1]
        
        # Fit the polynomial to the points
        p = Polynomial.fit(x, y, degree)
        coefficients = p.convert().coef
        
        return coefficients, p

    @staticmethod
    def extract_border_points(mask):
        """
        Extracts border points from a binary mask by identifying the topmost and bottommost points for each column.

        Parameters:
            mask (np.array): Binary mask where the object of interest is marked as 1.

        Returns:
            np.array: An array of border points (x, y).
        """
        rows, cols = mask.shape
        border_points = []
        
        for col in range(cols):
            col_points = np.where(mask[:, col] > 0)[0]
            if col_points.size > 0:
                min_y = np.min(col_points)
                max_y = np.max(col_points)
                border_points.append([col, min_y])
                border_points.append([col, max_y])
        
        return np.array(border_points)
        
    @staticmethod
    def standardize_points(points, is_upper):
        """
        Standardizes points by translating them to the origin and optionally rotating them.

        Parameters:
            points (np.array): Array of points (x, y) to standardize.
            is_upper (bool): If True, rotates the points by 180 degrees.

        Returns:
            np.array: The standardized points.
        """
        # Find the tightest bounding box
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        # Translate the points
        points[:, 0] -= min_x
        points[:, 1] -= min_y

        if is_upper:
            # Rotate by 180 degrees
            points = np.dot(points, np.array([[-1, 0], [0, -1]]))
            # Translate again
            points[:, 0] += (max_x - min_x)
            points[:, 1] += (max_y - min_y)
            
        return points

    def get_sclera_key_points(self, sclera_mask, iris_centroid_x, direction='r'):
        """
        Extracts key points on the sclera, including the superior, inferior, medial, and lateral points.

        Parameters:
            sclera_mask (np.array): Binary mask of the sclera.
            iris_centroid_x (float): The x-coordinate of the iris centroid.
            direction (str, optional): The direction of the eye ('r' for right, 'l' for left). Default is 'r'.

        Returns:
            tuple: A tuple containing the superior, inferior, medial, and lateral points (x, y).
        """
        # Identify all y (rows) and x (columns) coordinates where the mask is non-zero
        column = sclera_mask[:, int(iris_centroid_x)]
        y_indices = np.where(column > 0)[0]
    
        # The superior point is the minimum y index, and the inferior point is the maximum y index
        superior_point_y = y_indices.min()
        inferior_point_y = y_indices.max()
        
        # Return points as (x, y) pairs
        superior_point = (iris_centroid_x, superior_point_y)
        inferior_point = (iris_centroid_x, inferior_point_y)
        
        y_coords, x_coords = np.where(sclera_mask > 0)
        
        if direction == 'r':
            lateral_point = (x_coords.min(), y_coords[x_coords.argmin()])  # Min x
            medial_point = (x_coords.max(), y_coords[x_coords.argmax()])  # Max x
        else:
            medial_point = (x_coords.min(), y_coords[x_coords.argmin()])  # Min x
            lateral_point = (x_coords.max(), y_coords[x_coords.argmax()])  # Max x  

        return superior_point, inferior_point, medial_point, lateral_point

    def sclera_points(self, sclera, iris_centroid, direction='l'):
        """
        Extracts key scleral points, splits them into upper and lower sets, and fits polynomials to each set.

        Parameters:
            sclera (np.array): Binary mask of the sclera.
            iris_centroid (tuple): The centroid of the iris (x, y).
            direction (str, optional): The direction of the eye ('l' for left, 'r' for right). Default is 'l'.

        Returns:
            tuple: A tuple containing:
                - sup (tuple): The superior point of the sclera.
                - inf (tuple): The inferior point of the sclera.
                - med (tuple): The medial point of the sclera.
                - lat (tuple): The lateral point of the sclera.
                - upper_poly (np.array): The standardized coefficients of the upper polynomial fit.
                - lower_poly (np.array): The standardized coefficients of the lower polynomial fit.
        """
        sup, inf, med, lat = Sclera.get_sclera_key_points(self, sclera, iris_centroid[0], direction=direction)
        
        # Identify border points
        border_points = self.extract_border_points(sclera)

        # Split mask into upper and lower sets
        upper, lower = self.split_upper_lower(border_points, med, lat)

        # Standardize lid positions
        upper = self.standardize_points(upper, is_upper=True)
        lower = self.standardize_points(lower, is_upper=False)

        # Fit polynomials and get coefficients
        upper_poly, _ = self.fit_polynomial(upper)
        lower_poly, _ = self.fit_polynomial(lower)

        scaler = StandardScaler()
        
        # Standardize coefficients
        upper_poly = scaler.fit_transform(upper_poly.reshape(-1, 1)).flatten()
        lower_poly = scaler.fit_transform(lower_poly.reshape(-1, 1)).flatten()

        return sup, inf, med, lat, upper_poly, lower_poly

    
class Brows:
    """
    A class for extracting key points on the eyebrows, including superior and inferior points relative to 
    the medial canthus, lateral canthus, and iris center.
    """

    def __init__(self):
        pass

    def _find_eyebrow_point(self, eyebrow, reference_x, reference_point, dir='inf'):
        """
        Finds the closest point on the eyebrow to a reference point in either the inferior or superior direction.

        Parameters:
            eyebrow (np.array): Binary mask of the eyebrow.
            reference_x (int): The x-coordinate to reference for finding the closest point.
            reference_point (tuple): The reference point (x, y) to compare against.
            dir (str, optional): The direction to search ('inf' for inferior, 'sup' for superior). Default is 'inf'.

        Returns:
            tuple: The coordinates of the closest point found on the eyebrow.
        """
        candidates = np.where(eyebrow[:, int(reference_point[0])] == 1)[0]
        if dir == 'inf':
            return ExtractorTools.find_closest_if_no_match(candidates, eyebrow, reference_point, 'inferior')
        elif dir == 'sup':
            return ExtractorTools.find_closest_if_no_match(candidates, eyebrow, reference_point, 'superior')
        
    def get_eyebrow_points(self, eyebrow, lateral_canthus, medial_canthus, iris_center, landmarks, laterality='l'):    
        """
        Extracts and assigns the lateral, central, and medial eyebrow points (and their superior counterparts)
        to the landmarks dictionary.

        Parameters:
            eyebrow (np.array): Binary mask of the eyebrow.
            lateral_canthus (tuple): Coordinates of the lateral canthus (x, y).
            medial_canthus (tuple): Coordinates of the medial canthus (x, y).
            iris_center (tuple): Coordinates of the iris center (x, y).
            landmarks (dict): Dictionary to store the calculated landmark points.
            laterality (str, optional): Indicates whether the points belong to the left ('l') or right ('r') eye. Default is 'l'.

        Returns:
            None
        """
        brow_error = np.array([0, 0])

        # Extract key eyebrow points
        lat_eyebrow = self._find_eyebrow_point(eyebrow, lateral_canthus[0], lateral_canthus)
        medial_eyebrow = self._find_eyebrow_point(eyebrow, medial_canthus[0], medial_canthus)
        center_eyebrow = self._find_eyebrow_point(eyebrow, int(iris_center[0]), iris_center)

        sup_lat_eyebrow = self._find_eyebrow_point(eyebrow, lateral_canthus[0], lateral_canthus, dir='sup')
        sup_medial_eyebrow = self._find_eyebrow_point(eyebrow, medial_canthus[0], medial_canthus, dir='sup')
        sup_center_eyebrow = self._find_eyebrow_point(eyebrow, int(iris_center[0]), iris_center, dir='sup')
        
        try:
            # Assign the points to the landmarks dictionary
            landmarks[f'{laterality}_lat_eyebrow'] = lat_eyebrow
            landmarks[f'{laterality}_center_eyebrow'] = center_eyebrow
            landmarks[f'{laterality}_medial_eyebrow'] = medial_eyebrow

            landmarks[f'sup_{laterality}_lat_eyebrow'] = sup_lat_eyebrow
            landmarks[f'sup_{laterality}_center_eyebrow'] = sup_center_eyebrow
            landmarks[f'sup_{laterality}_medial_eyebrow'] = sup_medial_eyebrow
    
        except:
            # Handle exceptions by setting default error values
            landmarks[f'{laterality}_lat_eyebrow'] = brow_error
            landmarks[f'{laterality}_center_eyebrow'] = brow_error
            landmarks[f'{laterality}_medial_eyebrow'] = brow_error
            
            landmarks[f'sup_{laterality}_lat_eyebrow'] = brow_error
            landmarks[f'sup_{laterality}_center_eyebrow'] = brow_error
            landmarks[f'sup_{laterality}_medial_eyebrow'] = brow_error
            raise Warning('Error identifying brow points. This image may have odd measurements on the output file as all brow points are being set to the origin. Does the person in this image have eyebrows?')



class EyeFeatureExtractor:
    """
    A class for extracting and organizing key anatomical features of the eye, including the iris, sclera, 
    and eyebrow, from segmented masks.
    """

    def __init__(self, detections_dict, crop_image):
        """
        Initializes the EyeFeatureExtractor class with segmentation masks and the cropped image.

        Parameters:
            detections_dict (dict): Dictionary containing various segmented masks.
            crop_image (PIL.Image): The cropped image from which features are extracted.
        """
        self.pupil_iris_getter = PupilIris()
        self.sclera_getter = Sclera()
        self.brow_getter = Brows()
        self.landmarks = {}
        self.image_size = (crop_image.size[1], crop_image.size[0])
        self.detections_dict = detections_dict
        self.crop_image = crop_image

    def _cluster(self):
        """
        Clusters the segmented masks into individual components for the right and left sclera, iris, 
        and brow, and assigns them to instance variables.
        """
        self.right_sclera_mask, self.left_sclera_mask, self.right_iris_mask, self.left_iris_mask, self.right_brow_mask, self.left_brow_mask \
                = ExtractorTools.get_all_clusters(self.detections_dict, self.crop_image)
    
    def _iris(self):
        """
        Extracts the iris center, diameter, superior, and inferior points for both the left and right eyes 
        and stores them in the landmarks dictionary.
        """
        right_iris_centroid, right_iris_diameter, right_iris_superior, right_iris_inferior \
            = self.pupil_iris_getter.get_iris_info(self.right_iris_mask, self.right_sclera_mask, self.image_size)
        
        left_iris_centroid, left_iris_diameter, left_iris_superior, left_iris_inferior \
            = self.pupil_iris_getter.get_iris_info(self.left_iris_mask, self.left_sclera_mask, self.image_size)

        # Store the extracted iris features in the landmarks dictionary
        self.landmarks['right_iris_centroid'] = right_iris_centroid
        self.landmarks['right_iris_diameter'] = right_iris_diameter
        self.landmarks['right_iris_superior'] = right_iris_superior
        self.landmarks['right_iris_inferior'] = right_iris_inferior

        self.landmarks['left_iris_centroid'] = left_iris_centroid
        self.landmarks['left_iris_diameter'] = left_iris_diameter
        self.landmarks['left_iris_superior'] = left_iris_superior
        self.landmarks['left_iris_inferior'] = left_iris_inferior

    def _sclera(self):
        """
        Extracts key scleral points, including superior, inferior, medial, and lateral canthus, and fits 
        polynomials to the upper and lower borders of the sclera for both the left and right eyes.
        """
        left_sclera_superior, left_sclera_inferior,  \
        left_medial_canthus, left_lateral_canthus, \
        l_upper_poly, l_lower_poly \
            = self.sclera_getter.sclera_points(self.left_sclera_mask, self.landmarks['left_iris_centroid'], direction='l')
        
        # Store the extracted scleral features in the landmarks dictionary
        self.landmarks['left_sclera_superior'] = left_sclera_superior
        self.landmarks['left_sclera_inferior'] = left_sclera_inferior
        self.landmarks['left_medial_canthus'] = left_medial_canthus
        self.landmarks['left_lateral_canthus'] = left_lateral_canthus
        self.landmarks['left_upper_poly'] = l_upper_poly
        self.landmarks['left_lower_poly'] = l_lower_poly

        right_sclera_superior, right_sclera_inferior, \
        right_medial_canthus, right_lateral_canthus, \
        r_upper_poly, r_lower_poly \
            = self.sclera_getter.sclera_points(self.right_sclera_mask, self.landmarks['right_iris_centroid'], direction='r')

        # Store the extracted scleral features in the landmarks dictionary
        self.landmarks['right_sclera_superior'] = right_sclera_superior
        self.landmarks['right_sclera_inferior'] = right_sclera_inferior
        self.landmarks['right_medial_canthus'] = right_medial_canthus
        self.landmarks['right_lateral_canthus'] = right_lateral_canthus
        self.landmarks['right_upper_poly'] = r_upper_poly
        self.landmarks['right_lower_poly'] = r_lower_poly
        
    def _eyebrow(self):
        """
        Extracts key points on the eyebrows for both the left and right eyes and stores them in the landmarks dictionary.
        """
        self.brow_getter.get_eyebrow_points(self.left_brow_mask, \
                                            self.landmarks['left_lateral_canthus'], \
                                            self.landmarks['left_medial_canthus'], self.landmarks['left_iris_centroid'], self.landmarks, laterality='l')

        self.brow_getter.get_eyebrow_points(self.right_brow_mask, \
                                            self.landmarks['right_lateral_canthus'], \
                                            self.landmarks['right_medial_canthus'], self.landmarks['right_iris_centroid'], self.landmarks, laterality='r')

    def extract_features(self):
        """
        Main method to extract all relevant anatomical features from the image, including clustering the masks, 
        extracting iris and sclera features, and extracting eyebrow points. Stores all results in the landmarks dictionary.

        Returns:
            dict: A dictionary containing all extracted landmarks for the left and right eyes.
        """
        self._cluster()
        try:
            self._iris()
            self._sclera()
        except:
            raise Exception('Iris or sclera segmentation did not produce a usable mask on the left or right side. Unable to continue. Try acquiring a better image, or better yet, contribute to the project! ;)')
        
        self._eyebrow()

        return self.landmarks




