import numpy as np
import math

class AnatomyTools:
    """
    A collection of static methods for calculating distances, angles, and other geometric properties 
    related to anatomical landmarks.
    """

    @staticmethod
    def distance(point1, point2, stack=False):
        """
        Calculates the Euclidean distance between two points.

        Parameters:
            point1 (array-like): The first point (x, y).
            point2 (array-like): The second point (x, y).
            stack (bool, optional): If True, uses the x-coordinate of point1 and the y-coordinate of point2 
                                    to calculate the distance. Default is False.

        Returns:
            float: The Euclidean distance between the two points.
        """
        if stack:
            return np.linalg.norm(np.array([point1[0], point2[1]]) - np.array(point2)) 
        else:
            return np.linalg.norm(np.array(point1) - np.array(point2))

    @staticmethod
    def conditional_distance(cond, pointA, pointB, pointC=None):
        """
        Calculates a distance based on a condition. If the condition is true, it returns the distance between 
        pointA and pointB. If the condition is false, it returns the distance between pointA and pointC, 
        along with the distance between pointB and pointC.

        Parameters:
            cond (bool): Condition to decide which distance to calculate.
            pointA (array-like): The first point (x, y).
            pointB (array-like): The second point (x, y).
            pointC (array-like, optional): The third point (x, y) used if cond is False. Default is None.

        Returns:
            tuple: A tuple containing the primary distance and a secondary distance (0 if cond is True).
        """
        if cond:
            return AnatomyTools.distance(pointA, pointB), 0
        else:
            return AnatomyTools.distance(pointA, pointC), AnatomyTools.distance(pointB, pointC)

    @staticmethod
    def dot(vA, vB):
        """
        Calculates the dot product of two vectors.

        Parameters:
            vA (array-like): The first vector (x, y).
            vB (array-like): The second vector (x, y).

        Returns:
            float: The dot product of the two vectors.
        """
        return vA[0] * vB[0] + vA[1] * vB[1]

    @staticmethod
    def ang(lineA, lineB):
        """
        Calculates the angle between two lines in radians.

        Parameters:
            lineA (list): A list of two points defining the first line [(x1, y1), (x2, y2)].
            lineB (list): A list of two points defining the second line [(x1, y1), (x2, y2)].

        Returns:
            float: The angle between the two lines in radians.
        """
        # Get vector form of the lines
        vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
        vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]

        # Calculate the dot product of the vectors
        dot_prod = AnatomyTools.dot(vA, vB)

        # Calculate magnitudes of the vectors
        magA = AnatomyTools.dot(vA, vA) ** 0.5
        magB = AnatomyTools.dot(vB, vB) ** 0.5

        # Calculate the cosine of the angle between the vectors
        cos_ = dot_prod / magA / magB

        # Calculate the angle in radians
        angle = math.acos(cos_)

        return angle


class EyeMetrics:
    """
    Class to calculate various anatomical measurements from landmarks and segmentation masks.

    Attributes:
        landmarks (dict): Dictionary of key facial landmarks.
        mask_dict (dict): Dictionary of segmentation masks.
        measurements (dict): Dictionary to store calculated measurements.
    """

    def __init__(self, landmarks, mask_dict):
        """
        Initializes the EyeMetrics class with landmarks and segmentation masks.

        Parameters:
            landmarks (dict): Dictionary of key facial landmarks.
            mask_dict (dict): Dictionary of segmentation masks.
        """
        self.landmarks = landmarks
        self.mask_dict = mask_dict
        self.measurements = {}

    def horiz_fissure(self):
        """
        Calculates the horizontal palpebral fissure length (distance between the medial and lateral canthus)
        for both the left and right eyes.
        """
        left_horiz_pf = AnatomyTools.distance(self.landmarks['left_medial_canthus'], self.landmarks['left_lateral_canthus'], True)
        right_horiz_pf = AnatomyTools.distance(self.landmarks['right_medial_canthus'], self.landmarks['right_lateral_canthus'], True)
        self.measurements['left_horiz_fissure'] = left_horiz_pf
        self.measurements['right_horiz_fissure'] = right_horiz_pf

    def brow_heights(self):
        """
        Calculates the vertical distance between the medial, central, and lateral canthus to the corresponding points
        on the eyebrow for both the left and right sides, including the superior eyebrow heights.
        """
        # Left brow heights
        left_mc_bh = AnatomyTools.distance(self.landmarks['left_medial_canthus'], self.landmarks['l_medial_eyebrow'])
        left_central_bh = AnatomyTools.distance(self.landmarks['left_iris_centroid'], self.landmarks['l_center_eyebrow'])
        left_lc_bh = AnatomyTools.distance(self.landmarks['left_lateral_canthus'], self.landmarks['l_lat_eyebrow'])

        # Right brow heights
        right_mc_bh = AnatomyTools.distance(self.landmarks['right_medial_canthus'], self.landmarks['r_medial_eyebrow'])
        right_central_bh = AnatomyTools.distance(self.landmarks['right_iris_centroid'], self.landmarks['r_center_eyebrow'])
        right_lc_bh = AnatomyTools.distance(self.landmarks['right_lateral_canthus'], self.landmarks['r_lat_eyebrow'])

        # Superior left brow heights
        sup_left_mc_bh = AnatomyTools.distance(self.landmarks['left_medial_canthus'], self.landmarks['sup_l_medial_eyebrow'])
        sup_left_central_bh = AnatomyTools.distance(self.landmarks['left_iris_centroid'], self.landmarks['sup_l_center_eyebrow'])
        sup_left_lc_bh = AnatomyTools.distance(self.landmarks['left_lateral_canthus'], self.landmarks['sup_l_lat_eyebrow'])

        # Superior right brow heights
        sup_right_mc_bh = AnatomyTools.distance(self.landmarks['right_medial_canthus'], self.landmarks['sup_r_medial_eyebrow'])
        sup_right_central_bh = AnatomyTools.distance(self.landmarks['right_iris_centroid'], self.landmarks['sup_r_center_eyebrow'])
        sup_right_lc_bh = AnatomyTools.distance(self.landmarks['right_lateral_canthus'], self.landmarks['sup_r_lat_eyebrow'])

        # Store the measurements in the dictionary
        self.measurements['sup_left_medial_bh'] = sup_left_mc_bh
        self.measurements['sup_left_central_bh'] = sup_left_central_bh
        self.measurements['sup_left_lateral_bh'] = sup_left_lc_bh

        self.measurements['sup_right_medial_bh'] = sup_right_mc_bh
        self.measurements['sup_right_central_bh'] = sup_right_central_bh
        self.measurements['sup_right_lateral_bh'] = sup_right_lc_bh

        self.measurements['left_medial_bh'] = left_mc_bh
        self.measurements['left_central_bh'] = left_central_bh
        self.measurements['left_lateral_bh'] = left_lc_bh

        self.measurements['right_medial_bh'] = right_mc_bh
        self.measurements['right_central_bh'] = right_central_bh
        self.measurements['right_lateral_bh'] = right_lc_bh

    def scleral_show(self):
        """
        Calculates the margin reflex distance (MRD) and scleral show (SSS and ISS) measurements.
        """
        # Left eye measurements
        l_mrd_1, left_SSS = AnatomyTools.conditional_distance((self.landmarks['left_sclera_superior'][1] > self.landmarks['left_iris_superior'][1]), self.landmarks['left_iris_centroid'], self.landmarks['left_sclera_superior'], self.landmarks['left_iris_superior'])
        l_mrd_2, left_ISS = AnatomyTools.conditional_distance((self.landmarks['left_sclera_inferior'][1] < self.landmarks['left_iris_inferior'][1]), self.landmarks['left_iris_centroid'], self.landmarks['left_sclera_inferior'], self.landmarks['left_iris_inferior'])
        l_mrd_1 = l_mrd_1 + left_SSS
        l_mrd_2 = l_mrd_2 + left_ISS

        # Right eye measurements
        r_mrd_1, right_SSS = AnatomyTools.conditional_distance((self.landmarks['right_sclera_superior'][1] > self.landmarks['right_iris_superior'][1]) ,self.landmarks['right_iris_centroid'], self.landmarks['right_sclera_superior'], self.landmarks['right_iris_superior'])
        r_mrd_2, right_ISS = AnatomyTools.conditional_distance((self.landmarks['right_sclera_inferior'][1] < self.landmarks['right_iris_inferior'][1]), self.landmarks['right_iris_centroid'], self.landmarks['right_sclera_inferior'], self.landmarks['right_iris_inferior'])
        r_mrd_1 = r_mrd_1 + right_SSS
        r_mrd_2 = r_mrd_2 + right_ISS
        
        # Store the measurements in the dictionary
        self.measurements['left_mrd_1'] = l_mrd_1
        self.measurements['left_SSS'] = left_SSS
        self.measurements['left_mrd_2'] = l_mrd_2
        self.measurements['left_ISS'] = left_ISS
        self.measurements['right_mrd_1'] = r_mrd_1
        self.measurements['right_SSS'] = right_SSS
        self.measurements['right_mrd_2'] = r_mrd_2
        self.measurements['right_ISS'] = right_ISS
        
        # Vertical palpebral fissure (distance between the superior and inferior sclera)
        r_vpf = AnatomyTools.distance(self.landmarks['right_sclera_superior'], self.landmarks['right_sclera_inferior'])
        l_vpf = AnatomyTools.distance(self.landmarks['left_sclera_superior'], self.landmarks['left_sclera_inferior'])
        
        self.measurements['right_vert_pf'] = r_vpf
        self.measurements['left_vert_pf'] = l_vpf

    def canthal_height(self):
        """
        Calculates the canthal heights (vertical distance from iris line to canthus)
        for both medial and lateral canthus for both eyes.
        """
        # Extract canthus landmarks
        rmc = self.landmarks['right_medial_canthus']
        lmc = self.landmarks['left_medial_canthus']
        rlc = self.landmarks['right_lateral_canthus']
        llc = self.landmarks['left_lateral_canthus']
        
        # Iris center points
        r_iris_center = self.landmarks['right_iris_centroid']
        l_iris_center = self.landmarks['left_iris_centroid']

        # Define the iris line (line connecting the centers of the irises)
        iris_line_slope = (r_iris_center[1] - l_iris_center[1]) / (r_iris_center[0] - l_iris_center[0])
        iris_line_intercept = l_iris_center[1] - iris_line_slope * l_iris_center[0]

        def shortest_distance(point, slope, intercept):
            """
            Calculates the vertical distance from a point to a line defined by slope and intercept.

            Parameters:
                point (tuple): The (x, y) coordinates of the point.
                slope (float): The slope of the line.
                intercept (float): The y-intercept of the line.

            Returns:
                float: The vertical distance from the point to the line.
            """
            x, y = point
            y_line = slope * x + intercept
            distance = abs(y - y_line)
            return distance

        # Calculate the canthal heights for each canthus
        r_mch = shortest_distance(rmc, iris_line_slope, iris_line_intercept)
        r_lch = shortest_distance(rlc, iris_line_slope, iris_line_intercept)
        l_mch = shortest_distance(lmc, iris_line_slope, iris_line_intercept)
        l_lch = shortest_distance(llc, iris_line_slope, iris_line_intercept)

        # Store the measurements in the dictionary
        self.measurements['r_med_canthal_height'] = r_mch
        self.measurements['r_lat_canthal_height'] = r_lch
        self.measurements['l_med_canthal_height'] = l_mch
        self.measurements['l_lat_canthal_height'] = l_lch
                                
    def icd(self):
        """
        Calculates the intercanthal distance (ICD), interpupillary distance (IPD),
        and outer canthal distance (OCD) between the left and right eyes.
        """
        icd = AnatomyTools.distance(self.landmarks['left_medial_canthus'], self.landmarks['right_medial_canthus'], True)
        ipd = AnatomyTools.distance(self.landmarks['left_iris_centroid'], self.landmarks['right_iris_centroid'], True)
        ocd = AnatomyTools.distance(self.landmarks['left_lateral_canthus'], self.landmarks['right_lateral_canthus'], True)

        # Store the measurements in the dictionary
        self.measurements['icd'] = icd
        self.measurements['ipd'] = ipd
        self.measurements['ocd'] = ocd
         
    def canthal_tilt(self):
        """
        Calculates the canthal tilt for both eyes, which is the angle between the medial and lateral canthus
        relative to a horizontal line through the midpoint of the irises.
        """
        # Calculate the midpoint between the left and right iris centroids
        left_iris = self.landmarks['left_iris_centroid']
        right_iris = self.landmarks['right_iris_centroid']
        midline = [(left_iris[0] + right_iris[0]) / 2, (left_iris[1] + right_iris[1]) / 2]

        # Calculate left eye canthal tilt
        lmc_llc = [self.landmarks['left_medial_canthus'], self.landmarks['left_lateral_canthus']]
        mid_lmc = [[midline[0], self.landmarks['left_medial_canthus'][1]], self.landmarks['left_medial_canthus']]
        lct_rad = AnatomyTools.ang(lmc_llc, mid_lmc)
        lct_deg = math.degrees(lct_rad)

        # Calculate right eye canthal tilt
        rmc_rlc = [self.landmarks['right_medial_canthus'], self.landmarks['right_lateral_canthus']]
        mid_rmc = [self.landmarks['right_medial_canthus'], [midline[0], self.landmarks['right_medial_canthus'][1]]]
        rct_rad = AnatomyTools.ang(mid_rmc, rmc_rlc)
        rct_deg = math.degrees(rct_rad)
        rct_deg = 180 - rct_deg  # Subtract to get acute angle

        # Store the measurements in the dictionary
        self.measurements['left_canthal_tilt'] = lct_deg
        self.measurements['right_canthal_tilt'] = rct_deg

    def vert_dystop(self):
        """
        Calculates the vertical dystopia, which is the vertical misalignment between the left and right irises.
        """
        l_mc = [self.landmarks['left_iris_centroid'][0], self.landmarks['left_iris_centroid'][1]]
        r_mc = [self.landmarks['left_iris_centroid'][0], self.landmarks['right_iris_centroid'][1]]

        vert_dystop = AnatomyTools.distance(l_mc, r_mc)

        # Store the measurements in the dictionary
        self.measurements['vert_dystopia'] = vert_dystop
        self.measurements['left_vd_plot_point'] = l_mc
        self.measurements['right_vd_plot_point'] = r_mc

    def scleral_area(self):
        """
        Calculates the ratio of scleral area to iris area for both eyes.
        """
        # Calculate areas for the right eye
        right_iris_area = np.sum(self.mask_dict['right_iris'])
        right_sclera_area = np.sum(self.mask_dict['right_sclera_orig'])

        # Calculate areas for the left eye
        left_iris_area = np.sum(self.mask_dict['left_iris'])
        left_sclera_area = np.sum(self.mask_dict['left_sclera_orig'])
        
        # Calculate scleral area ratio (scleral area / iris area) for both eyes
        right_area = right_sclera_area / right_iris_area if right_iris_area > 0 else 0
        left_area = left_sclera_area / left_iris_area if left_iris_area > 0 else 0

        # Store the measurements in the dictionary
        self.measurements['right_scleral_area'] = right_area
        self.measurements['left_scleral_area'] = left_area
    
    def polynomial_update(self):
        """
        Updates the polynomial fit of the upper and lower eyelids for both eyes.
        """
        self.measurements['right_upper_poly'] = list(self.landmarks['right_upper_poly'])
        self.measurements['right_lower_poly'] = list(self.landmarks['right_lower_poly'])
        
        self.measurements['left_upper_poly'] = list(self.landmarks['left_upper_poly'])
        self.measurements['left_lower_poly'] = list(self.landmarks['left_lower_poly'])
            
    def run(self):
        """
        Runs all the measurement functions in the EyeMetrics class and returns the collected measurements.

        Returns:
            dict: Dictionary containing all the calculated measurements.
        """
        self.horiz_fissure()
        try:
            self.brow_heights()
        except KeyError:
            print('Eyebrow measurement skipped due to missing landmarks.')
            pass
        self.scleral_show()
        self.canthal_height()
        self.icd()
        self.canthal_tilt()
        self.vert_dystop()
        self.scleral_area()
        self.polynomial_update()

        return self.measurements
