import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image




        
# draw the lines on the face        
class LineAnnotator:
    def __init__(self, image, mask_dict, landmark_dict, image_name, measurements):
        self.image = image
        self.width, self.height = self.image.size
        self.masks = mask_dict
        self.landmarks = landmark_dict
        self.name = image_name
        self.measurements = measurements

    def _draw_line(self, point1, point2, color='black', linewidth=1, force_horizontal = False, linestyle = '-'):
        if force_horizontal:
            plt.plot([point1[0], point2[0]], [point2[1], point2[1]], color=color, linewidth=linewidth, linestyle = linestyle)
        else:        
            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color=color, linewidth=linewidth, linestyle = linestyle)
   
    def draw_landmarks(self):
        fig, ax = plt.subplots()
        
        ax.imshow(self.image)

        thickness_1 = 2
        thickness_2 = 3
        thickness_3 = 2
        
        brow_data = {
            'l_medial_eyebrow': ['left_medial_canthus', 'l_medial_eyebrow', 'purple', 2],
            'l_center_eyebrow': ['left_iris_centroid', 'l_center_eyebrow', 'purple', 2],
            'l_lat_eyebrow': ['left_lateral_canthus', 'l_lat_eyebrow', 'purple', 2],
            'r_medial_eyebrow': ['right_medial_canthus', 'r_medial_eyebrow', 'purple', 2],
            'r_center_eyebrow': ['right_iris_centroid', 'r_center_eyebrow', 'purple', 2],
            'r_lat_eyebrow': ['right_lateral_canthus', 'r_lat_eyebrow', 'purple', 2],
            
            'sup_l_medial_eyebrow': ['left_medial_canthus', 'sup_l_medial_eyebrow', 'black', 1],
            'sup_l_center_eyebrow': ['left_iris_centroid', 'sup_l_center_eyebrow', 'black', 1],
            'sup_l_lat_eyebrow': ['left_lateral_canthus', 'sup_l_lat_eyebrow', 'black', 1],
            'sup_r_medial_eyebrow': ['right_medial_canthus', 'sup_r_medial_eyebrow', 'black', 1],
            'sup_r_center_eyebrow': ['right_iris_centroid', 'sup_r_center_eyebrow', 'black', 1],
            'sup_r_lat_eyebrow': ['right_lateral_canthus', 'sup_r_lat_eyebrow', 'black', 1]
        }

        for key, (start, end, color, thickness) in brow_data.items():
            if not np.array_equal(self.landmarks[key], np.array([0, 0])):
                self._draw_line(self.landmarks[start], self.landmarks[end], color=color, linewidth=thickness)

        self._draw_line(self.landmarks['left_lateral_canthus'], self.landmarks['right_lateral_canthus'], color='darkgreen', linewidth=3.1, force_horizontal=True)  # OCD
        self._draw_line(self.landmarks['left_iris_centroid'], self.landmarks['right_iris_centroid'], color='red', linewidth=2.5, force_horizontal=True, linestyle='--')  # IPD
        self._draw_line(self.landmarks['left_medial_canthus'], self.landmarks['right_medial_canthus'], color='yellow', linewidth=2.5, force_horizontal=True)  # ICD
        self._draw_line(self.landmarks['left_medial_canthus'], self.landmarks['left_lateral_canthus'], color='deepskyblue', linewidth=thickness_1, force_horizontal=True)  # horiz fissure
        self._draw_line(self.landmarks['right_medial_canthus'], self.landmarks['right_lateral_canthus'], color='deepskyblue', linewidth=thickness_1, force_horizontal=True)  # horiz fissure
        
        scleral_show_data = {
            'left_SSS': ['left_sclera_superior', 'left_iris_superior', 'left_iris_centroid', 'lightblue'],
            'left_ISS': ['left_sclera_inferior', 'left_iris_inferior', 'left_iris_centroid', 'orange'],
            'right_SSS': ['right_sclera_superior', 'right_iris_superior', 'right_iris_centroid', 'lightblue'],
            'right_ISS': ['right_sclera_inferior', 'right_iris_inferior', 'right_iris_centroid', 'orange']
        }

        for key, (sclera, iris_dir, center, color) in scleral_show_data.items():
            if self.measurements[key] != 0:
                ax.plot([self.landmarks[sclera][0], self.landmarks[iris_dir][0]], [self.landmarks[sclera][1], self.landmarks[iris_dir][1]], color='blue', linewidth=thickness_3)
                ax.plot([self.landmarks[center][0], self.landmarks[iris_dir][0]], [self.landmarks[center][1], self.landmarks[iris_dir][1]], color=color, linewidth=thickness_3)
            else:
                ax.plot([self.landmarks[center][0], self.landmarks[sclera][0]], [self.landmarks[center][1], self.landmarks[sclera][1]], color=color, linewidth=thickness_3)


        # return annotated plt as numpy array
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_aspect('auto') 
        ax.axis('off')
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img_array = img_array.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)  
        img = Image.fromarray(img_array)
        img = img.resize((self.width, self.height), Image.NEAREST)
        img_array_resized = np.array(img)

        return img_array_resized
        
    def plot(self):
        img_annot = self.draw_landmarks()
        return img_annot
    
        

class Plotter:
    def __init__(self):
        pass
    
    def create_plots(self, image, mask_dict, landmark_dict, image_name, measurements):  
        line_annotator = LineAnnotator(image, mask_dict, landmark_dict, image_name,measurements)
        image_annot = line_annotator.plot()
        return image_annot


