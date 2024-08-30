
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pandas as pd
import os
from .pkg_utils import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mediapipe as mp 


def generate_label_plain(inputs):
    """
    Converts the output of a model into a plain label format by selecting the class with the highest score for each pixel.

    Parameters:
        inputs (torch.Tensor): A batch of model outputs with shape (batch_size, num_classes, height, width).

    Returns:
        np.array: A batch of label maps with shape (batch_size, height, width), where each pixel contains the predicted class.
    """
    pred_batch = []
    for input in inputs:
        input = input.view(1, 6, 256, 256)  # Reshape to match the expected input size
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)  # Get the class with the highest score for each pixel
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
            
    label_batch = []
    for p in pred_batch:
        label_batch.append(p.numpy())  # Convert to numpy array
                
    label_batch = np.array(label_batch)

    return label_batch

def extract_and_split_masks(combined_prediction):
    """
    Splits the combined prediction mask into separate masks for different anatomical structures and separates them into left and right sides.

    Parameters:
        combined_prediction (np.array): A combined prediction mask with shape (height, width), where each pixel is labeled with a class.

    Returns:
        dict: A dictionary containing left and right masks for each anatomical structure.
    """
    categories = {
        'sclera_orig': 2,
        'iris': 3,
        'brow': 1,
        'caruncle': 4,
        'lid': 5
    }
    midline_x = combined_prediction.shape[1] // 2
    masks_dict = {}
    for structure, value in categories.items():
        left_mask = np.where(combined_prediction == value, 1, 0)
        right_mask = np.where(combined_prediction == value, 1, 0)
        right_mask[:, midline_x:] = 0  # Clear the right side of the left mask
        left_mask[:, :midline_x] = 0   # Clear the left side of the right mask
        masks_dict[f'left_{structure}'] = left_mask
        masks_dict[f'right_{structure}'] = right_mask
    return masks_dict

def apply_color_map(mask, color_map):
    """
    Applies a color map to a binary mask, assigning specific colors to different class labels.

    Parameters:
        mask (np.array): A binary mask where each pixel is labeled with a class.
        color_map (dict): A dictionary mapping class labels to RGB color values.

    Returns:
        np.array: A colored mask with shape (height, width, 3).
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for value, color in color_map.items():
        colored_mask[mask == value] = color
    
    return colored_mask

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlays a colored mask onto an image with a specified transparency.

    Parameters:
        image (PIL.Image): The original image.
        mask (np.array): The mask to overlay, where each pixel is labeled with a class.
        alpha (float, optional): The transparency level of the overlay (0 = fully transparent, 1 = fully opaque). Default is 0.5.

    Returns:
        PIL.Image: The image with the colored mask overlay.
    """
    color_map = {
        0: [255, 0, 0],    # Red for background (or class 0)
        1: [0, 255, 0],    # Green for class 1
        2: [0, 0, 255],    # Blue for class 2
        3: [255, 255, 0],  # Yellow for class 3
        4: [255, 0, 255],  # Magenta for class 4
        5: [0, 255, 255],  # Cyan for class 5
    }
    colored_mask = apply_color_map(mask, color_map)
    image_np = np.array(image).astype(float)
    overlay = np.where(mask[..., None] > 0, image_np * (1 - alpha) + colored_mask * alpha, image_np)

    return Image.fromarray(overlay.astype(np.uint8))

def convert_measurements(measurements, landmarks):
    """
    Converts pixel measurements to millimeter measurements using a calibration factor based on iris diameter.

    Parameters:
        measurements (dict): A dictionary containing pixel measurements.
        landmarks (dict): A dictionary containing landmark points, including iris diameter.

    Returns:
        dict: A dictionary containing the converted measurements in millimeters.
    """
    # Calibration factor based on average iris diameter
    cf = 11.71 / ((landmarks['right_iris_diameter'] + landmarks['left_iris_diameter']) / 2)

    converted_measurements = {}
    special_keys = ['left_upper_poly', 'left_lower_poly', 'right_upper_poly', 'right_lower_poly',\
                     'left_canthal_tilt', 'right_canthal_tilt','right_vd_plot_point','left_vd_plot_point']

    for key, value in measurements.items():
        if key not in special_keys:
            converted_measurements[key] = value * cf  # Convert measurements to millimeters
        else:
            converted_measurements[key] = value  # Keep special measurements unchanged

    return converted_measurements

def clean_measurements(pix_measurements, mm_measurements, landmarks):
    """
    Cleans and prepares pixel and millimeter measurements for output, ensuring iris diameters are included.

    Parameters:
        pix_measurements (dict): A dictionary containing pixel measurements.
        mm_measurements (dict): A dictionary containing millimeter measurements.
        landmarks (dict): A dictionary containing landmark points, including iris diameter.

    Returns:
        tuple: Two pandas DataFrames, one for pixel measurements and one for millimeter measurements.
    """
    # Add iris diameters to the measurements
    pix_measurements['left_iris_diameter'] = landmarks['left_iris_diameter']
    pix_measurements['right_iris_diameter'] = landmarks['right_iris_diameter']
    
    mm_measurements['left_iris_diameter'] = landmarks['left_iris_diameter']
    mm_measurements['right_iris_diameter'] = landmarks['right_iris_diameter']
    
    # Remove unnecessary plot points from the measurements
    pix_measurements.pop('left_vd_plot_point', None)
    pix_measurements.pop('right_vd_plot_point', None)
    
    mm_measurements.pop('left_vd_plot_point', None)
    mm_measurements.pop('right_vd_plot_point', None)

    # Convert the measurements to DataFrames
    pix_df = pd.DataFrame(list(pix_measurements.items()), columns=['Measurement', 'Pixel Value'])
    mm_df = pd.DataFrame(list(mm_measurements.items()), columns=['Measurement', 'MM Value'])
    
    return pix_df, mm_df

class ResizeAndPad:
    """
    A class to resize an image to a specified width and pad it to a specified height, maintaining aspect ratio.
    """

    def __init__(self, output_size=(512, 512), fill=0, padding_mode='constant'):
        """
        Initializes the ResizeAndPad class with the desired output size, fill color, and padding mode.

        Parameters:
            output_size (tuple, optional): The desired output size (width, height). Default is (512, 512).
            fill (int or tuple, optional): The fill color for padding. Default is 0 (black).
            padding_mode (str, optional): The padding mode ('constant', 'edge', etc.). Default is 'constant'.
        """
        self.output_size = output_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Resizes and pads the input image to the specified output size.

        Parameters:
            img (PIL.Image): The input image to resize and pad.

        Returns:
            PIL.Image: The resized and padded image.
        """
        original_width, original_height = img.size
        
        # Calculate the new height to maintain aspect ratio
        new_height = int(original_height * (self.output_size[0] / original_width))
        
        # Resize the image
        img = img.resize((self.output_size[0], new_height), Image.NEAREST)

        # Calculate padding for top and bottom
        padding_top = (self.output_size[1] - new_height) // 2
        padding_bottom = self.output_size[1] - new_height - padding_top

        # Add padding to the image
        img = ImageOps.expand(img, (0, padding_top, 0, padding_bottom), fill=self.fill)
        
        return img

def transformer(dynamic_resize_and_pad, totensor, normalize):
    """
    Creates a composition of image transformations based on the provided flags.

    Parameters:
        dynamic_resize_and_pad (bool): Whether to apply dynamic resizing and padding.
        totensor (bool): Whether to convert the image to a tensor.
        normalize (bool): Whether to normalize the image.

    Returns:
        torchvision.transforms.Compose: A composition of the selected transformations.
    """
    options = []
    if dynamic_resize_and_pad:
        options.append(ResizeAndPad(output_size=(512, 512)))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(options)

def crop_and_resize(img):
    """
    Crops the input image into left and right halves and resizes each to 256x256.

    Parameters:
        img (PIL.Image): The input image to crop and resize.

    Returns:
        tuple: Two PIL.Images representing the left and right halves resized to 256x256.
    """
    mid = img.width // 2
    left_half = img.crop((0, 0, mid, img.height))

    right_half_start = mid if img.width % 2 == 0 else mid + 1
    right_half = img.crop((right_half_start, 0, img.width, img.height))

    left_resized = left_half.resize((256, 256))
    right_resized = right_half.resize((256, 256))
    return left_resized, right_resized

def transform_img_split(resize, totensor, normalize):
    """
    Creates a composition of image transformations including splitting the image into two halves.

    Parameters:
        resize (bool): Whether to resize the image halves.
        totensor (bool): Whether to convert the image halves to tensors.
        normalize (bool): Whether to normalize the image halves.

    Returns:
        torchvision.transforms.Compose: A composition of the selected transformations.
    """
    options = []
    if resize:
        options.append(transforms.Lambda(crop_and_resize))
    if totensor:
        options.append(transforms.Lambda(lambda imgs: (transforms.ToTensor()(imgs[0]), transforms.ToTensor()(imgs[1]))))
    if normalize:
        options.append(transforms.Lambda(lambda imgs: (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[0]), 
                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[1]))))
    transform = transforms.Compose(options)
    return transform

def confirm_call(img_path, img, img_name, model):
    """
    Confirms and loads the image from a given path or an existing PIL image.

    Parameters:
        img_path (str): The path to the image file.
        img (PIL.Image or np.ndarray): The image object or numpy array.
        img_name (str): The name of the image.
        model (str): The model name to check for validity.

    Returns:
        tuple: The loaded PIL image and the image name.
    """
    model_choices = ['custom', 'combine', 'cfd', 'celeb']
    
    if model not in model_choices:
        raise ValueError(f"Invalid model selection. Choose from: {', '.join(model_choices)}")
    
    if img is not None:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            raise TypeError("img must be a PIL Image or a numpy array.")

    if img is None and img_path is not None:
        img = Image.open(img_path)    
        img_name = os.path.splitext(os.path.basename(img_path))[0]
    else:
        raise Exception('No image or image path provided')

    if img_name == 'default':
        raise Warning('No image name provided. Image name is set to "default" ')
    
    return img, img_name
    
class MP:
    """
    A class to use MediaPipe FaceMesh for facial landmark detection on images.
    """

    def __init__(self):
        """
        Initializes the MP class with MediaPipe solutions for drawing and face mesh.
        """
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFacemesh = mp.solutions.face_mesh
    
    def map(self, img):
        """
        Maps facial landmarks from an image using MediaPipe FaceMesh.

        Parameters:
            img (np.array): The input image as a numpy array.

        Returns:
            list: A list of coordinates (x, y) for each detected facial landmark.
        """
        self.faceMesh = self.mpFacemesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
        drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        results = self.faceMesh.process(img)
        id_list = []
        if results.multi_face_landmarks:
            for landmark in results.multi_face_landmarks:
                for id, lm in enumerate(landmark.landmark):
                    [ih, iw, ic] = img.shape
                    px, py, pz = int(lm.x * iw), int(lm.y * ih), (lm.z * iw)
                    append = [px, py]
                    id_list.append(append)
        return id_list
    

def crop_image(image):
    """
    Crops the input image to focus on the eye region based on detected facial landmarks.

    Parameters:
        image (PIL.Image or np.array): The input image to crop.

    Returns:
        PIL.Image: The cropped image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    facemesher = MP()
    fm = facemesher.map(image)

    # Identify the ROI based on specific facial landmarks
    y_left = fm[345]
    y_right = fm[104]
    
    x_right = fm[139]
    x_left = fm[383]

    roi_top_right = [x_right[0], y_right[1]]
    roi_bottom_left = [x_left[0], y_left[1]]

    # Crop the image to the identified ROI
    cropped_image = image[int(roi_top_right[1]):int(roi_bottom_left[1]), int(roi_top_right[0]):int(roi_bottom_left[0])]
    cropped_image_pil = Image.fromarray(cropped_image)

    return cropped_image_pil
