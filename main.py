
from .segment_eyes import Tester
from .find_anatomy_from_masks import EyeFeatureExtractor
from .extract_periorbital_distances import EyeMetrics
from .periorbital_plotter import Plotter
from .pkg_utils import *

def run_periorbital_pipeline(img=None, img_path=None, full_face=False, model='combined', img_name='default', user_data_dict=None, user_segmentation_mask=None, user_merged_mask=None, just_segment=False):
    """
    Runs the periorbital segmentation and measurement pipeline.

    Parameters:
        img (PIL.Image, optional): Preloaded image object. Default is None.
        img_path (str, optional): Path to the image file to load. Default is None.
        full_face (bool, optional): Whether to crop the image to the full face. Default is False.
        model (str, optional): Model to use for segmentation ('combined', 'custom', etc.). Default is 'combined'.
        img_name (str, optional): Name of the image for identification purposes. Default is 'default'.
        user_data_dict (dict, optional): Dictionary with user-provided data for custom models. Default is None.
        user_segmentation_mask (np.array, optional): User-provided segmentation mask for custom models. Default is None.
        user_merged_mask (np.array, optional): User-provided merged mask for custom models. Default is None.
        just_segment (bool, optional): Flag to only return the semgentation results and not use measurement pipeline. Default is False. 

    Returns:
        dict: A dictionary containing:
            - 'pix_df' (DataFrame): DataFrame with pixel measurements.
            - 'mm_df' (DataFrame): DataFrame with measurements in millimeters.
            - 'segmentation' (np.array): Segmentation mask array.
            - 'merged' (np.array): Merged image array with segmentation overlays.
            - 'original' (PIL.Image): Resized original image.
            - 'annotated' (PIL.Image): Image annotated with landmarks and measurements.
    """
    # Confirm and load the image based on the provided path or object
    img, img_name = confirm_call(img_path, img, img_name, model)

    # Crop the image to the full face if the option is selected
    if full_face:
        img = crop_image(img)

    # If using a pre-trained or provided model, run the model on the image
    if model != 'custom':        
        test_obj = Tester(model_weights=model, image_name=img_name)
        predictions_dict, resize_img, segmentation_mask_array, merged_image_array = test_obj.test_single_image(img)
    else:
        # If using custom data, load the user-provided predictions and images
        predictions_dict = user_data_dict
        resize_img = img
        segmentation_mask_array = user_segmentation_mask
        merged_image_array = user_merged_mask
    
    # only do segmentation and not distance measruement 
    if just_segment:
        pix_df = None
        mm_df = None
        image_annot = None
        # Compile all results into a dictionary for easy access
        results = {
                'pix_df': pix_df,
                'mm_df' : mm_df,
                'segmentation' : segmentation_mask_array,
                'merged' : merged_image_array,
                'original' : resize_img,
                'annotated' : image_annot 
        }

        return results 
    
    else:
        # Extract anatomical landmarks from the predictions
        anatomy_grabber = EyeFeatureExtractor(predictions_dict, resize_img)
        landmarks = anatomy_grabber.extract_features()

        # Calculate periorbital measurements in pixels
        periorbital_calculator = EyeMetrics(landmarks, predictions_dict)
        measurements_pix = periorbital_calculator.run()

        # Convert pixel measurements to millimeters
        measurements_mm = convert_measurements(measurements_pix, landmarks)

        # Create annotated plots of the image with landmarks and measurements
        periorbital_plotter = Plotter()
        image_annot = periorbital_plotter.create_plots(resize_img, predictions_dict, landmarks, img_name, measurements_pix)

        # Clean and organize the measurements into DataFrames
        pix_df, mm_df = clean_measurements(measurements_pix, measurements_mm, landmarks)

        # Compile all results into a dictionary for easy access
        results = {
                'pix_df': pix_df,
                'mm_df' : mm_df,
                'segmentation' : segmentation_mask_array,
                'merged' : merged_image_array,
                'original' : resize_img,
                'annotated' : image_annot 
        }

        return results


if __name__ == '__main__':
    # Example usage of the periorbital pipeline
    results = run_periorbital_pipeline(img_path='8_cfd.jpg', full_face=False, model='cfd')
