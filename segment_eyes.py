import torch
import numpy as np
import torch.nn as nn
import cv2
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import os
from .pkg_utils import *
from huggingface_hub import hf_hub_download

class Tester(object):
    """
    A class for testing a pre-trained DeepLabV3 model on a single image, specifically designed 
    for periorbital segmentation tasks.

    Attributes:
        device (torch.device): The device to run the model on (CPU or GPU).
        model (torch.nn.Module): The DeepLabV3 model with a custom classifier for segmentation.
        name (str): The name of the image being processed.
    """

    def __init__(self, model_weights, image_name):
        """
        Initializes the Tester class, loads the model weights, and prepares the model for evaluation.

        Parameters:
            model_weights (str): The name of the model weights file to load.
            image_name (str): The name of the image being processed.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the DeepLabV3 model with default pretrained weights
        self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        
        # Replace the classifier with a custom one for the specific segmentation task
        self.model.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 6, kernel_size=(1, 1), stride=(1, 1))
        )
        
        self.model.to(self.device)

        # Download the model weights from the Hugging Face Hub
        model_weights = hf_hub_download(repo_id="grnahass/periorbital_segmentation", filename=model_weights + '.pth')
        
        # Load the model weights
        self.model.load_state_dict(torch.load(model_weights))
        self.model.eval()  # Set the model to evaluation mode
        self.name = image_name

    def test_single_image(self, image):
        """
        Tests the model on a single image, applying necessary transformations and generating predictions.

        Parameters:
            image (PIL.Image): The image to be segmented.

        Returns:
            tuple: A tuple containing:
                - data_dict (dict): The extracted masks for different anatomical regions.
                - transform_gt_plot (PIL.Image): The transformed image used for ground truth plotting.
                - combined_prediction (np.array): The predicted segmentation mask.
                - merged_image (PIL.Image): The original image with the segmentation overlay.
        """
        # Apply transformations to the image for model input and plotting
        transform = transform_img_split(resize=True, totensor=True, normalize=True)
        transform_gt_plot = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False)
        
        # Get the original image size
        original_size = image.size
        
        # Transform the image for left and right eye segmentation
        l_img, r_img = transform(image)
        
        # Get the combined prediction by splitting the face and processing each half
        combined_prediction = self.predict_split_face([l_img], [r_img], original_size, transform_gt_plot)
        
        # Overlay the predicted mask on the original image
        merged_image = overlay_mask_on_image(transform_gt_plot(image), combined_prediction, alpha=0.5)
        
        # Extract and split the masks for different anatomical features
        data_dict = extract_and_split_masks(combined_prediction)

        return data_dict, transform_gt_plot(image), combined_prediction, merged_image

    def predict_split_face(self, l_imgs, r_imgs, original_size, transform_plotting):
        """
        Predicts the segmentation masks for the left and right halves of the face, 
        stitches them together, and resizes the result to match the original image size.

        Parameters:
            l_imgs (list): A list containing the transformed left side image tensor.
            r_imgs (list): A list containing the transformed right side image tensor.
            original_size (tuple): The original image size (width, height).
            transform_plotting (callable): A transformation function to apply after stitching the prediction.

        Returns:
            np.array: The stitched and resized prediction mask.
        """
        # Stack images into a batch
        l_imgs = torch.stack(l_imgs) 
        r_imgs = torch.stack(r_imgs) 

        # Move images to the appropriate device (CPU/GPU)
        l_imgs = l_imgs.to(self.device)
        r_imgs = r_imgs.to(self.device)

        # Predict segmentation masks for left and right images
        l_labels_predict = self.model(l_imgs)['out'] 
        r_labels_predict = self.model(r_imgs)['out']  

        # Generate plain label masks (non-probabilistic)
        l_labels_predict_plain = generate_label_plain(l_labels_predict)
        r_labels_predict_plain = generate_label_plain(r_labels_predict)

        labels_predict_plain = []

        # Stitch the left and right predictions together
        for idx, (left_pred, right_pred) in enumerate(zip(l_labels_predict_plain, r_labels_predict_plain)):
            original_width, original_height = original_size
            mid = original_width // 2
            
            # Resize the left and right predictions to match their respective halves of the original image
            left_pred_resized = cv2.resize(left_pred, (mid, original_height), interpolation=cv2.INTER_NEAREST)
            right_pred_resized = cv2.resize(right_pred, (original_width - mid, original_height), interpolation=cv2.INTER_NEAREST)

            # Create an empty stitched image and place the left and right predictions
            stitched = np.zeros((original_height, original_width), dtype=np.uint8)
            stitched[:, :mid] = left_pred_resized
            stitched[:, mid:] = right_pred_resized
            
            # Apply transformation for plotting and add to the list of predictions
            resized_stitched = transform_plotting(Image.fromarray(stitched))
            labels_predict_plain.append(np.array(resized_stitched))

        # Return the stitched and resized prediction mask
        return labels_predict_plain[0]
