"""
This module provides functions and classes for generating saliency maps using 
various CAM methods (e.g., GradCAM) applied to a YOLO model's outputs. The code 
includes functionality for image preprocessing, setting up activations and gradients 
hooks, computing CAM weights, and visualizing the generated CAMs. The code also supports 
processing images in batches.

Note:
  - Replace any file paths (e.g., '/path/to/your/images', '/path/to/your/model/weights') 
    with generic placeholders to avoid exposing personal information.
  - The original logic of the code is preserved.
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil, sys, yaml
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression

from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.base_cam import BaseCAM

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def letterbox(im, new_shape=(800, 800), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad the image to meet the expected input dimensions while preserving aspect ratio.
    
    Args:
        im: Input image.
        new_shape (tuple): Desired shape.
        color (tuple): Padding color.
        auto (bool): Automatically determine padding.
        scaleFill (bool): Scale image to fill new shape (ignoring aspect ratio).
        scaleup (bool): Allow up-scaling of the image.
        stride (int): Stride for padding adjustment.
        
    Returns:
        im: Resized and padded image.
        ratio: Scaling ratio.
        (dw, dh): Width and height padding.
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        """
        Store activations and gradients from the target layers via hooks.
        
        Args:
            model: The model to inspect.
            target_layers (list): List of layers to register hooks on.
            reshape_transform: Function to reshape activations, if needed.
        """
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        """Save the activation outputs from the forward pass (prints shape only in debug mode)."""
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())
        
        # Print activation shape only in debug mode
        if hasattr(self, 'debug') and self.debug:
            print(f"Activation shape: {activation.shape}")

    def save_gradient(self, module, grad_input, grad_output):
        """Save the gradients from the backward pass and print their shape."""
        if not hasattr(grad_output[0], "requires_grad") or not grad_output[0].requires_grad:
            return

        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients.append(grad.cpu().detach())

        print(f"Gradient shape: {grad.shape}")

    def __call__(self, x):
        """
        Reset stored activations and gradients, then perform a forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            model_output: Output from the model.
        """
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        print("Model output:", model_output[0].shape)

        return model_output

    def release(self):
        """Remove all hook handles to free resources."""
        for handle in self.handles:
            handle.remove()


class GradCAM(BaseCAM):
    def get_cam_weights(self, input_tensor, target_layer, activations, gradients, target_category):
        """
        Compute the weights for CAM by averaging the gradients.
        
        Args:
            input_tensor: Input tensor to the model.
            target_layer: The target layer.
            activations: Activations of the target layer.
            gradients: Gradients from the target layer.
            target_category: Target category index (unused here).
            
        Returns:
            weights: The computed weight values.
        """
        # Add dimension checking and handling
        if len(gradients.shape) not in [2, 4]:
            raise ValueError(f"Gradients should have 2 or 4 dimensions, got {len(gradients.shape)}")
            
        # Choose aggregation method based on dimensionality
        if len(gradients.shape) == 4:
            weights = np.mean(gradients, axis=(2, 3))
        else:
            weights = np.mean(gradients, axis=1)
            
        return weights


class yolo_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        """
        Define a target module to extract relevant outputs based on confidence and bounding box data.
        
        Args:
            ouput_type (str): Type of output to extract ('class', 'box', or 'all').
            conf (float): Confidence threshold.
            ratio (float): Ratio determining the number of outputs to process.
        """
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        """
        Process the model output by iterating over the second dimension.
        
        For each detection:
          - If the confidence (fifth element) is below the threshold, break.
          - Depending on ouput_type, append confidence score and/or bounding box coordinates.
        
        Returns:
            final_result: Sum of stacked outputs with gradients enabled.
        """
        # Print input tensor properties
        post_result = data
        print(f"Input data shape: {post_result.shape}")
        print(f"Input data device: {post_result.device}")
        print(f"Input data requires_grad: {post_result.requires_grad}")
        
        result = []
        for i in trange(int(post_result.size(1) * self.ratio)):  # Iterate using the second dimension
            if float(post_result[0, i, 4]) < self.conf:  # Using the fifth element as the confidence score
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[0, i, 4])  # Append confidence score
            if self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(post_result[0, i, j])  # Append bounding box coordinates
        
        # Modification: Instead of direct sum, use torch.stack() and then torch.sum()
        if len(result) == 0:
            # Return a zero tensor with gradients enabled if no target is detected
            return torch.tensor(0.0, requires_grad=True, device=post_result.device)
        
        final_result = torch.sum(torch.stack(result))
        print(f"Output requires_grad: {final_result.requires_grad}")
        return final_result


class yolo_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, 
                 ratio, show_box, renormalize, colormap=cv2.COLORMAP_JET, 
                 alpha=0.5, use_rgb=True):
        """
        Initialize the YOLO heatmap visualization module.
        
        Args:
            weight (str): Path to the model weights (e.g., '/path/to/your/model/weights').
            device (str): Device identifier (e.g., 'cuda:0' or 'cpu').
            method (str): CAM method to use (e.g., 'GradCAM', 'EigenCAM').
            layer (list): List of target layer indices.
            backward_type (str): Type of backward processing for gradients.
            conf_threshold (float): Confidence threshold.
            ratio (float): Ratio for target selection.
            show_box (bool): Flag to indicate if detection boxes should be displayed.
            renormalize (bool): Flag to indicate if CAM should be renormalized within bounding boxes.
            colormap: OpenCV colormap to use.
            alpha (float): Transparency factor for heatmap overlay.
            use_rgb (bool): Whether to convert the heatmap to RGB.
        """
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = yolo_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(1, 3)).astype(int)
        
        # Add new visualization parameters
        self.colormap = colormap
        self.alpha = alpha
        self.use_rgb = use_rgb
        
        self.__dict__.update(locals())

    def draw_detections(self, box, color, name, img):
        """
        Draw bounding box and label on the image.
        
        Args:
            box (iterable): Bounding box coordinates.
            color (tuple): Color for the rectangle and text.
            name (str): Label to display.
            img (numpy.array): The original image.
        
        Returns:
            img: Image with drawn detections.
        """
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """
        Renormalize the CAM within specified bounding boxes for improved visualization.
        
        Args:
            boxes (list): List of bounding box coordinates.
            image_float_np (numpy.array): Normalized original image.
            grayscale_cam (numpy.array): Grayscale CAM.
        
        Returns:
            eigencam_image_renormalized: CAM image overlaid on the original image.
        """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def show_cam_on_image_colormap(self, img, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Generate an improved heatmap visualization by blending a colormap with the original image.
        
        Args:
            img (numpy.array): Normalized original image.
            cam (numpy.array): CAM heatmap.
            alpha (float): Transparency factor.
            colormap: OpenCV colormap.
        
        Returns:
            output (numpy.array): Image with heatmap overlay.
        """
        cam = np.float32(cam)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)  # Add small epsilon to avoid division by zero
        cam = np.uint8(255 * cam)
        cam = cv2.applyColorMap(cam, colormap)
        cam = np.float32(cam) / 255.0
        cam = cam[..., ::-1]  # Convert BGR to RGB
        
        # Blend original image with heatmap using alpha
        output = (1 - alpha) * np.float32(img) + alpha * cam
        output = output / np.max(output)
        return np.uint8(255 * output)

    def process(self, img_path, save_path):
        """
        Process a single image to generate and save a saliency (CAM) overlay.
        
        Args:
            img_path (str): Path to the input image (e.g., '/path/to/your/images').
            save_path (str): Path to save the output image.
        """
        # Image preprocessing
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image at path: {img_path}")
        
        # Preprocess image: resize using letterbox and convert to RGB
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        
        # Convert to tensor and adjust dimensions for model input
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0)
        tensor = tensor.to(self.device)

        try:
            # Obtain model output and CAM
            model_output = self.method.activations_and_grads(tensor)
            model_output = model_output[0].permute(0, 2, 1)
            
            # Generate CAM
            grayscale_cam = self.method(tensor, [self.target])
            grayscale_cam = grayscale_cam[0, :]
            
            # Save CAM results for each layer if multiple layers are specified
            if len(self.layer) > 1:
                base_name = os.path.splitext(save_path)[0]
                for i, layer_idx in enumerate(self.layer):
                    layer_cam = self.show_cam_on_image_colormap(
                        img, 
                        grayscale_cam[i],
                        alpha=self.alpha,
                        colormap=self.colormap
                    )
                    layer_save_path = f"{base_name}_layer{layer_idx}.png"
                    Image.fromarray(layer_cam).save(layer_save_path)
            else:
                # Single-layer CAM processing
                cam_image = self.show_cam_on_image_colormap(
                    img, 
                    grayscale_cam,
                    alpha=self.alpha,
                    colormap=self.colormap
                )
                Image.fromarray(cam_image).save(save_path)
            
        except Exception as e:
            print(f"Error during CAM generation: {e}")
            raise

    def __call__(self, img_path, save_path):
        """
        Process input images to generate and save CAM visualizations.
        
        If img_path is a directory, processes all images within it; otherwise, processes a single image.
        
        Args:
            img_path (str): Path to the input image or directory.
            save_path (str): Path or directory to save the output results.
        """
        logger.info(f"Processing images from {img_path}")
        
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            img_files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            for img_file in tqdm(img_files, desc="Generating CAMs", leave=False):
                full_img_path = os.path.join(img_path, img_file)
                full_save_path = os.path.join(save_path, img_file)
                try:
                    self.process(full_img_path, full_save_path)
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
        else:
            self.process(img_path, os.path.join(save_path, 'result.png'))

    def process_batch(self, img_paths, save_dir, batch_size=4):
        """
        Process images in batches.
        
        Args:
            img_paths (list): List of image file paths.
            save_dir (str): Directory to save the output images.
            batch_size (int): Number of images per batch.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i + batch_size]
            batch_imgs = []
            
            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = self.preprocess_image(img)
                batch_imgs.append(img)
                
            if not batch_imgs:
                continue
                
            # Process the batch tensor
            batch_tensor = torch.stack(batch_imgs).to(self.device)
            try:
                self.process_batch_tensor(batch_tensor, batch_paths, save_dir)
            except Exception as e:
                print(f"Error processing batch: {e}")

def get_params():
    """
    Retrieve configuration parameters for saliency map generation.
    
    Returns:
        dict: Configuration parameters.
    """
    params = {
        'weight': '/path/to/your/model/weights',  # Replace with the path to your model weights
        'device': 'cuda:0',
        'method': 'GradCAM',  # Options: GradCAM, EigenCAM, etc.
        'layer': [i],
        'backward_type': 'all',
        'conf_threshold': 0.5,
        'ratio': 0.1,
        'show_box': False,
        'renormalize': False,
        'colormap': cv2.COLORMAP_JET,  # Heatmap color mapping
        'alpha': 0.5,  # Heatmap transparency
        'use_rgb': True  # Use RGB color space for visualization
    }
    return params

def yolov10_reshape_transform(tensor):
    """
    Helper function to reshape the feature map tensor for a YOLOv10 model.
    
    Args:
        tensor: Input tensor representing feature maps.
    
    Returns:
        torch.Tensor: Reshaped tensor.
    """
    result = []
    for t in tensor:
        # Process YOLOv10 feature map outputs
        if len(t.shape) == 4:
            t = t.permute(0, 2, 3, 1)
        result.append(t)
    return torch.cat(result, dim=0)

if __name__ == '__main__':

    i = 19
    # Create model instance using specified parameters
    params = get_params()
    model = yolo_heatmap(**params)

    # Set output directory (replace with your desired output path)
    output_dir = f'/path/to/save/results/gradcam-layer_{i}'

    # Process images
    logger.info(f"Processing layer {i}")
    model('/path/to/your/images', output_dir)

  