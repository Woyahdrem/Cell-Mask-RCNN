"""
Mask R-CNN
Train on the Cell dataset

Written by Stefano Gatti

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=coco
    
    # Resume training a model that you had trained earlier
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=last
    
    # Train a new model starting from ImageNet weights
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=imagenet
"""

import os
import sys
import json
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = "" #TODO
assert len(ROOT_DIR) != 0, "Please specify the root directory (variable ROOT_DIR)"

# Import Mask RCNN
sys.path.append(ROOT_DIR) # To find the local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class CellConfig(Config):
    """Configuration for training on the cell dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cell"
    
    # The number of images per GPU depends on the size of the VRAM and the size of the images
    # Adjust as ceil( VRAM / max(img_size) )
    IMAGES_PER_GPU = 1
    
    # Number of classes (including background)
    # Find out how to use multiple labels
    NUM_CLASSES = 1 + 4 # Background + others
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    # May need adjusting due to high cell count
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    # May need adjusting due to high cell count
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Input image resizing
    # IMAGE_MIN_DIM is the size of the scaled shortest side
    # IMAGE_MAX_DIM is the maximum allowed size of the scaled longest side
    # May benefit from adjusting
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    
    # Number of color channels per image
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    # Right now we are using the usual RGB channels
    IMAGE_CHANNEL_COUNT = 3
    
    # Image mean (RGB)
    # Must have length equal to IMAGE_CHANNEL_COUNT
    # Values could depend on brightness of layer
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    
    # Maximum number of ground truth instances  and final detections
    # These must definitely be higher
    MAX_GT_INSTANCES = 100
    DETECTION_MAX_INSTANCES = 100
    

############################################################
#  Dataset
############################################################

class CellDataset(utils.Dataset):
    
    def load_cell(self, dataset_dir, subset, mode):
        """Load a subset of the Cell dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have multiple classes, one for each gene
        # What is the difference between source and class_name?
        self.add_class("cell", 1, "red")
        self.add_class("cell", 2, "white")
        self.add_class("cell", 3, "red and white")
        self.add_class("cell", 4, "negative")
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Load annotations
        # { "filename": "2019-04-03_RNAScope PAF OCT D380.lif [PAF OCT D380 FOXJ1 CFTR 1 1] Z1.jpg",
		#   "size": 328856,
		#   "regions": [
 		# 	{ "shape_attributes": {
 		# 	    "name": "rect",
 		# 	    "x": 409,
 		# 	    "y": 1017,
 		# 	    "width": 58,
 		# 	    "height": 78},
 		# 	  "region_attributes": {
 		# 			"gene expression": "red"
        #     }},
        #     ... more regions...
        #   },
        #   "size": 328856
        # }
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        
        # Skip unannotated images
        annotations = [a for a in annotations if a['regions']]
        
        # Add the images
        for a in annotations:
            # We are using Bounding Boxes instead of Masks
            # To use the masks we will convert the boxes in polygons and use them as masks
            polygons = [r['shape_attributes'] for r in a['regions']]
            
            # Since we are using multiple classes, we also need to know what class each annotation belongs to
            classes = [r['region_attributes'] for r in a['regions']]
            
            # Next, we need to load the image path and the image size
            # if the dataset becomes too big, having the values directly in the json becomes necessary
            image_path = os.path.join(dataset_dir, a['filename'])
            img = skimage.io.imread(image_path)
            height, width = img.shame[:2]
            
            self.add_image(
                "cell",
                image_id=a["filename"],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                classes=classes)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cell dataset image, delegate to parent class
        info = self.image_info[image_id]
        if info["source"] != "cell":
            return super(self.__class__, self).load_mask(image_id)
        
        # Convert the BBox to a bitmap mask of shape
        # [height, width, instance_count]
        # Right now we are converting the Bounding Box into a rectangular mask
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        class_ids = np.zeros(len(info["polygons"]))
        
        # Set the masks for each instance
        # At the same time, set class_ids to the corresponding class
        for i, (p, c) in enumerate(zip(info["polygons"], info["classes"])):
            rr, cc = skimage.draw.rectangle(start=(p["y"], p["x"]), extent=(p["height"], p["width"]))
            mask[rr, cc, i] = 1
            if c["gene expression"] == "red": class_ids[i] = 1
            elif c["gene expression"] == "white": class_ids[i] = 2
            elif c["gene expression"] == "red and white": class_ids[i] = 3
            elif c["gene expression"] == "negative": class_ids[i] = 4
            else: print(f"Class not recognized: {c['gene expression']} in image {image_id}")
        
        # Return mask and class ID array
        return mask.astype(np.bool), class_ids


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model):
    """Train the model."""
    
    # Training dataset
    dataset_train = CellDataset()
    dataset_train.load_cell(args.dataset, "train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = CellDataset()
    dataset_val.load_cell(args.dataset, "val")
    dataset_val.prepare()
    
    # In the future the regex to select the correct layers is this
    # layers = r"(conv1)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)"
    
    # Define the augmentation of for the dataset
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])
    
    # Finally, train the model
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learing_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads',
                augmentation=augmentation)


############################################################
#  Model
############################################################

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect gene expression in cells.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' (or 'detect' once implemented)")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cell/dataset/",
                        help='Directory of the Cell dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image on which yo detect cells')
    args = parser.parse_args()
    
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image, "Provide --image or --video to apply color splash"
    
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    
    # Configurations
    if args.command == "train":
        config = CellConfig()
    else:
        class InferenceConfig(CellConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()
    
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
    
    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    
    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        # TODO
        print("Not yet implemented.")
    else:
        print(f"'{args.command}' is not recognized. \nUse 'train' (or 'detect' once implemented)".format(args.command))