import os
import re
import json
import skimage.io
import skimage.color
import shutil
import numpy as np

ROOT_DIR = os.path.abspath("../../")
CELL_CHANNEL = "CELL"
GENE_CHANNELS = ["WHITE", "RED", "GREEN"]


def stack_layers(file_path):
    """
    Loads the images of the channels and composes them into a single matrix

    Parameters
    ----------
    file_path : String
        Absolute path of the image to load.

    Returns
    -------
    img . Numpy array
        H * W * C array of the grayscale channels
    """
    # Create the base image
    file_name = re.sub("\.jpg", " " + CELL_CHANNEL + ".jpg", file_path)
    img = skimage.io.imread(file_name)
    img = img[:, :, 0]

    # Load the channels
    for i, suffix in enumerate(GENE_CHANNELS):
        file_name = re.sub("\.jpg", " " + suffix + ".jpg", file_path)
        if os.path.isfile(file_name):
            l = skimage.io.imread(file_name)
            l = l[:, :, 0]
            img = np.dstack((img, l))
        else:
            l = np.zeros(img.shape[:-1], dtype=img.dtype)
            img = np.dstack((img, l))

    # Return the final array
    return img


def load_dataset(old_dataset_dir, new_dataset_dir):
    annotations = json.load(open(os.path.join(old_dataset_dir, "via_export_json.json")))
    filenames = [a["filename"] for a in list(annotations.values()) if a["regions"]]  # Need only the filenames

    for name in filenames:
        original_image_path = os.path.join(old_dataset_dir, name)

        image = stack_layers(original_image_path)

        new_image_path = os.path.join(new_dataset_dir, name)
        np.save(open(new_image_path, "wb"), image)

    shutil.copy2(os.path.join(old_dataset_dir, "via_export_json.json"),
                 os.path.join(new_dataset_dir, "via_export_json.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Copy a Dataset of standard images into a dataset of special multi-channel images.')

    parser.add_argument('--src', required=True,
                        metavar="/path/to/src/dataset",
                        help="Path to the folder containing the dataset.")
    parser.add_argument('--dst', required=True,
                        metavar="/path/to/dst/dataset",
                        help="Path to the folder where the new dataset will be saved.")
    args = parser.parse_args()

    # Validate the source folders
    src_train_dir = os.path.join(args.src, "train")
    src_val_dir = os.path.join(args.src, "val")
    assert os.path.isdir(src_train_dir), "The source dataset folder does not contain a train folder."
    assert os.path.isdir(src_val_dir), "The source dataset folder does not contain a val folder."

    # Create the destination folders if needed
    dst_train_dir = os.path.join(args.dst, "train")
    dst_val_dir = os.path.join(args.dst, "val")
    if not os.path.isdir(args.dst):
        os.mkdir(args.dst)
    if not os.path.isdir(dst_train_dir):
        os.mkdir(dst_train_dir)
    if not os.path.isdir((dst_val_dir)):
        os.mkdir(dst_val_dir)

    load_dataset(src_train_dir, dst_train_dir)
    load_dataset(src_val_dir, dst_val_dir)