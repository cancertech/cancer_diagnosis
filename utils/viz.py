import tqdm

import cv2
import numpy as np

import scipy

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt

import imageio

import os

import pdb

import glob


def viz_segmentation_countour(img, label_mask, border_color=[1, 0, 0], border_width=5, output_dir:str="output/"):
    """Visuzlie the segmentation in countour fashion.
    
    Args:
        img (np.array): [h, w, 3] RGB image
        label_mask (np.array): [h, w] label for the image
        border_color (list): an array with 3 numbers for the border color
        border_width (int): the border width for the visualization
        
    Returns:
        img2 (np.array): [h, w, 3] RGB image where the countours are in the 
        defined color.
    """
    binary_mask_shift_x = np.roll(label_mask, 1, axis=0)
    binary_mask_shift_y = np.roll(label_mask, 1, axis=1)
    gradient = 2 * label_mask - binary_mask_shift_x - binary_mask_shift_y
    gradient_mask = gradient != 0
    
    
    if border_width > 1:
        kernel = np.ones((border_width, border_width),np.uint8)
        gradient_mask = cv2.dilate(gradient_mask.astype(np.uint8), kernel, iterations=1)
    
    gradient_mask = gradient_mask.astype(bool)
    
    countours_img = img.copy()
    
    if len(countours_img.shape) >= 3:
        countours_img[gradient_mask, :] = border_color
    else:        
        countours_img[gradient_mask] = border_color

    
    if output_dir is not None:
        imageio.imsave(os.path.join(output_dir, "countour_with_gradient.jpg"),
                   countours_img)
    
    return countours_img


def save_superpixels(img, label_mask, kmeans_id, output_dir:str="output/"):
    """Save each superpixel segmentation
    
    This function takes lots of time, and we advice to run this function
    inside a separate thread so that the main thread would not be stucked.

    Args:
        img (np.array): [h, w, 3] RGB image
        label_mask (np.array): [h, w] gray scale image corresponding to the rgb image
        
        
    Returns:
        features (np.array): [h, w, 4], where each channel corresponds to a feature 
        extraction method.


    Examples::
            >>> img = imageio.imread("../test.jpg")
            >>> features = extract_features_for_slice(img)
            >>> print(feature)
            >>> print(features.shape)
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
            
    # Note: Beibin.
    # This for loop runs very slow, because we need to iterate few thousand times
    # A better method is to use multi-threading
    for label_id in tqdm.tqdm(np.unique(label_mask)):        
        #%
        binary_mask =  label_mask == label_id
        mask_3_channels = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
    
        patch =  mask_3_channels * img # img where only the patch has color
        patch_cropped = patch[np.ix_(binary_mask.any(axis=1),binary_mask.any(axis=0))]
        
        class_ = kmeans_id[label_id]
        
        if not os.path.exists(os.path.join(output_dir, str(class_))):
            os.mkdir(os.path.join(output_dir, str(class_)))
        
        imageio.imsave(os.path.join(output_dir, str(class_), "rst_%d.jpg" % label_id), patch_cropped)
        

def superpixel_in_html(output_dir="../output/"):
    """Convert everything in the output folder to HTML
    The output dir is the input dir
    
    """
    folders = glob.glob(os.path.join(output_dir, "*"))
    
    f = open(os.path.join(output_dir, "rst.html"), "w")
    for folder_name in folders:
        if not os.path.isdir(folder_name): continue
        f.write("<div>\n")
        
        for imgname in glob.glob(os.path.join(folder_name, "*.jpg")):
            imgname = imgname.replace("\\", "/")
            line = '<img src="%s">' % imgname
            
            f.write(line + "\n")
        f.write("</div><hr>\n\n\n")
        
        



# %%
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
