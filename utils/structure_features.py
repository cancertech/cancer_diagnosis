import os
import sys
sys.path.append(os.path.dirname(__file__))

from superpixel_cooccurence import neighbours 
from superpixel_classification import assign_sp_cls

import imageio
from skimage.segmentation import slic

import scipy

import cv2
import numpy as np

import matplotlib

matplotlib.use("agg") # avoid QT error

import matplotlib.pyplot as plt
from copy import deepcopy

import random
import pdb
import time
import datetime

VIZ = True


def test_plot(imgpath):
    plt.plot([1,2,3], [1,2,3])
    plt.title("Test")
    plt.savefig(imgpath)


def find_ducts_from_semantic(mask, duct_cls=[1, 2, 5, 7], out_name=None):
    """
    Find the ducts by using semantic labels (either ground truth or predicted).
    
    Args:
        mask (np.array): [h, w] semantic segmentation mask, where each pixel represents the corresponding
                pixel-wise semantic labels
        duct_cls (list): a list of classes should be considered as part of a duct.
        
    Output:
        ducts (np.array): [h, w] labled image, where each pixel correspond to the
            duct ID. For duct ID = 0, it is background
    """
    
    binary_mask = np.zeros(mask.shape, dtype=np.uint8)
    for cls_id in duct_cls:
        binary_mask += mask == cls_id 

    binary_mask = binary_mask > 0

    
    # Erode the image first to remove noise
    kernel = np.ones((50,50),np.uint8)
    erosion = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations = 1)
    
    
    # Get the connected components
    ret, labels = cv2.connectedComponents(erosion.astype(np.uint8))

    # save the results to images (if necessary)
    if out_name is not None:
        cv2.imwrite(out_name, labels)
        
    return labels

def get_sp_ids_at_border(binary_mask, sp_mask):
    """
    We can find the ID for superpixels that lies on the border of the binary_mask.
    
    Note that we only need the out border of a strided (empty) shape.
    
    Args:
        binary_mask (np.array): binary mask where True means the pixel is in the 
            duct (structure).
    
    Output:
        sp_ids (list): a list of superpixel IDs that lies at the border of the
            given binary_mask
    """

    # Fill holes in the binary mask
    h, w, = binary_mask.shape
    filled_mask = np.zeros((h + 2, w + 2), np.uint8) # this mask should be 2 pixel wider/higher than the input
    cv2.floodFill(binary_mask.astype(np.uint8), filled_mask, (0,0), 1)     # line 27
    filled_mask = filled_mask - 1
    
    filled_mask = filled_mask[1:h + 1, 1:w + 1] # remove the redundant 2 pixels in width/height
    
    
    # find contour
    try:
        # CV3 code
        contours, hierarchy = cv2.findContours(filled_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        # CV2 code
        im2, contours, hierarchy = cv2.findContours(filled_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    # Note that there should be only 1 contours in this case, because we already
    # filled the mask with holes
    hull = [cv2.convexHull(_, False) for _ in contours]
    if len(hull) > 1 and VIZ:
        print("More than 1 contour/hull found!!!")
        plt.imshow(filled_mask)
        plt.show()

    # Get borders by using Convex Hulls
    drawing = np.zeros(filled_mask.shape, np.uint8)
    cv2.drawContours(drawing, hull, contourIdx=0, color=1, thickness=1, lineType=8)
    borders = np.where(drawing)
    
#    # Get borders with simple border (gradient)
#    edges = cv2.Canny(filled_mask.astype(np.uint8), 100,200)
#    borders = np.where(edges)
    
    sp_ids = []
    for i in range(len(borders[0])):
        sp_ids.append(sp_mask[borders[0][i], borders[1][i]])
    
    sp_ids = list(set(sp_ids))
    
    return sp_ids

def get_layers_from_duct(known_sps, sp_label, neighbours):
    """
    Get the inner and outer layer from the known superpixels. The algorithm
    is straightforward: 
        1. find all unknown neighours of the known superpixels. 
        2. for each neighbour, it is in inner layer if it is closer to the duct center;
           otherwise, it is in outer layer.
    
    Note: this function will be called multiple times in get_all_layers
    
    Args:
        known_sps (list): a list of superpixel ID, which are known superpixels
            around the duct
        sp_label (np.array): an array with shape [h, w] where each pixel is the
            corresponding superpixel ID
        neighbours (dict): a dictionary that stores the information of neighbourhood.
            
    Output:
        inner_layer (list): a list of cluster ID which lies one layer inside
            the known superpixels
        outer_layer (list): a list of cluster ID which lies one layer outside 
            the known superpixels.
    """
    # Get the (binary) mask for known super pixels
    binary_mask = np.zeros(sp_label.shape, dtype=np.uint8)
    for cls_id in known_sps:
        binary_mask += sp_label == cls_id 
    binary_mask = binary_mask > 0

    # Get the center coordinate (mass center) of known superpixels
    mass_center = scipy.ndimage.measurements.center_of_mass(binary_mask) # (center_y, center_x)


    # Store results
    inner_layer = set()
    outer_layer = set()
    
    for sp in known_sps:
        sp_binary_mask = sp_label == sp
        sp_center = scipy.ndimage.measurements.center_of_mass(sp_binary_mask)
        
        # eucleadian distance to mass center 
        distance_sp_to_center = np.linalg.norm(np.array(mass_center) - np.array(sp_center))
        
        for sp_neigh in neighbours[sp]:
            # process "sp_neigh" if we haven't done so
            if sp_neigh not in known_sps and sp_neigh not in inner_layer \
            and sp_neigh not in outer_layer:
                sp_binary_mask_temp = sp_label == sp_neigh
                sp_center_temp = scipy.ndimage.measurements.center_of_mass(sp_binary_mask_temp)
                temp_distance = np.linalg.norm(np.array(mass_center) - np.array(sp_center_temp))

                if temp_distance <= distance_sp_to_center:
                    inner_layer.add(sp_neigh)
                else:
                    outer_layer.add(sp_neigh)
                    

    return list(inner_layer), list(outer_layer)


def get_all_layers_for_one_duct(sp_at_border, sp_label, neighs, nlayers=5):
    """
    Get the superpixel IDs for all the layers. 
    
    Args:
        sp_at_border: a list of superpixel ID, which lies on the duct border
        sp_label (np.array): an array with shape [h, w] where each pixel is the
            corresponding superpixel ID
        neighs (dict): a dictionary that stores the information of neighbourhood.
        nlayers (int): number of layers we need to get from both the side and 
            the outside from the duct border.
            
    Output:
        layers (list): the list has "nlayers * 2 + 1" elements, where the middle
            element is the sp_at_border. From the first element to the last element,
            the element is an embedded array of superpixel IDs that are from the
            outer-most layer to the inner-most layer.
            If one layer does not contain any elements, then the superpixel IDs
            should be empty.
    """
    
    n_total_layers = nlayers * 2 + 1
    known_sps = sp_at_border    

    # from the first index to the last is: the most outer layer to the most inner layer    
    layers = [[]] * n_total_layers    
    layers[nlayers] = deepcopy(sp_at_border)
    
    for _ in range(nlayers):
#        print(time.ctime(), "start getting layers: ", _)
        inner_layer, outer_layer = get_layers_from_duct(known_sps, sp_label, neighs)
        layers[nlayers - _ - 1] = deepcopy(outer_layer)
        layers[nlayers + _ + 1] = deepcopy(inner_layer)
        known_sps = known_sps + inner_layer + outer_layer
        
    return layers

def structure_features_for_duct(rgb_img, seg_mask, duct_mask, 
                                num_pixels_per_seg=3000,
                                n_seg_cls=8, nlayers=5, viz_name=None):
    """
    Get the structure feature for one duct. The features are just histogram (count)
    for each layer in the structure.
    
    Args:
        rgb_img (np.array): array with [h, w, 3] that is the RGB image
        seg_mask (np.array): array with [h, w] that corresponds to the semantic
            segmentation mask.
        duct_mask (np.array): boolean array with [h, w], where the True is the 
            region of the duct.
        num_pixels_per_seg (int): the (average) number of pixels in each superpixel.
        n_seg_cls (int): number fo semantic segmentation classes
        nlayers (int): number of layers we need to get from both the side and 
            the outside from the duct border.
    
    Output:
        features (list): a list for all structure features. The length of this
            list is $n_seg_cls * (nlayers * 2 + 1)$
    """


    # Get superpixel labels
#    print(time.ctime(), "begin slic", rgb_img.shape)
    h, w, _ = rgb_img.shape
    num_segs = int(h * w / num_pixels_per_seg)
    open(viz_name + ".log", "w").write("%s: Begin the Duct\n" % str(datetime.datetime.now()))

    sp_mask = slic(rgb_img, n_segments=num_segs)
#    print(time.ctime(), "end slic")
    neighs = neighbours(sp_mask) # dictioary of neighbours
    sp_ids = get_sp_ids_at_border(duct_mask, sp_mask)
    open(viz_name + ".log", "a").write("%s: SLIC finished\n" % str(datetime.datetime.now()))
    
    layers = get_all_layers_for_one_duct(sp_ids, sp_mask, neighs, nlayers=5)    
#    print(time.ctime(), "end getting layers")
    open(viz_name + ".log", "a").write("%s: Layers Got!\n" % str(datetime.datetime.now()))
    
    features = []
    sp_id_to_seg_cls = assign_sp_cls(sp_mask, seg_mask)
    
    for layer in layers:
        counts = [0] * n_seg_cls
        open(viz_name + ".log", "a").write("%s: Layer: %s\n" % (str(datetime.datetime.now()), str(layer)))
        for sp in layer:
            cls_id = sp_id_to_seg_cls[sp]
            counts[cls_id] += 1        
        features += counts
     
#    print(time.ctime(), "end getting features")


    
    try:
        if VIZ and viz_name is not None:
    #        plt.show()
            # Viz for Debug
            plt.cla(); plt.clf(); # must need to clean it to save memory and speed before every plots
            plt.subplot(1, 4, 1)
            plt.imshow(rgb_img)
            plt.subplot(1, 4, 2)
            plt.imshow(duct_mask)
            
            viz_mask = np.zeros(sp_mask.shape, dtype=np.uint8)
            for idx in sp_ids:
                locs = sp_mask == idx
                viz_mask[locs] = 1
            plt.subplot(1, 4, 3)
            plt.imshow(viz_mask)
            
            viz = np.zeros(sp_mask.shape, dtype=np.uint8)
            for layer_id in range(len(layers)):
                for sp in layers[layer_id]:
                    viz[sp_mask == sp] = layer_id + 1 # use zero for background
            plt.subplot(1, 4, 4)
            plt.imshow(viz, vmin=1, vmax=nlayers * 2 + 1, cmap="Set3")
    
            plt.savefig(viz_name, bbox_inches="tight", dpi=300)
        
    except:
        pass
    # Get features from the layers


    # print(features)
    
    return features


def structure_features_for_roi(rgb_img, seg_mask, nlayers=5, n_seg_cls=8, 
                               num_pixels_per_seg=3000,
                               duct_cls=[1, 2, 5, 7], duct_label_name=None):
    """
    Get the structure features for an ROI image. If the ROI image contains 
    multiple ducts. We use the results (summation) from all ducts.
    
    
    Args:
        rgb_img (np.array): array with [h, w, 3] that is the RGB image
        seg_mask (np.array): a matrix with [h, w] where each pixel is the semantic
            segmentation label.
        nlayers (int): number of layers we need to get from both the side and 
            the outside from the duct border.
        n_seg_cls (int): number fo semantic segmentation classes
        num_pixels_per_seg (int): the (average) number of pixels in each superpixel. 
        duct_cls (list): a list of classes should be considered as part of a duct.
        
    Output:
        features (list): a list of features
    """    
    assert(rgb_img.shape[0] == seg_mask.shape[0])
    assert(rgb_img.shape[1] == seg_mask.shape[1])
    if duct_label_name is not None:
        # if we want to save the duct labels, we must use PNG (lossless compression)
        assert(duct_label_name.endswith(".png"))

        ducts = find_ducts_from_semantic(seg_mask, duct_cls=duct_cls, out_name=duct_label_name.replace(".png", "sp_duct.png"))
    else:
        ducts = find_ducts_from_semantic(seg_mask, duct_cls=duct_cls)

    features_all = []
    
    h, w, _ = rgb_img.shape

        
    approx_border_size_for_out_layers = int(np.sqrt(num_pixels_per_seg) * 2 * nlayers * 1.5)
    
    # we need at least 1 superpixel for each layer (in average)
    THRESHOLD_SMALLEST_DUCT = num_pixels_per_seg * nlayers * 2
    
    duct_sizes = []
    
    logname = duct_label_name + ".log"
    open(logname, "w").write("%s: Begin\n" % str(datetime.datetime.now()))


    for duct_id in range(1, np.max(ducts)):
        open(logname, "a").write("%s: Begin process %d (of %d)\n" % (str(datetime.datetime.now()), duct_id, np.max(ducts)))
        binary = ducts == duct_id

        duct_sizes.append(np.sum(binary))
        if np.sum(binary) < THRESHOLD_SMALLEST_DUCT:
            # Region too small. we do not care.
            continue
        
        where = np.array(np.where(binary))
        h1, w1 = np.amin(where, axis=1)
        h2, w2 = np.amax(where, axis=1)
        
        h1 = max(0, h1 - approx_border_size_for_out_layers)
        h2 = min(h, h2 + approx_border_size_for_out_layers)
    
        w1 = max(0, w1 - approx_border_size_for_out_layers)
        w2 = min(w, w2 + approx_border_size_for_out_layers)
        
        sub_rgb_img = rgb_img[h1:h2, w1:w2]
        sub_semantic_mask = seg_mask[h1:h2, w1:w2]
        sub_duct_mask = binary[h1:h2, w1:w2]
        
        if duct_label_name is not None:
            viz_name = duct_label_name.replace(".png", "_%06d.png" % duct_id)
        else:
            viz_name = None
        feature = structure_features_for_duct(sub_rgb_img, sub_semantic_mask, sub_duct_mask, 
                                    num_pixels_per_seg=num_pixels_per_seg,
                                    n_seg_cls=n_seg_cls, nlayers=nlayers, viz_name=viz_name)
    
        features_all.append(feature)
    

    open(logname, "a").write("%s: Finished Process!\n" % str(datetime.datetime.now()))

    try:

        if VIZ and viz_name is not None:
            plt.cla(); plt.clf();
            plt.hist(duct_sizes)
            plt.axvline(THRESHOLD_SMALLEST_DUCT, color="k")
            plt.title("Duct Size Histogram")
            plt.savefig(viz_name.replace(".png", "_duct_size_hist.png"))
    except Exception as e:
        print(e)

    open(logname, "a").write("%s: Features All: %s\n" % (str(datetime.datetime.now()), str(features_all)))
    print(features_all)
    features_all = np.array(features_all)    
    features = np.sum(features_all, axis=0)        
    return features.tolist(), features_all

# %%
if __name__ == "__main__":
    rgb_img = imageio.imread("../test/test.jpg")
    seg_mask = imageio.imread("../test/ynet_out.png")
#    ducts = find_ducts_from_semantic(seg_mask)
#    plt.imshow(ducts)
    
    rst = structure_features_for_roi(rgb_img, seg_mask, nlayers=5, n_seg_cls=8, 
                               num_pixels_per_seg=3000,
                               duct_cls=[1, 2, 5, 7], duct_label_name="../test/test_structure_duct.png")

    
