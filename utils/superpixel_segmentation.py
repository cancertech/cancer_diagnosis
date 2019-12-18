import os
import sys
sys.path.append(os.path.dirname(__file__))

import color_conversion
import feature_extraction
import viz

import numpy as np
    
from  sklearn.cluster import KMeans
    
import imageio

import tqdm
import os
from skimage.transform import resize

from skimage.segmentation import slic
import time

import pdb

import matplotlib.pyplot as plt


def process_img(infname, output_base_dir, num_pixels_per_seg, slic_shrink_scale, 
                K:int=6, downsample=1):
    """ Run the Superpixel Pipeline
    
    Here, we use uint8 [0-255] to represent images for consistency. If the 
    result of some function is not uint8, cast it to uint8.
    
    Args:
        infname (str): Location of the input image
        output_base_dir (str): Location to store output files
        num_pixels_per_seg (int): number of pixels need for each segmentation
        slic_shrink_scale (int): scale to shrink the image for SuperPixel
            algorithm. Larger number means the result image will be smaller 
            and the quality will be poorer. This resize input image for SLIC, 
            and do not affect other parts of this pipeline.
        K (int): number of classes needed for k-means algorithm.

    Returns:
        overlay (np.array): [h, w, 3]. An overlay image for output segmentation 
        and input image.
    """
    tic = time.time()
    start_time = "%d" % tic
    
    output_dir = os.path.join(output_base_dir, start_time)
    output_dir = os.path.realpath(output_dir)
    
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    img = imageio.imread(infname)
    
    num_segments = img.shape[0] * img.shape[1] // num_pixels_per_seg

    if downsample > 1:
        img = resize(img, (img.shape[0] // downsample, img.shape[1] // downsample), preserve_range=True)
        img = img.astype(np.uint8)
    tic = time.time()
    
    
    he = color_conversion.rgb2he(img).astype(np.uint8)
    imageio.imsave(os.path.join(output_dir, 'he.jpg'), he)
    img_features = feature_extraction.extract_features_for_slice(img)
    
    img_small = resize(img, (img.shape[0] // slic_shrink_scale, img.shape[1] // slic_shrink_scale))
    he_small = resize(he, (img.shape[0] // slic_shrink_scale, img.shape[1] // slic_shrink_scale))
    
    label_small = slic(he_small, n_segments=num_segments)
    toc = time.time()
    
    
    # Scale the label back to large size.
    # order=0 means warping take less 
    # preserve_ragne so that the labels are not normalzied to 0-1.
    img_label = resize(label_small, (img.shape[0], img.shape[1]), order=0, preserve_range=True)
    img_label = img_label.astype(np.uint64)
    
    print("It takes %.2f seconds for SLIC" % (toc - tic))
    viz.viz_segmentation_countour(img, img_label, border_color=[1, 0, 0], output_dir=output_dir)
    plt.imshow(img_label)

            
    X = feature_extraction.hist_features_for_all_superpixel(img_features, img_label)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    
    toc = time.time()
    
    print("The Whole Segmentation finished in %.2f seconds!" % (toc - tic))
    
    # View k-means result in real image
    colors = np.array([
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255]
            ])
    
    
    
    rst = img_small.copy()
    
    
    for l in np.unique(kmeans.labels_):
        superpixel_ids = []
        for i in range(len(kmeans.labels_)):
            if kmeans.labels_[i] == l:
                superpixel_ids.append(i)
        
        for super_id in tqdm.tqdm(superpixel_ids):
            rst[label_small==super_id, :] = colors[l]                
    
    rst_big = resize(rst, (img.shape[0], img.shape[1]), order=0, preserve_range=True)
        
    plt.imshow(rst_big)
        
        
    imageio.imsave(os.path.join(output_dir, 'final_segmentation.jpg'), rst_big)
    
    # %%
    overlay = img.astype(float) * 0.7 + rst_big * 0.3
    plt.imshow(overlay)
    
    imageio.imsave(os.path.join(output_dir,'overlay.jpg'), overlay)
    
    # %%  Visualization for Convenience. Not necessary for result.
    
    # This takes several minutes
    viz.save_superpixels(img, img_label, kmeans.labels_, output_dir=output_dir)
    
    
    viz.superpixel_in_html(output_dir=output_dir)

    return overlay
