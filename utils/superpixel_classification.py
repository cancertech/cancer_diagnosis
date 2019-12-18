import os
import sys
sys.path.append(os.path.dirname(__file__))
from superpixel_frequency import sp_cls_count
from superpixel_cooccurence import co_occurence, neighbours
from viz import viz_segmentation_countour

import numpy as np
from collections import Counter
from skimage.segmentation import slic
from PIL import Image
import tqdm

import time

import pdb

# %%

pallete = [ 255, 255, 255,
            130, 0, 130,
            0, 0, 130,
            255, 150, 255,
            150 ,150 ,255,
            0 ,255 ,0,
            255, 255 ,0,
            255, 0, 0,
            0, 0, 0] # the last class (9-th class) is for border
    

# %%    
def assign_sp_cls(sp_labels, mask):
    """
    Use majority voting to assign a class label for each superpixel.
    
    
    Args:
        sp_labels (np.array): size [h, w], where each pixel represent the super
            pixel ID. 
        mask (np.array): size [h, w], where each pixel represent the class ID
            for that pixel. Here, we use zero-indexing, which means the class
            are in range [0, k)
    
    Output:
        sp_cls (dict): the key is the superpixel ID, and the value is the
            class ID for the corresponding superpixel. There are n elements
            in the sp_cls. Here, we use zero-indexing, which means the class
            are in range [0, k)
    """
#    k = np.max(mask)
    n = np.max(sp_labels) + 1

    sp_cls = {}
    
    for i in range(n):
        sp_mask_i = sp_labels == i
        values = mask[sp_mask_i] # cluster IDs
        c = Counter(list(values))        
        sp_cls[i] = c.most_common()[0][0]
        
    return sp_cls


    

def mask_to_superpixel_co_occurence(img, seg_mask, tile_size=5000, 
                                    num_pixels_per_seg=3000, viz_fname=None):
    """
    Get SuperpixelFrequency features and SuperpixelCo-occurrence features for
    an image with semantic segmentation mask.
    
    The process follows these procedures:
        - Step 1: Get neighbours for each superpixel
        - Step 2: Get the class label (by majority voting) for each pxiel
        - Step 3: Get the co-occurence features by using the information 
                extracted above.

    Args:
        img (np.array): size [h, w, 3]
        seg_mask (np.array): size [h, w]
        tile_size (int): the size (width and height) for each tile.
        num_pixels_per_seg (int): the (average) number of pixels in each
            superpixel. 
        viz_fname (str): a image filename, whose image will store the visualization
            of the final semantic segmentation based on the superpixel majority
            voting.
    
    Output:
        freq_features (list): an array of frequency features        
        cooccr_features (list): an array of co-occurence features
    """
    
    assert(img.shape[0] == seg_mask.shape[0])
    assert(img.shape[1] == seg_mask.shape[1])
    
    # the following two arrays will store results for each tile
    freqs = []
    cooccrs = []
    
    h, w, _ = img.shape
    
    print(time.ctime(), "Running Superpixel Segmentation ...")
    
    if viz_fname is not None:
        viz_img = np.zeros(seg_mask.shape, dtype=np.uint8)
    
    for row_i in tqdm.tqdm(range(0, h, tile_size)):
        row_end = min(row_i + tile_size, h)
        for col_i in range(0, w, tile_size):
            col_end = min(col_i + tile_size, w)
            
            tile_img = img[row_i:row_end, col_i:col_end, :]
            tile_seg = seg_mask[row_i:row_end, col_i:col_end]

            num_segments_in_tile = tile_img.shape[0] * tile_img.shape[1] // num_pixels_per_seg
            tile_sp_labels = slic(tile_img, n_segments=num_segments_in_tile)

            tile_neigh = neighbours(tile_sp_labels)
            tile_sp_cls = assign_sp_cls(tile_sp_labels, tile_seg)
            tile_cooccr = co_occurence(tile_sp_labels, tile_sp_cls, tile_neigh, k=8)
            
            tile_sp_freq = sp_cls_count(tile_sp_cls, n_seg_cls=8)
            
            freqs.append(tile_sp_freq)
            cooccrs.append(tile_cooccr.reshape(-1))
            
            if viz_fname is not None:
                viz_img_tile = viz_img[row_i:row_end, col_i:col_end] 
                for sp_id in range(np.max(tile_sp_labels)):
                    sp_mask = tile_sp_labels == sp_id
                    viz_img_tile[sp_mask] = tile_sp_cls[sp_id]
                    
                # add border for each superpixel
                viz_img_tile = viz_segmentation_countour(viz_img_tile, 
                                                         tile_sp_labels, 
                                                         border_color=8, # class ID
                                                         border_width=5, 
                                                         output_dir=None)

                # Place the viz back
                viz_img[row_i:row_end, col_i:col_end] = viz_img_tile
            
    if viz_fname is not None:
        outpil = Image.fromarray(np.uint8(viz_img))
        outpil.putpalette(pallete)
        outpil.save(viz_fname)
    
    freqs = np.array(freqs)
    cooccrs = np.array(cooccrs)
    
    freq_features = np.sum(freqs, axis=0)
    cooccr_features = np.sum(cooccrs, axis=0)
        
    return freq_features.tolist(), cooccr_features.tolist()
    

# %%
if __name__ == "__main__":
    import imageio
    num_pixels_per_seg = 3000
    
    img = imageio.imread("../test/test.jpg")
    mask = imageio.imread("../test/ynet_out.png")
    mask = mask.astype(np.uint8)
    
    a, b = mask_to_superpixel_co_occurence(img, mask, tile_size=3000, viz_fname="../test/test_seg_on_sp.png")
    
    print(a)
    print(b)
#    pdb.set_trace()
    
    # %%
#
#    num_segments = img.shape[0] * img.shape[1] // num_pixels_per_seg
#    
#    sp_labels = slic(img, n_segments=num_segments)
#    
#    # Function-wise Test
#    neigh = neighbours(sp_labels) # it takes about 18.3 seconds on CPU
#    sp_cls = assign_sp_cls(sp_labels, mask) # it teaks about 35 seconds on CPU
#    cooccr = co_occurence(sp_labels, sp_cls, neigh) # less than 1 seconds on CPU
#    
#    import pickle
#    pickle.dump(cooccr, open("../test/sp_cooccr.pickle", "wb"))
    
