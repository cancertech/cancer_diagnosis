import tqdm

from skimage.color import rgb2gray, rgb2lab
from skimage import feature
import numpy as np
import imageio

import pdb

import time
import multiprocessing
from multiprocessing.pool import ThreadPool



def extract_features_map_for_slice(img, gray=None, lab=None, verbose=False):
    """Extract features for a medical slice. The output will have four channels
    with LBP and LAB features.


    Args:
        img (np.array): [h, w, 3] RGB image
        gray (np.array): [h, w, 1] gray scale image corresponding to the rgb image
        lab (np.array): [h, w, 3] lab image
        verbose (bool): print more messages or not
        
    Returns:
        features (np.array): [h, w, 4], where each channel corresponds to a feature 
        extraction method. The four features are: LBP, LAB 1, LAB 2, LAB 3.


    Examples::
            >>> img = imageio.imread("../test.jpg")
            >>> features = extract_features_for_slice(img)
            >>> print(feature)
            >>> print(features.shape)
    """
    
    h, w, _ = img.shape
    
    if verbose:
        print("Image shape", h, w)
    
    if gray is None:
        gray = rgb2gray(img)
    lbp = feature.local_binary_pattern(gray, 8, 1)
    lbp = lbp.reshape(h, w, 1) # reshape 2D array to 3D array

    if lab  is None:
        lab = rgb2lab(img)
    
    all_features = np.concatenate([lbp, lab], axis=2)            
    
    return all_features


def patch_feature_histogram(img, mask, ranges=None, nbins:int=64):
    """Compute color histogram for the small patch in a big image
    The patch can be arbitray shape, and disconnected.


    Args:
        img (np.array): [h, w, d] array, where each channel is a feature space. It can be rgb image
                        lbp, lab, or any kind of channel.
        mask (np.array): [h, w] boolean mask that defines which region to use
        ranges (np.array): [d, 2] array, where each row represent the min and
                max for the corresponding channel in img.
        nbins (int): number of bins to use for the histogram

    Returns:
        features (np.array): [d * nbin]. A 1D array.

    Examples::
            >>> img = imageio.imread("../test.jpg")
            >>> mask = np.zeros(shape=[img.shape[0], img.shape[1]])
            >>> mask[100:200, 30: 90] = 1
            >>> hist = patch_feature_histogram(features, mask)
            >>> print(features.shape)
    """
    h, w, d = img.shape
    n = np.sum(mask) # number of pixels in the mask
    
    if ranges is not None:
        assert(ranges.shape == (d, 2))

    vals = img[mask.astype(bool), :] # [n, d] array
    assert(vals.shape[0] == n)

    rst = []

    for i in range(d):
        if ranges is not None:
            r = (ranges[i, 0], ranges[i, 1])
        else:
            r = (np.min(img[:, :, i]), np.max(img[:, :, i]))
        hist, edges_ = np.histogram(vals[:, i], range=r, bins=nbins)
        hist = hist / np.sum(hist) # normalize it to have the percetage instead of count
        rst += hist.reshape(-1).tolist()

    rst = np.array(rst).reshape(-1)
    return rst


def roi_feature_histogram(img, x0, y0, x1, y1, ranges=None, nbins:int=64):
    """Compute color histogram for the ROI in a big image
    The patch can be arbitray shape, and disconnected.


    Args:
        img (np.array): [h, w, d] array, where each channel is a feature space. It can be rgb image
                        lbp, lab, or any kind of channel.
        mask (np.array): [h, w] boolean mask that defines which region to use
        x0 (int): upper left x
        y0 (int): upper left y
        x1 (int): lower right x
        y1 (int): lower right y
        nbins (int): number of bins to use for the histogram

    Returns:
        features (np.array): [d * nbin]. A 1D array.

    Examples::
            >>> img = imageio.imread("../test.jpg")
            >>> mask = np.zeros(shape=[img.shape[0], img.shape[1]])
            >>> mask[100:200, 30: 90] = 1
            >>> hist = patch_feature_histogram(features, mask)
            >>> print(features.shape)
    """
    h, w, d = img.shape

    vals = img[y0:y1, x0:x1, :] # [n, d] array

    rst = []

    for i in range(d):
        if ranges is not None:
            r = (ranges[i, 0], ranges[i, 1])
        else:
            r = (np.min(img[:, :, i]), np.max(img[:, :, i]))
        
        hist, edges_ = np.histogram(vals[:, i], range=r, bins=nbins)
        hist = hist / np.sum(hist) # normalize it to have the percetage instead of count
        rst += hist.reshape(-1).tolist()

    rst = np.array(rst).reshape(-1)
    return rst

# %%

#def worker(img, masks, label_id, return_dict):
#    binary_mask =  masks == label_id
#    feature = patch_feature_histogram(img, binary_mask)
#    return_dict[label_id] = feature
#    return feature


def worker(img, masks, label_id):
    binary_mask = masks == label_id
    feature = patch_feature_histogram(img, binary_mask)
    return feature
    
    
#def hist_features_for_all_superpixel(img, masks):
#    tic = time.time()
#    
#    manager = multiprocessing.Manager()
#    return_dict = manager.dict()
#    jobs = []
#    for label_id in tqdm.tqdm(np.unique(masks)):        
#        p = multiprocessing.Process(target=worker, args=(img, masks, label_id, return_dict))
#        jobs.append(p)
#        p.start()
#    print(jobs)
#    for proc in jobs:
#        proc.join()
#    print(return_dict.keys())
#    
#    toc = time.time()
#    print("It takes %.2f seconds for feature extraction" % (toc - tic))
#    
#    return return_dict


def hist_features_for_all_superpixel(img, masks, workers:int=16):
    """
    This function takes about 4 minutes. 2x faster than without multi-thread

    
    """
    tic = time.time()

    pool = ThreadPool(workers)
    results = []
    for label_id in np.unique(masks):        
        results.append(pool.apply_async(worker, args=(img, masks, label_id)))
    
    pool.close()
    pool.join()
    
    results = [r.get() for r in results]
    
    toc = time.time()
    print("It takes %.2f seconds for feature extraction" % (toc - tic))

    # X is [n, p] array.     
    X = np.concatenate([_.reshape(1, -1) for _ in results], axis=0)    
    
    return X
    
    
# %%
if __name__ == "__main__":
    img = imageio.imread("../test.jpg")
    mask = np.zeros(shape=[img.shape[0], img.shape[1]])
    mask[100:200, 30: 90] = 1
    hist = patch_feature_histogram(img, mask)
    print(hist)
    print(hist.shape)

    hist2 = roi_feature_histogram(img, 30, 90, 100, 200)

    img = imageio.imread("../test.jpg")
    features = extract_features_map_for_slice(img)
    print(features)
    print(features.shape)

