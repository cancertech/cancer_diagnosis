from skimage.feature.texture import greycomatrix
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(__file__))
import feature_extraction

TAU = 0.7

def get_seg_features(img, ncls=8):
    """Extract features (based on segmentation results) for a labelled image

    Args:
        img (np.array): [h, w, 3] rgb image
        ncls (int): number of classes
            
    Returns:
        features (list): a list of features
    """
    
    vals = img.reshape(-1)
    freq_feats = [np.sum(vals == i) / vals.shape[0] for i in range(ncls)]
    
    m = greycomatrix(image=img, distances=[1],
                     angles=[0, np.pi / 2], levels=ncls,
                     symmetric=False, normed=True)
    m = np.squeeze(m)
    co_occ_feats = m.reshape(-1)

    return freq_feats + list(co_occ_feats)

    

def get_rgb_features(img, ncls=8):
    """Extract features (based on RGB channles) for a labelled image

    Args:
        img (np.array): [h, w] labelled image
        ncls (int): number of classes
            
    Returns:
        features (list): a list of features
    """
#    pdb.set_trace()
    feature_map = feature_extraction.extract_features_map_for_slice(img)
    x = feature_extraction.roi_feature_histogram(feature_map, 
                                                 0, 0, 
                                                 img.shape[1], img.shape[0], 
                                                 ranges=(0, 255))        
    return list(x)


# %%
def segmentation_features_from_csv(csvname, input_col_names, feature_cols):
    """Extract features from a CSV file, which is generated from the CNN process

    Args:
        csvname (str): path of the CSV file
        col_names (list): a list of column names
        feature_cols (list): a list of column names, which will be used for 
            feature extraction

    Returns:
        features (list): a list of features    
    """

    ncols = len(input_col_names)
    
    data = pd.read_csv(csvname, names=input_col_names)
    data["volume"] = (data.x1 - data.x0) * (data.y1 - data.y0) # number of pixels in the tile
    data.volume = data.volume / data.volume.min() # normalize it
    
    max_dx_prob = data[["dx_prob_0", "dx_prob_1", "dx_prob_2", "dx_prob_3", "dx_prob_4"]].max(axis=1)
    
    mask = max_dx_prob > TAU
    
    data_sub = data[mask]
    k = mask.sum()
    if k < 10:
        print("\n", "Warning", csvname, "has only %d tiles for features" % k)
        return [0] * ncols
    
    features = data_sub[feature_cols].values.reshape(k, -1) * data_sub.volume.values.reshape(k, -1)
    features = np.sum(features, axis=0) / data_sub.volume.sum() # normalize the features for each tile
    features = list(features.reshape(-1)) # cast to list
    
    for dx_prob in ["dx_prob_0", "dx_prob_1", "dx_prob_2", "dx_prob_3", "dx_prob_4"]:
        hist = np.histogram(data[dx_prob], bins=10, range=(0,1))[0]
        features += list(hist / np.sum(hist))
    
    return features
    
    