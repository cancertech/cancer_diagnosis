import os
import sys
sys.path.append(os.path.dirname(__file__))
from cascade_ml import run_model
from structure_features import structure_features_for_roi, test_plot


import numpy as np



from PIL import Image


import time


import time
import os


Image.MAX_IMAGE_PIXELS = np.Inf

# %% Global Constant
N_LAYERS = 5
N_SEG_CLS = 8
NUM_PIX_PER_SEG = 3000
DUCT_CLS = [1, 2, 5, 7]

# %%
def get_features(roi_img_path, mask_img_path, out_dir, min_duct_size=50):
    b_ = os.path.basename(roi_img_path)
    b_ = b_[:b_.rfind(".")]
    o_imgname = os.path.join(out_dir, b_ + ".png")    

    img = Image.open(roi_img_path)    
    img = np.array(img)
    mask = Image.open(mask_img_path)
    mask = np.array(mask).astype(np.uint8)

    of = roi_img_path.replace(".jpg", "_structure_features.csv")
    
    if not os.path.exists(of):
        print("Calculating structure features. This may take few minutes.")
        feats = structure_features_for_roi(img, mask, nlayers=N_LAYERS, 
                                       n_seg_cls=N_SEG_CLS, 
                                       num_pixels_per_seg=NUM_PIX_PER_SEG,
                                       duct_cls=DUCT_CLS, 
                                       duct_label_name=o_imgname)       
        f = open(of, "w")        
        f.write(str(feats[0]) + "\n")
        for duct_info in feats[1]:
            f.write(",".join([str(_) for _ in duct_info]) + "\n")
        f.close()        
        
    print("Loading structure features")
    text = open(of, "r").read()        
    data = text.split("\n")[1:]
    data = [_ for _ in data if len(_) > 10]
    data = ["[" + _ + "]" for _ in data]
    data = [eval(_) for _ in data]
    data = np.array(data)

    size_per_duct = np.sum(data, axis=1)
    # Get ducts with proper size
    big_duct_idx = size_per_duct > min_duct_size
    data_filtered = data[big_duct_idx, :]
    features = np.sum(data_filtered, axis=0)

    print(time.ctime(), b_, "DONE!!!!")
        
    # Normalize the feature vector
    features = np.array(features) / np.sum(features)

    return features

# %%
def classify_one(models, mask_img_path):
    """
    Args:
        mask_img_path (str): the mask image path
        
    Output:
        dx (int): the class for diagnosis
        dx_name (str): the name for diagnosis class
    """
    roi_img_path = mask_img_path.replace("_seg_label.png", ".jpg")
    out_dir = os.path.dirname(mask_img_path)
    
    features = get_features(roi_img_path, mask_img_path, out_dir).reshape(1, -1)

    return run_model(models, features)
    


# %%
def classify_files(models, mask_img_paths):
    """
    Need the mask_img_path to ends with _seg_label.png, and the actual ROI 
    images in the same folder.
    
    Args:
        models (dict): a dictionary of models, which is loaded from pickle
        mask_img_paths (list): a list of image masks from the semantic segmentation
    
    Output:
        results (dict): a dictionary for results. The key is roi name, and 
            the value is a pair for (dx, dx_name)
    """
    
    rst = {}
    for mask_path in mask_img_paths:
        roi_name = os.path.basename(mask_path)
        assert(roi_name.rfind("_seg_label.png") >= 0)
        roi_name = roi_name[:roi_name.rfind("_seg_label.png")]
        dx, dx_name = classify_one(models, mask_path)

        rst[roi_name] = (dx, dx_name)
    return rst
