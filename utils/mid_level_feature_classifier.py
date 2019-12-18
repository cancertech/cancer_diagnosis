#import os
#import sys
#sys.path.append(os.path.dirname(__file__))

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC

import numpy as np
#from skimage.feature.texture import greycomatrix

import sklearn.linear_model
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
from PIL import Image
from sklearn.decomposition import PCA, LatentDirichletAllocation

import os
import glob
import re
import matplotlib.pyplot as plt

import tqdm
import pandas as pd
import re
import random

import pdb

import pickle

# %%
def classify_one(models, cooccurence_csv):
    """
    Args:
        cooccurence_csv (str): the CSV file path for the co-occurrence feautres
        
    Output:
        dx (int): the class for diagnosis
        dx_name (str): the name for diagnosis class
    """
    # Get the features from the CSV file
    cooc_features = open(cooccurence_csv, "r").read().split(",")
    freq_features = open(cooccurence_csv.replace("Cooccurrence", "Frequency"), "r").read().split(",")
    cooc_features = np.array([float(_.strip().rstrip()) for _ in cooc_features])
    freq_features = np.array([float(_.strip().rstrip()) for _ in freq_features])
    
    # Get the results only for upper triangle matrix (including diagonal).
    # Other parts will be all zero
    cooc_features = np.triu(cooc_features.reshape(8, 8), k=0).reshape(-1)
    
    # Normalize features
    cooc_features = cooc_features / np.sum(cooc_features)
    freq_features = freq_features / np.sum(freq_features)
    
    features = np.array(cooc_features.tolist() + freq_features.tolist()).reshape(1, -1)

    # First decision
    for experiment in  ["Invasive v.s. Noninvasive",
                        "Atypia and DCIS v.s. Benign",
                        "DCIS v.s. Atypia"]:
        pca = models[experiment + " PCA"]
        if pca is not None:
            features = pca.transform(features).reshape(1, -1)
        model = models[experiment + " model"]
        rst = model.predict(features)[0]
    
        if rst:
            if experiment == "Invasive v.s. Noninvasive":
                return 4, "Invasive"
            if experiment == "Atypia and DCIS v.s. Benign":
                return 1, "Benign"
            if experiment == "DCIS v.s. Atypia":
                return 3, "DCIS"
            raise("programming error! unknown experiment")

    if experiment == "DCIS v.s. Atypia" and not rst:
        return 2, "Atypia"
    
    raise("programming error 2! Unknown experiment and rst")


def classify_files(models, cooccurence_csv_paths):
    """
    Args:
        models (dict): a dictionary of models, which is loaded from pickle
        cooccurence_csv_paths (list): a list of feature names for the Cooccurrence 
            features
    
    Output:
        results (dict): a dictionary for results. The key is roi name, and 
            the value is a pair for (dx, dx_name)
    """
    
    rst = {}
    for csvname in cooccurence_csv_paths:
        dx, dx_name = classify_one(models, csvname)
        roi_name = os.path.basename(csvname)
        roi_name = roi_name[:roi_name.rfind("_")]
        
        rst[roi_name] = (dx, dx_name)
    return rst



if __name__ == "__main__":
    models = pickle.load(open("../mid_level_classifier_weights.pickle", "rb"))
#    rst = classify_one("../output/1180_crop_0_SuperpixelCooccurrence.csv")
    rst = classify_files(models, ["../output/1180_crop_0_SuperpixelCooccurrence.csv"])
    print(rst)