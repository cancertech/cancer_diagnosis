# # Training Machine Learning Models for Expert-extracted features
# 
# 
# Beibin Li
# 05/31/2020
# 
# I have written many versions of diagnosis for different combinations of features.
# 
# Here, I put all these version together to this file for training.



from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import numpy as np

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
import sys


STRUCTURE_FEATURE_DIR = "Structure_features_2/"
SP_FEATURE_DIR = "SP_features_out_3000/"

MIN_DUCT_SIZE_FOR_STRUCTURE = 500

INCLUDE_STRUCTURE_FEATURES = True
INCLUDE_FREQUENCY_FEATURES = False
INCLUDE_COOCCURENCE_FEATURES = False
# INCUDE_PIXEL_FEATURE = False # We should not support this feature

LABEL_CSV = "../ezgi_data/diagnosis.xlsx"
NUM_RUNS = 100

OUT_DIR = "dx_rst_2020_05_31" 


TEMP_FEATURE_CSV = os.path.join(OUT_DIR, "features_debug.csv")


# Assert, we need at least one feature
assert(INCLUDE_STRUCTURE_FEATURES or 
       INCLUDE_FREQUENCY_FEATURES or 
       INCLUDE_COOCCURENCE_FEATURES)


# %%

def parse_features():
    
    # Remove existing output file in the cache.
    if os.path.exists(TEMP_FEATURE_CSV):
        os.remove(TEMP_FEATURE_CSV) 

    if LABEL_CSV.endswith(".xlsx"):
        df = pd.read_excel(LABEL_CSV)
    else:
        df = pd.read_csv(LABEL_CSV)

    df = df[["Filename", "Diagnosis"]]

    #  Get features
    feat_cols = None # array to store the names of all features

    for csvname in tqdm.tqdm(glob.glob(os.path.join(STRUCTURE_FEATURE_DIR, "*_structure_features.csv"))):
    #    print(csvname)
        name = os.path.basename(csvname).replace("_structure_features.csv", "")
        idx = df["Filename"] == name
        if (idx.sum()) != 1:
            print("Error occured!", name, idx.sum())
            continue    

        # Get the features for the CSV file
        text = open(csvname, "r").read()

        if len(text) < 100:
            print(csvname, "missing!")
            continue
        else:
            data = text.split("\n")[1:]
            data = [_ for _ in data if len(_) > 10]
            data = ["[" + _ + "]" for _ in data]
            data = [eval(_) for _ in data]
            data = np.array(data)

            size_per_duct = np.sum(data, axis=1)

            # Get ducts with proper size
            big_duct_idx = size_per_duct > MIN_DUCT_SIZE_FOR_STRUCTURE
            data_filtered = data[big_duct_idx, :]
            features = np.sum(data_filtered, axis=0)

            # Get the biggest 3 ducts
            # big_duct_idx = np.argsort(size_per_duct)[-3:]
            # features = data[big_duct_idx, :]
            # features = np.sum(features, axis=0)
            # pdb.set_trace()

            # Normalize the feature vector
            structure_features = np.array(features) / np.sum(features)
            del features
            # features = features.reshape(1, -1)
            # pdb.set_trace()


        # Get the Co-occurence and Frequency features
        cooccurence_csv = os.path.join(SP_FEATURE_DIR, name + "_SuperpixelCooccurrence.csv")
        cooc_features = open(cooccurence_csv, "r").read().split(",")
        freq_features = open(cooccurence_csv.replace("Cooccurrence", "Frequency"), "r").read().split(",")
        cooc_features = np.array([float(_) for _ in cooc_features])
        freq_features = np.array([float(_) for _ in freq_features])

        # Get the results only for upper triangle matrix (including diagonal).
        # Other parts will be all zero
        cooc_features = np.triu(cooc_features.reshape(8, 8), k=0).reshape(-1)

        # Normalize features
        cooc_features = cooc_features / np.sum(cooc_features)
        freq_features = freq_features / np.sum(freq_features)

        
        # Combine the necessary features
        features = []        
        if INCLUDE_STRUCTURE_FEATURES:
            features += structure_features.tolist()
        if INCLUDE_FREQUENCY_FEATURES:
            features += freq_features.tolist()
        if INCLUDE_COOCCURENCE_FEATURES:
            features += cooc_features.tolist()
            
            
        # Initalize the feature names for df    
        if feat_cols is None:
            feat_cols = []
            if INCLUDE_STRUCTURE_FEATURES:
                feat_cols += ["structure_feature_%d" % _ for _ in range(len(structure_features))]
            if INCLUDE_FREQUENCY_FEATURES:
                feat_cols += ["frequencySP_feature_%d" % _ for _ in range(len(freq_features))]
            if INCLUDE_COOCCURENCE_FEATURES:
                feat_cols += ["cooccurenceSP_feature_%d" % _ for _ in range(len(cooc_features))]
                
            for colname in feat_cols:
                df[colname] = None

        df.loc[idx, feat_cols] = features

        del data, features

    df.to_csv(TEMP_FEATURE_CSV, index=False) # debug only
    
    return df



# %% Machine Learning Function
def train_val_models(l1_lambda, model_name="svc"):
    ofile = open(os.path.join(OUT_DIR, "loocv_result.tsv"), "a")
    final_models = {}
    for experiment in ["Invasive v.s. Noninvasive", "Atypia and DCIS v.s. Benign", "DCIS v.s. Atypia"]:
        accs = []
        sens = []
        specs = []

        df = pd.read_csv(TEMP_FEATURE_CSV)
        df = df.dropna(axis=0)

        df["num_preds_correct"] = 0
        df["num_preds"] = 0

        reals = []
        preds = []

        df["raw_dx"] = df.Diagnosis

        if experiment == "Invasive v.s. Noninvasive":
            # Invasive v.s. Noninvasive (acc = 0.9361702127659575)
            df.Diagnosis = df.Diagnosis == 4
        elif experiment == "Atypia and DCIS v.s. Benign":
            # Atypia and DCIS v.s. Benign (acc = 0.706766917293233)
            df = df[df.Diagnosis.isin([1, 2, 3])]
            df.Diagnosis = df.Diagnosis == 1
        elif experiment == "DCIS v.s. Atypia":
            # DCIS v.s. Atypia (acc = 0.7708333333333334)
            df = df[df.Diagnosis.isin([2, 3])]
            df.Diagnosis = df.Diagnosis == 3

        print("Shape:", experiment, df.shape)

        feat_cols = [_ for _ in df.keys() if _.find("_feature_") >= 0]
        for experiment_run_id in tqdm.tqdm(range(NUM_RUNS)):

            # Leave-one-out-cross-validation
            for filename in df.Filename.unique():
                train = df[df.Filename != filename]
                test = df[df.Filename == filename]

                # up-sample
                n_per_cls = max(train.Diagnosis.value_counts()[0], train.Diagnosis.value_counts()[1]) * 2
                tmp_0 = train[train.Diagnosis == False].sample(n_per_cls, replace=True)
                tmp_1 = train[train.Diagnosis == True].sample(n_per_cls, replace=True)
                train = pd.concat([tmp_0, tmp_1])

                # Assert the validation set is NOT in the training set
                assert(test.Filename.tolist()[0] not in train.Filename.tolist())

                train_features = train[feat_cols].values
                test_features = test[feat_cols].values
                
                # Normalize features for each column
                scaler = preprocessing.StandardScaler().fit(train_features)
                train_features = scaler.transform(train_features)
                test_features = scaler.transform(test_features)
                
                train_labels = train["Diagnosis"].values.reshape(-1)
                test_labels = test["Diagnosis"].values.reshape(-1)

                if train_features.shape[0] < train_features.shape[1]:
                    dim = min(20, train_features.shape[0])
                    pca = PCA(n_components=dim)
                    pca.fit(train_features)
                    train_features = pca.transform(train_features)
                    test_features = pca.transform(test_features)
                else:
                    pca = None


                model_name = model_name.lower()
                if model_name == "svc" or model_name == "svm":
                    model = SVC(C=l1_lambda, gamma="scale", kernel="poly", degree=3, decision_function_shape="ovr", shrinking=True)
                elif model_name == "svcrbf":
                    model = SVC(C=l1_lambda, gamma="scale", kernel="rbf", decision_function_shape="ovr", shrinking=True)
                elif model_name == "logistic":
                    model = sklearn.linear_model.LogisticRegression(class_weight="balanced", C=l1_lambda, penalty="l1", solver='liblinear') # liblinear for l1 loss
                elif model_name == "rf" or model_name == "random forest":
                    model = RandomForestClassifier()
                elif model_name == "ada":
                    model = sklearn.ensemble.AdaBoostClassifier()
                elif model_name == "bagging":
                    model = sklearn.ensemble.BaggingClassifier()
                    
                model.fit(train_features, train_labels)
                pred = model.predict(test_features)
                reals += list(test_labels)
                preds += list(pred)            

                try:
                    df.loc[df.Filename == filename, "num_preds_correct"] += (test_labels == pred)[0]
                    df.loc[df.Filename == filename, "num_preds"] += 1
                except:
                    import pdb
                    print("error")
                    pdb.set_trace()
            #    print("Accuracy is:", acc)
            acc =  np.sum(np.array(reals) == np.array(preds)) / len(reals)
            sklearn.metrics.classification_report(reals, preds)

            cm = confusion_matrix(reals, preds)

            TN = cm[0][0]
            FN = cm[1][0]
            TP = cm[1][1]
            FP = cm[0][1]

            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)

            accs.append(acc)
            sens.append(sensitivity)
            specs.append(specificity)
        # End of all runs

        if sys.platform.find("win") >= 0:
            # Plot figures if we are using Windows
            plt.hist(acc)
            plt.show()
            plt.hist(sens)
            plt.show()
            plt.hist(specs)
            plt.show()

        report = [experiment, "Acc:", np.mean(accs), "Sensitivity:", np.mean(sens), "Specificity:", np.mean(specs),
                 "min_duct_size", MIN_DUCT_SIZE_FOR_STRUCTURE, "l1_lambda", l1_lambda,
                  '"%s"' % str(model), '"%s"' % str(pca), 
                  INCLUDE_STRUCTURE_FEATURES, INCLUDE_FREQUENCY_FEATURES, INCLUDE_COOCCURENCE_FEATURES]
        print(report)
        ofile.write("\t".join([str(_) for _ in report]) + "\n")
        
        final_models[experiment + " PCA"] = pca
        final_models[experiment + " model"] = model
    ofile.close()

    
    pickle_name = "weights_structure_%d_frequency_%d_cooccurence_%d_%f_%s.pickle" % (INCLUDE_STRUCTURE_FEATURES, 
                                               INCLUDE_FREQUENCY_FEATURES,
                                               INCLUDE_COOCCURENCE_FEATURES, l1_lambda, model_name)
    pickle.dump(final_models, open(os.path.join(OUT_DIR, pickle_name), "wb"))

    # %%
    cm = confusion_matrix(reals, preds)
    print(cm)



