import os
import glob
import sklearn
import pickle

# %%
model_p = "../models"
models = glob.glob("../models/*.pkl")

# %%
#for modelname in models:

# %%
kmeans_filename = os.path.join(model_p, 'kmeans.pkl')

loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))

# %%
hcluster_filename = os.path.join(model_p, 'hcluster.pkl')
loaded_hcluster = pickle.load(open(hcluster_filename, 'rb'))

import sklearn.cluster


model = sklearn.cluster.AgglomerativeClustering()

old_dict = loaded_hcluster.__dict__

for k in old_dict.keys():
    model.__dict__[k] = old_dict[k]
    

pickle.dump(model, open(hcluster_filename, 'wb'))