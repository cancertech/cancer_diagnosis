import sys

sys.path.append("../utils/")

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,100).__str__()
os.environ["CV_IO_MAX_IMAGE_PIXELS"] = pow(2,100).__str__() # OpenCV 4.0+


import ml_for_dx

import tkinter as tk
import tkinter.filedialog

import numpy as np

import os
import glob
import re
import random
import psutil
import cv2



from structure_features import structure_features_for_roi

# from PIL import Image
# Image.MAX_IMAGE_PIXELS = None


# %% Helper function to process structure features
def get_structure_features(dirname, rgbname):
    b_ = os.path.basename(rgbname)
    b_ = b_[:b_.rfind(".")]
    print("Basename:", b_)
    
    of = os.path.join(dirname, b_ + "_structure_features.csv")
    
    if os.path.exists(of):
        print("The structure features already extracted for:", rgbname)
        return
    else:
        print("Now, erxtracting structure features for:", rgbname)

    o_imgname = os.path.join(dirname, b_ + "_out.png")
    
    img = cv2.imread(rgbname)    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mask = cv2.imread(os.path.join(dirname, b_ + "_seg_label.png"), 0)

    
    feats = structure_features_for_roi(img, mask, nlayers=5, n_seg_cls=8, 
                           num_pixels_per_seg=3000,
                           duct_cls=[1, 2, 5, 7], duct_label_name=o_imgname)
    
    
    f = open(of, "w")        
    f.write(str(feats[0]) + "\n")
    for duct_info in feats[1]:
        f.write(",".join([str(_) for _ in duct_info]) + "\n")
    f.close()


# %%


# %% GUI Title and head
window = tk.Tk()

l = tk.Label(window, 
    text='Train Diagnosis Models',
    font=('Arial', 24),
    width=50, height=1)
l.grid(row=0, column=0, columnspan=3)


# %% Get Data Path
def get_root_dir():
    d =  tkinter.filedialog.askdirectory(initialdir = "diagnosis_example/",
                              title = "Select Data Directory")

    root_path_var.set(d)
    return d

root_path_button = tk.Button(window, text="Select Data Directory", 
                            width=30, height=1,
              command=get_root_dir)
root_path_button.grid(row=1, column=0)

root_path_var = tk.StringVar()
root_path_var.set("diagnosis_example/")
root_path_label = tk.Label(window, textvariable=root_path_var)
root_path_label.grid(row=1, column=1)





# %% Get Diagnosis Label file
default_label_path = 'diagnosis_example/diagnosis.csv'
def get_csv_path():
    mpath = tkinter.filedialog.askopenfilename(
        initialdir="diagnosis.csv",
        title="Select label CSV file",
        default=default_label_path,
        filetypes=(("CSV", "*.csv"),
                   ("Excel", "*.xlsx"),
                   ("all files", "*.*")))

    label_path_var.set(mpath)
    return mpath



label_paths_button = tk.Button(window, text="Select Diagnosis Label File", width=30, height=1,
              command=get_csv_path)
label_paths_button.grid(row=2, column=0)

label_path_var = tk.StringVar()
label_path_var.set(default_label_path)

label_paths_label = tk.Label(window, textvariable=label_path_var)
label_paths_label.grid(row=2, column=1)


# %% Select which model to use
model_name_idx = tk.IntVar()
model_name_idx.set(0)  # initializing the choice, i.e. Python

ml_model_names = ['SVC', "Random Forest", "Ada"]

def ShowChoice():
    print(model_name_idx.get())

tk.Label(window, 
         text="Choose the Machine Learning Model",
         justify = tk.LEFT,
         padx = 20).grid(row=3, column=0)


base_row = 3
for val, model_name in enumerate(ml_model_names):
    tk.Radiobutton(window, 
                  text=model_name,
                  padx = 20, 
                  variable=model_name_idx, 
                  command=ShowChoice,
                  value=val).grid(row=base_row, column=1 + val)



# %% Select lambda if using SVC
def create_slider(row, title, v0, v1, resolution=1):
    tmp_label = tk.Label(window, text=title)
    tmp_label.grid(row=row, column=0, columnspan=1)

    tmp_slider = tk.Scale(from_=v0, to=v1, orient=tk.HORIZONTAL, resolution=resolution)
    tmp_slider.set((v0 + v1 / 2))
    tmp_slider.grid(row=row, column=1, columnspan=1)
    
    return tmp_slider

lam_slider = create_slider(5, "Penalty for SVC", v0=1, v1=1000)


num_runs_slider = create_slider(6, "Number of Runs", v0=1, v1=100)


# %% Select which feature to use
tk.Label(window, text="Select Features:").grid(row=11, column=0)
use_structure = tk.IntVar()
tk.Checkbutton(window, text="Use Structure Features", variable=use_structure).grid(row=11, column=1)
use_freq = tk.IntVar()
tk.Checkbutton(window, text="Use Frequency Features", variable=use_freq).grid(row=11, column=2)
use_cooc = tk.IntVar()
tk.Checkbutton(window, text="Use Co-occurence Features", variable=use_cooc).grid(row=11, column=3)

# %% Run the training
def print_error(msg):
            
    msg_window = tk.Tk()
        
    tk.Message(msg_window, 
                   text=msg,
                   width=800,
                   bg="white",
                   font=('Arial', 20)
                   ).pack() 
    
    
def run_training():

    BASE_DIR = root_path_var.get().replace("\\", "/")
    
    lam = int(lam_slider.get()) # batch size
    
    model_idx = model_name_idx.get()
    model_name = ml_model_names[model_idx]
    
    csvname = label_path_var.get().replace("\\", "/")
    
    print("Root Dir", BASE_DIR)
    print("Lambda", lam)
    print("model_idx", model_idx)
    print("model_name", model_name)
    print("csvname", csvname)
    
    print("Features:", use_structure.get(), use_freq.get(), use_cooc.get())

    if use_structure.get() or use_freq.get() or use_cooc.get():
        print("Good, we have at least one feature")
    else:
        print_error("You need AT LEAST ONE feature! Please select some!")
        return
        
    ml_for_dx.STRUCTURE_FEATURE_DIR = BASE_DIR
    ml_for_dx.SP_FEATURE_DIR = BASE_DIR
    ml_for_dx.OUT_DIR = BASE_DIR

    ml_for_dx.MIN_DUCT_SIZE_FOR_STRUCTURE = 500
    
    ml_for_dx.INCLUDE_STRUCTURE_FEATURES = bool(use_structure.get())
    ml_for_dx.INCLUDE_FREQUENCY_FEATURES = bool(use_freq.get())
    ml_for_dx.INCLUDE_COOCCURENCE_FEATURES = bool(use_cooc.get())
    
    ml_for_dx.LABEL_CSV = csvname
    ml_for_dx.NUM_RUNS = int(num_runs_slider.get())
    
    ml_for_dx.TEMP_FEATURE_CSV = os.path.join(ml_for_dx.OUT_DIR, 
                                              "features_debug.csv")

    if use_structure.get():
        for maskname in glob.glob(os.path.join(BASE_DIR, "*_seg_label.png")):
            rgbname = maskname.replace("_seg_label.png", ".jpg")
            get_structure_features(BASE_DIR, rgbname)


    
    print(ml_for_dx)
    
    # Begin ML
    ml_for_dx.parse_features()
    ml_for_dx.train_val_models(lam, model_name)
    
    
    print("Done!")
    
    
    
run_button = tk.Button(window, text="Begin Training", width=40, height=2,
                      command=run_training)
run_button.grid(row=15, column=0, columnspan=3)


# %%

window.mainloop()