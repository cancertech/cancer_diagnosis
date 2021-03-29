import sys

sys.path.append("../YNet/")
sys.path.append("../YNet/stage1/")
sys.path.append("../YNet/stage2/")

import train_seg_utils

import tkinter as tk
import tkinter.filedialog

import numpy as np

import os
import glob
import re
import random
import psutil


# %% GUI Title and head
window = tk.Tk()

l = tk.Label(window, 
    text='Train Semantic Segmentation',
    font=('Arial', 24),
    width=50, height=1)
l.grid(row=0, column=0, columnspan=3)


# %% Get Data Path
def get_root_dir():
    d =  tkinter.filedialog.askdirectory(initialdir = "segmentation_example/",
                              title = "Select Data Directory")

    root_path_var.set(d)
    return d

root_path_button = tk.Button(window, text="Select Data Directory", 
                            width=30, height=1,
              command=get_root_dir)
root_path_button.grid(row=1, column=0)

root_path_var = tk.StringVar()
root_path_var.set("segmentation_example/")
root_path_label = tk.Label(window, textvariable=root_path_var)
root_path_label.grid(row=1, column=1)


# %%

def create_slider(row, title, v0, v1, resolution=1):
    tmp_label = tk.Label(window, text=title)
    tmp_label.grid(row=row, column=0, columnspan=1)

    tmp_slider = tk.Scale(from_=v0, to=v1, orient=tk.HORIZONTAL, resolution=resolution)
    tmp_slider.set((v0 + v1 / 2))
    tmp_slider.grid(row=row, column=1, columnspan=1)
    
    return tmp_slider

bs_slider = create_slider(2, "Batch Size", v0=1, v1=30)
max_epochs_slider = create_slider(3, "Epochs", v0=10, v1=1000)

num_class_slider = create_slider(4, "Number of Classes", v0=2, v1=20)

lr_slider = create_slider(5, "Learning rate", v0=1e-4, v1=1.0, resolution=1e-4)
# %%

def print_error(msg):
            
    msg_window = tk.Tk()
        
    tk.Message(msg_window, 
                   text=msg,
                   width=800,
                   bg="white",
                   font=('Arial', 20)
                   ).pack() 
    
    
# %% Run the training
def run_training():

    BASE_DIR = root_path_var.get()
    
        
    num_classes = int(num_class_slider.get())
    bs = int(bs_slider.get()) # batch size

    
    # image_size = 512
    max_epochs = int(max_epochs_slider.get())
    lr = 1e-2
    
    print("Root Dir", BASE_DIR)
    print("Slider:", bs)
    print("Epochs:", max_epochs)
    
    # Split data to train and valid
    train_seg_utils.create_train_valid_txt_for_segmentation(BASE_DIR)
    
    
    # Check parameter integrity
    if not os.path.exists(BASE_DIR):
        print_error("Base dir %s does not exists" % BASE_DIR)
    
    if not os.path.exists(os.path.join(BASE_DIR, "train.txt")) or not os.path.exists(os.path.join(BASE_DIR, "val.txt")):
        print_error("train.txt or val.txt doesn't exist")
    
    rst = os.system("python ../YNet/stage1/main.py --data_dir {basedir} --classes {classes} --batch_size {batch_size} --lr {lr} --max_epochs {max_epochs}".format(
        basedir = BASE_DIR, classes=num_classes, batch_size=bs, lr=lr, max_epochs=max_epochs))
    
    print("Command result:", rst)

    
run_button = tk.Button(window, text="Begin Training", width=40, height=2,
                      command=run_training)
run_button.grid(row=10, column=0, columnspan=3)


# %%

window.mainloop()

