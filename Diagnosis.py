from utils.mid_level_feature_classifier import classify_files
import numpy as np

import tkinter as tk
import tkinter.filedialog

import pickle
import os
import pdb


# %% Define the feature names and features to use
model = pickle.load(open("./models/mid_level_classifier_weights.pickle", "rb"))

DX_LABELS = ["Benign", "Atypia", "DCIS", "Invasive"]

# %%

window = tk.Tk()

l = tk.Label(window, 
    text='Diagnosis Prediction',
    font=('Arial', 24),
    width=50, height=1)
l.grid(row=0, column=0, columnspan=3)



# %%
def get_csv_paths():
    if not os.path.exists("output"):
        os.mkdir("output")
    fn =  tkinter.filedialog.askopenfilenames(initialdir = "output/",
                              title = "Select CSV files",
                              filetypes = (("Co-occurence Features", "*SuperpixelCooccurrence.csv"),
                                           ("all files","*.*")))

    csv_paths_var.set(fn)
    return fn

csv_paths_button = tk.Button(window, text="Select CSV files", width=30, height=1,
              command=get_csv_paths)
csv_paths_button.grid(row=2, column=0)

csv_paths_var = tk.StringVar()
csv_paths_var.set("CSV Paths Goes Here")
csv_paths_label = tk.Label(window, textvariable=csv_paths_var)
csv_paths_label.grid(row=2, column=1)



# %%
def begin_dx_classification():
    csv_paths = eval(csv_paths_var.get())
    print(csv_paths)

    results = {}
    rst_txt =  ""
    preds = []
    
    rst = classify_files(model, csv_paths)
    
    for k in rst.keys():
        pred, pred_label = rst[k]
        rst_txt += "%s Prediction: %d (%s)\n" % (k, pred, pred_label)
        preds.append(pred)
        
    max_pred = np.max(preds)
    rst_txt += "\nFinal Prediction: %d (%s)\n" % (max_pred, DX_LABELS[max_pred - 1])

    print("Classification All Done!", csv_paths)
    print(results)

    # Show result
    window_rst = tk.Tk()
    tk.Message(window_rst, text=rst_txt, 
            width=800, bg="snow2", fg="red",font=("Arial", 20)
            ).pack()
    
    return results


go_button = tk.Button(window, text="Begin Classification", width=40, height=2,
                      command=begin_dx_classification)
go_button.grid(row=7, column=0, columnspan=3)


# %%
window.mainloop()
