from utils.segmentation_features import segmentation_features_from_csv
import numpy as np

import tkinter as tk
import tkinter.filedialog

import pickle
import os
import pdb

# %% Define the feature names and features to use
model = pickle.load(open("./models/dx_mlp_model.pickle", "rb"))

DX_LABELS = ["Benign", "Atypia", "DCIS", "Invasive"]

# Define the feature names
ncols = 147
input_col_names = ["row_id", "col_id", "x0", "x1", "y0", "y1",]    

# columns that need to be extracted from the DF
extract_cols = ["dx_prob_0", "dx_prob_1", "dx_prob_2", "dx_prob_3", "dx_prob_4",]
extract_cols += ["feature_%d" % _ for _ in range(ncols - 11)]    
input_col_names += extract_cols

# columns that will be used for ML
# feat_cols = extract_cols + ["dx_prob_dx_%d_hist_%d" % (dx_id, hist_id) for dx_id in range(5) for hist_id in range(10)]    
# feat_cols = input_col_names + feat_cols
# 
# %%

window = tk.Tk()

l = tk.Label(window, 
    text='Diagnosis Prediction',
    font=('Arial', 24),
    width=50, height=1)
l.grid(row=0, column=0, columnspan=3)



# %%
def get_csv_paths():
    fn =  tkinter.filedialog.askopenfilenames(initialdir = "./",
                              title = "Select CSV files",
                              filetypes = (("CSV", "*.csv"),
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
    
    for csv_path in csv_paths:
        try:
            features = segmentation_features_from_csv(csv_path, input_col_names, extract_cols)
            features = np.array(features).reshape(1, -1)
            pred = model.predict(features)[0]
            # prob = model.predict_proba(features)[0][pred - 1]
            pred_label = DX_LABELS[pred - 1]
            preds.append(pred)
            results[csv_path] = pred
            rst_txt += "%s Preidciton: %d (%s)\n" % (os.path.basename(csv_path), pred, pred_label)
        except Exception as e:
            window2 = tk.Tk()
            tk.Message(window2,
                       text=str(csv_path) + " Error encountered!\n" + str(e),
                       width=800,
                       bg="white",
                       font=('Arial', 20)
                       ).pack()
    
            continue

    max_pred = np.max(preds)
    rst_txt += "\nFinal Preidciton: %d (%s)\n" % (max_pred, DX_LABELS[max_pred - 1])

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
