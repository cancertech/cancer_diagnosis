from run_ynet import process

import tkinter as tk
import tkinter.filedialog

import psutil
import os
import pdb

# %%

window = tk.Tk()

l = tk.Label(window, 
    text='Semantic Segmentation',
    font=('Arial', 24),
    width=50, height=1)
l.grid(row=0, column=0, columnspan=3)




# %%
default_model_path = os.path.abspath('YNet/stage2/pretrained_model_st2/ynet_c1.pth')
def get_model_path():
    mpath = tkinter.filedialog.askopenfilename(
        initialdir="./YNet/stage2/pretrained_model_st2/",
        title="Select Model file",
        default=default_model_path,
        filetypes=(("PyTorch Model", "*.pth"),
        ("all files", "*.*")))

    model_path_var.set(mpath)
    return mpath

#model_path_button = tk.Button(window, text="Select CNN Model", width=30, height=1,
#              command=get_model_path)
#model_path_button.grid(row=1, column=0)
model_path_var = tk.StringVar()
model_path_var.set(default_model_path)
#model_path_label = tk.Label(window, textvariable=model_path_var)
#model_path_label.grid(row=1, column=1)



# %%
def get_img_paths():
    fn =  tkinter.filedialog.askopenfilenames(initialdir = "./",
                              title = "Select Image files",
                              filetypes = (("all files","*.*"),
                                           ("TIFF", "*.tiff"), 
                                           ("jpeg files","*.jpg"), 
                                           ("png files","*.png")))

    img_paths_var.set(fn)
    return fn

img_paths_button = tk.Button(window, text="Select Images", width=30, height=1,
              command=get_img_paths)
img_paths_button.grid(row=2, column=0)

img_paths_var = tk.StringVar()
img_paths_var.set("Image Path Goes Here")
img_paths_label = tk.Label(window, textvariable=img_paths_var)
img_paths_label.grid(row=2, column=1)

# %%
def get_out_dir():
    d =  tkinter.filedialog.askdirectory(initialdir = "./",
                              title = "Select Output Directory")

    out_path_var.set(d)
    return d

out_path_button = tk.Button(window, text="Select Output Directory", 
                            width=30, height=1,
              command=get_out_dir)
out_path_button.grid(row=3, column=0)

out_path_var = tk.StringVar()
out_path_var.set("./")
out_path_label = tk.Label(window, textvariable=out_path_var)
out_path_label.grid(row=3, column=1)

# %%  Get batch size
available_memory_GB = psutil.virtual_memory().available / 1024 ** 3

if available_memory_GB > 8: # GB
    default_batch_size = 10
elif available_memory_GB > 6:
    default_batch_size = 5
else:
    default_batch_size = 1
    

bs_label = tk.Label(window, text="Batch Size")
bs_slider = tk.Scale(from_=1, to=20, orient=tk.HORIZONTAL)
bs_slider.set(default_batch_size)
bs_label.grid(row=5, column=0, columnspan=1)
bs_slider.grid(row=5, column=1, columnspan=1)

# %%
def begin_task():
    model_path = model_path_var.get()
    img_paths = eval(img_paths_var.get())
    print(img_paths)
    out_dir = out_path_var.get()
    bs = int(bs_slider.get()) # batch size
    
    for img_path in img_paths:
    
        basename = os.path.basename(img_path)
        name = basename[:basename.rfind(".")]
        
        window2 = tk.Tk()
        
        try:
#            pdb.set_trace()
            process(img_path, img_arr=None, batch_size=bs,
                model=None, weight_loc=model_path, 
                output_dir=out_dir, output_prefix=name)
        except Exception as e:
            tk.Message(window2,
                       text=str(img_path) + " Error encountered!\n" + str(e),
                       width=800,
                       bg="white",
                       font=('Arial', 20)
                       ).pack()
    
            continue
    
    
    tk.Message(window2, 
                   text="Success!",
                   width=800,
                   bg="white",
                   font=('Arial', 20)
                   ).pack() 
    
    print("Segmentation All Done!", img_paths)


go_button = tk.Button(window, text="Begin Segmentation", width=40, height=2,
                      command=begin_task)
go_button.grid(row=7, column=0, columnspan=3)

# %%
l = tk.Label(window, 
    text="""
    Make sure 
    
    For any problems, please contact beibin@uw.edu
    """,
    font=('Arial', 12),
    width=50, height=1)
l.grid(row=9, column=0, columnspan=3)

# %%

window.mainloop()
