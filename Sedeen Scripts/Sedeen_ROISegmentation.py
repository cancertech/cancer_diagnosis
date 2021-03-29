import sys
import os
sys.path.append('../')
from utils.pixel_classification import process
from tkinter import *
from tkinter import messagebox, filedialog, scrolledtext

import os
import pdb
import numpy as np
import subprocess


class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see(END)
        self.widget.configure(state="disabled")
        root.update()

root = Tk()
root.title('MLCD')
root.geometry('800x600')
l = Label(root, text='Semantic Segmentation', font=('Arial', 24), width=50, height=1)
l.place(relx=0.1, relwidth=0.8, rely=0.05)
text = scrolledtext.ScrolledText(root, width=50, height=15, relief='solid', padx=2, pady=2, 
    spacing1=5, spacing2=8, spacing3=5, state=DISABLED, wrap=WORD)
text.place(relx=0.05, relwidth=0.9, rely=0.15, relheight=0.8)
sys.stdout = TextRedirector(text, "stdout")
sys.stderr = TextRedirector(text, "stderr")
root.update()


class ROISegmentation:
    """
    class Segmenting ROI.

    """
    def __init__(self, pathtoimage, output_path, batchsize=5):
        """
        Generate Segmentation Object
        """
        self._image_path = pathtoimage
        # self._output_path = os.path.dirname(output_path)

        self._model_path = os.path.abspath('../YNet/stage2/pretrained_model_st2/ynet_c1.pth')

        self._batch_size = batchsize


    def run(self):
        MODEL_PATH = self._model_path
        IMG_PATHS = self._image_path
        print('Image Paths:')
        print(IMG_PATHS)
        # OUT_DIR = self._output_path
        bs = int(self._batch_size)

        messagebox.showinfo('ROI Semantic Segmentation','Segmentation might take a long time. Please wait patiently.')

        text_str = ''

        for img_path in IMG_PATHS:
            text_str += img_path + '\n'
            basename = os.path.basename(img_path)
            name = basename[:basename.rfind(".")]
            out_dir = os.path.dirname(img_path)

            try:
                process(img_path, img_arr=None, batch_size=bs,
                    model=None, weight_loc=MODEL_PATH, 
                    output_dir=out_dir, output_prefix=name)
            except Exception as e:
                window2 = Tk()
                Message(window2, text=str(img_path) + " Error encountered!\n" + str(e),
                    width=800,
                    bg="white",
                    font=('Arial', 20)).pack()
                continue

        print("Segmentation All Done!", IMG_PATHS)

        text_str += "Segmentation All Done!"
        # if root.destory() is commented out, sedeen will stuck and crash later
        root.destroy()

        os.system('python ../utils/display_imageflow.py '+' '.join(IMG_PATHS))

        return text_str.encode()


# image_path = ['C:/ITCR/cancer_diagnosis-master/output/1180_copy_0.jpg', 'C:/ITCR/cancer_diagnosis-master/output/1180_copy_1.jpg']
# bsize = 5
# seg = ROISegmentation(image_path, bsize)
# result = seg.run()
# print(result)
# root.mainloop()