import tkinter as tk
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class DisplayImageFlow:
    def __init__(self, master, imagepath):
        self.imagepath = imagepath
        self.colormap = mpimg.imread('../docs/tutorial_img/seg_color_map.png')
        self.index = 0

        self.master = master
        master.geometry('1200x800')
        master.title('MLCD')
        title = tk.Label(master, text='ROI Semantic Segmentation Result', font=('Arial', 24), height=1)
        title.place(relx=0.1, rely=0.05, relwidth=0.8)

        self.canvas = tk.Canvas(master, bg='white', width=800, height=350)
        self.figure = plt.figure()
        # Initialize Images and label
        name = self.get_image()
        self.create_form(self.figure)

        self.image_label = tk.Label(self.master, relief='ridge', font=('Arial', 16),
                                    borderwidth=3, height=1)
        self.image_label.config(text=name)
        # self.image_label.pack(fill=tk.X)
        self.image_label.place(relx=0.2, rely=0.9, relwidth=0.6)

        self.next = tk.Button(master, command=self.read_image_next, text="Next", borderwidth=1)
        self.prev = tk.Button(master, command=self.read_image_prev, text="Previous", borderwidth=1)
        # Initialize button status
        self.prev.config(state=tk.DISABLED)
        if len(imagepath) == 1:
            self.next.config(state=tk.DISABLED)
        self.next.place(relx=0.51, rely=0.95, relwidth=0.16)
        self.prev.place(relx=0.35, rely=0.95, relwidth=0.16)

        self.master.mainloop()

    def get_image(self):
        # paths of images to display
        imgpath = self.imagepath[self.index]
        dir = os.path.dirname(imgpath)
        base = os.path.basename(imgpath)
        name, _ = os.path.splitext(base)
        seg_img = os.path.join(dir, name + '_seg_viz.png')
        sp_img = os.path.join(dir, name + '_seg_sp_viz.png')
        # create figure
        plt.clf()
        plt.subplot(1, 4, 1)
        plt.title('ROI', fontsize=8)
        plt.imshow(mpimg.imread(imgpath))
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.title('Segmentation', fontsize=8)
        plt.imshow(mpimg.imread(seg_img))
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.title('Segmentation\n in Superpixels', fontsize=8)
        plt.imshow(mpimg.imread(sp_img))
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.title('Color Map', fontsize=8)
        plt.imshow(self.colormap)
        plt.axis('off')
        return name

    def create_form(self, figure):
        self.canvas = FigureCanvasTkAgg(figure, self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(relx=0.05, relwidth=0.9, rely=0.1, relheight=0.8)

    def read_image_next(self):
        self.index += 1
        name = self.get_image()
        # self.create_form(self.figure)
        self.canvas.draw()
        self.image_label.config(text=name)
        if self.index == len(self.imagepath)-1:
            self.next.config(state=tk.DISABLED)
        if self.index > 0:
            self.prev.config(state=tk.NORMAL)

    def read_image_prev(self):
        self.index -= 1
        name = self.get_image()
        self.canvas.draw()
        # self.create_form(self.figure)
        self.image_label.config(text=name)
        if self.index == 0:
            self.prev.config(state=tk.DISABLED)
        if self.index < len(self.imagepath)-1:
            self.next.config(state=tk.NORMAL)


if __name__ == '__main__':
    imagepath = sys.argv[1:]
    # imagepath = ['C:\\ITCR\\cancer_diagnosis-master\\output\\1180_copy_0.jpg', 'C:\\ITCR\\cancer_diagnosis-master\\output\\1180_copy_1.jpg', 'C:\\ITCR\\cancer_diagnosis-master\\output\\1180_copy_2.jpg']

    root = tk.Tk()
    dsp = DisplayImageFlow(root, imagepath)


