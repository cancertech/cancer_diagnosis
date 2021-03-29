from tkinter import *
from tkinter import messagebox, filedialog
from tkinter.ttk import Progressbar
sys.path.append('../')
from utils.bag import Bag
from utils.util import get_feat_from_image, get_histogram_cluster
from utils.classifier import *
from utils.cluster import predict_kmeans
import numpy as np
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2 ** 64)
import cv2

class ROIWindowClassifier:
    """
    class Classify ROI Window.

    """

    def __init__(self, pathtoimage, output_path):
        """
        Generate Classifier Object
        """
        self._image_path = pathtoimage
        # if output_path is void, the output image will be saved in the same folder as input
        if output_path:
            self._output_basename = os.path.basename(output_path)
            self._output_folder = os.path.dirname(output_path)
        else:
            bn = os.path.basename(pathtoimage)
            bn = os.path.splitext(bn)[0]
            self._output_basename = bn + '_marked.jpg'
            self._output_folder = os.path.dirname(pathtoimage)

        self._root = Tk()
        self._root.title('MLCD')
        self._root.geometry('350x150')
        l = Label(self._root, text='ROI Window Classifier Progress', font=('Arial', 12), width=30, height=2)
        l.pack()

        self._model_path = '../models/'
        self._openslide_flag = True

        try:
            import openslide
        except (ImportError, OSError):
            self._openslide_flag = False
            import warnings
            warnings.warn('Cannot support SVS format and large TIF files', ImportWarning)

        self._progressbar = Progressbar(self._root, orient=HORIZONTAL, length=250, mode='determinate')
        self._progressbar.pack()


    def run(self):
        self._root.update()
        model_p = self._model_path
        output_p = self._output_folder
        output_bn = self._output_basename
        clf_filename = os.path.join(model_p, 'clf.pkl')
        kmeans_filename = os.path.join(model_p, 'kmeans.pkl')

        if not os.path.exists(clf_filename):
            clf_filename = filedialog.askopenfilename(initialdir="./",
                                                      title="Select Trained SVM Model File (Clf.pkl)",
                                                      filetypes=(("Pickle File", "*.pkl"),
                                                                 ("all files", "*.*")))
        clf = model_load(clf_filename)

        if not os.path.exists(kmeans_filename):
            kmeans_filename = filedialog.askopenfilename(initialdir="./",
                                                         title="Select Trained K-Means Model File (kmeans.pkl)",
                                                         filetypes=(("Pickle File", "*.pkl"), ("all files", "*.*")))
        loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))

        self._progressbar['value'] = 0
        # percent['text'] = "{}%".format(self._progressbar['value'])
        self._root.update()

        filename, ext = os.path.splitext(self._image_path)

        if ext == '.SVS':  # file format in .SVS, must be open with openslide
            if self._openslide_flag:
                im_os = openslide.Openslide(self._image_path)
                im_size = (im_os.dimensions[1], im_os.dimensions[0])
                if im_size[0] * im_size[1] > pow(2, 64):
                    # big image
                    im = None
                    im_BGR = None
                else:
                    im = im_os.read_region((0, 0), 0, im_os.dimensions).convert('RGB')
                    im = np.array(im, dtype=np.uint8)
                    im_BGR = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            else:
                messagebox.showerror("Error", "Unsupported format without openslide: {}".format(ext))
                return
        else:  # other formats, open with opencv
            # f = open("C:/ITCR/cancer_diagnosis-master/testoutput/ROIwindowlog.txt", 'a')
            # f.write("ROIWindowC_0\n")
            # f.close()
            im_BGR = cv2.imread(self._image_path)
            # f = open("C:/ITCR/cancer_diagnosis-master/testoutput/ROIwindowlog.txt", 'a')
            # f.write("ROIWindowC_1\n")
            # f.close()
            if im_BGR is None:
                messagebox.showerror("Error", "CV2 image read error: image must have less than 2^64 pixels")
                return
            im = cv2.cvtColor(im_BGR, cv2.COLOR_RGB2BGR)
            im = np.array(im, dtype=np.uint8)
            im_size = (im.shape[0], im.shape[1])

        if im is not None:
            output = np.empty((im.shape[0], im.shape[1]))
            bags = Bag(img=im, size=3600, overlap_pixel=2400, padded=True)
        elif self._openslide_flag and im_BGR is None:
            bags = Bag(h=im_size[0], w=im_size[1], size=3600, overlap_pixel=2400, padded=True)
            output = np.empty(im_size)
        else:
            messagebox.showerror("Error", "image read fail")
            return

        bn = os.path.basename(self._image_path)
        bn = os.path.splitext(bn)[0]
        feat_outname = os.path.join(os.path.dirname(self._image_path), '{}_feat.pkl'.format(bn))

        if os.path.exists(feat_outname):
            feat = pickle.load(open(feat_outname, 'rb'))
            precomputed = True
        else:
            feat = np.zeros([len(bags), 40])
            precomputed = False
            messagebox.showinfo('ROIWindowClassifier','Features are not precomputed. The following process might take 20 minutes or more!')

        result = np.zeros(len(bags))
        for i in range(len(bags)):
            self._progressbar['value'] = min((float(i + 1) / len(bags)) * 100, 100)
            self._root.update()

            if not precomputed:
                if bags.img is not None:
                    bag = bags[i][0]
                else:
                    bbox = bags.bound_box(i)
                    size_r = bbox[1] - bbox[0]
                    size_c = bbox[3] - bbox[2]
                    top_left_x = max(bbox[2] - bags.left, 0)
                    top_left_y = max(bbox[0] - bags.top, 0)
                    top_left = (top_left_x, top_left_y)
                    bag = im_os.read_region(top_left, 0, (size_c, size_r)).convert('RGB')
                    bag = np.array(bag, dtype=np.uint8)
                try:
                    feat_words = get_feat_from_image(None, False, 120, image=bag)
                    cluster_words = predict_kmeans(feat_words, loaded_kmeans)
                    hist_bag = get_histogram_cluster(cluster_words, dict_size=40)
                except np.linalg.LinAlgError:
                    result[i] = 0
                    hist_bag = [0] * 40
                    hist_bag[23] = 900
                feat[i, :] = hist_bag
                pickle.dump(feat, open(feat_outname, 'wb'))

            result[i] = model_predict(clf, [feat[i, :]])
            bbox = bags.bound_box(i)
            bbox[0] = max(0, min(bbox[0] - bags.top, im_size[0] - 1))
            bbox[1] = max(0, min(bbox[1] - bags.top, im_size[0] - 1))
            bbox[2] = max(0, min(bbox[2] - bags.left, im_size[1] - 1))
            bbox[3] = max(0, min(bbox[3] - bags.left, im_size[1] - 1))
            output[bbox[0]:bbox[1], bbox[2]:bbox[3]] = result[i]

        # draw bounding box and save
        output *= 255
        output = np.array(output, dtype=np.uint8)
        # save feature
        pickle.dump(feat, open(feat_outname, 'wb'))

        if im_BGR is None and self._openslide_flag:
            # if image is very large, scale by 8
            output = cv2.resize(output, None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_AREA)
            im_BGR = im_os.get_thumbnail((im_os.dimensions[0] // 8, im_os.dimensions[1] // 8)).convert('RGB')
            im_BGR = np.array(im_BGR, dtype=np.uint8)
            im_BGR = cv2.cvtColor(im_BGR, cv2.COLOR_RGB2BGR)

        contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        final = im_BGR.copy()
        final = cv2.drawContours(final, contours, -1, (0, 0, 255), 15)
        marked_outname = os.path.join(output_p, output_bn)
        cv2.imwrite(marked_outname, final)

        # save jpeg for segmentation
        count = 0
        bboxes = []
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            img = im_BGR[y:y + h, x:x + w, :]
            bboxes += [[y, y + h, x, x + w]]
            roi_outname = os.path.join(output_p, '{}_{}.jpg'.format(bn, count))
            count += 1
            cv2.imwrite(roi_outname, img)

        display_im = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        # draw_area = final.copy()
        # scale_side = max(draw_area.shape[0], draw_area.shape[1])
        # if scale_side > 800:
        #     scale_factor = float(scale_side) / 500
        #     final_resized = cv2.resize(draw_area, None, fx=1 / scale_factor, fy=1 / scale_factor,
        #                                interpolation=cv2.INTER_AREA)
        # else:
        #     final_resized = draw_area
        # display_im2 = cv2.cvtColor(final_resized, cv2.COLOR_BGR2RGB)
        self._root.destroy()

        # Save roi contours
        contours_path = os.path.join(output_p, '{}_contours.txt'.format(bn))
        f = open(contours_path, 'w')
        for cont in contours:
            for point in cont:
                point = point.squeeze()
                f.write('({},{}) '.format(point[0], point[1]))
            f.write('\n')
        f.close()

        return contours_path.encode()
        # return display_im.astype(np.uint8)
        # return final.astype(np.uint8)


# image_path = 'C:/ITCR/cancer_diagnosis-master/data/1180_copy.jpg'
# output_path = 'C:/ITCR/cancer_diagnosis-master/testoutput/test.jpg'
# classifier = ROIWindowClassifier(image_path, output_path)
# classifier.run()