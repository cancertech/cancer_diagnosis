import os
import sys
sys.path.append(os.path.dirname(__file__))
from segmentation_features import get_seg_features
from superpixel_classification import mask_to_superpixel_co_occurence
sys.path.insert(0, '../YNet/stage2/')
sys.path.insert(0, 'YNet/stage2/')

import Model as Net

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.autograd import Variable
from PIL import Image

import cv2

import sys
import time
import math
import os
import pdb

pallete = [ 255, 255, 255,
            130, 0, 130,
            0, 0, 130,
            255, 150, 255,
            150 ,150 ,255,
            0 ,255 ,0,
            255, 255 ,0,
            255, 0, 0]
    
    
def process(imagename, img_arr=None, batch_size=10, model=None, 
            weight_loc='../YNet/stage2/pretrained_model_st2/ynet_c1.pth',
            output_dir="out", output_prefix="unknown"):
    """ Run the CNN Segmentation
    
    The intermediate results will be saved into _seg_label.png (segmentation label), 
    _seg_viz.png (segmentation visualization), and CSV (features for all tiles) files.
    These files can be used for future diagnosis prediction.
    
    Note: we assume the input image is at least 384 x 384 pixel.
    Here, we use uint8 [0-255] to represent images for consistency. If the 
    result of some function is not uint8, cast it to uint8.
    
    Args:
        imagename (str): Location of the input image
        img_arr (numpy.array): if the image array is given, then we do not read
                            from the image name
        batch_size (int): batch size for the CNN. It depends on the hardware.
                         With larger batch size, the CNN can finish running
                         faster.
        model (torch.Module): the pyTorch model. If the model is given, then
                   this function would not load the model from hard drive.
        weight_loc (str): if the model is not specified, load the weights of 
                        model from this location.
        output_dir (str): output directory
        output_prefix (str): the prefix for the output files (e.g. subject id,
                      image id, roi id, etc.)
        
    Returns:
        output (np.array): The predicted segmentation with [h, w] shape
    """
    
    # %% Load Model
    if model is None:
        model = Net.ResNetC1_YNet(8, 5)
        model.load_state_dict(torch.load(weight_loc, map_location="cpu"))
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    output_log_name = os.path.join(output_dir, output_prefix + "_seg.log")
    output_mask_name = os.path.join(output_dir, output_prefix + "_seg_label.png")
    output_rgb_name = os.path.join(output_dir, output_prefix + "_seg_viz.png")
    output_sp_name = os.path.join(output_dir, output_prefix + "_seg_sp_viz.png")
    of_freq = os.path.join(output_dir, output_prefix + "_SuperpixelFrequency.csv")
    of_cooc = os.path.join(output_dir, output_prefix + "_SuperpixelCooccurrence.csv")
    
    # %% Setup constants
    aoi_width = 384
    aoi_height = 384 
    
    aoi_center_w = 128
    aoi_center_h = 128
    
    border_w = (aoi_width - aoi_center_w) // 2
    border_h = (aoi_height - aoi_center_h) // 2
    

    
    # %%
    if img_arr is None:
        whole_img = cv2.imread(imagename).astype(np.float32)
        whole_img /=255
    else:
        whole_img = img_arr.copy()        
    
    
    # assumption, the whole image has at least width and height for one patch
    assert(whole_img.shape[0] > aoi_height and whole_img.shape[1] > aoi_width)
    
        
    image_dict = {}
    
    nrows = math.ceil((whole_img.shape[0] - 2 * border_h) / aoi_center_h) 
    ncols = math.ceil((whole_img.shape[1] - 2 * border_w)/ aoi_center_w)
    
    for row_id in range(nrows):
        for col_id in range(ncols):
            idx = (row_id, col_id)
            
    
            h0 = aoi_center_h * row_id
            h1 = h0 + aoi_height
            if h1 > whole_img.shape[0]: # the aoi exceed the whole image
                h1 = whole_img.shape[0]
                h0 = h1 - aoi_height 
    
    
            w0 = aoi_center_w * col_id
            w1 = w0 + aoi_width
            if w1 > whole_img.shape[1]: # the aoi exceed the whole image
                w1 = whole_img.shape[1]
                w0 = w1 - aoi_width 
    
            img_aoi = whole_img[h0:h1, w0:w1, :]
            
            assert(img_aoi.shape == (aoi_height, aoi_width, 3))
            
            image_dict[idx] = img_aoi
    
    
    # %%
    fstream = open(output_log_name, "w")
    
    tic = time.time()
    output = np.zeros(shape=[whole_img.shape[0], whole_img.shape[1]])
#    output_dx_prob = np.zeros(shape=[whole_img.shape[0], whole_img.shape[1], 5])
    
    # The dx should be (5, 1) for each tile/instance, but I reshaped/expanded to (h, w, 5) so that
    # it can be viewed as an image format. Note that there are only 4 dx labels, but we use 
    # 5 here because the legacy code use 1 - 4 to represent classes, and hence 0 is redundant.

    # We use batch here, so that the code can run slightly faster
    aoi_locs = list(image_dict.keys()) # the (row_id, col_id) pairs
    aoi_idx = 0  # an iterator for the aois

    while aoi_idx < len(aoi_locs): 
        # Note: we cannot use for loop here because we are not sure about the
        # batch size. We have another inner for loop.
        imgs_this_batch = []
        
        if aoi_idx % (batch_size * 100) == 0:
            print(aoi_idx, "out of", len(aoi_locs), "locations have done")
            fstream.flush()

        # Setup the batch for processing
        for batch_i in range(batch_size):
            if aoi_idx >= len(aoi_locs):
                break
            img = image_dict[aoi_locs[aoi_idx]] # get the aoi image section
            img = img.transpose((2, 0, 1))
            img = img.reshape(1, 3, aoi_width, aoi_height)
            imgs_this_batch.append(img)
            aoi_idx += 1

        # Cast numpy batch to PyTorch batch
        img_tensor = torch.from_numpy(np.concatenate(imgs_this_batch))
        img_variable = Variable(img_tensor)
        if torch.cuda.is_available():
            img_variable = img_variable.cuda()

        # Process
        img_out, sal_out = model(img_variable)
        # diagnostic_lvls = torch.argmax(sal_out, dim=1)
        sal_out = torch.softmax(sal_out, dim=1)
        sal_out = sal_out.detach().cpu().numpy()

        # Store the result batch to "all_outputs"
        num_imgs_in_batch = img_variable.shape[0]
        for batch_i, img_idx in enumerate(range(aoi_idx - num_imgs_in_batch, aoi_idx)):
            
            # Clean and organize the output from CNN
            img_out_norm = img_out[batch_i]
            prob, classMap = torch.max(img_out_norm, 0)
#            classMap_numpy = classMap.data.cpu().numpy()
            segClassMap_np = classMap.data.cpu().numpy()[border_h:-border_h, border_w:-border_w]
#            im_pil = Image.fromarray(np.uint8(classMap_numpy))
#            im_pil.putpalette(pallete)
#            all_outputs[aoi_locs[img_idx]] = im_pil
            
            # reshape to the same size of the image tile
            expanded_dx_prob = np.ones((img_out.shape[2], img_out.shape[3], 5)) * sal_out[batch_i]
            expanded_dx_prob = expanded_dx_prob[border_h:-border_h, border_w:-border_w] # cut the border

            # Put the result into the numpy            
            row_id, col_id = aoi_locs[img_idx]

            h0 = aoi_center_h * row_id
            h1 = h0 + aoi_height
            if h1 > whole_img.shape[0]: # the aoi exceed the whole image
                h1 = whole_img.shape[0]
                h0 = h1 - aoi_height 
        
            w0 = aoi_center_w * col_id
            w1 = w0 + aoi_width
            if w1 > whole_img.shape[1]: # the aoi exceed the whole image
                w1 = whole_img.shape[1]
                w0 = w1 - aoi_width 

            output[h0+border_h:h1-border_h, w0+border_w:w1-border_w] = segClassMap_np
#            output_dx_prob[h0+border_h:h1-border_h, w0+border_w:w1-border_w, :] = np.array(expanded_dx_prob)
            
            # Write into fstream
            out_data = [row_id, col_id, w0+border_w, w1-border_w, h0+border_h, h1-border_h] 
            out_data += list(sal_out[batch_i].reshape(-1))
            out_data += get_seg_features(segClassMap_np)
            fstream.write(",".join([str(_) for _ in out_data]))
            fstream.write("\n")
            
#            del segClassMap_np, im_pil, expanded_dx_prob # release some memory
        del img_variable, img_out, sal_out, prob, classMap # release some memory. not sure helpful or not

    toc = time.time()
    print("it takes %.2f seconds to run the CNN" % (toc - tic))
    
    fstream.close()
    # %% Save all results
    # Save to image
    cv2.imwrite(output_mask_name, output)

    outpil = Image.fromarray(np.uint8(output))
    outpil.putpalette(pallete)
    outpil.save(output_rgb_name)
    
    # %% Get the superpixel features
#    pdb.set_trace()
    whole_img = (whole_img * 255).astype(np.uint8)
    output = output.astype(np.uint8)
    freq, cooc = mask_to_superpixel_co_occurence(whole_img, output, tile_size=5000,
                                                 viz_fname=output_sp_name)
    
    open(of_freq, "w").write(",".join([str(_) for _ in freq]))
    open(of_cooc, "w").write(",".join([str(_) for _ in cooc]))

    print(time.ctime())
    print("*" * 30, "Done saving", imagename, "*" * 30)

    return output
    
    

# %%
if __name__ == "__main__":
    # filename = "../ezgi_data/alternative_consensus_roi_jpg/2238_1.jpg"
    filename = "../test/test.jpg"
    filename = "../output/1180_crop_0.jpg"
#    filename = "../test/1883/1883_1.jpg"
    process(filename, img_arr=None, batch_size=10, model=None, output_dir="../output", output_prefix="1180_crop_0")
#    process("test/1883/1883_2.jpg", img_arr=None, batch_size=16, model=None, output_dir="out2", output_prefix="1883_2")
