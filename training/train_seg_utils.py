
import numpy as np

import os
import glob

suffix = [".jpg", ".jpeg", ".png", ".tiff", ".tif"]



def create_train_valid_txt_for_segmentation(base_dir):

    if not os.path.exists(os.path.join(base_dir, "train.txt")) or not os.path.exists(os.path.join(base_dir, "val.txt")):
        img_dir = os.path.join(base_dir, "images/")
        lab_dir = os.path.join(base_dir, "labels/")
    
        images = [glob.glob(os.path.join(img_dir, "*" + s_)) for s_ in suffix]
        images = [_ for arr in images for _ in arr] # flatten the list
        
        
        labels = []
        err_imgs = []
        
        for imgname in images:
            try:
                basename = os.path.basename(imgname[:imgname.rfind(".")]) # filename without dir and suffix
                labelname = glob.glob(os.path.join(lab_dir, basename + "*"))[0]
                labels.append(labelname)
            except Exception as e:
                print(e)
                print("We will ignore:", imgname, "because we could not find its label")
                err_imgs.append(imgname)
            
        [images.remove(_) for _ in err_imgs]
        
        # Now, we have images and labels
        
    
        
        # Use relative path
        images = [os.path.join("images", os.path.basename(_)) for _ in images]
        labels = [os.path.join("labels", os.path.basename(_)) for _ in labels]
    
        images = [_.replace("\\", "/") for _ in images] 
        labels = [_.replace("\\", "/") for _ in labels]     
        
        n = len(images)
        num_train = int(0.7 * n)
        
        train_idx = np.random.choice(range(n), num_train, replace=False)
        
        
        train_f = open(os.path.join(base_dir, "train.txt"), "w")
        valid_f = open(os.path.join(base_dir, "val.txt"), "w")
        
        for i in range(n):
            if i in train_idx:
                f = train_f
            else:
                f = valid_f
                
            f.write("%s, %s, 0\n" % (images[i], labels[i]))
        train_f.close()
        valid_f.close()


