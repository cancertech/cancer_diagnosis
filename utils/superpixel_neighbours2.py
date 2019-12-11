import numpy as np
from collections import defaultdict, Counter
from skimage.segmentation import slic


def neighbours(sp_labels):
    """
    Get the boolean adjacency matrix for the input superpixel labels. 
    This function runs in O(h * w) time and O(n * s) space, where h = height of 
    the image, w = width of the image, n = number of superpixels in the image,
    and s = average number of pixels in each superpixel.

    Here, we use zero-indexing for everything (e.g. the superpixel ID are in 
                                               range [0, n) )

    
    
    Args:
        sp_labels (np.array): size [h, w], where each pixel represent the super
            pixel ID.
        
    Output:
        neigh (dict): key is the superpixel ID, and the value is a set for its
            neighbours.
    
    """
    h, w = sp_labels.shape
    
    neigh = defaultdict(set)
    
    for i in range(1, h):
        for j in range(1, w):
            curr = sp_labels[i, j] # current pixel
            
            # Check the left pixel
            left = sp_labels[i, j - 1]
            neigh[curr].add(left)
            neigh[left].add(curr)
            
            # Check the upper pixel
            upper = sp_labels[i - 1, j]
            neigh[curr].add(upper)
            neigh[upper].add(curr)
            
            # Check the upper left pixel
            upperleft = sp_labels[i - 1, j - 1]
            neigh[curr].add(upperleft)
            neigh[upperleft].add(curr)
    
    return neigh
    
def assign_sp_cls(sp_labels, mask):
    """
    Use majority voting to assign a class label for each superpixel.
    
    
    Args:
        sp_labels (np.array): size [h, w], where each pixel represent the super
            pixel ID. 
        mask (np.array): size [h, w], where each pixel represent the class ID
            for that pixel. Here, we use zero-indexing, which means the class
            are in range [0, k)
    
    Output:
        sp_cls (dict): the key is the superpixel ID, and the value is the
            class ID for the corresponding superpixel. There are n elements
            in the sp_cls. Here, we use zero-indexing, which means the class
            are in range [0, k)
    """
#    k = np.max(mask)
    n = np.max(sp_labels) + 1

    sp_cls = {}
    
    for i in range(n):
        sp_mask_i = sp_labels == i
        values = mask[sp_mask_i] # cluster IDs
#        pdb.set_trace()
        c = Counter(list(values))        
        sp_cls[i] = c.most_common()[0][0]
        
    return sp_cls


def co_occurence(sp_labels, sp_cls, neigh=None, k=None):
    """
    Get the co-occurence features for the superpixel.
    Here, we use zero-indexing, which means the superpixel ID are in range [0, n)
    
    The time complexity is O(n * s)
    
    Args:
        sp_labels (np.array): size [h, w], where each pixel represent the super
            pixel ID. 
        sp_cls (dict): the key is the superpixel ID, and the value is the
            class ID for the corresponding superpixel. There are n elements
            in the sp_cls. Here, we use zero-indexing, which means the class
            are in range [0, k)
        neigh (dict): key is the superpixel ID, and the value is a set for its
            neighbours.
        k (int): number of classes for each pixel. If None, then use the max
            value in the sp_cls.
    
    Output:
        cooccr (np.array): co-occurence features with size (k, k), where k is 
        the number of different classes for superpixels.
    """    
    if k is None:
        k = np.max(list(sp_cls.values())) + 1
    
    cooccr = np.zeros(shape=(k, k), dtype=np.uint64)

    for sp in neigh.keys():
        neighbours_sp = neigh[sp]
        
        for sp2 in neighbours_sp:
            if sp == sp2:
                continue # we do not count occurence for its self.

            cls_i = sp_cls[sp]
            cls_j = sp_cls[sp2]
            
            cooccr[cls_i, cls_j] += 1
            cooccr[cls_j, cls_i] += 1        

    return cooccr

def superpixel_co_occurence(sp_labels, mask, k=None):
    """
    Get the superpixel co-occurence feature matrix. The process follows these
    procedures:
        - Step 1: Get neighbours for each superpixel
        - Step 2: Get the class label (by majority voting) for each pxiel
        - Step 3: Get the co-occurence features by using the information 
                extracted above.
                
    Important Note:
        we use dictionary to store the adjacency matrix to save memory, because
        the adjacency matrix is sparse and would waste lots of memory. 
        
    Let's do another math here, say the ROI is (73000, 53000, 3), which is the 
    largest ROI in the UW breast cancer dataset, and it takes more than 100 GB
    to save on disk without compression. 
    Suppose s = 3000 is the threshold for superpixel.
    Then, $n = h * w / s = math.ceil(73000 * 53000 / 3000) = 1289667$.
    Then, we only need n * 64 * 10 /1024/1024/1024 = 0.77 GB memory to build 
    the "neigh" dictionary (suppose we use uint64 long long (64 bytes) to 
    represent superpixel ID, and assume each superpixel has average 10
    neighbours)
    
        
    Args:
        sp_labels (np.array): size [h, w], where each pixel represent the super
            pixel ID. 
        mask (np.array): size [h, w], where each pixel represent the class ID
            for that pixel. Here, we use zero-indexing, which means the class
            are in range [0, k)
        k (int): number of classes for each pixel. If None, then use the max
            value in the mask.
    
    Output:
        sp_cls (dict): the key is the superpixel ID, and the value is the
            class ID for the corresponding superpixel. There are n elements
            in the sp_cls. Here, we use zero-indexing, which means the class
            are in range [0, k)
    """
    
    neigh = neighbours(sp_labels)
    sp_cls = assign_sp_cls(sp_labels, mask)
    cooccr = co_occurence(sp_labels, sp_cls, neigh)
    
    return cooccr

# %%
if __name__ == "__main__":
    import imageio
    num_pixels_per_seg = 3000
    
    img = imageio.imread("../test/test.jpg")
    mask = imageio.imread("../test/ynet_out.png")
    mask = mask.astype(np.uint8)

    num_segments = img.shape[0] * img.shape[1] // num_pixels_per_seg
    
    sp_labels = slic(img, n_segments=num_segments)
    
    # Function-wise Test
    neigh = neighbours(sp_labels) # it takes about 18.3 seconds on CPU
    sp_cls = assign_sp_cls(sp_labels, mask) # it teaks about 35 seconds on CPU
    cooccr = co_occurence(sp_labels, sp_cls, neigh) # less than 1 seconds on CPU
    
    import pickle
    pickle.dump(cooccr, open("../test/sp_cooccr.pickle", "wb"))

    # Full Procedure Test
    superpixel_co_occurence(sp_labels, mask, k=8) # it takes about 56 seconds on CPU
    