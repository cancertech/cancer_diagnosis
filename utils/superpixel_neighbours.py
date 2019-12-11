import numpy as np
from collections import Counter
from skimage.segmentation import slic


def neighbours(sp_labels):
    """
    Get the boolean adjacency matrix for the input superpixel labels. 
    This function runs in O(h * w) time and O(n^2) space, where h = height of 
    the image, w = width of the image, n = number of superpixels in the image.

    Here, we use zero-indexing for everything (e.g. the superpixel ID are in 
                                               range [0, n) )

    
    
    Args:
        sp_labels (np.array): size [h, w], where each pixel represent the super
            pixel ID.
        
    Output:
        neigh (np.array): boolean numpy array with size (n, n), where n is the 
            number of superpixels in the image. if neigh[i, j] is 1, it means
            superpixel i and superpixel j are adjacent (neighbour) to each other.
    
    """
    h, w = sp_labels.shape
    n = np.max(sp_labels) + 1
    
    neigh = np.zeros(shape=(n, n), dtype=np.ubyte)
    
    for i in range(1, h):
        for j in range(1, w):
            curr = sp_labels[i, j] # current pixel
            
            # Check the left pixel
            left = sp_labels[i, j - 1]
            neigh[curr, left] = 1
            
            # Check the upper pixel
            upper = sp_labels[i - 1, j]
            neigh[curr, upper] = 1
            
            # Check the upper left pixel
            upperleft = sp_labels[i - 1, j - 1]
            neigh[curr, upperleft] = 1
    
    # The adjacency matrix should be symmetric. 
    # Get the maximum value with diagnomal
    for i in range(n):
        for j in range(n):
            neigh[i, j] = max(neigh[i, j], neigh[j, i])
    
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
    
    Args:
        sp_labels (np.array): size [h, w], where each pixel represent the super
            pixel ID. 
        sp_cls (dict): the key is the superpixel ID, and the value is the
            class ID for the corresponding superpixel. There are n elements
            in the sp_cls. Here, we use zero-indexing, which means the class
            are in range [0, k)
        neigh (np.array): boolean numpy array with size (n, n), where n is the 
            number of superpixels in the image. if neigh[i, j] is 1, it means
            superpixel i and superpixel j are adjacent (neighbour) to each other.
        k (int): number of classes for each pixel. If None, then use the max
            value in the sp_cls.
    
    Output:
        cooccr (np.array): co-occurence features with size (k, k), where k is 
        the number of different classes for superpixels.
    """
    n = np.max(sp_labels) + 1
    
    if k is None:
        k = np.max(list(sp_cls.values())) + 1
    
    cooccr = np.zeros(shape=(k, k), dtype=np.uint64)
        
    for i in range(n):
        for j in range(n):
            if neigh[i, j] == 0:
                continue # not adjacent. We do not care
            if i == j:
                continue # We should not count diagonal.
                
            cls_i = sp_cls[i]
            cls_j = sp_cls[j]
            
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
        We know that $n = (h * w) / s$, where s is the average size of each 
    superpixel.
    The "s" is usually a constant (e.g. 3000 in Ezgi Mercan's paper) and small. 
    So, when the image is large and s is relative small, then n and the required
    memory O(n^2) will be large. It is possible that the result will explode
    the computer memory if ROI is large. During research, scientists have good
    computers with over 100 GB memory and do not need to worry, 
    but it is a terrible problem in production. 
    A naive solution is to cut the big image into different tiles,
    and then calculate the superpixel co-occurence for each individual tile, 
    but this method has bordering effects at the edge of tiles which is 
    not accurate enough.
    
    Let's do some math here, say the ROI is (73000, 53000, 3), which is the 
    largest ROI in the UW breast cancer dataset, and set s = 3000.
    Then, $n = h * w / s = math.ceil(73000 * 53000 / 3000) = 1289667$.
    Then, we need n**2/1024/1024/1024 = 1549.01 GB memory to build the adjacency
    matrix. Even if we can use virtual memory (which is extremely slow disk I/O), 
    it is hard to buy a harddrive with such large volume. 
    
    Here, we can use some properties of superpixels. The superpixel in the 
    first row of the image would not be adjacent to any other superpixels in the
    bottom rows of the image. So, we can calculate the class of the superpixel
    and its neighbours on-the-fly, and remove the superpixel from the memory
    when we get all of its neighbours. In this way, we do not need to build the
    adjacency matrix, and we just need O(n) for superpixel classes and O(k^2)
    for the co-occurence matrix. Moreover, this solution gives accurate result
    rather than fuzzy result in the tile solution. However, we need some
    careful thinking, algorithm, and implementation for this solution.
    
        
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
    neigh = neighbours(sp_labels) # it takes about 15 seconds on CPU
    sp_cls = assign_sp_cls(sp_labels, mask) # it teaks about 35 seconds on CPU
    cooccr = co_occurence(sp_labels, sp_cls, neigh) # it takes about 7 seconds on CPU
    
    import pickle
    pickle.dump(cooccr, open("../test/sp_cooccr.pickle", "wb"))

    # Full Procedure Test
    superpixel_co_occurence(sp_labels, mask, k=8) # it takes about 1 minute on CPU
    