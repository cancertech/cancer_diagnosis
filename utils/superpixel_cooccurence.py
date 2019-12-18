import numpy as np
from collections import defaultdict

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


