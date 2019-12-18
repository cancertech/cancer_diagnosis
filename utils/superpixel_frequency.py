

def sp_cls_count(sp_cls, n_seg_cls=8):
    """
    Get the count (number of superpixels) for each segmentation class
    
    Args:
        sp_cls (dict): the key is the superpixel ID, and the value is the
            class ID for the corresponding superpixel. There are n elements
            in the sp_cls. Here, we use zero-indexing, which means the class
            are in range [0, k)
    
        n_seg_cls (int): number of segmentation classes

    Output:
        counts (list): a list for the count, where each index is the count
            for the corresponding segmentation class. The length of the list
            equals to the number of semantic segmentation classes.
    """
    counts = [0] * n_seg_cls
    
    for k in sp_cls.keys():
        counts[sp_cls[k]] += 1
        
    return counts
