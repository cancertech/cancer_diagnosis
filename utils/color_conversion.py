import numpy as np


# Construct the HE conversion matrix
M = np.array([ [0.65, 0.70, 0.29],
     [0.07, 0.99, 0.11],
     [0.27, 0.57, 0.78] 
     ]
    )

# Normalize to have unit length 1
M = np.divide(M, np.sum(M, axis=1).reshape(3,1))
M[np.isnan(M)] = 0
M = np.linalg.inv(M)


def rgb2he(img):
    """Convert an RGB image the HED image.

    Args:
        img (np.array): [h, w, 3] RGB image
        
    Returns:
        he (np.array): [h, w, 3]. The first dimension is H, the second is E, and the third is D.

    Examples::
            >>> img = imageio.imread("../test.jpg")
            >>> he = rgb2he(img)
            >>> print(he)
            >>> print(he.shape)
    """
    # TODO: HE color conversion and normalization according to https://github.com/mitkovetta/staining-normalization
    return np.matmul(img, M.T)


# %%
if __name__ == "__main__":
    from PIL import Image
    
    x = Image.open("../test.jpg")

    y = rgb2he(np.array(x))
    
    
    import matplotlib.pyplot as plt
    
    plt.imshow(y[:, :, 0], cmap="gray")
    plt.show()
    plt.imshow(y[:, :, 1], cmap="gray")
    plt.show()
    plt.imshow(y[:, :, 2], cmap="gray")
    plt.show()    
    # %%
#    y0 = Image.fromarray(y[:, :, 0].astype(np.uint8))
    y0 = Image.fromarray((y[:, :, 0]  - np.min(y[:, :, 0])/ (np.max(y[:, :, 0] - np.min(y[:, :, 0]))) * 255).astype(np.uint8))
    y0.save("test_h.jpg")
    

    
#    y1 = Image.fromarray(y[:, :, 1].astype(np.uint8))
    y1 = Image.fromarray((y[:, :, 1]  - np.min(y[:, :, 1])/ (np.max(y[:, :, 1] - np.min(y[:, :, 1]))) * 255).astype(np.uint8))
    y1.save("test_e.jpg")