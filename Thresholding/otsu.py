import numpy as np
import cv2
import matplotlib.pyplot as plt

def otsu(img):

    # Calculate the histogram
    hist = np.histogram(img, bins=256, range=(0, 255))[0]

    # Calculate total number of pixels
    total = np.sum(hist)

    ThresholdList = np.arange(0, 256)
    var_intra = np.zeros(256)

    for Threshold in ThresholdList:
        # Calculate background weight
        w_bg = np.sum(hist[0:Threshold]) / total if total > 0 else 0
        # Calculate foreground weight
        w_fg = np.sum(hist[Threshold:256]) / total if total > 0 else 0
        # Calculate background mean
        m_bg = (np.sum(hist[0:Threshold] * np.arange(0, Threshold)) / np.sum(hist[0:Threshold])) if np.sum(hist[0:Threshold]) > 0 else 0
        # Calculate foreground mean
        m_fg = (np.sum(hist[Threshold:256] * np.arange(Threshold, 256)) / np.sum(hist[Threshold:256])) if np.sum(hist[Threshold:256]) > 0 else 0
        # Calculate variance of background
        var_bg = (np.sum(hist[0:Threshold] * (np.arange(0, Threshold) - m_bg)**2) / np.sum(hist[0:Threshold])) if np.sum(hist[0:Threshold]) > 0 else 0
        # Calculate variance of foreground
        var_fg = (np.sum(hist[Threshold:256] * (np.arange(Threshold, 256) - m_fg)**2) / np.sum(hist[Threshold:256])) if np.sum(hist[Threshold:256]) > 0 else 0

        # Calculate intra-class variance
        var_intra[Threshold] = w_bg * var_bg + w_fg * var_fg    

    # Get the threshold that minimizes intra-class variance
    Threshold = np.argmin(var_intra)

    return Threshold

if "__main__" == __name__:
    img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

    Threshold = otsu(img)
    print("Threshold: ", Threshold)

    # Apply thresholding
    img[img > Threshold] = 255
    img[img <= Threshold] = 0

    # Shpw the images original and thresholded
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.title('Thresholded')
    plt.show()