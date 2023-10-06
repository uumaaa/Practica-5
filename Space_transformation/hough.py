import numpy as np
from Thresholding import otsu
from Border_detection import border_detection
def houghTransform(image, error):

    # Define the values of theta (angle) and rho (distance from origin)
    theta_res = 1  # Resolution of theta in degrees
    rho_res = 1    # Resolution of rho in pixels

    # Define theta and rho ranges
    theta = np.deg2rad(np.arange(-90, 90, theta_res))
    height, width, _ = image.shape
    max_rho = int(np.sqrt(height**2 + width**2))
    rho = np.arange(-max_rho, max_rho, rho_res)

    # Create Hough space
    hough_space = np.zeros((2 * max_rho, len(theta)), dtype=np.uint64)

    # Find edges
    edges = border_detection.canny_bordering(image)

    # Get edge coordinates
    y_idx, x_idx = np.where(edges > 0)

    # Calculaten the Hough space
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        for t_idx in range(len(theta)):
            r = int(x * np.cos(theta[t_idx]) + y * np.sin(theta[t_idx]))
            hough_space[r + max_rho, t_idx] += 1
    
    # Get the max and min values of the Hough space
    max_value = int(np.max(hough_space))
    min_value = int(np.min(hough_space))

    # Calculate the threshold with Otsu's method getting the maximun intra-class variance
    threshold = otsu.otsu(hough_space) - (max_value) * error # 0 <= error <= 1

    # Get the coordinates of the peaks
    y_peaks, x_peaks = np.where(hough_space > threshold)

    # Draw lines in the image
    for i in range(len(x_peaks)):
        r = rho[y_peaks[i]]
        t = theta[x_peaks[i]]
        a = np.cos(t)
        b = np.sin(t)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))