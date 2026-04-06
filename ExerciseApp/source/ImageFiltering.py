import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cv2
import Utilities


# apply median filter
def applyMedianFilter(img, kSize):
    if kSize % 2 == 0 or kSize < 1:  
        raise ValueError("kSize must be a positive odd number")
    
    img = Utilities.ensure_one_channel_grayscale_image(img)
    filtered_img = img.copy()
    pad = kSize // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='edge') 

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            patch = padded[y:y + kSize, x:x + kSize]
            filtered_img[y, x] = np.median(patch)  
    return filtered_img


# create a moving average kernel of arbitrary size
def createMovingAverageKernel(kSize):
    kernel = np.ones((kSize, kSize)) / (kSize * kSize)
    return kernel


def gaussian(x, y, sigmaX, sigmaY, meanX, meanY):
    exponent = -0.5 * (((x - meanX) ** 2 / sigmaX ** 2) + ((y - meanY) ** 2 / sigmaY ** 2))
    result = np.exp(exponent) / (2 * np.pi * sigmaX * sigmaY)
    return result


# create a gaussian kernel of arbitrary size
def createGaussianKernel(kSize, sigma=None):
    if sigma is None:
        sigma = kSize / 6  # rule of thumb: 3-sigma covers the kernel
    
    kernel = np.zeros((kSize, kSize))
    center = kSize // 2
    
    for y in range(kSize):
        for x in range(kSize):
            kernel[y, x] = gaussian(x, y, sigma, sigma, center, center)
    
    kernel /= kernel.sum()  # normalize so values sum to 1
    return kernel


# create a sobel kernel in x direction of size 3x3
def createSobelXKernel():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

# create a sobel kernel in y direction of size 3x3
def createSobelYKernel():
    return np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]], dtype=np.float32)


def applyKernelInSpatialDomain(img, kernel):
    img = Utilities.ensure_one_channel_grayscale_image(img)
    kSize = kernel.shape[0]
    pad = kSize // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')
    filtered_img = np.zeros_like(img, dtype=np.float32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            patch = padded[y:y + kSize, x:x + kSize]
            filtered_img[y, x] = np.sum(patch * kernel)

    return np.clip(filtered_img, 0, 255).astype(np.uint8)

# Extra: create an integral image of the given image
def createIntegralImage(img):
    img = Utilities.ensure_one_channel_grayscale_image(img)
    integral_image = np.cumsum(np.cumsum(img, axis=0), axis=1).astype(np.float64)
    return integral_image


# Extra: apply the moving average filter by using an integral image
def applyMovingAverageFilterWithIntegralImage(img, kSize):
    img = Utilities.ensure_one_channel_grayscale_image(img)
    integral = createIntegralImage(img)
    pad = kSize // 2
    filtered_img = np.zeros_like(img, dtype=np.float64)

    # pad the integral image so border pixels are handled cleanly
    integral = np.pad(integral, ((pad + 1, pad), (pad + 1, pad)), mode='edge')

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # corners of the rectangle in the padded integral image
            y1, y2 = y, y + kSize
            x1, x2 = x, x + kSize
            total = (integral[y2, x2]
                   - integral[y1, x2]
                   - integral[y2, x1]
                   + integral[y1, x1])
            filtered_img[y, x] = total / (kSize * kSize)

    return np.clip(filtered_img, 0, 255).astype(np.uint8)

# Extra:
def applyMovingAverageFilterWithSeperatedKernels(img, kSize):
    img = Utilities.ensure_one_channel_grayscale_image(img)
    
    # 1D kernel of shape (kSize,) normalized
    kernel_1d = np.ones(kSize) / kSize
    
    pad = kSize // 2
    temp = np.pad(img, ((0, 0), (pad, pad)), mode='edge').astype(np.float64)
    
    # pass 1: horizontal
    h_filtered = np.zeros_like(img, dtype=np.float64)
    for x in range(img.shape[1]):
        h_filtered[:, x] = np.dot(temp[:, x:x + kSize], kernel_1d)
    
    temp = np.pad(h_filtered, ((pad, pad), (0, 0)), mode='edge')
    
    # pass 2: vertical
    filtered_img = np.zeros_like(img, dtype=np.float64)
    for y in range(img.shape[0]):
        filtered_img[y, :] = np.dot(temp[y:y + kSize, :].T, kernel_1d)
    
    return np.clip(filtered_img, 0, 255).astype(np.uint8)

def run_runtime_evaluation(img):
    kernel_sizes = range(3, 16, 2)  # 3, 5, 7, 9, 11, 13, 15
    
    times_convolution = []
    times_separated = []
    times_integral = []

    for kSize in kernel_sizes:
        # --- Faltung ---
        start = dt.datetime.now()
        kernel = createMovingAverageKernel(kSize)
        applyKernelInSpatialDomain(img, kernel)
        times_convolution.append((dt.datetime.now() - start).total_seconds())

        # --- Separierbare Filter ---
        start = dt.datetime.now()
        applyMovingAverageFilterWithSeperatedKernels(img, kSize)
        times_separated.append((dt.datetime.now() - start).total_seconds())

        # --- Integralbild ---
        start = dt.datetime.now()
        applyMovingAverageFilterWithIntegralImage(img, kSize)
        times_integral.append((dt.datetime.now() - start).total_seconds())

    # --- Plot ---
    plt.figure()
    plt.plot(kernel_sizes, times_convolution, label='Faltung')
    plt.plot(kernel_sizes, times_separated, label='Separierbare Filter')
    plt.plot(kernel_sizes, times_integral, label='Integralbild')
    plt.xlabel('Kernelgröße w')
    plt.ylabel('Zeit (Sekunden)')
    plt.title('Laufzeitvergleich Moving-Average-Filter')
    plt.legend()
    plt.savefig('plot.png')
    plt.show()