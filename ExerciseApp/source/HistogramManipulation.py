import cv2
import numpy as np
import Utilities


def applyLUT(img, lut):
    result = img.copy()

    # LUT auf gültigen Bereich bringen
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # Grauwertbild per Lookup transformieren
    result = lut[result]

    return result.astype(np.uint8)


def equalizeHistogram(img):
    result = img.copy()

    # Histogramm berechnen
    hist = calculateHistogram(img, 256)

    # Kumulatives Histogramm
    cdf = np.cumsum(hist)

    # kleinsten Grauwert mit Häufigkeit > 0 finden
    cdf_min = 0
    for v in cdf:
        if v > 0:
            cdf_min = v
            break

    total_pixels = img.shape[0] * img.shape[1]

    lut = np.zeros(256, dtype=np.uint8)

    if total_pixels == cdf_min:
        # Sonderfall: Bild hat nur einen einzigen Grauwert
        lut = Utilities.create_identity_lut().astype(np.uint8)
    else:
        for i in range(256):
            lut[i] = np.clip(
                round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255),
                0,
                255
            )

    result = applyLUT(img, lut)
    return result, lut


def findMinMaxPos(histogram):
    minPos = 0
    maxPos = 255

    # erstes Histogramm-Bin > 0
    for i in range(len(histogram)):
        if histogram[i] > 0:
            minPos = i
            break

    # letztes Histogramm-Bin > 0
    for i in range(len(histogram) - 1, -1, -1):
        if histogram[i] > 0:
            maxPos = i
            break

    return minPos, maxPos


def stretchHistogram(img):
    result = img.copy()
    lut = Utilities.create_identity_lut().astype(np.uint8)

    hist = calculateHistogram(img, 256)
    minPos, maxPos = findMinMaxPos(hist)

    if maxPos == minPos:
        # Sonderfall: konstantes Bild
        return result, lut

    for i in range(256):
        value = (i - minPos) * 255.0 / (maxPos - minPos)
        lut[i] = np.clip(round(value), 0, 255)

    result = applyLUT(img, lut)
    return result, lut


def calculateHistogram(img, nrBins):
    histogram = np.zeros([nrBins], dtype=int)

    # Falls Farbbild: in Graubild umwandeln
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            val = int(img[y, x])
            if 0 <= val < nrBins:
                histogram[val] += 1

    return histogram

def apply_log(img):
    lut = Utilities.create_identity_lut().astype(np.float32)

    # Normieren auf [0,1], dann Log-Mapping
    for i in range(256):
        x = i / 255.0
        lut[i] = 255.0 * np.log1p(x) / np.log(2.0)

    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return applyLUT(img, lut), lut


def apply_exp(img):
    lut = Utilities.create_identity_lut().astype(np.float32)

    # Exponentialfunktion auf [0,1]
    # so normiert, dass 0 -> 0 und 1 -> 255
    e_minus_1 = np.e - 1.0
    for i in range(256):
        x = i / 255.0
        lut[i] = 255.0 * (np.exp(x) - 1.0) / e_minus_1

    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return applyLUT(img, lut), lut


def apply_inverse(img):
    lut = Utilities.create_identity_lut().astype(np.uint8)

    for i in range(256):
        lut[i] = 255 - i

    return applyLUT(img, lut), lut


def apply_threshold(img, threshold):
    lut = Utilities.create_identity_lut().astype(np.uint8)

    threshold = int(np.clip(threshold, 0, 255))

    for i in range(256):
        lut[i] = 0 if i < threshold else 255

    return applyLUT(img, lut), lut


def apply_contrast_sigmoid(img, factor):
    x = np.linspace(0, 1, 256)

    # Sigmoid um 0.5 zentriert
    lut = 1.0 / (1.0 + np.exp(-factor * (x - 0.5)))
    lut = lut * 255.0

    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return applyLUT(img, lut), lut


def apply_contrast(img, factor):
    lut = Utilities.create_identity_lut().astype(np.float32)

    # lineare Kontraständerung um Mittelgrau 128
    for i in range(256):
        lut[i] = factor * (i - 128) + 128

    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return applyLUT(img, lut), lut


def apply_exposure(img, ev):
    lut = Utilities.create_identity_lut().astype(np.float32)

    # EV-Schritte: Multiplikation mit 2^ev
    exposure_factor = 2.0 ** ev

    for i in range(256):
        lut[i] = i * exposure_factor

    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return applyLUT(img, lut), lut