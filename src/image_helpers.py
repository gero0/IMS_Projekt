import numpy as np
import math
from PIL import Image


def pixel_to_YCbCr(color):
    (r, g, b) = color
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    Cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    # Make sure values don't escape [0, 255] range
    Y = int(min(max(0,Y),255))
    Cb = int(min(max(0,Cb),255))
    Cr = int(min(max(0,Cr),255))

    return [Y, Cb, Cr]

def image_to_YCbCr(image):
    converted = np.zeros(shape=image.shape)

    rows = image.shape[0]
    cols = image.shape[1]

    for x in range(0, cols):
        for y in range(0, rows):
            converted[y, x] = pixel_to_YCbCr(image[y, x])

    return np.uint8(converted)

def split_YCbCr(image):
    (x, y, _z) = image.shape

    rows = image.shape[0]
    cols = image.shape[1]

    Y = np.zeros(shape=(x,y))
    Cb = np.zeros(shape=(x,y))
    Cr = np.zeros(shape=(x,y))

    for x in range(0, cols):
        for y in range(0, rows):
            data = image[y, x]
            Y[y, x] = data[0]
            Cb[y, x] = data[1]
            Cr[y, x] = data[2]

    return (Y, Cb, Cr)

def pad_image(image):
    width, height = image.size
    w_8 = math.ceil(width / 8)
    h_8 = math.ceil(height / 8)

    padded_width = w_8 * 8
    padded_height = h_8 * 8

    padded = Image.new(image.mode, (padded_width, padded_height), "black")
    padded.paste(image, (0,0))
    return padded

# image musut be np.array
def block_segment(image, blocksize):
    (x, y, _z) = image.shape

    segments = []

    for iy in range(0, y, blocksize):
        row = []
        for ix in range(0, x, blocksize):
            segment = image[ ix:(ix + blocksize), iy:(iy + blocksize), :]
            row.append(segment)
        segments.append(row)

    return np.array(segments)

