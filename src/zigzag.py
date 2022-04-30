import numpy as np


def diagonal_upright(image, point, values, blocksize):
    while point[1] != blocksize - 1 and point[0] != 0:
        values = np.append(values, image[point])
        point = (point[0] - 1, point[1] + 1)

    return values, point


def diagonal_downleft(image, point, values, blocksize):
    while point[1] != 0 and point[0] != blocksize - 1:
        values = np.append(values, image[point])
        point = (point[0] + 1, point[1] - 1)

    return values, point


def zigzag(image, blocksize=8):
    point = (0, 0)
    values = []

    while point != (blocksize - 1, blocksize - 1):
        values, point = diagonal_upright(image, point, values, blocksize)
        values = np.append(values, image[point])

        if point[1] < blocksize - 1:
            point = (point[0], point[1] + 1)
        else:
            point = (point[0] + 1, point[1])

        values, point = diagonal_downleft(image, point, values, blocksize)
        values = np.append(values, image[point])

        if point[0] < blocksize - 1:
            point = (point[0] + 1, point[1])
        else:
            point = (point[0], point[1] + 1)

    values = np.append(values, image[point])

    return values