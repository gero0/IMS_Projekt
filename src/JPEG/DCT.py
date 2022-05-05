import math
import numpy as np


def dct_alpha(x):
    if x == 0:
        return 1.0 / math.sqrt(2)
    else:
        return 1.0


def sum_cosines(block, u, v, shapeX, shapeY):
    cos_sum = 0

    for x in range(0, shapeX):
        for y in range(0, shapeY):
            cos_sum += (
                block[x, y]
                * math.cos(((2 * x + 1) * u * math.pi) / 16)
                * math.cos(((2 * y + 1) * v * math.pi) / 16)
            )

    return cos_sum


def DCT_2D(block):
    (x, y) = block.shape
    transformed = np.zeros(shape=block.shape)

    # We gonna need more precision for that
    block = np.float32(block)
    # Subtract the midpoint to bring the values to [-128, 127] range
    block = block - 128

    for u in range(0, x):
        for v in range(0, y):
            G = 0.25 * dct_alpha(u) * dct_alpha(v) * sum_cosines(block, u, v, x, y)
            transformed[u, v] = G

    return transformed
