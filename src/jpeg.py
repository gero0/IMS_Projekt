import numpy as np
import image_helpers as im
from DCT import DCT_2D
from PIL import Image

# quantization table for 50% quality
Q = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]


def main():
    np.set_printoptions(precision=2, suppress=True)

    # test = np.array([
    #     [52, 55, 61, 66, 70, 61, 64, 73],
    #     [63, 59, 55, 90, 109, 85, 69, 72],
    #     [62, 59, 68, 113, 144, 104, 66, 73],
    #     [63, 58, 71, 122, 154, 106, 70, 69],
    #     [67, 61, 68, 104, 126, 88, 68, 70],
    #     [79, 65, 60, 70, 77, 68, 58, 75],
    #     [85, 71, 64, 59, 55, 61, 65, 83],
    #     [87, 79, 69, 68, 65, 76, 78, 94],
    # ])

    # T = DCT_2D(test)
    # B = np.int8(np.around(T / Q))

    # print(T)
    # print(B)

    image = Image.open("./img/test_dice.png")
    img = im.pad_image(image)
    I = np.asarray(img)

    I_c = im.image_to_YCbCr(I)
    components = im.split_YCbCr(I_c)

    segments = segment_components(components)
    transformed_segments = transform_segments(segments)
    quantized_segments = quantize_segments(transformed_segments)

    print(quantized_segments)

    # img_re = Image.fromarray(I_c, mode="YCbCr")
    # img_re.show("After_conversion")


def segment_components(components, blocksize=8):
    Y, Cb, Cr = components
    seg_Y = im.block_segment(Y, blocksize)
    seg_Cb = im.block_segment(Cb, blocksize)
    seg_Cr = im.block_segment(Cr, blocksize)
    return (seg_Y, seg_Cb, seg_Cr)


def transform_segments(segments):
    segments_Y, segments_Cb, segments_Cr = segments
    t_Y = [DCT_2D(segment) for segment in segments_Y]
    t_Cb = [DCT_2D(segment) for segment in segments_Cb]
    t_Cr = [DCT_2D(segment) for segment in segments_Cr]
    return (t_Y, t_Cb, t_Cr)

def quantize_segments(segments):
    segments_Y, segments_Cb, segments_Cr = segments
    q_Y = [np.int8(np.around(T / Q)) for T in segments_Y]
    q_Cb = [np.int8(np.around(T / Q)) for T in segments_Cb]
    q_Cr = [np.int8(np.around(T / Q)) for T in segments_Cr]
    return (q_Y, q_Cb, q_Cr)

if __name__ == "__main__":
    main()
