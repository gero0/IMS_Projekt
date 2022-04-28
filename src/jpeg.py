import numpy as np
import image_helpers as im
from DCT import DCT_2D
from PIL import Image

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

    image = Image.open("./img/test_dice.png")
    img = im.pad_image(image)
    I = np.asarray(img)

    I_c = im.image_to_YCbCr(I)
    components = im.split_YCbCr(I_c)

    segments = segment_components(components)
    transformed_segments = transform_segments(segments)

    print(transformed_segments)

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


if __name__ == "__main__":
    main()

