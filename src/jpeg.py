import numpy as np
from JPEG.constants import *
import image_helpers as im
import math
from JPEG.DCT import DCT_2D
from JPEG.jpeg_file import write_to_file
from JPEG.zigzag import zigzag
from JPEG.constants import *


def compress_image(image, filename="output_jpeg.jpg"):
    I_c = im.image_to_YCbCr(image)
    components = im.split_YCbCr(I_c)

    s = I_c.shape
    width = s[1]  # type: ignore
    height = s[0]  # type: ignore

    print("Segmenting...")
    segments = segment_components(components)
    print("DCT...")
    transformed_segments = transform_segments(segments)
    print("Quantization...")
    quantized_segments = quantize_segments(transformed_segments)
    print("Zigzag...")
    zigzags = zigzag_segments(quantized_segments)
    print("Separating DC and AC...")
    dc, ac = separate_components(zigzags)
    print("Encoding DC components...")
    encoded_dc = encode_dc_components(dc)
    print("Runlength encoding AC components...")
    re_segments = runlength_encode_segments(ac)

    write_to_file(filename, width, height, encoded_dc, re_segments)


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
    q_Y = [np.int8(np.around((T / Q_lumi) + 0.01)) for T in segments_Y]
    q_Cb = [np.int8(np.around((T / Q_chroma) + 0.01)) for T in segments_Cb]
    q_Cr = [np.int8(np.around((T / Q_chroma) + 0.01)) for T in segments_Cr]
    return (q_Y, q_Cb, q_Cr)


def zigzag_segments(segments):
    segments_Y, segments_Cb, segments_Cr = segments
    z_Y = [np.int8(zigzag(T)) for T in segments_Y]
    z_Cb = [np.int8(zigzag(T)) for T in segments_Cb]
    z_Cr = [np.int8(zigzag(T)) for T in segments_Cr]
    return (z_Y, z_Cb, z_Cr)


def separate_components(zigzags):
    (z_Y, z_Cb, z_Cr) = zigzags

    dc_Y = [z[0] for z in z_Y]
    ac_Y = [z[1:] for z in z_Y]

    dc_Cb = [z[0] for z in z_Cb]
    ac_Cb = [z[1:] for z in z_Cb]

    dc_Cr = [z[0] for z in z_Cr]
    ac_Cr = [z[1:] for z in z_Cr]

    return (dc_Y, dc_Cb, dc_Cr), (ac_Y, ac_Cb, ac_Cr)


def encode_dc_components(dc_components):
    (c_Y, c_Cb, c_Cr) = dc_components
    e_Y = encode_dc(c_Y, True)
    e_Cb = encode_dc(c_Cb, False)
    e_Cr = encode_dc(c_Cr, False)
    return (e_Y, e_Cb, e_Cr)


def bits_required(n):
    if n == 0:
        return 0

    return int(math.log2(abs(n)) + 1)


def val_to_bitstream(n):
    if n == 0:
        return ""

    binstr = bin(abs(n))[2:]

    def invert(c):
        if c == "0":
            return "1"
        return "0"

    if n < 0:
        new_binstr = [invert(c) for c in binstr]
        binstr = "".join(new_binstr)

    return binstr


def encode_dc(dc_comp, luminance=False):
    encoded = []
    previous = 0
    for comp in dc_comp:
        value = comp - previous
        previous = comp

        category = bits_required(value)
        binstream = val_to_bitstream(value)

        if luminance:
            code_word = DC_lumi_coeff[category]
        else:
            code_word = DC_chroma_coeff[category]

        encoded.append(code_word + binstream)

    return encoded


def runlength_encode_segments(ac_segments):
    (ac_Y, ac_Cb, ac_Cr) = ac_segments
    re_Y = [runlength_encode(e, True) for e in ac_Y]
    re_Cb = [runlength_encode(e, False) for e in ac_Cb]
    re_Cr = [runlength_encode(e, False) for e in ac_Cr]
    return (re_Y, re_Cb, re_Cr)


def runlength_encode(ac_components, luminance=False):
    last_nonzero = -1
    for i, component in enumerate(ac_components):
        if component != 0:
            last_nonzero = i

    symbols = []
    runlength = 0

    for i, component in enumerate(ac_components):
        if i > last_nonzero:
            if luminance:
                symbol = AC_lumi_coeff[(0, 0)]
            else:
                symbol = AC_chroma_coeff[(0, 0)]
            symbols.append(symbol)
            break
        elif component == 0 and runlength < 15:
            runlength += 1
        else:
            size = bits_required(component)
            index = (runlength, size)
            if luminance:
                symbol = AC_lumi_coeff[index]  # type: ignore
            else:
                symbol = AC_chroma_coeff[index]  # type: ignore

            symbol = symbol + val_to_bitstream(component)
            symbols.append(symbol)

            runlength = 0

    return symbols
