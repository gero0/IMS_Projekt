import struct
import numpy as np
from JPEG.constants import *
from JPEG.zigzag import zigzag


def write_to_file(filename, width, height, encoded_dc, encoded_ac):
    file = open(filename, "wb")

    write_soiapp(file, width, height)
    write_dqt(file)
    write_sof(file, width, height)
    write_dht(file)
    write_sos(file)
    write_image_data(file, encoded_dc, encoded_ac)

    # write EOF
    file.write(bytearray([0xFF, 0xD9]))

    file.close()


def write_soiapp(file, width, height):
    # ==SOI segment==
    file.write(bytearray([0xFF, 0xD8]))
    # ==APP0 segment==
    file.write(bytearray([0xFF, 0xE0]))
    file.write(bytearray([0x00, 0x10]))  # SECTION LENGTH
    file.write(b"JFIF\0")  # identifier
    file.write(bytearray([0x01, 0x01]))  # version

    file.write(bytearray([0x02, 0x00, 0x1C, 0x00, 0x1C]))  # pixel density

    file.write(bytearray([0, 0]))  # Thumbnail size - 0,0


def write_dqt(file):
    # lumi table
    file.write(bytearray([0xFF, 0xDB]))  # DQT marker
    file.write(bytearray([0x00, 0x43]))  # section length
    file.write(bytearray([0x00]))  # 8-bit precision, id0
    Q_zigzag = np.uint8(zigzag(np.array(Q_lumi)))
    # write qtable in zigzag order
    file.write(bytearray(Q_zigzag))

    # chroma table
    file.write(bytearray([0xFF, 0xDB]))  # DQT marker
    file.write(bytearray([0x00, 0x43]))  # section length
    file.write(bytearray([0x01]))  # 8-bit precision, id1
    Q_zigzag = np.uint8(zigzag(np.array(Q_chroma)))
    # write qtable in zigzag order
    file.write(bytearray(Q_zigzag))


def write_sof(file, width, height):
    file.write(bytearray([0xFF, 0xC0]))  # SOF marker
    file.write(bytearray([0x00, 0x11]))  # section length - 17 bytes
    file.write(bytearray([0x08]))  # precision of samples

    # vertical resolution of image (number of lines)
    s = struct.pack(">H", height)
    (first, second) = struct.unpack(">BB", s)
    file.write(bytearray([first, second]))

    # horizontal resolution of image (samples per line)
    s = struct.pack(">H", width)
    (first, second) = struct.unpack(">BB", s)
    file.write(bytearray([first, second]))

    file.write(bytearray([0x03]))  # number of components
    # Component data
    file.write(bytearray([0x01, 0x11, 0x0]))  # Y - id 1, 1:1 sampling factor, Qtable 0
    file.write(bytearray([0x02, 0x11, 0x01]))  # Cb - id 2, 1:1, Qtable 1
    file.write(bytearray([0x03, 0x11, 0x01]))  # Cb - id 3, 1:1, Qtable 1


def write_dht(file):
    # DC lumi
    file.write(bytearray([0xFF, 0xC4]))  # DHT Marker
    file.write(bytearray([0x00, 0x1F]))  # segment length
    file.write(bytearray([0x00]))  # DC table, ID 0
    file.write(bytearray([x for x in HT_DC_Y]))  # DC HT for luminance
    # AC lumi
    file.write(bytearray([0xFF, 0xC4]))  # DHT Marker
    file.write(bytearray([0x00, 0xB5]))  # segment length
    file.write(bytearray([0x10]))  # AC table, ID 0
    file.write(bytearray([x for x in HT_AC_Y]))
    # DC chroma
    file.write(bytearray([0xFF, 0xC4]))  # DHT Marker
    file.write(bytearray([0x00, 0x1F]))  # segment length
    file.write(bytearray([0x01]))  # DC table, ID 1
    file.write(bytearray([x for x in HT_DC_CH]))  # DC HT for luminance
    # AC chroma
    file.write(bytearray([0xFF, 0xC4]))  # DHT Marker
    file.write(bytearray([0x00, 0xB5]))  # segment length
    file.write(bytearray([0x11]))  # AC table, ID 1
    file.write(bytearray([x for x in HT_AC_CH]))


def write_sos(file):
    file.write(bytearray([0xFF, 0xDA]))  # SOS Marker
    file.write(bytearray([0x00, 0x0C]))  # segment length
    file.write(bytearray([0x03]))  # number of components
    file.write(bytearray([0x01, 0x00]))  # component Y HT table
    file.write(bytearray([0x02, 0x11]))  # component Cb HT table
    file.write(bytearray([0x03, 0x11]))  # component Cr HT table
    file.write(
        bytearray([0x00, 0x3F, 0x00])
    )  # spectral selection, not used in JPEG baseline


def write_image_data(file, encoded_dc, encoded_ac):
    bitstream = ""
    block_count = len(encoded_dc[0])

    for i in range(0, block_count):
        for c in range(0, 3):
            bitstream += encoded_dc[c][i]
            for x in encoded_ac[c][i]:
                bitstream += x

    bytes = []

    string_index = 0
    while string_index < len(bitstream):
        if string_index + 8 >= len(bitstream):
            number = bitstream[string_index:]
            for _i in range(0, 8 - len(number)):
                number += "1"
        else:
            number = bitstream[string_index : string_index + 8]

        bytes.append(np.uint8(int(number, 2)))

        string_index += 8

    for byte in bytes:
        file.write(byte)
        if byte == 0xFF:
            file.write(np.uint8(0x0))
