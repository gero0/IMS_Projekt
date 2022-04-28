import numpy as np
import image_helpers as im
from PIL import Image


if __name__ == "__main__":
    image = Image.open('./img/test_dice.png')
    
    img = im.pad_image(image)

    I = np.asarray(img)

    I_c = im.image_to_YCbCr(I)

    segments = im.block_segment(I_c, 8)

    img_re = Image.fromarray(I_c, mode="YCbCr")
    img_re.show("After_conversion")