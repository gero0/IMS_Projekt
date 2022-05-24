import numpy as np
import image_helpers as im
from jpeg import compress_image
from PIL import Image

def main():
    image = Image.open("./img/test_dice.png")
    rgb_im = image.convert('RGB')
    img = im.pad_image(rgb_im)
    I = np.asarray(img)

    print("Compressing using JPEG...")
    compress_image(I)
    print("Done!")

if __name__ == "__main__":
    main()
