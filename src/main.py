import sys
import numpy as np
import image_helpers as im
from jpeg import compress_image
from PIL import Image

def main():
    try:
        filename = sys.argv[1]
        image = Image.open(filename)
    except:
        print("Error opening input file")
        exit()
    
    rgb_im = image.convert('RGB')
    img = im.pad_image(rgb_im)
    I = np.asarray(img)

    print("Compressing using JPEG...")
    compress_image(I)
    print("Done!")

if __name__ == "__main__":
    main()
