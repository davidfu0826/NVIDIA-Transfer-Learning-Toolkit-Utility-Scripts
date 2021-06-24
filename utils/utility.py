import os
from PIL import Image

def get_img_metadata(img_path):
    im = Image.open(img_path)
    img_w, img_h = im.size

    return img_w, img_h

def kitti_labelpath2imagepath(txt_path):
    # Get image metadata
    img_path = txt_path.replace("/labels/", "/images/")
    if os.path.isfile(img_path.replace(".txt", ".jpg")):
        img_path = img_path.replace(".txt", ".jpg")
    elif os.path.isfile(img_path.replace(".txt", ".png")):
        img_path = img_path.replace(".txt", ".png")
    else:
        print(f"The corresponding image {img_path} [.png, .jpg] does not exist for this text file {txt_path}")
        assert False 
    
    return img_path