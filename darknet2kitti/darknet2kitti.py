import os
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

OUTPUT_DIR = "output"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", help="input to convert to kitti format (directory or file .txt)")

    parser.add_argument('-l', '--label', help="darknet labels (.label)")
    parser.add_argument('-l', '--tlt', help="Make compatible with NVIDIA Transfer Learning Toolkit 3.0", action="store_true")
    #parser.add_argument('-w', "--walk", help="os.walk inside the directory")
    args = parser.parse_args()

    with open(args.label, 'r') as f:
        labels = [line.replace("\n", "") for line in f.readlines()]
        
    # Find all text files
    text_files = list()
    for dirpath, _, files in os.walk(args.input):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join (dirpath, filename)
                text_files.append(filepath)

    # Convert all darknet label files into kitti label format 
    for text_file in tqdm(text_files):

        # Read darknet label
        with open(text_file, "r") as f:
            lines = f.readlines()

        # Read corresponding image (to get width and height)
        img_file = text_file.replace("labels", "images").replace(".txt", ".jpg")
        w, h = Image.open(img_file).size

        lines = [line.replace("\n", "").split() for line in lines]
        if len(lines) != 0:
            assert isinstance(lines[0][0], str)
            for line in lines:
 
                assert len(line) == 5
                line[0] = int(line[0])
                line[1] = float(line[1])
                line[2] = float(line[2])
                line[3] = float(line[3])
                line[4] = float(line[4])  

            assert isinstance(lines[0][0], int)
            assert isinstance(lines[0][1], float)

        # write as kitti label, (0-based index) [class_name, 0.0, 0, 0.0, left, top, right, bottom, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt        
        kitti_bboxes = list()
        for class_idx, x_center, y_center, bbox_width, bbox_height in lines:
            assert len(line) == 5
            class_name = labels[class_idx]
            left =   int(w*(x_center - bbox_width/2))
            top =    int(h*(y_center - bbox_height/2))
            right =  int(w*(x_center + bbox_width/2))
            bottom = int(h*(y_center + bbox_height/2))

            if not w >= right:
                right = w        
            if not h >= bottom:
                bottom = h      

            assert 0 <= left
            assert 0 <= top
            assert 0 <= right
            assert 0 <= bottom
            assert w >= left
            assert h >= top
            assert w >= right
            assert h >= bottom

            if args.tlt:
                kitti_bboxes.append(f"{class_name} 0.0 0 0.0 {left} {top} {right} {bottom} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n")
            else:
                kitti_bboxes.append(f"{class_name} 0.0 0 0.0 {left} {top} {right} {bottom} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n")

        kitti_txt_path = os.path.join(OUTPUT_DIR, text_file)
        Path(os.path.dirname(kitti_txt_path)).mkdir(parents=True, exist_ok=True)
        with open(kitti_txt_path, 'w') as f:
            for kitti_line in kitti_bboxes:
                f.write(kitti_line)

    print(f"Output saved to {kitti_txt_path}")
    print("Finished.")
