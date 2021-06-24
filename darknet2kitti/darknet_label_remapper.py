import os
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

OUTPUT_DIR = "output"

if __name__ == "__main__":
    """Remaps labels.

    Prerequisite:
    - label file (Darknet .label)
    - new label file (Darknet .label)
    - remapper file (see example file sample_remap.label)
    - an object detection dataset (Darknet format)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", help="input to convert to kitti format (directory or file)")

    parser.add_argument('-l', '--label', help="darknet labels (.label)")
    parser.add_argument('-nl', '--new-label', help="new darknet labels (.label)")
    parser.add_argument('-r', '--remapper', help="remapper (.label)")
    parser.add_argument('-w', "--walk", help="os.walk inside the directory")
    args = parser.parse_args()

    # Read old labels
    with open(args.label, 'r') as f:
        labels = [line.replace("\n", "") for line in f.readlines()]

    # Read new labels
    with open(args.new_label, 'r') as f:
        new_labels = [line.replace("\n", "") for line in f.readlines()]
    new_label2idx = {label:i for i, label in enumerate(new_labels)}
    print(new_label2idx)
    print(args)

    # Put label remapper into dictionary
    with open(args.remapper) as f:
        lines = f.readlines()
    lines = [line.replace("\n", "").split("-") for line in lines]
    for line in lines:
        assert len(line) == 2
    new_class_mapper = {line[0]:line[1] for line in lines}

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

        lines = [line.split() for line in lines]
        for line in lines:
            class_idx = int(line[0])
            class_name = labels [class_idx]
            new_class_name = new_class_mapper [class_name]
            new_class_idx = new_label2idx [new_class_name]
            line[0] = str(new_class_idx)
            assert len(line) == 5


        lines = [" ".join(line) for line in lines]

        new_txt_path = os.path.join(OUTPUT_DIR, os.path.basename(text_file))
        Path(os.path.dirname(new_txt_path)).mkdir(parents=True, exist_ok=True)
        with open(new_txt_path, 'w') as f:
            if "1277107089Image000056.txt" == os.path.basename(text_file):
                print(new_txt_path)
                print(lines)
            #for line in lines:
            #    f.write(line)
            f.write("\n".join(lines))

    print(f"Saved to {OUTPUT_DIR}")
    print("Finished.")
