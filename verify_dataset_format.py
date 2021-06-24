import os, argparse

from utils.parse import parse_kitti_txt, parse_darknet_txt, parse_darknet_label_file

def get_all_image_paths(root_dir):
    img_paths = list()
    for path, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(path, file)
            if file_path.endswith(".jpg") or file_path.endswith(".png"):
                img_paths.append(file_path)
    return img_paths

def get_all_txt_paths(root_dir):
    txt_paths = list()
    for path, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(path, file)
            if file_path.endswith(".txt"):
                txt_paths.append(file_path)
    return txt_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify dataset format.')
    parser.add_argument('--dir-path', type=str, required=True,
                        help='path to dataset')
    parser.add_argument('--format', type=str, required=True,
                    help='dataset format e.g. kitti, darknet')
    parser.add_argument('--names', type=str, default=None,
                    help='label file for darknet format')
    parser.add_argument('--ignore-assert', action="store_true",
                    help='Ignore assertions')
    args = parser.parse_args()
    print(args)

    root_dir = args.dir_path
    img_paths = get_all_image_paths(root_dir)
    txt_paths = get_all_txt_paths(root_dir)

    if not len(img_paths) >= len(txt_paths):
        print(f"len(img_paths)={len(img_paths)}, len(txt_paths)={len(txt_paths)}, len(img_paths) >= len(txt_paths) -> {len(img_paths) >= len(txt_paths)}")
        assert False # Number of txt files must be less than number of images
    print(f"Image files found: \t{len(img_paths)}\nText files found: \t{len(txt_paths)}")

    for txt_path in txt_paths:
        jpg_path = txt_path.replace("/labels/", "/images/").replace(".txt", ".jpg")
        png_path = txt_path.replace("/labels/", "/images/").replace(".txt", ".png")
        if os.path.isfile(jpg_path) or os.path.isfile(png_path):
            pass
        else:
            print(f"There is no corresponding image for {txt_path}")	
            assert False # For every text file, the corresponding image must exist

    # Example
    #   car        0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
    #   cyclist    0.00 0 -2.46 665.45 160.00 717.93 217.99 1.72 0.47 1.65  2.45 1.35 22.10 -2.35
    #   pedestrian 0.00 2  0.21 423.17 173.67 433.17 224.03 1.60 0.38 0.30 -5.87 1.63 23.11 -0.03
    #   0          1    2  3    4      5      6      7      8    9    10    11   12   13    14
    # Bounding box coordinates valid with respect to image?
    if args.format.lower() == "kitti":
        print("KITTI format selected.")
        classes = list()
        for txt_path in txt_paths:
            # Parse data
            if args.ignore_assert:
                bboxes = parse_kitti_txt(txt_path, check_validity=False)
            else:
                bboxes = parse_kitti_txt(txt_path)

            for bbox in bboxes:
                classes.append(bbox[0])
        print(f"Classes found: {set(classes)}")
        


    elif args.format.lower() == "darknet":
        print("Darknet format selected.")
        class_indicies = list()
        for txt_path in txt_paths:
            # Parse data
            if args.ignore_assert:
                bboxes = parse_darknet_txt(txt_path, check_validity=False)
            else:
                bboxes = parse_darknet_txt(txt_path)

            for bbox in bboxes:
                class_indicies.append(bbox[0])

        if args.names is not None:
            idx2name = parse_darknet_label_file(args.names)
            class_names = list()
            for class_idx in set(class_indicies):
                class_name = idx2name[class_idx]
                class_names.append(class_name)

            print(f"Classes indicies found: {set(class_names)}.")

        else:
            print(f"Classes indicies found: {set(class_indicies)}. Tip: Set '--names' if you want class names")




    else:
        print(f"Format {args.format} not supported. Select from following formats []")
