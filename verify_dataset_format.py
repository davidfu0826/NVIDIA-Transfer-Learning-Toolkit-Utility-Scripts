import os, argparse
from PIL import Image

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

if __name__ == "__name__":
    parser = argparse.ArgumentParser(description='Verify dataset format.')
    parser.add_argument('--dir-path', type=str, required=True,
                        help='path to dataset')
    parser.add_argument('--format', type=str, required=True,
                    help='dataset format e.g. kitti, darknet')
    args = parser.parse_args()
    print(args)

    root_dir = args.dir_path
    img_paths = get_all_image_paths(root_dir)
    txt_paths = get_all_txt_paths(root_dir)

    assert len(img_paths) >= len(txt_paths) # Number of txt files must be less than number of images
    print(f"Image files found: \t{len(img_paths)}\nText files found: \t{len(txt_paths)}")

    for txt_path in txt_paths:
        jpg_path = txt_path.replace("/labels/", "/images/").replace(".txt", ".jpg")
        png_path = txt_path.replace("/labels/", "/images/").replace(".txt", ".png")

        assert os.path.isfile(jpg_path) and os.path.isfile(png_path) # For every text file, the corresponding image must exist

    # Data format valid?
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
            with open(txt_path) as f:
                lines = f.readlines()
            lines = [line.replace('\n', '').split() for line in lines]

            # Get image metadata
            img_path = txt_path.replace("/labels/", "/images/")
            if os.path.isfile(img_path.replace(".txt", ".jpg")):
                img_path = img_path.replace(".txt", ".jpg")
            elif os.path.isfile(img_path.replace(".txt", ".png")):
                img_path = img_path.replace(".txt", ".png")
            else:
                print(f"The corresponding image {img_path} [.png, .jpg] does not exist for this text file {txt_path}")
                assert False 

            im = Image.open(img_path)
            img_w, img_h = im.size

            # Convert to correct format
            for line in lines:
                assert len(line) == 15 # Number of columns is correct

                # Data type is correct
                line[0] = str(line[0])
                line[1] = float(line[1])
                line[2] = int(line[2])
                line[3] = float(line[3])
                line[4] = float(line[4])
                line[5] = float(line[5])
                line[6] = float(line[6])
                line[7] = float(line[7])
                line[8] = float(line[8])
                line[9] = float(line[9])
                line[10] = float(line[10])
                line[11] = float(line[11])
                line[12] = float(line[12])
                line[13] = float(line[13])
                line[14] = float(line[14])

                # [0 to image width]
                assert 0 <= line[4]
                assert line[4] < img_w

                # [0 to image_height]
                assert 0 <= line[5]
                assert line[5] < img_h

                # [top_left, image_width]
                assert line[4] <= line[6]
                assert line[5] <= img_w

                # [bottom_right, image_height]
                assert line[5] <= line[7]
                assert line[7] < img_h

                classes.add(line[0])
        print(f"Classes found: {set(classes)}")
        
    elif args.format.lower() == "darknet":
        print("Darknet format selected.")
    else:
        print(f"Format {args.format} not supported. Select from following formats []")