from .utility import kitti_labelpath2imagepath, get_img_metadata

def parse_kitti_txt(txt_path, check_validity=True):
    if check_validity:
        img_path = kitti_labelpath2imagepath(txt_path)
        img_w, img_h = get_img_metadata(img_path)

    with open(txt_path) as f:
        lines = f.readlines()
    lines = [line.replace('\n', '').split() for line in lines]
    # Convert to correct format
    for line in lines:

        assert len(line) == 15 or len(line) == 16 # Number of columns is correct

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

        if check_validity:
            #print(txt_path)
            #print("bbox", line[4:8], "img size:", img_w, img_h)
            # [0 to image width]
            assert 0 <= line[4]
            assert line[4] <= img_w

            # [0 to image_height]
            assert 0 <= line[5]
            assert line[5] <= img_h

            # [top_left, image_width]
            assert line[4] <= line[6]
            assert line[5] <= img_w

            # [bottom_right, image_height]
            assert line[5] <= line[7]
            assert line[7] <= img_h
    return lines

def parse_darknet_txt(txt_path, check_validity=True):
    if check_validity:
        img_path = kitti_labelpath2imagepath(txt_path)
        img_w, img_h = get_img_metadata(img_path)

    with open(txt_path) as f:
        lines = f.readlines()
    lines = [line.replace('\n', '').split() for line in lines]
    # Convert to correct format
    for line in lines:
        assert len(line) == 5 # Number of columns is correct

        # Data type is correct
        line[0] = int(line[0])
        line[1] = float(line[1])
        line[2] = float(line[2])
        line[3] = float(line[3])
        line[4] = float(line[4])

        if check_validity:
            # [0 to 1]
            assert 0 <= line[1]
            assert line[1] <= 1

            # [0 to 1]
            assert 0 <= line[2]
            assert line[2] < 1

            # [0 to 1]
            assert 0 <= line[3]
            assert line[3] <= 1

            # [0 to 1]
            assert 0 <= line[4]
            assert line[4] <= 1

    return lines

def parse_darknet_label_file(label_file_path):
    with open(label_file_path) as f:
        class_names = f.readlines()
    return {idx: class_name.replace("\n", '') for idx, class_name in enumerate(class_names)}
