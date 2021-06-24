import os, argparse
from pathlib import Path

import cv2

from utils.utility import get_img_metadata
from utils.parse import parse_kitti_txt, parse_darknet_txt, parse_darknet_label_file

def put_bbox_on_img(img, classes, bboxes):
    """
    img: np.array[H,W,3]  - Numpy array representing the image
    classes: List         - List with labels
    bboxes: List[List]    - List with [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
    """
    for label, (x1,y1,x2,y2) in zip(classes, bboxes):

        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, label, (x1,y1-7), fontFace=font, fontScale=1, color=(0,0,255), thickness=2)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), color=(0,0,255), thickness=3)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify dataset format.')
    parser.add_argument('--img-path', type=str, default=None,
                        help='path to dataset')
    parser.add_argument('--txt-path', type=str, default=None,
                    help='path to dataset')
    parser.add_argument('--format', type=str, required=True,
                    help='dataset format e.g. kitti, darknet')
    parser.add_argument('--names', type=str, default=None,
                    help='label file for darknet format')
    args = parser.parse_args()
    print(args)

    if args.img_path is None:
        if args.txt_path is None:
            print("Please define either --img-path or --txt-path")
        else:
            txt_path = args.txt_path
            img_path = txt_path.replace("/labels/", "/images/").replace(".txt", ".jpg")
            if not os.path.isfile(img_path):
                img_path = txt_path.replace("/labels/", "/images/").replace(".txt", ".png")
    else:
        img_path = args.img_path
        if args.txt_path is None:
            txt_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".png", ".txt")
        else: 
            txt_path = args.txt_path

    if args.format.lower() == "kitti":
        bboxes = parse_kitti_txt(txt_path)
        classes = [bbox[0] for bbox in bboxes]
        bboxes = [[int(bbox[4]), int(bbox[5]), int(bbox[6]), int(bbox[7])] for bbox in bboxes]
        
        print(img_path, bboxes, classes)
        img = cv2.imread(img_path)
        img = put_bbox_on_img(img, classes, bboxes)

        Path("output/").mkdir(exist_ok=True, parents=True)
        save_path = os.path.join("output", os.path.basename(img_path))
        cv2.imwrite(save_path, img)
        print(f"Saving results to {save_path}")

    elif args.format.lower() == "darknet":
        bboxes = parse_darknet_txt(txt_path)
        img_w, img_h = get_img_metadata(img_path)
        
        class_indices = [int(bbox[0]) for bbox in bboxes]
        if args.names is not None:
            idx2name = parse_darknet_label_file(args.names)
            classes = [idx2name[idx] for idx in class_indices]
        else:
            classes = [str(idx) for idx in class_indices]
            print("Tip: Set '--names' if you want class names")
        bboxes = [[img_w*float(bbox[1]), img_h*float(bbox[2]), img_w*float(bbox[3]), img_h*float(bbox[4])] for bbox in bboxes]
        bboxes = [[int(x_center-bbox_w/2), int(y_center-bbox_h/2), int(x_center+bbox_w/2), int(y_center+bbox_h/2)] for x_center, y_center, bbox_w, bbox_h in bboxes]
        
        print(img_path, bboxes, classes)
        img = cv2.imread(img_path)
        img = put_bbox_on_img(img, classes, bboxes)

        Path("output/").mkdir(exist_ok=True, parents=True)
        save_path = os.path.join("output", os.path.basename(img_path))
        cv2.imwrite(save_path, img)
        print(f"Saving results to {save_path}")

    else:
        print(f"Format not supported: {format}")