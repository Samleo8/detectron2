import cv2
import json
import os, sys
import argparse

USAGE_PROMPT = """
$ python3 visualise.py --images <dir/to/images> --bboxes <path/to/json> 

Example:
$ python visualise.py --images ../../data/171024_pose1/hdImgs/00_00/ --bboxes ../../data/171024_pose1/detections/171024_pose1/00_00.json
"""

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    
    parser.add_argument(
        "--images",
        help="Directory to images"
    )
    parser.add_argument(
        "--bboxes",
        help="Path to JSON file containing BBOXes for that image"
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()    
    imageDir = args.images
    bboxJSONName = args.bboxes

    if imageDir is None or bboxJSONName is None:
        print(USAGE_PROMPT)
        exit()

    if not os.path.exists(imageDir):
        raise FileNotFoundError(f"Cannot find directory to images {imageDir}")

    if not os.path.isdir(imageDir):
        raise NotADirectoryError(f"Invalid directory for images {imageDir}")

    if not os.path.exists(imageDir):
        raise FileNotFoundError(f"Cannot find directory to images {imageDir}")

    if not os.path.isdir(imageDir):
        raise NotADirectoryError(f"Invalid directory for images {imageDir}")

    # Parse JSON file
    with open(bboxJSONName, 'r') as f:
        bboxJSON = json.load(f)

    imageDirArr = sorted(os.listdir(imageDir))

    for i, imgName in enumerate(imageDirArr):
        if not imgName.endswith(".jpg") and not imgName.endswith(".png"): 
            continue

        imgPath = os.path.join(imageDir, imgName)
        img = cv2.imread(imgPath)
        detections = bboxJSON[i]

        color = (255, 0, 0)
        thickness = 2
        for bbox in detections:
            t, l, b, r, conf = map(int, bbox)
            startPoint = (l, t)
            endPoint = (r, b)

            img = cv2.rectangle(img, startPoint, endPoint, color, thickness)

        cv2.imshow("Detections", img)
        c = cv2.waitKey(0)

        if c % 255 == 'q':
            break

cv2.destroyAllWindows()