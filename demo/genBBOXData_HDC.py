# Python script for generating BBOX Data for HDC dataset
# Adapted from detect/demo.py

'''
USAGE:

$ python3 genBBOXData_HDC.py \
    --config-file <config_file> \
    --input  <input images dir> \
    --output <output dir>

DEFAULT:

$ python3 genBBOXData_HDC.py \
    --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
    --input  ../../learnable-triangulation-pytorch/data/hdc \
    --output ../../learnable-triangulation-pytorch/data/pretrained/hdc/mrcnn-detections \
    --opts MODEL.WEIGHTS ../model_final_280758.pkl

More info with 
$ python3 genBBOXData_HDC --help
'''

import argparse
import multiprocessing as mp
import os
import json
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

from predictor import VisualizationDemo

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        help="Root directory to dataset"
    )
    parser.add_argument(
        "--output",
        help="A directory to save output visualizations"
    )
    parser.add_argument(
        "--logs",
        default="-",
        help="A directory to save error or log data"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def singleFrameBBOX(predictions, sortByConf=False):
    pred_boxes = predictions["instances"].pred_boxes.tensor.cpu().detach().numpy()
    pred_classes = predictions["instances"].pred_classes.cpu().detach().numpy()
    confidences = predictions["instances"].scores.cpu().detach().numpy()

    bboxes = []

    for i, bbox in enumerate(pred_boxes):
        cls = pred_classes[i]
        conf = confidences[i]

        # Not human/persons
        if cls != 0:
            continue

        left, top, right, bottom = tuple(bbox)

        bboxes.append([left, top, right, bottom, conf])

    def getConfidence(bbox):
        return float(bbox[4])

    if sortByConf:
        bboxes.sort(key=getConfidence, reverse=True)

    return bboxes

forceSinglePerson = False
ignore_actions = []
ignore_cameras = []

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()    
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    '''
    Data organisation

    [ACTION_NAME]/n[CAMERA_NO]/hdImgs/[SUB_ACTION]_[FRAME_NO].jpg
    '''
    hdc_root = args.input
    output_dir = args.output
    logs_dir = args.logs

    DEBUG = False if logs_dir == "-" else True

    if DEBUG:
        multiple_detections_file = os.path.join(logs_dir, "multiple_detections.txt")

        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)
        else:
            # Clear files
            with open(multiple_detections_file, "w") as f:
                f.write("")

    assert os.path.isdir(hdc_root), f"Input {hdc_root} must be a directory!"
    assert os.path.isdir(output_dir), f"Output {output_dir} must be a directory!"

    print("Detectron configured!")

    # Get images
    for action_name in os.listdir(hdc_root):
        # Make sure that this is actually a scene
        # and not sth like 'scripts' or 'matlab'
        if 'output_' not in action_name:
            continue

        if action_name in ignore_actions:
            continue

        action_dir = os.path.join(hdc_root, action_name)

        # Ensure is a proper directory
        if not os.path.isdir(action_dir):
            print(f"{action_dir} is not a directory")
            continue

        # Search through all cameras first
        for camera_name in os.listdir(action_dir):
            if camera_name in ignore_cameras:
                continue
            
            # Find the cameras
            images_dir = os.path.join(action_dir, camera_name, 'frames')

            if not os.path.isdir(images_dir):
                print(f"Image directory {images_dir} does not exist")
                continue

            # Populate frames dictionary
            images_dir_cam = os.path.join(images_dir, camera_name)
            output_dir_cam = os.path.join(output_dir, action_name)
            output_dir_cam_file = os.path.join(output_dir_cam, camera_name + ".json")

            # Dont overwrite
            if os.path.exists(output_dir_cam_file):
                continue

            images_sorted = sorted(os.listdir(images_dir_cam))

            bbox_by_camera = []

            print(f"Working on action {action_name}, camera {camera_name} with {len(images_sorted)} images...")

            for img_path in images_sorted:
                full_img_path = os.path.join(images_dir_cam, img_path)

                try: 
                    img = read_image(full_img_path, format="BGR")
                    predictions, visualized_output = demo.run_on_image(img)
                
                    # frame_name = img_path.replace(f'{camera_name}_', '').replace('.jpg', '')

                    bboxes = singleFrameBBOX(predictions, sortByConf=forceSinglePerson)

                    if len(bboxes) == 0:
                        bboxes = [ [0,0,0,0,0] ]

                    if len(bboxes) > 1 and forceSinglePerson:
                        print(f"[ALERT] {img_path} has multiple detections of people ({len(bboxes)})!")

                        if DEBUG:          
                            with open(multiple_detections_file, 'a') as f:
                                f.write(full_img_path + "\n")
                except:
                    bboxes = [ [0,0,0,0,0] ]

                if forceSinglePerson:
                    bbox_by_camera.append(bboxes[0])
                else:
                    bbox_by_camera.append(bboxes)

            print("Done!")

            os.makedirs(output_dir_cam, exist_ok=True)
            with open(output_dir_cam_file, 'w') as f:
                f.write(str(bbox_by_camera))
                print(f"Write to {f.name} complete")
