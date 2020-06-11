# Python script for generating BBOX Data for CMU dataset
# Adapted from detect/demo.py

'''
USAGE:

$ python3 genBBOXData \ 
    --config_file <config_file> \
    --input  <input images dir> \
    --output <output dir> \

DEFAULT:

$ python3 genBBOXData \ 
    --config_file ./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
    --input  ~/learnable-triangulation-pytorch/data/cmupanoptic \
    --output ~/learnable-triangulation-pytorch/data/pretrained/mrcnn-detections/cmu \
    --opts MODEL.WEIGHTS ../model_final_280758.pkl

More info with 
$ python3 genBBOXData --help
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
        nargs="+",
        help="",
    )
    parser.add_argument(
        "--output",
        help="A directory to save output visualizations"
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

def singleFrameBBOX(predictions):
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

        bboxes.append(list(left, top, right, bottom, conf))

    return bboxes

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()    
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    '''
    Data organisation

    [ACTION_NAME]/hdImgs/[CAMERA_NO]/[CAMERA_NO]_[FRAME_NO].jpg
    '''
    cmu_dance_root = args.input
    output_dir = args.output

    assert os.path.isdir(cmu_dance_root), f"Input {cmu_dance_root} must be a directory!"
    assert os.path.isdir(output_dir), f"Output {output_dir} must be a directory!"

    # Get images
    for action_name in os.listdir(cmu_dance_root):
        # Make sure that this is actually a scene
        # and not sth like 'scripts' or 'matlab'
        if '_dance' not in action_name and '_moonbaby' not in action_name:
            continue

        action_dir = os.path.join(cmu_dance_root, action_name)

        # Ensure is a proper directory
        if not os.path.isdir(action_dir):
            if DEBUG:
                print(f"{action_dir} does not exist")
            continue

        # Find the cameras
        images_dir = os.path.join(action_dir, 'hdImgs')

        if not os.path.isdir(images_dir):
            if DEBUG:
                print(f"Image directory {images_dir} does not exist")
            continue

        for camera_name in os.listdir(images_dir):
            # Populate frames dictionary
            images_dir_cam = os.path.join(images_dir, camera_name)
            output_dir_cam = os.path.join(output_dir, action_name)
            output_dir_cam_file = os.path.join(output_dir_cam, camera_name + ".json")

            images_sorted = os.listdir(images_dir_cam).sort()

            bbox_by_camera = []

            for img_path in images_sorted:
                img = read_image(img_path, format="BGR")
                predictions, visualized_output = demo.run_on_image(img)
                
                frame_name = img_path.replace(f'{camera_name}_', '').replace('.jpg', '')

                bboxes = singleFrameBBOX(predictions)

                if len(bboxes) == 0:
                    bboxes = [0,0,0,0,0]

                if len(bboxes) > 1:
                    print(f"[ALERT] Camera {camera_name} Frame {frame_name} has multiple detections of people ({len(bboxes)})!")

                bbox_by_camera.append(bboxes[0])
    
            os.makedirs(output_dir_cam)
            with open(output_dir_cam, 'w') as f:
                json.dump(bbox_by_camera, f)
