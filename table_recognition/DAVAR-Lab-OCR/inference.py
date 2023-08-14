"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_pub.py
# Abstract       :    Script for inference

# Current Version:    1.0.1
# Date           :    2021-09-23
##################################################################################################
"""
import torch
import sys, os
import mmcv
import cv2
import json
import jsonlines
import numpy as np
from tqdm import tqdm
from davarocr.davar_table.utils import TEDS, format_html
from davarocr.davar_common.apis import inference_model, init_model
import glob

out_dir="/results/lgpma"
os.makedirs(out_dir, exist_ok=True)

def obtain_ocr_results(img_path, model_textdet, model_rcg):
    """obtrain ocr results of table.
    """

    def crop_from_bboxes(img, bboxes, expand_pixels=(0, 0, 0, 0)):
        """crop images from original images for recognition model
        """
        ret_list = []
        for bbox in bboxes:
            max_x, max_y = min(img.shape[1], bbox[2] + expand_pixels[3]), min(img.shape[0], bbox[3] + expand_pixels[1])
            min_x, min_y = max(0, bbox[0] - expand_pixels[2]), max(0, bbox[1] - expand_pixels[0])
            if len(img.shape) == 2:
                crop_img = img[min_y: max_y, min_x: max_x]
            else:
                crop_img = img[min_y: max_y, min_x: max_x, :]
            ret_list.append(crop_img)

        return ret_list

    ocr_result = {'bboxes': [], 'confidence': [], 'texts': []}

    # single-line text detection
    text_bbox, text_mask = inference_model(model_textdet, img_path)[0]
    text_bbox = text_bbox[0]
    for box_id in range(text_bbox.shape[0]):
        score = text_bbox[box_id, 4]
        box = [int(cord) for cord in text_bbox[box_id, :4]]
        ocr_result['bboxes'].append(box)
        ocr_result['confidence'].append(score)

    # single-line text recognition
    origin_img = mmcv.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    cropped_img = crop_from_bboxes(origin_img, ocr_result['bboxes'], expand_pixels=(1, 1, 3, 3))
    rcg_output = inference_model(model_rcg, cropped_img)
    ocr_result['texts'] = rcg_output['text']

    return ocr_result

with torch.no_grad():
    # path setting
    savepath = "prediction" # path to save prediction
    config_file = 'configs/lgpma_pub.py' # config path
    checkpoint_file = 'maskrcnn-lgpma-pub-e12-pub.pth' # model path

    # loading model from config file and pth file
    model = init_model(config_file, checkpoint_file)

    #text_det_model.pth  text_rcg_model_e8.pth
    config_det = 'configs/ocr_models/det_mask_rcnn_r50_fpn_pubtabnet.py'
    checkpoint_det = 'text_det_model.pth'
    config_rcg = 'configs/ocr_models/rcg_res32_bilstm_attn_pubtabnet_sensitive.py'
    checkpoint_rcg = 'text_rcg_model_e8.pth'
    model_det = init_model(config_det, checkpoint_det)
    if 'postprocess' in model_det.cfg['model']['test_cfg']:
        model_det.cfg['model']['test_cfg'].pop('postprocess')
    model_rcg = init_model(config_rcg, checkpoint_rcg)

    for img_path in glob.glob("/data/TablesJPG/*"): 
        result_ocr = obtain_ocr_results(img_path, model_det, model_rcg)
        torch.cuda.empty_cache()
        model.cfg['model']['test_cfg']['postprocess']['ocr_result'] = [result_ocr]
        result = inference_model(model, img_path)[0]['html']
        torch.cuda.empty_cache()
        
        html_file_path = os.path.join(out_dir, os.path.basename(img_path).replace('.jpg', '.htm'))
        with open(html_file_path, 'w', encoding='utf-8') as f:
            # write to html file
            f.write(result)
     
