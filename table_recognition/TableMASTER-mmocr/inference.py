"""
Source: https://github.com/JiaquanYe/TableMASTER-mmocr.git

Weights: available (Google Drive)
Patches: replace SyncBN to BN in config file psenet_r50_fpnf_600e_pubtabnet.py
"""

import os
from argparse import ArgumentParser

import torch
from mmcv.image import imread

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401

import sys, os, glob
import codecs
import glob
import time
import pickle
import numpy as np
from tqdm import tqdm

from table_recognition.table_inference import Detect_Inference, Recognition_Inference, End2End, Structure_Recognition
from table_recognition.match import Matcher

class EvalMatcher(Matcher):
    def __init__(self, end2end_result_dict, structure_master_result_dict):
        self.end2end_results = end2end_result_dict
        self.structure_master_results = structure_master_result_dict


def htmlPostProcess(text):
    text = '<html><body><table>' + text + '</table></body></html>'
    return text


pse_config='./configs/textdet/psenet/psenet_r50_fpnf_600e_pubtabnet.py'
master_config='./configs/textrecog/master/master_lmdb_ResnetExtra_tableRec_dataset_dynamic_mmfp16.py'
tablemaster_config='./configs/textrecog/master/table_master_ResnetExtract_Ranger_0705.py'
pse_checkpoint='./checkpoints/pse_epoch_600.pth'
master_checkpoint='./checkpoints/master_epoch_6.pth'
tablemaster_checkpoint='./checkpoints/tablemaster_best.pth'

out_dir="/results/tablemaster"
os.makedirs(out_dir, exist_ok=True)

# main process
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# table structure predict
tablemaster_inference = Structure_Recognition(tablemaster_config, tablemaster_checkpoint)

for img_path in glob.glob("/data/TablesJPG/*"): 
    html_file_path = os.path.join(out_dir, img_path.replace('.jpg', '.htm'))
    if os.path.exists(html_file_path): continue
    pse_inference = Detect_Inference(pse_config, pse_checkpoint)
    master_inference = Recognition_Inference(master_config, master_checkpoint)
    end2end = End2End(pse_inference, master_inference)
    end2end_result, end2end_result_dict = end2end.predict(img_path)
    torch.cuda.empty_cache()
    del pse_inference
    del master_inference
    del end2end
    
    tablemaster_result, tablemaster_result_dict = tablemaster_inference.predict_single_file(img_path)
    torch.cuda.empty_cache()
    #del tablemaster_inference

    # merge result by matcher
    matcher = EvalMatcher(end2end_result_dict, tablemaster_result_dict)
    match_results = matcher.match()
    merged_results = matcher.get_merge_result(match_results)

    # save predict result
    for k in merged_results.keys():
        html_file_path = os.path.join(out_dir, k.replace('.jpg', '.htm'))
        print(html_file_path)
        with open(html_file_path, 'w', encoding='utf-8') as f:
            # write to html file
            html_context = htmlPostProcess(merged_results[k])
            f.write(html_context)
