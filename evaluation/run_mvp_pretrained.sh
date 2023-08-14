#!/bin/bash

python run_analysis.py --html-dir ../evaluation/results/groundtruth --run-teds --run-bleu --run-rouge 
python run_analysis.py --html-dir ../evaluation/results/lgpma --run-teds --run-bleu --run-rouge 
python run_analysis.py --html-dir ../evaluation/results/tablemaster --run-teds --run-bleu --run-rouge 
