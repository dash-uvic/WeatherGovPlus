"""
Script to evaluate modules using TEDS and BLEU/ROUGE
"""

#1. Load docker image
#2. Load dataset 
#4. Run runner (inference.py) on docker 
#5. Get HTML output
#6. Calculate the TEDS score
#7. Convert HTML output to NLP token string (see TEDS code)
#8. Run token string through MVP
#9. Calculate BLEU and ROUGE score
#10. Log results

import os, sys
from datasets import load_dataset
from transformers import MvpTokenizer, MvpForConditionalGeneration
import evaluate
import datasets, pandas as pd
import numpy as np
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
from transformers import pipeline
from glob import glob
from utils import convert, teds
import json
import pprint
pp = pprint.PrettyPrinter()

import argparse
desc="Table Recognition to Table Summarization"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--run-teds", action="store_true",
        help="Run TEDS evaluation")
parser.add_argument("--run-bleu", action="store_true",
        help="Run BLEU evaluation")
parser.add_argument("--run-rouge", action="store_true",
        help="Run ROUGE evaluation")
parser.add_argument("-gs", "--gold-summ", type=str, 
        default="./wg/",
        help="Table Summarization gold directory")
parser.add_argument("-gr", "--gold-rec", type=str,
        default="./WEATHERGOV_PLUS/TablesHTML_htags/",
        help="Table Recognition gold directory")
parser.add_argument("-hd", "--html-dir", type=str,
        default="results/lgpma",
        help="Table Recognition predicted HTML files")
parser.add_argument("--prefix", type=str,
        default="Describe the following data: ",
        help="Tokenizer string prefix")
parser.add_argument("--debug", action="store_true",
        help="turn on debugging")
args = parser.parse_args()

mapping     = os.path.join(os.path.abspath(args.gold_summ), "MappingWG2Orig", "testtgt_indices2orig.txt")
summ_gold   = os.path.join(os.path.abspath(args.gold_summ), "test.tgt")
reco_gold   = os.path.abspath(args.gold_rec)
html_dir    = os.path.abspath(args.html_dir)
input_nlp   = os.path.join(html_dir, "test.src")
prefix      = args.prefix  

from itertools import groupby
def convert_html_to_nlp_input(html_dir, output_fn, overwrite=False):
    ignore_cells = [";", "..", "-", "--"]
    html2tok = convert.HTML2Token(ignore_cells=ignore_cells, n_jobs=4)
    table_group = groupby(sorted(glob(os.path.join(html_dir, "*.htm"))), lambda string: string.split('_')[-2])
    
    for group in table_group:
        output_fn = os.path.join(html_dir, f"{int(group[0].replace('Table', ''))}.src")
        if not overwrite and os.path.exists(output_fn): continue
        
        all_tables = []
        for htm_file in sorted(group[1]): 
            print(f"Procesing {htm_file}")
            with open(htm_file) as fp:
                tables = fp.read() 
            nlp_input = html2tok.convert(tables)
            all_tables.append(nlp_input)
     
        
        with open(output_fn, "w") as fp:
            for table in all_tables:
                fp.write(f"'{table}'\n")


def eval_table_summarization(metric_strs, pred_path):
    print(pred_path)
    out_path = os.path.join(pred_path, "bleu_rouges")
    os.makedirs(out_path, exist_ok=True)
    
    metrics = []
    stemmers = []
    for metric_str in metric_strs:
        metrics.append(evaluate.load(metric_str))
        use_stemmer = False
        if metric_str == "rouge":
            use_stemmer = True
        stemmers.append(use_stemmer)

    mapper = {}
    with open(mapping, "r") as fh:
        for idx, line in enumerate(fh.readlines()):
            mapper[line.rstrip()] = idx
    
    print(f"Loading gold summaries {summ_gold}")
    x = np.loadtxt(summ_gold, delimiter=None,quotechar="'", dtype=str)
    target = pd.DataFrame(data = x, columns = ["gold"])
    pipe = pipeline(task="summarization", model="RUCAIBox/mvp-data-to-text", tokenizer="RUCAIBox/mvp", device=0, batch_size=1)
    
    for pred_src in sorted(glob(os.path.join(html_dir, "*.src"))):
        result_dict = {}
        #get the associated index for this HTML data src
        table_id = os.path.splitext(os.path.basename(pred_src))[0]
        line_no = mapper[table_id]
        table_tgt = target.iloc[line_no]
        
        #Load *.src for base image
        print(f"Loading converted HTML predictions {pred_src}")
        data=pd.read_csv(pred_src, quotechar="'", header=None, names=["input"])
        data['input'] = prefix + data['input'].astype(str)
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    print(data["input"]) 

        data = data.assign(gold=table_tgt["gold"])
        test_dset = datasets.Dataset.from_pandas(data)

        tokenizer_kwargs = {'max_length': 1024, 'truncation' : True}

        reference = table_tgt["gold"]
       
        result_dict[table_id] = []
        fname = os.path.join(out_path, f"{os.path.basename(pred_path)}_bleu_rouge_{os.path.basename(pred_src)}.txt")
        print(f"Creating raw prediction output: {fname}")
        with open(fname, "w") as fh:
            for idx, prediction in enumerate(pipe(KeyDataset(test_dset, "input"), **tokenizer_kwargs)):
                fh.write(f'\'{prediction[0]["summary_text"]}\'\n')
        
                results = {}
                for metric, stemmer in zip(metrics, stemmers):
                    if stemmer:
                        res = metric.compute(predictions=[prediction[0]["summary_text"]],references=[reference], use_stemmer=stemmer)
                    else:
                        res = metric.compute(predictions=[prediction[0]["summary_text"]],references=[reference])

                    results = {**results, **res}
                result_dict[table_id].append(results)
        
        assert len(result_dict[table_id]) == 20 

        fname = os.path.join(out_path, f"{os.path.basename(pred_path)}_bleu_rouge_{os.path.basename(pred_src)}.json")
        print(f"Saving bleu/rouge results: {fname}")
        with open(fname, "w") as fh:
            json.dump(result_dict, fh, indent=6)
        

def eval_table_recognition(html_dir):
    pred_json_obj = {}
    true_json_obj = {}
    for htm_file in sorted(glob(os.path.join(html_dir, "*.htm"))): 
        print(f"Procesing {htm_file}")
        bname = os.path.basename(htm_file)
        with open(htm_file) as fp:
            html = fp.read()
        pred_json_obj[bname] = html
        
        gt_htm_file = os.path.join(reco_gold, bname)
        with open(gt_htm_file) as fp:
            html = fp.read()
        true_json_obj[bname] = { "html" : html }
   
    teds_metric = teds.TEDS(n_jobs=4)
    scores = teds_metric.batch_evaluate(pred_json_obj, true_json_obj)
    pp.pprint(scores)
    with open(f"{os.path.basename(html_dir)}_teds.json", "w") as fh:
        json.dump(scores, fh, indent=6)

    return scores

if __name__ == "__main__":
    if args.run_teds:
        eval_table_recognition(html_dir)
    if args.run_bleu or args.run_rouge:
        print(f"Creating {input_nlp}")
        convert_html_to_nlp_input(html_dir, input_nlp, overwrite=args.debug)
        
        if args.run_bleu and args.run_rouge:
            eval_table_summarization(["bleu", "rouge"], html_dir)
        elif args.run_rouge:
            eval_table_summarization(["rouge"], html_dir)
        else:
            eval_table_summarization(["bleu"], html_dir)
