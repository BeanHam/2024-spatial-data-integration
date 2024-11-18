import json
import argparse
import evaluate
import numpy as np
import pandas as pd

from utils import *
from tqdm import tqdm
from openai import OpenAI
from typing import Iterable
from datasets import load_dataset
from os import path, makedirs, getenv

#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join')
    parser.add_argument('--finetuned', type=str, default='True')
    parser.add_argument('--fewshot', type=str, default='True')
    parser.add_argument('--key', type=str, default='123abc')
    args = parser.parse_args()
    args.model_path=MODEL_PATHS[args.model_id]
    args.save_path=f'inference_results/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)  
        
    # ----------------------
    # Load Data
    # ----------------------
    print('Downloading and preparing data...')
    test = load_dataset(args.dataset, split='test')
    fp = load_dataset(args.dataset, split='fp')
    fn = load_dataset(args.dataset, split='fn')
    
    #-----------------------
    # load model & tokenizer
    #-----------------------
    print('Inference...')
    client = OpenAI(api_key=args.key)
    if args.finetuned=='True':
        test_outputs = evaluate_gpt(test, client, args.model_path, 'False')
        fp_outputs = evaluate_gpt(fp, client, args.model_path, 'False')
        fn_outputs = evaluate_gpt(fn, client, args.model_path, 'False')
    else:
        model_outputs = evaluate_gpt(test, client, args.model_id, args.fewshot)
        fp_outputs = evaluate_gpt(fp, client, args.model_id, args.fewshot)
        fn_outputs = evaluate_gpt(fn, client, args.model_id, args.fewshot)
    
    np.save(args.save_path+f"{args.model_id}_finetuned_{args.finetuned}_fewshot_{args.fewshot}_test.npy", test_outputs)
    np.save(args.save_path+f"{args.model_id}_finetuned_{args.finetuned}_fewshot_{args.fewshot}_fp.npy", fp_outputs)
    np.save(args.save_path+f"{args.model_id}_finetuned_{args.finetuned}_fewshot_{args.fewshot}_fn.npy", fn_outputs)
    
if __name__ == "__main__":
    main()
