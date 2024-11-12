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
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join')
    parser.add_argument('--finetuned', type=str, default='True')
    parser.add_argument('--key', type=str, default='llama3')
    args = parser.parse_args(args=[])
    args.save_path=f'inference_results/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)  
        
    # ----------------------
    # Load Data
    # ----------------------
    print('Downloading and preparing data...')
    data = get_dataset_slices(args.dataset)
    test = data['test']
    
    #-----------------------
    # load model & tokenizer
    #-----------------------
    print('Inference...')
    client = OpenAI(api_key=args.key)
    if args.finetuned=='True':
        model_outputs = evaluate_gpt(test, client, 'ft:gpt-4o-2024-08-06:uw-howe-lab::ASVjfaKB')
    else:
        model_outputs = evaluate_gpt(test, client, 'gpt-4o)
    np.save(args.save_path+f"gpt4o_finetuned_{args.finetuned}.npy", model_outputs)

if __name__ == "__main__":
    main()
