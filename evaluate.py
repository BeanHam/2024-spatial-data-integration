import json
import torch
import argparse
import numpy as np
import pandas as pd
import transformers

from utils import *
from tqdm import tqdm
from datasets import load_dataset
from os import path, makedirs, getenv
from huggingface_hub import login as hf_login


#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='beanham/gibberish')
    parser.add_argument('--device', type=str, default='cuda', help='The device to mount the model on.')
    parser.add_argument('--hf_token_var', type=str, default='[your token]', help='hf login token')
    parser.add_argument('--epoch', type=int, default=1, help='hf login token')
    args = parser.parse_args()
    args.model='llama3'
    args.suffix = MODEL_SUFFIXES[args.model]
    args.save_path=f'inference_results/'
    if args.hf_token_var:
        hf_login(token=getenv(args.hf_token_var))
    if not path.exists(args.save_path):
        makedirs(args.save_path)
        
    print('Downloading and preparing data...')
    data = get_dataset_slices(args.dataset)    
    train_data = data['train']
    train_data.set_format(type='torch', device=args.device)
    
    print('=====================')
    print(f'Epoch: {args.epoch}...')
    print('=====================')

    #-----------------------
    # load model & tokenizer
    #-----------------------
    print('Getting model and tokenizer...')
    model, tokenizer = get_model_and_tokenizer(args.model_id,
                                               gradient_checkpointing=False,
                                               quantization_type='4bit',
                                               device='auto')
    model = PeftModel.from_pretrained(model, f'beanham/gibberishepoch_{args.epoch}')

    #------------
    # inference
    #------------
    model.eval()
    model_outputs, metrics  = evaluate_model(model=model,
                                             tokenizer=tokenizer,
                                             data=train_data,
                                             max_new_tokens=16,
                                             remove_suffix=args.suffix)
            
    for k, v in metrics.items(): print(f'   {k}: {v}')
    with open(args.save_path+f"gibberish_{args.epoch}.json", 'w') as f: json.dump(metrics, f)
    np.save(args.save_path+f"gibberish_{args.epoch}.npy", model_outputs)

if __name__ == "__main__":
    main()
