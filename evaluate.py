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
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join')
    parser.add_argument('--finetuned', type=str, default='False')
    parser.add_argument('--fewshot', type=str, default='False')
    parser.add_argument('--use_model_prompt_defaults', type=str, default='llama3')
    args = parser.parse_args()    
    args.suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]
    args.model_path = MODEL_PATHS[args.use_model_prompt_defaults]
    args.save_path=f'inference_results/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)    
    hf_login()    
        
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
    print('Getting model and tokenizer...')
    model, tokenizer = get_model_and_tokenizer(args.model_id,
                                               gradient_checkpointing=False,
                                               quantization_type='4bit',
                                               device='auto')
    if args.finetuned=='True':
        model = PeftModel.from_pretrained(model, args.model_path)

    #------------
    # inference
    #------------
    model.eval()
    if args.finetuned=='True':
        test_outputs  = evaluate_model(model=model,tokenizer=tokenizer,data=test,max_new_tokens=10,remove_suffix=args.suffix,'False')
        fp_outputs  = evaluate_model(model=model,tokenizer=tokenizer,data=fp,max_new_tokens=10,remove_suffix=args.suffix,'False')
        fn_outputs  = evaluate_model(model=model,tokenizer=tokenizer,data=fn,max_new_tokens=10,remove_suffix=args.suffix,'False')
    else:
        test_outputs  = evaluate_model(model=model,tokenizer=tokenizer,data=test,max_new_tokens=5,remove_suffix=args.suffix,args.fewshot)
        fp_outputs  = evaluate_model(model=model,tokenizer=tokenizer,data=fp,max_new_tokens=10,remove_suffix=args.suffix,args.fewshot)
        fn_outputs  = evaluate_model(model=model,tokenizer=tokenizer,data=fn,max_new_tokens=10,remove_suffix=args.suffix,args.fewshot)
        
    np.save(args.save_path+f"{args.model_id}_finetuned_{args.finetuned}_fewshot_{args.fewshot}_test.npy", test_outputs)
    np.save(args.save_path+f"{args.model_id}_finetuned_{args.finetuned}_fewshot_{args.fewshot}_fp.npy", fp_outputs)
    np.save(args.save_path+f"{args.model_id}_finetuned_{args.finetuned}_fewshot_{args.fewshot}_fn.npy", fn_outputs)

if __name__ == "__main__":
    main()
