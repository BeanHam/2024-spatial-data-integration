import gc
import torch
import argparse
import numpy as np

from utils import *
from prompts import *
from tqdm import tqdm
from os import path, makedirs
from datasets import load_dataset
from unsloth import FastLanguageModel
from huggingface_hub import login as hf_login

## formating function
def formatting_prompts_func(example):
    input       = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])
    output      = ""
    text = alpaca_prompt.format(instruction, input, output)
    return { "text" : text}  

## evaluation function
def evaluate(model, tokenizer, data):
    outputs=[]
    for text in tqdm(data['text']):
        start_decode = len(tokenizer.encode(text, truncation=True, max_length=2048))        
        inputs = tokenizer(text, return_tensors = "pt", max_length=2048).to("cuda")
        response = model.generate(**inputs, max_new_tokens = 10)
        response = tokenizer.decode(response[0][start_decode:])
        outputs.append(response)
    return outputs
    
#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # Parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='llama3')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join_dataset')
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()    
    args.save_path=f'inference_results/gt/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)
        
    # ----------------------
    # Load & Process Data
    # ----------------------
    print('Downloading and preparing data...')    
    data = load_dataset(args.dataset)
    test = data['test'].map(formatting_prompts_func)
    
    #---------------------------
    # loop through metric values
    #---------------------------
    args.model_path = MODEL_PATHS_WEAK_LABELS[f"{args.model_id}_gt"]
    args.save_name = f"{args.model_id}_gt.npy"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True
    )
    FastLanguageModel.for_inference(model)
    outputs=evaluate(model, tokenizer, test)
    np.save(args.save_path+args.save_name, outputs)
    
        
if __name__ == "__main__":
    main()
