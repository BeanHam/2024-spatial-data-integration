import gc
import torch
import argparse
import numpy as np

from utils import *
from tqdm import tqdm
from os import path, makedirs
from unsloth import FastLanguageModel
from huggingface_hub import login as hf_login
from datasets import load_dataset,concatenate_datasets

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
        inputs = tokenizer(text, return_tensors = "pt").to("cuda")
        response = model.generate(**inputs, max_new_tokens = 10)
        response = tokenizer.decode(response[0]).split('Response')[1]
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
    parser.add_argument('--metric_name', type=str, default='degree')
    parser.add_argument('--proportion', type=float, default=0.1)
    args = parser.parse_args()
    
    args.save_path=f'inference_results/{args.metric_name}/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)
    
    if args.metric_name == 'degree':
        args.metric_values = [5]
    elif args.metric_name == 'distance':
        args.metric_values = [1,2,3,4,5]
    hf_login()    
        
    # ----------------------
    # Load & Process Data
    # ----------------------
    print('Downloading and preparing data...')    
    data = load_dataset(args.dataset)
    data = data.map(formatting_prompts_func)
    data = concatenate_datasets([data['train'], data['val'], data['test']])
    data=data.train_test_split(train_size=args.proportion, seed=100)
    test = data['test']    
    
    #---------------------------
    # loop through metric values
    #---------------------------
    for metric_value in args.metric_values:
        print('=====================================================')
        print(f'{args.metric_name}: {metric_value}...')        
        print('   -- Getting model and tokenizer...')
        args.model_path = MODEL_PATHS[f"{args.model_id}_{args.metric_name}_{metric_value}_nonfixed_{args.proportion}"]
        args.save_name = f"{args.model_id}_{args.metric_name}_{metric_value}_nonfixed_{args.proportion}"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_path,
            max_seq_length = args.max_seq_length,
            dtype = None,
            load_in_4bit = True
        )
        FastLanguageModel.for_inference(model)
        outputs=evaluate(model, tokenizer, test)
        np.save(args.save_path+args.save_name+".npy", outputs)
        
        ## clear memory for next metric value
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    main()
