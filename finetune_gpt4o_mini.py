import os
import sys
import time
import json
import torch
import string
import datasets
import argparse
import numpy as np
from utils import *
from tqdm import tqdm
from os import path, makedirs, getenv, mkdir
from huggingface_hub import login as hf_login

import openai
from openai import OpenAI

def format_for_finetuning(user_input: str,
                          assistant_output: str,
                          system_prompt: str) -> str:
    """
    Format data in JSON for fine-tuning an OpenAI chatbot model.
    """    
    
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": instruction}, 
                {"role": "user", "content": user_input}, 
                {"role": "assistant", "content": assistant_output}
            ]
        }
    )

def process_train_data(train, metric_name, metric_value):
    
    if metric_name == 'degree':
        positive = train.filter(lambda x: ((x['label']==1) & (x['min_angle']<=metric_value)) )
        negative = train.filter(lambda x: ((x['label']==0) & (x['min_angle']>metric_value)) )
        new_train=concatenate_datasets([positive,negative]).shuffle()
    elif metric_name == 'distance':
        positive = train.filter(lambda x: ((x['label']==1) & (x['euc_dist']>=metric_value)) )
        negative = train.filter(lambda x: ((x['label']==0) & (x['euc_dist']<metric_value)) )
        new_train=concatenate_datasets([positive,negative]).shuffle()
        
    return new_train

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='gpt-4o-mini-2024-07-18', help='The model ID to fine-tune.')
    parser.add_argument('--OPENAI_API_KEY', type=str, help='API key to finetune GPT-4o')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join_dataset')
    parser.add_argument('--formatted_data_dir', type=str, default='formatted_data')
    parser.add_argument('--metric_name', type=str, default='degree')
    parser.add_argument('--metric_value', type=int, default=1)    
    args = parser.parse_args()
    hf_login()    
    if not path.exists(args.formatted_data_dir):
        makedirs(args.formatted_data_dir)        
    
    # ----------------------
    # Load Data
    # ----------------------
    print('Formatting data for fine-tuning...')      
    data = load_dataset(args.dataset)
    train = data['train']
    train = process_train_data(train, args.metric_name, args.metric_value)
    val = data['val']                         
    train_formatted = '\n'.join(
        [format_for_finetuning(
            "Sidewalk: "+str(train['sidewalk'][i])+"\nRoad: "+str(train['road'][i])
            "Label: "+str(train['label'][i]),
            instruction
        ) for i in tqdm(range(len(train)))]
    )
    val_formatted = '\n'.join(
        [format_for_finetuning(
            "Sidewalk: "+str(val['sidewalk'][i])+"\nRoad: "+str(val['road'][i])
            "Label: "+str(val['label'][i]),
            instruction
        ) for i in tqdm(range(len(val)))]
    )
    
    # ----------------------------------
    # Write the formatted data to a file
    # ----------------------------------
    print('Writing formatted data to file...')
    with open(path.join(args.formatted_data_dir, f'train_{args.metric_name}_{args.metric_value}.jsonl'), 'w') as f:
        f.write(train_formatted)
    with open(path.join(args.formatted_data_dir, f'val_{args.metric_name}_{args.metric_value}.jsonl'), 'w') as f:
        f.write(val_formatted)

    # ----------------------------------
    # Set the OpenAI API key and create a client
    # ----------------------------------        
    client = OpenAI(api_key=args.key)

    # Create the training dataset
    train_response = client.files.create(
        file=open(path.join(args.formatted_data_dir, f'train_{args.metric_name}_{args.metric_value}.jsonl'), "rb"),
        purpose="fine-tune"
    )
    val_response = client.files.create(
        file=open(path.join(args.formatted_data_dir, f'val_{args.metric_name}_{args.metric_value}.jsonl'), "rb"),
        purpose="fine-tune"
    )
    
    # Create the fine-tuning job
    job_response = client.fine_tuning.jobs.create(
        training_file=train_response.id,
        validation_file=val_response.id,
        model=args.model_id,
        hyperparameters={
            "n_epochs": 5,
        }
    )
    
    print('Wait for the fine-tuning job to complete...')
    print('You may close the terminal now...')