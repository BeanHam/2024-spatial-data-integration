import json
import argparse
from utils import *
from tqdm import tqdm
from os import path, makedirs
from huggingface_hub import login as hf_login
from datasets import load_dataset,concatenate_datasets

def format_for_finetuning(system_prompt: str,
                          user_input: str,
                          assistant_output: str) -> str:
    """
    Format data in JSON for fine-tuning an OpenAI chatbot model.
    """    
    
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": system_prompt}, 
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
    elif metric_name == 'area':
        positive = train.filter(lambda x: ((x['label']==1) & (x['max_area']>=metric_value)) )
        negative = train.filter(lambda x: ((x['label']==0) & (x['max_area']<metric_value)) )
        new_train=concatenate_datasets([positive,negative]).shuffle()
        
    return new_train

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beanham/spatial_union_dataset')
    parser.add_argument('--formatted_data_dir', type=str, default='formatted_data')
    parser.add_argument('--metric_name', type=str, default='degree')
    args = parser.parse_args()
    if not path.exists(args.formatted_data_dir):
        makedirs(args.formatted_data_dir)
    if args.metric_name == 'degree':
        args.metric_values = [1,2,3,4,5]
    elif args.metric_name == 'distance':
        args.metric_values = [0.5,0.6,0.7,0.8,0.9]
    
    # ----------------------
    # Load Data
    # ----------------------
    for v in args.metric_values:
        print('=====================================================')
        print(f'{args.metric_name}: {v}...')                
        print('  -- Formatting data for fine-tuning...')
        data = load_dataset(args.dataset)
        train = data['train']
        val = data['val']
        train = process_train_data(train, args.metric_name, v)
        train_formatted = '\n'.join(
            [format_for_finetuning(
                gpt_instruction,
                "Sidewalk 1: "+str(train['sidewalk'][i])+"\nSidewalk 2: "+str(train['road'][i]),
                "Label: "+str(train['label'][i])
            ) for i in tqdm(range(len(train)))]
        )
        val_formatted = '\n'.join(
            [format_for_finetuning(
                gpt_instruction,
                "Sidewalk 1: "+str(val['sidewalk'][i])+"\nSidewalk 2: "+str(val['road'][i]),
                "Label: "+str(val['label'][i])
            ) for i in tqdm(range(len(val)))]
        )
        
        # ----------------------------------
        # Write the formatted data to a file
        # ----------------------------------
        print('Writing formatted data to file...')
        with open(path.join(args.formatted_data_dir, f'train_{args.metric_name}_{v}.jsonl'), 'w') as f:
            f.write(train_formatted)
        with open(path.join(args.formatted_data_dir, f'val_{args.metric_name}_{v}.jsonl'), 'w') as f:
            f.write(val_formatted)
