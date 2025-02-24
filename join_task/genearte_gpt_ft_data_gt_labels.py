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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join_dataset')
    parser.add_argument('--formatted_data_dir', type=str, default='formatted_data')
    args = parser.parse_args()
    if not path.exists(args.formatted_data_dir):
        makedirs(args.formatted_data_dir)
    
    # ----------------------
    # Load Data
    # ----------------------
    data = load_dataset(args.dataset)
    train = data['train']
    val = data['val']
    train_formatted = '\n'.join(
        [format_for_finetuning(
            gpt_instruction,
            "Sidewalk: "+str(train['sidewalk'][i])+"\nRoad: "+str(train['road'][i]),
            "Label: "+str(train['label'][i])
        ) for i in tqdm(range(len(train)))]
    )
    val_formatted = '\n'.join(
        [format_for_finetuning(
            gpt_instruction,
            "Sidewalk: "+str(val['sidewalk'][i])+"\nRoad: "+str(val['road'][i]),
            "Label: "+str(val['label'][i])
        ) for i in tqdm(range(len(val)))]
    )
    
    # ----------------------------------
    # Write the formatted data to a file
    # ----------------------------------
    print('Writing formatted data to file...')
    with open(path.join(args.formatted_data_dir, f'train_gt.jsonl'), 'w') as f:
        f.write(train_formatted)
    with open(path.join(args.formatted_data_dir, f'val_gt.jsonl'), 'w') as f:
        f.write(val_formatted)
