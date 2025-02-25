import os
import json
import base64
import argparse
import requests
import numpy as np

from utils import *
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from visual_prompts import *
from os import path, makedirs
from itertools import product
from datasets import load_dataset

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def write_completion_request(text, base64_image, model):
    """
    Compose completion request.
    """    
    completion = {
        "model": model,
        "messages": [
            {"role": "user",
             "content": [
                 {"type": "text", "text": text},
                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
             ]}
        ],
        "temperature": 0,
        "max_tokens": 10,
        "top_p":1
    }
    return completion
    
def evaluate_gpt_4o_vision(data, model, index, path, api_web, headers):
    model_outputs = []
    for i in tqdm(range(len(data))):
        img_path=path+f'{index[i]}.png'
        base64_image = encode_image(img_path)
        completion = write_completion_request(data['text'][i], base64_image, model)
        response = requests.post(api_web, headers=headers, json=completion)
        model_outputs.append(response.json()['choices'][0]['message']['content'])
    return model_outputs

#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='4o_mini')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join_dataset')
    parser.add_argument('--key', type=str, default='openaikey')
    args = parser.parse_args()
    args.save_path=f'inference_results/base/{args.model_id}_visual/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)
        
    args.model_repo = MODEL_REPOS[args.model_id]
    data = load_dataset(args.dataset)
    img_path='../../2024-spatial-data-integration-exp/join_imgs/small_imgs/'
    api_web = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json","Authorization": f"Bearer {args.key}"}

    ## load configs
    methods = ['zero_shot', 'few_shot']    
    modes = ['no_heur', 'with_heur_hint', 'with_heur_value']
    heuristics = ['angle', 'distance', 'comb']
    configs=['_'.join(i) for i in list(product(methods, modes, heuristics))]
    configs.remove('zero_shot_no_heur_angle')
    configs.remove('zero_shot_no_heur_distance')
    configs.remove('zero_shot_no_heur_comb')
    configs.remove('few_shot_no_heur_angle')
    configs.remove('few_shot_no_heur_distance')
    configs.remove('few_shot_no_heur_comb')
    configs.append('zero_shot_no_heur')
    configs.append('few_shot_no_heur')
    configs=['zero_shot_no_heur']
    
    ## load test set index
    index=[]
    with open('../../2024-spatial-data-integration-exp/join_index/test.txt', 'r') as file:
        for line in file:
            index.append(line.strip())
    
    #-----------------------------
    # loop through parameters
    #-----------------------------
    for config in configs:
        print('=================================')
        print(f'Config: {config}...')
        def formatting_prompts_func(example):
            output = ""
            if 'value_angle' in config:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])
            elif 'value_distance' in config:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_distance: "+str(example['min_euc_dist'])    
            elif 'value_comb' in config:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])+"\nmin_distance: "+str(example['min_euc_dist'])        
            else:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])
            text = base_alpaca_prompt.format(base_instruction, input, output)
            return { "text" : text}

        base_instruction=INSTRUCTIONS[config]
        test = data['test'].map(formatting_prompts_func)
        args.save_name = f"{args.model_id}_{config}.npy"        
        outputs = evaluate_gpt_4o_vision(test, args.model_repo, index, img_path, api_web, headers)
        np.save(args.save_path+args.save_name, outputs)
        
if __name__ == "__main__":
    main()
