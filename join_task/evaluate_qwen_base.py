import time
import argparse
import numpy as np

from utils import *
from prompts import *
from tqdm import tqdm
from itertools import product
from openai import OpenAI
from os import path, makedirs
from datasets import load_dataset

def evaluate_qwen(data, client, model):    
    model_outputs = []            
    for i in tqdm(range(len(data))):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": data['text'][i]},
            ],
            temperature=0,
            max_tokens=10,
            top_p=1
        )
        model_outputs.append(response.choices[0].message.content)
        time.sleep(0.2)
    return model_outputs

#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='qwen')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join_dataset')
    parser.add_argument('--key', type=str, default='qwenkey')
    args = parser.parse_args()
    args.save_path=f'inference_results/base/{args.model_id}/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)
        
    args.model_repo = MODEL_REPOS[args.model_id]
    client = OpenAI(api_key=args.key,
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    data = load_dataset(args.dataset)
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
        test = data['test'].select(range(10)).map(formatting_prompts_func)
        args.save_name = f"{args.model_id}_{config}.npy"
        outputs = evaluate_qwen(test, client, args.model_repo)
        np.save(args.save_path+args.save_name, outputs)
        
if __name__ == "__main__":
    main()
