import argparse
import numpy as np

from utils import *
from prompts import *
from tqdm import tqdm
from itertools import product
from openai import OpenAI
from os import path, makedirs
from datasets import load_dataset

def evaluate(data, client, model):
    model_outputs = []            
    for i in tqdm(range(len(data))):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": data['text'][i]}],
            temperature=0,
            max_tokens=10,
            top_p=1
        )
        model_outputs.append(response.choices[0].message.content)
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
    parser.add_argument('--key', type=str, default='key')
    args = parser.parse_args()
    args.save_path=f'inference_results/base/{args.model_id}/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)
        
    args.model_repo = MODEL_REPOS[args.model_id]
    if args.model_id=='4o_mini':
        client = OpenAI(api_key=args.key)
    elif args.model_id=='qwen':
        client = OpenAI(api_key=args.key,base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    data = load_dataset(args.dataset)
    configs=list(INSTRUCTIONS.keys())
    configs=[
        'few_shot_with_heur_value_all'
    ]
    
    #-----------------------------
    # loop through parameters
    #-----------------------------
    for config in configs:
        print('=================================')
        print(f'Config: {config}...')
        
        def formatting_prompts_func(example):
            output = ""
            if config in ['zero_shot_with_heur_value_angle', 'few_shot_with_heur_value_angle']:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])
            elif config in ['zero_shot_with_heur_value_distance', 'few_shot_with_heur_value_distance']:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_distance: "+str(example['min_euc_dist'])
            elif config in ['zero_shot_with_heur_value_area', 'few_shot_with_heur_value_area']:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmax_area: "+str(example['max_area'])
            elif config in ['zero_shot_with_heur_value_angle_distance', 'few_shot_with_heur_value_angle_distance']:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])+"\nmin_distance: "+str(example['min_euc_dist'])
            elif config in ['zero_shot_with_heur_value_angle_area', 'few_shot_with_heur_value_angle_area']:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])+"\nmax_area: "+str(example['max_area'])
            elif config in ['zero_shot_with_heur_value_distance_area', 'few_shot_with_heur_value_distance_area']:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_distance: "+str(example['min_euc_dist'])+"\nmax_area: "+str(example['max_area'])    
            elif config in ['zero_shot_with_heur_value_all', 'few_shot_with_heur_value_all']:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])+"\nmin_distance: "+str(example['min_euc_dist'])+\
                        "\nmax_area: "+str(example['max_area'])
            else:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])
            text = base_alpaca_prompt.format(base_instruction, input, output)
            return { "text" : text}
            
        base_instruction=INSTRUCTIONS[config]
        test = data['test'].map(formatting_prompts_func)
        outputs = evaluate(test, client, args.model_repo)
        args.save_name = f"{args.model_id}_{config}.npy"
        np.save(args.save_path+args.save_name, outputs)
        
if __name__ == "__main__":
    main()
