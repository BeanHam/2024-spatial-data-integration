import argparse
import numpy as np

from utils import *
from tqdm import tqdm
from openai import OpenAI
from os import path, makedirs
from datasets import load_dataset

def evaluate_gpt_4o_series(data, client, model):
    
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
    return model_outputs

def evaluate_gpt_o1_o3(data, client, model):
    
    model_outputs = []            
    for i in tqdm(range(len(data))):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": data['text'][i]},
            ],
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
    parser.add_argument('--key', type=str, default='openaikey')
    args = parser.parse_args()
    args.save_path=f'inference_results/base/{args.model_id}/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)
        
    args.model_repo = MODEL_REPOS[args.model_id]
    client = OpenAI(api_key=args.key)
    data = load_dataset(args.dataset)
    methods = ['zero_shot', 'few_shot']
    modes = ['no_exp', 'with_exp']

    #-----------------------------
    # loop through methods & modes
    #-----------------------------
    for method in methods:
        for mode in modes:
            print('=================================')
            print(f'Method: {method}...')
            print(f'Mode: {mode}...')
            
            def formatting_prompts_func(example):
                output = ""
                if method=='zero_shot':                
                    if mode=='no_exp':
                        input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])
                        text = base_alpaca_prompt.format(instruction_no_exp, input, output)
                    else:
                        input = "Sidewalk: "+str(example['sidewalk'])+\
                                "\nRoad: "+str(example['road'])+\
                                "\nmin_angle: "+str(example['min_angle'])+\
                                "\nmin_distance: "+str(example['euc_dist'])
                        text = base_alpaca_prompt.format(instruction_with_exp, input, output)
                else:
                    if mode=='no_exp':
                        input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])                        
                        text = base_alpaca_prompt.format(instruction_no_exp+example_one_no_exp+example_two_no_exp, input, output)
                    else:
                        input = "Sidewalk: "+str(example['sidewalk'])+\
                                "\nRoad: "+str(example['road'])+\
                                "\nmin_angle: "+str(example['min_angle'])+\
                                "\nmin_distance: "+str(example['euc_dist'])
                        text = base_alpaca_prompt.format(instruction_with_exp+example_one_with_exp+example_two_with_exp, input, output)
                return { "text" : text}
            
            test = data['test'].map(formatting_prompts_func)
            if '4o' in args.model_repo:
                outputs = evaluate_gpt_4o_series(test, client, args.model_repo)
            else:
                outputs = evaluate_gpt_o1_o3(test, client, args.model_repo)
            args.save_name = f"{args.model_id}_{method}_{mode}"
            np.save(args.save_path+args.save_name+".npy", outputs)

if __name__ == "__main__":
    main()
