import argparse
import anthropic
import numpy as np

from utils import *
from prompts import *
from tqdm import tqdm
from openai import OpenAI
from itertools import product
from os import path, makedirs
from datasets import load_dataset

def review_generation(data, client, model):    
    model_outputs = []
    if 'claude' in model:
        for i in tqdm(range(len(data))):
            response = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": data['text'][i]}],
                temperature=0,
                max_tokens=500,
                top_p=1
            )
            model_outputs.append(response.content[0].text)
    else:
        for i in tqdm(range(len(data))):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": data['text'][i]}],
                temperature=0,
                max_tokens=500,
                top_p=1
            )
            model_outputs.append(response.choices[0].message.content)        
    return model_outputs

def improve_generation(data, client, model):    
    model_outputs = []
    if 'claude' in model:
        for i in tqdm(range(len(data))):
            response = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": data['text'][i]}],
                temperature=0,
                max_tokens=10,
                top_p=1
            )
            model_outputs.append(response.content[0].text)
    else:
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
    parser.add_argument('--dataset', type=str, default='beanham/spatial_union_dataset')
    parser.add_argument('--key', type=str, default='key')
    args = parser.parse_args()
    args.save_path=f'inference_results/base/{args.model_id}_correction/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)
        
    data = load_dataset(args.dataset)
    args.model_repo = MODEL_REPOS[args.model_id]
    if args.model_id in ['4o_mini', '4o']:
        client = OpenAI(api_key=args.key)
    elif args.model_id in ['qwen_plus', 'qwen_max']:
        client = OpenAI(api_key=args.key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    elif args.model_id in ['claude']:        
        client = anthropic.Anthropic(api_key=args.key)        
    args.metric_values = [
        'random',
        #'worst_single',
        #'best_single', 
        #'worst_comb',
        #'best_comb', 
        #'worst_all', 
        #'best_all'
    ]
    configs=[
        #'few_shot_with_heur_hint_angle_area',
        'few_shot_with_heur_value_angle_area'
    ]    
    
    for config in configs:
        print('=================================')
        print(f'Config: {config}...')
        
        for metric_value in args.metric_values:
            print('---------------------------------')
            print(f'   Metric Value: {metric_value}...')
            
            args.metric_value = metric_value
            args.save_name = f"{args.model_id}_{metric_value}_{config}_correction.npy"
            args.review_name = f"{args.model_id}_{metric_value}_{config}_correction_reviews.npy"

            def generate_weak_labels(example):
                if args.metric_value == 'worst_single':
                    pred=int(example['min_angle']<=4)
                elif args.metric_value == 'best_single':
                    pred=int(example['max_area']>=0.8)
                elif args.metric_value == 'worst_comb':
                    pred=int((example['max_area']>=0.9)&(example['min_angle']<=1))
                elif args.metric_value == 'best_comb':
                    pred=int((example['max_area']>=0.5)&(example['min_angle']<=3))
                else:
                    pred=np.random.randint(0,2)    
                return { "pred" : pred}
                
            def review_formatting_func(example):
                output = example['pred']
                if config in ['zero_shot_with_heur_value_angle_area', 'few_shot_with_heur_value_angle_area']:
                    input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                            "\nmin_angle: "+str(example['min_angle'])+"\nmax_area: "+str(example['max_area'])
                else:
                    input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])
                text = base_alpaca_prompt_review.format(base_instruction, input, output)
                return { "text" : text}

            def improve_formatting_func(example):
                output = example['pred']
                review = example['review']
                if config in ['zero_shot_with_heur_value_angle_area', 'few_shot_with_heur_value_angle_area']:
                    input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                            "\nmin_angle: "+str(example['min_angle'])+"\nmax_area: "+str(example['max_area'])
                else:
                    input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])   
                text = base_alpaca_prompt_improve.format(base_instruction, input, output, review)
                return { "text" : text}
                
            base_instruction=INSTRUCTIONS[config]
            if args.metric_value=='random':
                np.random.seed(100)
                test = data['test'].add_column('pred', np.random.randint(0,2,len(data['test'])))
            else:
                test = data['test'].map(generate_weak_labels)
            test = test.map(review_formatting_func)

            ## review generation
            reviews = review_generation(test, client, args.model_repo)
            test = test.add_column('review', reviews)
            test = test.map(improve_formatting_func)

            ## improved outputs generation
            outputs = improve_generation(test, client, args.model_repo)
            np.save(args.save_path+args.review_name, reviews)
            np.save(args.save_path+args.save_name, outputs)
            
if __name__ == "__main__":
    main()
