import argparse
import numpy as np

from utils import *
from prompts import *
from tqdm import tqdm
from itertools import product
from openai import OpenAI
from os import path, makedirs
from datasets import load_dataset

def review_generation(data, client, model):    
    model_outputs = []            
    for i in tqdm(range(len(data))):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": data['text'][i]}],
            temperature=0,
            max_tokens=200,
            top_p=1
        )
        model_outputs.append(response.choices[0].message.content)
    return model_outputs

def improve_generation(data, client, model):    
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
    parser.add_argument('--key', type=str, default='openaikey')
    args = parser.parse_args()
    args.save_path=f'inference_results/base/{args.model_id}_ec/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)

    data = load_dataset(args.dataset)
    configs=['zero_shot_with_heur_value_comb',             
             'few_shot_with_heur_value_comb']
    
    args.model_repo = MODEL_REPOS[args.model_id]
    client = OpenAI(api_key=args.key)      

    ## evaluate on a subset
    np.random.seed(100)
    subset_index=np.random.randint(0, len(data['test']), 1000)
    
    for config in configs:
        print('=================================')
        print(f'Config: {config}...')
        
        args.save_name = f"{args.model_id}_comb_{config}_ec.npy"
        args.review_name = f"{args.model_id}_comb_{config}_ec_reviews.npy"
        
        def generate_weak_labels(example):
            pred=int((example['min_euc_dist']>=2)&(example['min_angle']<=10))
            return { "pred" : pred}
            
        def review_formatting_func(example):
            output = example['pred']
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
            text = base_ec_review_alpaca_prompt.format(base_instruction, input, output)
            return { "text" : text}
            
        def improve_formatting_func(example):
            output = example['pred']
            review = example['review']
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
            text = base_ec_improve_alpaca_prompt.format(base_instruction, input, output, review)
            return { "text" : text}
        
        base_instruction=INSTRUCTIONS[config]
        test = data['test'].select(subset_index).map(generate_weak_labels)
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
