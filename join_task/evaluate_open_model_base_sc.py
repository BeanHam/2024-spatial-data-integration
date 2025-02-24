import argparse
import numpy as np

from utils import *
from prompts import *
from tqdm import tqdm
from itertools import product
from os import path, makedirs
from datasets import load_dataset
from unsloth import FastLanguageModel

## evaluation function
def review_generation(model, tokenizer, data, max_sequence_length):
    outputs=[]
    for text in tqdm(data['text']):
        start_decode = len(tokenizer.encode(text, truncation=True, max_length=max_sequence_length))        
        inputs = tokenizer(text, return_tensors = "pt", max_length=max_sequence_length).to("cuda")
        response = model.generate(**inputs, max_new_tokens = 100)
        response = tokenizer.decode(response[0][start_decode:])
        outputs.append(response)
    return outputs

## evaluation function
def improve_generation(model, tokenizer, data, max_sequence_length):
    outputs=[]
    for text in tqdm(data['text']):
        start_decode = len(tokenizer.encode(text, truncation=True, max_length=max_sequence_length))        
        inputs = tokenizer(text, return_tensors = "pt", max_length=max_sequence_length).to("cuda")
        response = model.generate(**inputs, max_new_tokens = 10)
        response = tokenizer.decode(response[0][start_decode:])
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
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    args.save_path=f'inference_results/base/{args.model_id}/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)
        
    data = load_dataset(args.dataset)    
    configs=['zero_shot_with_heur_value_angle', 
             'few_shot_with_heur_value_angle']
    
    #-----------------------------
    # load model
    #-----------------------------
    args.model_repo = MODEL_REPOS[args.model_id]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_repo,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True
    )
    FastLanguageModel.for_inference(model)
    
    #-----------------------------
    # loop through parameters
    #-----------------------------
    for config in configs:
        print('=================================')
        print(f'Config: {config}...')
        args.save_name = f"{args.model_id}_{config}_sc.npy"
        pred=np.load(f'inference_results/base/{args.model_id}/{args.model_id}_{config}.npy', allow_pickle=True)
        pred=[int(i) for i in pred]
        base_instruction=INSTRUCTIONS[config]
        
        def review_formatting_func(example):
            output = example['pred']
            if 'value_angle' in config:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])
            elif 'value_distance' in config:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_distance: "+str(example['euc_dist'])    
            elif 'value_comb' in config:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])+"\nmin_distance: "+str(example['euc_dist'])        
            else:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])
            text = base_sc_review_alpaca_prompt.format(base_instruction, input, output)
            return { "text" : text}
            
        def improve_formatting_func(example):
            output = example['pred']
            review = example['review']
            if 'value_angle' in config:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])
            elif 'value_distance' in config:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_distance: "+str(example['euc_dist'])    
            elif 'value_comb' in config:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])+"\nmin_distance: "+str(example['euc_dist'])        
            else:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])
            text = base_sc_improve_alpaca_prompt.format(base_instruction, input, output, review)
            return { "text" : text}
                    
        test = data['test'].add_column('pred', pred).map(review_formatting_func)

        ## review generation
        reviews = review_generation(model, tokenizer, test, args.max_seq_length)
        test = test.add_column('review', reviews)
        test = test.map(improve_formatting_func)
        
        ## improved outputs generation
        outputs = improve_generation(model, tokenizer, test, args.max_seq_length)
        np.save(args.save_path+args.save_name, outputs)
        
if __name__ == "__main__":
    main()
