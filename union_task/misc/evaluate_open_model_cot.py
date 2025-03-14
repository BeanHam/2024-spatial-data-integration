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
def evaluate(model, tokenizer, data, max_sequence_length):
    outputs=[]
    for text in tqdm(data['text']):
        start_decode = len(tokenizer.encode(text, truncation=True, max_length=max_sequence_length))        
        inputs = tokenizer(text, return_tensors = "pt", max_length=max_sequence_length).to("cuda")
        response = model.generate(**inputs, max_new_tokens = 500)
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
    parser.add_argument('--dataset', type=str, default='beanham/spatial_union_dataset')
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    args.save_path=f'inference_results/base/{args.model_id}/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)
    data = load_dataset(args.dataset)
    configs=list(COT_INSTRUCTIONS.keys())

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
        
        def formatting_prompts_func(example):
            output = ""
            if config in ['zero_shot_with_heur_value_angle_cot', 'few_shot_with_heur_value_angle_cot']:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])
            elif config in ['zero_shot_with_heur_value_area_cot', 'few_shot_with_heur_value_area_cot']:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmax_area: "+str(example['max_area'])
            elif config in ['zero_shot_with_heur_value_angle_area_cot', 'few_shot_with_heur_value_angle_area_cot']:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])+\
                        "\nmin_angle: "+str(example['min_angle'])+"\nmax_area: "+str(example['max_area'])
            else:
                input = "Sidewalk: "+str(example['sidewalk'])+"\nRoad: "+str(example['road'])
            text = base_alpaca_prompt.format(base_instruction, input, output)
            return { "text" : text}
            
        base_instruction=COT_INSTRUCTIONS[config]
        test = data['test'].map(formatting_prompts_func)
        outputs=evaluate(model, tokenizer, test, args.max_seq_length)
        args.save_name = f"{args.model_id}_{config}.npy"        
        np.save(args.save_path+args.save_name, outputs)
        
if __name__ == "__main__":
    main()
