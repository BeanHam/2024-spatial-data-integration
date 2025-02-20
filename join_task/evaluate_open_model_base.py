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
def evaluate(model, tokenizer, data):
    outputs=[]
    for text in tqdm(data['text']):
        start_decode = len(tokenizer.encode(text, truncation=True, max_length=2048))        
        inputs = tokenizer(text, return_tensors = "pt", max_length=2048).to("cuda")
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
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    args.save_path=f'inference_results/base/{args.model_id}/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)    
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
            text = base_alpaca_prompt.format(base_instruction, input, output)
            return { "text" : text}
        base_instruction=INSTRUCTIONS[config]
        test = data['test'].select(range(10)).map(formatting_prompts_func)
        args.save_name = f"{args.model_id}_{config}.npy"
        outputs=evaluate(model, tokenizer, test)
        np.save(args.save_path+args.save_name, outputs)
        
if __name__ == "__main__":
    main()
