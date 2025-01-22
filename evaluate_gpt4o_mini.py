import argparse
import numpy as np

from utils import *
from tqdm import tqdm
from openai import OpenAI
from os import path, makedirs
from datasets import load_dataset

def evaluate_gpt(data, client, model):
    
    model_outputs = []            
    for i in tqdm(range(len(data))):
        
        user_input="Sidewalk: "+str(data['sidewalk'][i])+"\nRoad: "+str(data['road'][i])
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": gpt_instruction},
                {"role": "user", "content": user_input},
            ],
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
    parser.add_argument('--metric_name', type=str, default='degree')
    args = parser.parse_args()

    args.save_path=f'inference_results/{args.metric_name}/'
    if not path.exists(args.save_path):
        makedirs(args.save_path)

    if args.metric_name == 'degree':
        args.metric_values = [1,2,5,10,20]
    elif args.metric_name == 'distance':
        #args.metric_values = [1,2,3,4,5]
        args.metric_values = [4,5]
        
    # ----------------------
    # Load Data
    # ----------------------
    print('Downloading and preparing data...')
    test = load_dataset(args.dataset, split='test')
    client = OpenAI(api_key=args.key)

    #---------------------------
    # loop through metric values
    #---------------------------
    for v in args.metric_values:
        print('=====================================================')
        print(f'{args.metric_name}: {v}...')
        args.model_path = MODEL_PATHS[f"{args.model_id}_{args.metric_name}_{v}"]
        args.save_name = f"{args.model_id}_{args.metric_name}_{v}"
        outputs = evaluate_gpt(test, client, args.model_path)
        np.save(args.save_path+args.save_name+".npy", outputs)

if __name__ == "__main__":
    main()
