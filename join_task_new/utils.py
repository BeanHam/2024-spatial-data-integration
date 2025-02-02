instruction="You are a helpful geospatial analysis assistant! I will provide you with a pair of (sidewalk, road) GeoJson information. Please help me determine if the sidewalk runs alongside the paired road, specifically checking if the road is adjacent to and approximately parallel with the sidewalk. If the sidewalk is alongside the road, return 1; otherwise, return 0."

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

gpt_instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful geospatial analysis assistant! I will provide you with a pair of (sidewalk, road) GeoJson information. Please help me determine if the sidewalk runs alongside the paired road, specifically checking if the road is adjacent to and parallel with the sidewalk. If the sidewalk is alongside the road, return 1; otherwise, return 0."""

MODEL_REPOS = {
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
    'llama3': 'meta-llama/Llama-3.1-8B-Instruct',
    'phi': 'microsoft/Phi-3.5-mini-instruct',
    'qwen': 'Qwen/Qwen2.5-7B-Instruct',
    'gpt-4o-mini-2024-07-18': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::AUloL5Yu'
}

MODEL_PATHS = {
    'llama3_degree_1': 'beanham/spatial_join_new_llama3_degree_1',
    'llama3_degree_2': 'beanham/spatial_join_new_llama3_degree_2',
    'llama3_degree_5': 'beanham/spatial_join_new_llama3_degree_5',
    'llama3_degree_10': 'beanham/spatial_join_new_llama3_degree_10',
    'llama3_degree_20': 'beanham/spatial_join_new_llama3_degree_20',
}

MODEL_STEPS_PATHS = {
    'llama3_degree_step_1': 'outputs_llama3/degree/5/checkpoint-753',
    'llama3_degree_step_2': 'outputs_llama3/degree/5/checkpoint-1506',
    'llama3_degree_step_3': 'outputs_llama3/degree/5/checkpoint-2259',
    'llama3_degree_step_4': 'outputs_llama3/degree/5/checkpoint-3012',
    'llama3_degree_step_5': 'outputs_llama3/degree/5/checkpoint-3765',
}
