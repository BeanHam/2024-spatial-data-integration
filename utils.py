instruction="You are a helpful geospatial analysis assistant! I will provide you with a pair of (sidewalk, road) GeoJson information. Please help me determine if the sidewalk runs alongside the paired road, specifically checking if the road is adjacent to and parallel with the sidewalk. If the sidewalk is alongside the road, return 1; otherwise, return 0."
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

MODEL_REPOS = {
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
    'llama3': 'meta-llama/Llama-3.1-8B-Instruct',
    'phi': 'microsoft/Phi-3.5-mini-instruct',
    'qwen': 'Qwen/Qwen2.5-7B-Instruct',
    'gpt-4o-mini-2024-07-18': 'ft:gpt-4o-mini-2024-07-18:uw-howe-lab::AUloL5Yu'
}

