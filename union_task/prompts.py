## =================
## base evaluation
## =================
base_alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response: 
{}"""

## ===============================
## base evaluation -- instructions
## ===============================


## -------------------
## no heuristics
## -------------------
instruction_no_heur="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format. Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially. If they do, return 1. Otherwise, return 0. No explanation is needed."""

instruction_no_heur_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format. Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially. If they do, return 1. Otherwise, return 0. Please solve the task step by step."""

## ------------------------
## with heuristic hint
## ------------------------

## -----
## angle
## -----
instruction_heur_hint_angle="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format. Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following condition:

- Parallelism: The two sidewalks should be approximately parallel, with only a small angular difference in their orientations.

If the condition is satisfied, return 1. Otherwise, return 0. No explanation is needed. """

instruction_heur_hint_angle_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format. Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following condition:

- Parallelism: The two sidewalks should be approximately parallel, with only a small angular difference in their orientations.

If the condition is satisfied, return 1. Otherwise, return 0. Please solve the task step by step."""

## -----
## area
## -----
instruction_heur_hint_area="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format. Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following condition:

- Overlap: The two sidewalks must fully or partially overlap. Simply connecting at the endpoints does not count as an intersection.

If the condition is satisfied, return 1. Otherwise, return 0. No explanation is needed. """

instruction_heur_hint_area_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format. Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following condition:

- Overlap: The two sidewalks must fully or partially overlap. Simply connecting at the endpoints does not count as an intersection.

If the condition is satisfied, return 1. Otherwise, return 0. Please solve the task step by step."""

## -------------------------
## combination (angle, area)
## -------------------------
instruction_heur_hint_angle_area="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format. Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following conditions:

- Parallelism: The two sidewalks should be approximately parallel, with only a small angular difference in their orientations.
- Overlap: The two sidewalks must fully or partially overlap. Simply connecting at the endpoints does not count as an intersection.

If both conditions are satisfied, return 1. Otherwise, return 0. No explanation is needed. """

instruction_heur_hint_angle_area_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format. Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following conditions:

- Parallelism: The two sidewalks should be approximately parallel, with only a small angular difference in their orientations.
- Overlap: The two sidewalks must fully or partially overlap. Simply connecting at the endpoints does not count as an intersection.

If both conditions are satisfied, return 1. Otherwise, return 0. Please solve the task step by step."""

## ---------------------------
## with heuristic value
## ---------------------------

## -----
## angle
## -----
instruction_heur_value_angle="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format, along with a key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.

Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following condition:

- Parallelism: The two sidewalks should be approximately parallel, with only a small angular difference in their orientations. The min_angle value provides a measure of this alignment.

If the condition is satisfied, return 1. Otherwise, return 0. No explanation is needed."""

instruction_heur_value_angle_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format, along with a key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.

Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following condition:

- Parallelism: The two sidewalks should be approximately parallel, with only a small angular difference in their orientations. The min_angle value provides a measure of this alignment.

If the condition is satisfied, return 1. Otherwise, return 0. Please solve the task step by step."""

## -----
## area
## -----
instruction_heur_value_area="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format, along with a key statistics:

- max_area: The maximum percentage of overlapping area relative to both sidewalks, considering a 10-meter buffer.

Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following condition:

- Overlap: The two sidewalks must fully or partially overlap. Simply connecting at the endpoints does not count as an intersection. The max_area values help quantify this overlap. 

If the condition is satisfied, return 1. Otherwise, return 0. No explanation is needed."""

instruction_heur_value_area_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk 1, sidewalk 2) geometries in GeoJSON format, along with a key statistics:

- max_area: The maximum percentage of overlapping area relative to both sidewalks, considering a 10-meter buffer.

Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following condition:

- Overlap: The two sidewalks must fully or partially overlap. Simply connecting at the endpoints does not count as an intersection. The max_area values help quantify this overlap. 

If the condition is satisfied, return 1. Otherwise, return 0. Please solve the task step by step."""


## -------------------------
## combination (angle, area)
## -------------------------
instruction_heur_value_angle_area="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with two key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.
- max_area: The maximum percentage of overlapping area relative to the sidewalk and road, considering a 10-meter buffer.

Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following conditions:

- Parallelism: The two sidewalks should be approximately parallel, with only a small angular difference in their orientations. The min_angle value provides a measure of this alignment.
- Overlap: The two sidewalks must fully or partially overlap. Simply connecting at the endpoints does not count as an intersection. The max_area values help quantify this overlap. 

If both conditions are satisfied, return 1. Otherwise, return 0. No explanation is needed."""


instruction_heur_value_angle_area_cot="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with two key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.
- max_area: The maximum percentage of overlapping area relative to the sidewalk and road, considering a 10-meter buffer.

Your task is to determine whether these two geometries represent the same sidewalk, either fully or partially, by evaluating the following conditions:

- Parallelism: The two sidewalks should be approximately parallel, with only a small angular difference in their orientations. The min_angle value provides a measure of this alignment.
- Overlap: The two sidewalks must fully or partially overlap. Simply connecting at the endpoints does not count as an intersection. The max_area values help quantify this overlap. 

If both conditions are satisfied, return 1. Otherwise, return 0. Please solve the task step by step."""

## ===============================
## base evaluation -- examples
## ===============================

## -------------------
## no heuristics
## -------------------
examples_no_heur="""

### First Exmaple:
Sidewalk 1: {'coordinates': [[-122.1731058, 47.5709015], [-122.1743964, 47.5709069]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1743856, 47.570905], [-122.1743848, 47.5710029]], 'type': 'LineString'}
Response: 0

### Second Exmaple:
Sidewalk 1: {'coordinates': [[-122.17248400000001, 47.570692699999995], [-122.17136010000002, 47.5706601], [-122.1698608, 47.5706113]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1700114, 47.5706215], [-122.1703076, 47.5706256]], 'type': 'LineString'}
Response: 1"""

## -------------------
## with heuristic hint
## -------------------
examples_heur_hint="""

### First Exmaple:
Sidewalk 1: {'coordinates': [[-122.1731058, 47.5709015], [-122.1743964, 47.5709069]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1743856, 47.570905], [-122.1743848, 47.5710029]], 'type': 'LineString'}
Response: 0

### Second Exmaple:
Sidewalk 1: {'coordinates': [[-122.17248400000001, 47.570692699999995], [-122.17136010000002, 47.5706601], [-122.1698608, 47.5706113]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1700114, 47.5706215], [-122.1703076, 47.5706256]], 'type': 'LineString'}
Response: 1"""

## -------------------
## with heuristic value
## -------------------

## negative example: train[0]
## positive example: train[1]

## angle
examples_heur_value_angle="""

### First Exmaple:
Sidewalk 1: {'coordinates': [[-122.1731058, 47.5709015], [-122.1743964, 47.5709069]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1743856, 47.570905], [-122.1743848, 47.5710029]], 'type': 'LineString'}
min_angle: 89.77154191724516
Response: 0

### Second Exmaple:
Sidewalk 1: {'coordinates': [[-122.17248400000001, 47.570692699999995], [-122.17136010000002, 47.5706601], [-122.1698608, 47.5706113]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1700114, 47.5706215], [-122.1703076, 47.5706256]], 'type': 'LineString'}
min_angle: 0.8684260514779112
Response: 1"""

## area
examples_heur_value_area="""

### First Exmaple:
Sidewalk 1: {'coordinates': [[-122.1731058, 47.5709015], [-122.1743964, 47.5709069]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1743856, 47.570905], [-122.1743848, 47.5710029]], 'type': 'LineString'}
max_area: 0.5473559863066643
Response: 0

### Second Exmaple:
Sidewalk 1: {'coordinates': [[-122.17248400000001, 47.570692699999995], [-122.17136010000002, 47.5706601], [-122.1698608, 47.5706113]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1700114, 47.5706215], [-122.1703076, 47.5706256]], 'type': 'LineString'}
max_area: 0.9832547250026938
Response: 1"""


## combination (angle, area)
examples_heur_value_angle_area="""

### First Exmaple:
Sidewalk 1: {'coordinates': [[-122.1731058, 47.5709015], [-122.1743964, 47.5709069]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1743856, 47.570905], [-122.1743848, 47.5710029]], 'type': 'LineString'}
min_angle: 89.77154191724516
max_area: 0.5473559863066643
Response: 0

### Second Exmaple:
Sidewalk 1: {'coordinates': [[-122.17248400000001, 47.570692699999995], [-122.17136010000002, 47.5706601], [-122.1698608, 47.5706113]], 'type': 'LineString'}
Sidewalk 2: {'coordinates': [[-122.1700114, 47.5706215], [-122.1703076, 47.5706256]], 'type': 'LineString'}
min_angle: 0.8684260514779112
max_area: 0.9832547250026938
Response: 1"""


INSTRUCTIONS = {
    'zero_shot_no_heur': instruction_no_heur,    
    'zero_shot_with_heur_hint_angle': instruction_heur_hint_angle,
    'zero_shot_with_heur_hint_area': instruction_heur_hint_area,
    'zero_shot_with_heur_hint_angle_area': instruction_heur_hint_angle_area,    
    'zero_shot_with_heur_value_angle': instruction_heur_value_angle,
    'zero_shot_with_heur_value_area': instruction_heur_value_area,    
    'zero_shot_with_heur_value_angle_area': instruction_heur_value_angle_area,
    
    'few_shot_no_heur': instruction_no_heur + examples_no_heur,    
    'few_shot_with_heur_hint_angle': instruction_heur_hint_angle + examples_heur_hint,
    'few_shot_with_heur_hint_area': instruction_heur_hint_area + examples_heur_hint,
    'few_shot_with_heur_hint_angle_area': instruction_heur_hint_angle_area + examples_heur_hint,    
    'few_shot_with_heur_value_angle': instruction_heur_value_angle + examples_heur_value_angle,
    'few_shot_with_heur_value_area': instruction_heur_value_area + examples_heur_value_area,  
    'few_shot_with_heur_value_angle_area': instruction_heur_value_angle_area + examples_heur_value_angle_area,
}

COT_INSTRUCTIONS = {
    'zero_shot_no_heur_cot': instruction_no_heur_cot,    
    'zero_shot_with_heur_hint_angle_cot': instruction_heur_hint_angle_cot,
    'zero_shot_with_heur_hint_area_cot': instruction_heur_hint_area_cot,
    'zero_shot_with_heur_hint_angle_area_cot': instruction_heur_hint_angle_area_cot,    
    'zero_shot_with_heur_value_angle_cot': instruction_heur_value_angle_cot,
    'zero_shot_with_heur_value_area_cot': instruction_heur_value_area_cot,    
    'zero_shot_with_heur_value_angle_area_cot': instruction_heur_value_angle_area_cot,
    
    'few_shot_no_heur_cot': instruction_no_heur_cot + examples_no_heur,    
    'few_shot_with_heur_hint_angle_cot': instruction_heur_hint_angle_cot + examples_heur_hint,
    'few_shot_with_heur_hint_area_cot': instruction_heur_hint_area_cot + examples_heur_hint,
    'few_shot_with_heur_hint_angle_area_cot': instruction_heur_hint_angle_area_cot + examples_heur_hint,    
    'few_shot_with_heur_value_angle_cot': instruction_heur_value_angle_cot + examples_heur_value_angle,
    'few_shot_with_heur_value_area_cot': instruction_heur_value_area_cot + examples_heur_value_area,  
    'few_shot_with_heur_value_angle_area_cot': instruction_heur_value_angle_area_cot + examples_heur_value_angle_area,
}

## ============================
## self correction prompts
## ============================

base_alpaca_prompt_review = """### Instruction:
{}

### Input:
{}

### Response: {}

### Review:
The above response is generated using heuristics. Please carefully review and identify if there are any problems with the above response."""

base_alpaca_prompt_improve = """### Instruction:
{}

### Input:
{}

### Response: {}

### Review:
The above response is generated using heuristics. Please carefully review and identify if there are any problems with the above response.

{}

### Improve:
Based on your review, please give your final response, either 0 or 1. 

### Response: """



## ============================
## finetuning instructions
## ============================
instruction="You are a helpful geospatial analysis assistant! I will provide you with a pair of (sidewalk, road) GeoJson information. Please help me determine if the sidewalk runs alongside the paired road, specifically checking if the road is adjacent to and parallel with the sidewalk. If the sidewalk is alongside the road, return 1; otherwise, return 0."

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
