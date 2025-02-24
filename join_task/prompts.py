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
instruction_no_heur="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road. If it does, return 1. Otherwise, return 0. No explaination is needed."""

## -------------------
## with heuristic hint
## -------------------

## angle
instruction_heur_hint_angle="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations.

If the condition is satisfied, return 1. Otherwise, return 0. No explaination is needed. """

## distance
instruction_heur_hint_distance="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. 

If the condition is satisfied, return 1. Otherwise, return 0. No explaination is needed. """

## combination
instruction_heur_hint_comb="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format. Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations.
- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. 

If both conditions are satisfied, return 1. Otherwise, return 0. No explaination is needed. """

## -------------------
## with heuristic value
## -------------------

## angle
instruction_heur_value_angle="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with a key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.

If the condition is satisfied, return 1. Otherwise, return 0. No explaination is needed."""

## distance
instruction_heur_value_distance="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with a key statistics:

- min_distance: The minimum distance (in meters) between the sidewalk and the road.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following condition:

- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. The min_distance value helps quantify this proximity.

If the condition is satisfied, return 1. Otherwise, return 0. No explaination is needed."""

## combination
instruction_heur_value_comb="""You are a helpful geospatial analysis assistant. I will provide you with a pair of (sidewalk, road) geometries in GeoJSON format, along with two key statistics:

- min_angle: The minimum angle (in degrees) between the sidewalk and the road.
- min_distance: The minimum distance (in meters) between the sidewalk and the road.

Your task is to determine whether the sidewalk runs alongside the road by evaluating the following conditions:

- Parallelism: The sidewalk should be approximately parallel to the road, with only a small angle difference between their orientations. The min_angle value provides a measure of this alignment.
- Clearance: The sidewalk and road must not overlap or intersect, and they must maintain a certain distance apart. The min_distance value helps quantify this proximity.

If both conditions are satisfied, return 1. Otherwise, return 0. No explaination is needed."""


## ===============================
## base evaluation -- examples
## ===============================

## -------------------
## no heuristics
## -------------------
examples_no_heur="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
Response: 1"""

## -------------------
## with heuristic hint
## -------------------
examples_heur_hint="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
Response: 1"""


## -------------------
## with heuristic value
## -------------------
## angle
examples_heur_value_angle="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
min_angle: 86.12658269425465
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
min_angle: 9.973873687169487
Response: 1"""

## distance
examples_heur_value_distance="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
min_distance: 0.6112785794641761
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
min_distance: 8.72605420848234
Response: 1"""

## combination
examples_heur_value_comb="""

### First Exmaple:
Sidewalk: {'coordinates': [[-122.15646960000001, 47.58741259999999], [-122.1562564, 47.58744089999999]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1563888, 47.5874271], [-122.1563897, 47.5874341], [-122.1564949, 47.5890663], [-122.1564975, 47.5890982]], 'type': 'LineString'}
min_angle: 86.12658269425465
min_distance: 0.6112785794641761
Response: 0

### Second Exmaple:
Sidewalk: {'coordinates': [[-122.13341579999998, 47.54698270000001], [-122.1334011, 47.5468383]], 'type': 'LineString'}
Road: {'coordinates': [[-122.1328993, 47.5458957], [-122.1329478, 47.5460104], [-122.1330183, 47.5461317], [-122.1330885, 47.5462402], [-122.1333795, 47.5466214], [-122.1334411, 47.5467369], [-122.1334757, 47.5468199], [-122.1335148, 47.5469582]], 'type': 'LineString'}
min_angle: 9.973873687169487
min_distance: 8.72605420848234
Response: 1"""


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

INSTRUCTIONS = {
    'zero_shot_no_heur': instruction_no_heur,
    'zero_shot_with_heur_hint_angle': instruction_heur_hint_angle,
    'zero_shot_with_heur_hint_distance': instruction_heur_hint_distance,
    'zero_shot_with_heur_hint_comb': instruction_heur_hint_comb,
    'zero_shot_with_heur_value_angle': instruction_heur_value_angle,
    'zero_shot_with_heur_value_distance': instruction_heur_value_distance,
    'zero_shot_with_heur_value_comb': instruction_heur_value_comb,
    'few_shot_no_heur': instruction_no_heur+examples_no_heur,
    'few_shot_with_heur_hint_angle': instruction_heur_hint_angle+examples_heur_hint,
    'few_shot_with_heur_hint_distance': instruction_heur_hint_distance+examples_heur_hint,
    'few_shot_with_heur_hint_comb': instruction_heur_hint_comb+examples_heur_hint,
    'few_shot_with_heur_value_angle': instruction_heur_value_angle+examples_heur_value_angle,
    'few_shot_with_heur_value_distance': instruction_heur_value_distance+examples_heur_value_distance,
    'few_shot_with_heur_value_comb': instruction_heur_value_comb+examples_heur_value_comb,
}

## ============================
## self correction prompts
## ============================

base_sc_review_alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response: {}

### Review:
Please review and identify if there are any problems with the above response."""

base_sc_improve_alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response: {}

### Review:
Please review and identify if there are any problems with the above response.

{}

### Improve:
Based on your review, please improve your response. 

### Response: """