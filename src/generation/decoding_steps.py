from enum import Enum


class DecodingStep(Enum):
    STEP_START_TO_PROMPT = 1  # Forces: {\n  "prompt": "
    STEP_PROMPT_VALUE = 2  # Forces: prompt exact value
    STEP_TO_NAME = 3  # Forces: ",\n  "name": "

    AI_NAME_VALUE = 4  # AI Choice: fn_add_numbers or fn_greet...

    STEP_TO_PARAMS = 5  # Forces: ",\n  "parameters": {

    AI_PARAMS_KEYS = 6  # AI Choice: params via the function chosen
    AI_PARAMS_VALS = 7  # AI Choice: filter per type (number, string, boolean)

    STEP_END = 8  # Forces: }\n}
