from pprint import pprint
import requests

from spotlighting import spotlighting
from eval import evaluate_output

############################## GLOBAL VARIABLES AND DEFS ##############################

API_KEY = "2446f4aba26f829a8e1238df75c078d7adb237fac3b6b077ac82a940d990bca8"
API_ENDPOINT = "https://api.together.xyz/v1/completions"
MODEL = "meta-llama/Llama-2-7b-chat-hf"
PROMPT_PATH = "prompts/prompt_os.txt"

METHODS = ['dm', 'ec']
ENC = "rot13"
MALICIOUS = 1

def load_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()

def get_model_response(input_text):
    payload = {
        "model": MODEL,
        "prompt": input_text,
        "max_tokens": 200,
        "stop": ["</s>"]
    }
        
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    return response.json()["choices"][0]["text"]

def apply_spotlighting_and_evaluate(input):
    sys_prompt, user_input, sys_prompt2, label, password = input

    prompt = load_prompt(PROMPT_PATH)
    sys_prompt += "\n" + prompt
    spotlighted_input = spotlighting(sys_prompt, user_input, METHODS, sys_prompt2=sys_prompt2, encoding_method=ENC)
    full_input = spotlighted_input + "\n\n Response: "
    output = get_model_response(full_input)

    return evaluate_output(output, password)

def non_neural_eval(input):
    if input is None:
        return MALICIOUS
    
    return apply_spotlighting_and_evaluate(input)
