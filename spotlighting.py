import base64
from pprint import pprint

############################## DELIMITING & DATAMARKING ##############################

### delimiting: mark where user input starts and ends
def delimiting(user_input):
    if user_input is None:
        return ""
    return f"[START]{user_input}[END]"

### datamarking: replace spaces with special char (^)
def datamarking(user_input):
    if user_input is None:
        return ""
    return user_input.translate(str.maketrans(" ", "^"))


############################## ENCODING ##############################

def encode_base64(user_input):
    return base64.b64encode(user_input.encode()).decode()

def encode_rot13(user_input):
    return user_input.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", 
                                              "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"))

def encode_binary(user_input):
    return " ".join(format(ord(c), "08b") for c in user_input)

### determines which encoding to use and calls helper function
def encoding(user_input, method="base64"):
    if user_input is None:
        return ""
    
    if method == "rot13":
        return encode_rot13(user_input)
    elif method == "binary":
        return encode_binary(user_input)
    else: # default is base64
        return encode_base64(user_input)


############################## SCAFFOLDING ##############################

### determines which spotlighting methods are used to generate prompt to append to system prompt
def generate_spotlight_prompt(methods, encoding_method):
    phrases = []
    if "dl" in methods:
        phrases.append("enclosed within [START] and [END] tags")
    if "dm" in methods:
        phrases.append("interleaved with the special character \"ˆ\" between every word")
    if "ec" in methods:
        phrases.append(f"encoded in {encoding_method}" if encoding_method else "encoded")

    # handle each case separately for grammatical precision
    if len(phrases) == 1:
        prompt = f"The user input is {phrases[0]} to help you distinguish the user input from system prompts."
    elif len(phrases) == 2:
        prompt = f"The user input is {phrases[0]} and {phrases[1]} to help you distinguish the user input from system prompts."
    else:
        prompt = f"The user input is {phrases[0]}, {phrases[1]}, and {phrases[2]} to help you distinguish the user input from system prompts."

    return prompt


### given a list of spotlighting methods in `methods`, transform the user input and concatenate all LLM inputs
### note: sys_prompt2 is only for tensortrust_extraction
def spotlighting(sys_prompt, user_input, methods, sys_prompt2=None, encoding_method="base64"):
    if "dl" in methods:
        spotlighted_input = delimiting(user_input)
    if "dm" in methods:
        spotlighted_input = datamarking(user_input)
    if "ec" in methods:
        spotlighted_input = encoding(user_input, encoding_method)

    spotlight_prompt = generate_spotlight_prompt(sys_prompt, encoding_method)

    if sys_prompt2 == None:
        input = f"{sys_prompt}\n{spotlight_prompt}\n{spotlighted_input}" 
    else: # tensortrust_extraction has sys_prompt1 and sys_prompt2
        input = f"{sys_prompt}\n{spotlight_prompt}\n{spotlighted_input}\n{sys_prompt2}"
    
    return input