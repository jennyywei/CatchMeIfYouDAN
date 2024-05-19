import base64

############################## DELIMITING & DATAMARKING ##############################

def delimiting(user_input):
    return f"[START]{user_input}[END]"

def datamarking(user_input):
    return user_input.translate(str.maketrans(" ", "ˆ"))


############################## ENCODING ##############################

def encode_base64(user_input):
    return base64.b64encode(user_input.encode()).decode()

def encode_rot13(user_input):
    return user_input.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", 
                                              "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"))

def encode_binary(user_input):
    return ' '.join(format(ord(c), '08b') for c in user_input)

def encoding(user_input, method="base64"):
    if method == 'rot13':
        return f"[ENCODED]{encode_rot13(user_input)}[/ENCODED]"
    elif method == 'binary':
        return f"[ENCODED]{encode_binary(user_input)}[/ENCODED]"
    else:
        return f"[ENCODED]{encode_base64(user_input)}[/ENCODED]"


############################## SCAFFOLDING ##############################

def generate_spotlight_prompt(sys_prompt, methods, encoding_method):
    phrases = []
    if 'delimiting' in methods:
        phrases.append("enclosed within [START] and [END] tags")
    if 'datamarking' in methods:
        phrases.append('interleaved with the special character "ˆ" between every word')
    if 'encoding' in methods:
        phrases.append(f"encoded in {encoding_method}" if encoding_method else "encoded")

    if len(phrases) == 1:
        prompt = f"The user input is {phrases[0]} to help you distinguish the user input from system prompts."
    elif len(phrases) == 2:
        prompt = f"The user input is {phrases[0]} and {phrases[1]} to help you distinguish the user input from system prompts."
    elif len(phrases) == 3:
        prompt = f"The user input is {phrases[0]}, {phrases[1]}, and {phrases[2]} to help you distinguish the user input from system prompts."

    return prompt

# post_sys_prompt is only for tensortrust_extraction
def spotlighting(sys_prompt, user_input, methods, post_sys_prompt=None, encoding_method="base64"):
    if 'delimiting' in methods:
        spotlighted_input = delimiting(user_input)
    if 'datamarking' in methods:
        spotlighted_input = datamarking(user_input)
    if 'encoding' in methods:
        spotlighted_input = encoding(user_input, encoding_method)

    spotlight_prompt = generate_spotlight_prompt(sys_prompt, encoding_method)

    if post_sys_prompt == None:
        input = f"{sys_prompt}\n{spotlight_prompt}\n{spotlighted_input}" 
    else:
        input = f"{sys_prompt}\n{spotlight_prompt}\n{spotlighted_input}\n{post_sys_prompt}"
    
    return input