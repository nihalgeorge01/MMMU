import re
import ast
import os
import json
from PIL import Image
from tqdm import tqdm
import sys
import json
import sys
from openai import OpenAI
import base64
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset

MMMU_SUBSETS = [
    'Accounting', 
    'Agriculture', 
    'Architecture_and_Engineering', 
    'Art', 
    'Art_Theory', 
    'Basic_Medical_Science', 
    'Biology', 
    'Chemistry', 
    'Clinical_Medicine', 
    'Computer_Science', 
    'Design', 
    'Diagnostics_and_Laboratory_Medicine', 
    'Economics', 
    'Electronics', 
    'Energy_and_Power', 
    'Finance', 
    'Geography', 
    'History', 
    'Literature', 
    'Manage', 
    'Marketing', 
    'Materials', 
    'Math', 
    'Mechanical_Engineering', 
    'Music', 
    'Pharmacy', 
    'Physics', 
    'Psychology', 
    'Public_Health', 
    'Sociology'
]

if len(sys.argv) == 6:
    MODEL_NAME = sys.argv[1]
    BASE_URL = sys.argv[2]
    MODE = sys.argv[3]
    DATASET = sys.argv[4]
    SUBSET = sys.argv[5]
else:
    print("Usage: python infer_mlc.py <MODEL_NAME> <BASE_URL> <MODE> <DATASET> <SUBSET>, example: python infer_mlc.py 'Phi-3.5-vision-instruct' http://127.0.0.1:8000 direct MMMU/MMMU Accounting")
    sys.exit(1)

API_KEY = 'your_api_key_here'
WORKERS = 1
TEMPERATURE = 0.0

# Load prompts from YAML file
import yaml
with open("prompts.yaml", "r") as file:
    prompt_config = yaml.safe_load(file)[MODE]


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    question = f"{question}\n{parsed_options}\n{prompt_config['standard']}"
    return question

def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)

def origin_mmmu_doc_to_visual(doc):
    visual = []
    for i in range(1,8):
        if not doc[f'image_{i}']:
            break
        visual.append(doc[f'image_{i}'])
    return visual

def vision_mmmu_doc_to_visual(doc):
    return [doc['image']]

def load_model(model_name=MODEL_NAME, base_url=BASE_URL, api_key=API_KEY, model=MODEL_NAME):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    return model_components

def request(prompt, timeout=120, max_tokens=128, base_url="", api_key="", model=MODEL_NAME, model_name=MODEL_NAME):
    client = OpenAI(base_url=base_url, api_key=api_key)
    include_system = False
    response = client.chat.completions.create(
        model=model,
        messages = [{"role": "system", "content": "You're a useful assistant."}] * include_system \
        + [{"role": "user", "content": prompt, "temperature": TEMPERATURE}],
        stream=False, max_tokens=max_tokens, timeout=timeout)
    return response

def encode_pil_image(pil_image):
    # Create a byte stream object
    buffered = BytesIO()
    # Save the PIL image object as a byte stream in PNG format
    pil_image.save(buffered, format="PNG")
    # Get the byte stream data and perform Base64 encoding
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# Function to create interleaved content of texts and images
def make_interleave_content(texts_or_image_paths):
    content = []
    for text_or_path in texts_or_image_paths:
        if isinstance(text_or_path, str):
            text_elem = {
                "type": "text",
                "text": text_or_path
            }
            content.append(text_elem)
        else:
            base64_image = encode_pil_image(text_or_path)
            image_elem = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            content.append(image_elem)
    return content

# Function to send request with images and text
def request_with_images(texts_or_image_paths, timeout=60, max_tokens=300, base_url=BASE_URL, api_key=API_KEY, model=MODEL_NAME):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f"You are a useful assistant and a large language model.",
                "role": "user",
                "content": make_interleave_content(texts_or_image_paths)
            }
        ],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE
    }

    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
    return response.json()

def infer(prompts, max_tokens=4096, use_vllm=False, **kwargs):
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)
    
    if isinstance(prompts, list):
        prompts = prompts[0]
    
    try:
        if isinstance(prompts, dict) and 'images' in prompts:
            prompts, images = prompts['prompt'], prompts['images']
            response_tmp = request_with_images([prompts, *images], max_tokens=max_tokens, base_url=base_url, api_key=api_key, model=model)
            print
            response = response_tmp["choices"][0]["message"]["content"]
        else:
            response = request(prompts, base_url=base_url, api_key=api_key, model=model, model_name=model_name)["choices"][0]["message"]["content"]
    except Exception as e:
        response = {"error": str(e)}
    
    return response


def process_prompt(data, model_components):
    if 'standard' in SUBSET or DATASET == "MMMU/MMMU":
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif SUBSET == 'vision':
        prompt = prompt_config['vision']
        images = vision_mmmu_doc_to_visual(data)
    else:
        print("Unexpected dataset/subset, inspect code")

    return infer({"prompt": prompt, "images": images}, max_tokens=4096, **model_components), data

def run_and_save():
    def save_results_to_file(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for output, data in results:
                data['response'] = output
                data = {k: v for k, v in data.items() if not (k.startswith('image'))}
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
    
    def save_results_to_file_append(results, output_path):
        with open(output_path, 'a', encoding='utf-8') as outfile:
            for output, data in results:
                data['response'] = output
                data = {k: v for k, v in data.items() if not (k.startswith('image'))}
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
    
    def process_and_save_part(part_data, part_name, split_name, model_components):
        print(f"Begin processing {part_name}")
        results = []
        sanitized_model_name = "-".join("-".join(MODEL_NAME.split('/')).split('.'))
        sanitized_dataset_name = "-".join("-".join(DATASET.split('/')).split('.'))
        output_path = f"./output/{sanitized_model_name}_{sanitized_dataset_name}-{part_name}-{split_name}_{MODE}.jsonl"
        
        done_ct = 0
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    done_ct += 1
            print(f"Loaded existing results for {part_name}")
        start_idx = done_ct
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(process_prompt, data, model_components) for data in part_data.select(list(range(start_idx,len(part_data))))]
            for future in tqdm(futures, desc=f"Processing {part_name}"):
                result, data = future.result()
                results.append((result, data))

        save_results_to_file_append(results, output_path)
        return output_path

    temp_files = []
    subsets_to_run = [SUBSET]
    splits_to_run = ['test']
    if DATASET == "MMMU/MMMU":
        splits_to_run = ["validation"]
        if SUBSET == "all":
            subsets_to_run = MMMU_SUBSETS
    
    model_components = load_model(model_name=MODEL_NAME, base_url=BASE_URL, api_key=API_KEY, model=MODEL_NAME)
    
    for sub in subsets_to_run:
        for spl in splits_to_run:
            dataset = load_dataset(DATASET, sub, split=spl)
            temp_files.append(process_and_save_part(dataset, sub, spl, model_components))

    
    temp_files.append(process_and_save_part(dataset, SUBSET, model_components))

def base64_local_image(image_path) -> str:
    with Image.open(image_path) as img:
        # Convert the image to a bytes buffer
        buffer = BytesIO()
        img.save(buffer, format="JPEG") # You can change the format to JPG or others if needed
        buffer.seek(0)
        # Convert the image data to base64
        image_data_str = base64.b64encode(buffer.read()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{image_data_str}"

    return data_url

def prepare_image_buffer():
    image_path = "/ssd1/nihaljog/4096x4096.jpg"
    base64_image = base64_local_image(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": f"You are a useful assistant and a large language model.",
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the vibe of this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 10,
        "temperature": TEMPERATURE
    }

    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
    return response.json()

def main():
    prepare_image_buffer()
    run_and_save()


if __name__ == '__main__':  
    main()
