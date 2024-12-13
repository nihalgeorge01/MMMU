# MMMU-Pro

## Overview

This folder contains inference scripts for the [MMMU-Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) dataset. 
0. `infer_mlc.py`: For model inference on MLC serving API
1. `infer_xxx.py`: For other model inference
2. `evaluate.py`: For evaluating inference results

Make sure to configure the necessary model and data files before use.

## Script Descriptions
### 0. Special Instructions for `infer_mlc.py`
First setup the serving. Docs of MLC-LLM are useful. An example command is below, but the paths, port and other options can be changed to your requirement
```
$ mlc_llm serve ./dist/Phi-3.5-vision-instruct-q4f32_1-MLC/ --model-lib ./dist/libs/Phi-3.5-vision-instruct-q4f32_1-cuda.so --port 3333
```
Next, run the inference. Currently supports MMMU and MMMU-Pro. Specify 'all' to run all subsets (only for MMMU). Otherwise, the options are 'standard' and 'vision' for MMMU-Pro and 30 subsets for MMMU like Accounting, Math, etc.
```
$ python infer/infer_mlc.py "Phi-3.5-vision-instruct-q4f32_1-MLC" http://127.0.0.1:3333/v1 direct MMMU/MMMU all
```
Outputs will be stored in ./output/ . It will be a file with a list of JSONs, 1 JSON (= 1 example) per line  
Currently MLC serving may hit OOM and not recover. In that case, kill both commands, delete rows in output file containing the error, and rerun both of the previous commands
### 1. Model Inference Script: `infer_xxx.py`

This script loads a specified model and performs inference. To run the script, use the following steps:

```bash
cd mmmu-pro
python infer/infer_xxx.py [MODEL_NAME] [MODE] [SETTING]
```

- **`[MODEL_NAME]`**: Specify the model's name (e.g., `gpt-4o`). Ensure the corresponding model files are available in the required directory.
- **`[MODE]`**: Choose the prompt mode:
  - `cot` (Chain of Thought): The model processes the problem step-by-step.
  - `direct`: The model directly provides the answer.
- **`[SETTING]`**: Select the inference task setting:
  - `standard(10 options)`: Uses the standard format of augmented MMMU with ten options.
  - `standard(4 options)`: Uses the standard format of augmented MMMU with four options.
  - `vision`: Uses a screenshot or photo form of augmented MMMU.

**Example**:

```bash
python infer/infer_gpt.py gpt-4o cot vision
```

This example runs the `gpt-4o` model in chain-of-thought (`cot`) mode using the `vision` setting of augmented MMMU. The inference results will be saved to the `./output` directory.

### 2. Evaluation Script: `evaluate.py`

This script evaluates the results generated from the inference step. To run the evaluation, use the following command:

```bash
cd mmmu-pro
python evaluate.py
```

Once executed, the script will:
- Load the inference results from the `./output` directory.
- Generate and display the evaluation report in the console.
- Save the evaluation report to the `./output` directory.

## Additional Information

- Make sure the model and data files are properly configured before running the scripts.
- To adjust parameters, edit the relevant sections in the script files as needed.
