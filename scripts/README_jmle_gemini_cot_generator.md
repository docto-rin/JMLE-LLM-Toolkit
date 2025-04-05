# JMLE Gemini CoT Dataset Generator

A tool for generating Chain-of-Thought (CoT) datasets for Japanese Medical Licensing Examination (JMLE) using generative AI. This tool leverages the Google Gemini-2.5-Pro API to generate detailed reasoning processes and answers from medical exam questions.

## Features

- AI-driven thought process expansion based on medical examination questions
- Generates structured datasets containing problem IDs, questions, choices, and answers
- Automatically separates successful and failed generations into distinct datasets
- Supports both system prompt and user prompt methodologies
- Tracks and monitors samples and progress during generation
- Automatic uploading to Hugging Face

## Prerequisites

- Python 3.8 or higher
- Required packages: dotenv, huggingface_hub, datasets, openai, py2slack, pandas, tqdm
- Google Gemini API key (set as environment variable `GEMINI_API_KEY`)
- Hugging Face API token (required for uploading, set as environment variable `HF_TOKEN`)

## Usage

This script can be used as follows:

```bash
# Basic usage (specify only sample size, run with default settings)
uv run scripts/jmle_gemini_cot_generator.py --sample_size 5

# Use "system prompt + user prompt" method
uv run scripts/jmle_gemini_cot_generator.py --sample_size 5 --use_system_prompt

# Example specifying all settings via command line arguments
uv run scripts/jmle_gemini_cot_generator.py \
    --dataset longisland3/NMLE \
    --sample_size 500 \
    --output_dir my_jmle_cot_dataset \
    --save_interim \
    --interim_interval 50 \
    --upload_to_hub \
    --hub_repo username/JMLE-CoT-Complete \
    --private_repo \
    --notify_slack \
    --use_system_prompt
```

## Prompt Methodology Selection

This tool supports two prompt methodologies:

1. **Unified Prompt Method** (default)
   - Integrates all instructions and problem content into a single user prompt
   - Simpler implementation that works with most LLM models

2. **System Prompt + User Prompt Method**
   - Places role and output format instructions in the system prompt
   - Places only problem content in the user prompt
   - More structured approach that may improve performance with some models

Add the `--use_system_prompt` flag to use the second method.

## Command Line Arguments

| Argument | Description | Default Value |
|----------|-------------|---------------|
| `--dataset` | Hugging Face dataset path to use | longisland3/NMLE |
| `--sample_size` | Number of samples to process (0=all) | 0 |
| `--output_dir` | Local directory for saving output | JMLE-CoT-gemini-2.5-pro-dataset |
| `--save_interim` | Whether to save interim results | False |
| `--interim_interval` | Interval for saving interim results (number of samples) | 100 |
| `--upload_to_hub` | Whether to upload to Hugging Face | True |
| `--hub_repo` | Repository name for upload | doctorin/JMLE-CoT-gemini-2.5-pro-dataset |
| `--private_repo` | Whether to make the repository private | True |
| `--notify_slack` | Whether to send Slack notifications | False |
| `--use_system_prompt` | Use system prompt + user prompt method | False |

## Output Format

This tool generates the following outputs:

1. **Datasets Format Dataset**
   - `train` split: Successfully generated samples
   - `unmatched` split: (if exists) Failed generated samples
   
2. **CSV Files**
   - `successful_samples.csv`: Successfully generated samples
   - `failed_samples.csv`: Failed generated samples
   
3. **Interim Results** (if `--save_interim` is specified)
   - Saves intermediate results in CSV format at specified intervals

## Generated Data Format

Each sample includes the following fields:

- `id`: Problem ID from the original dataset
- `question`: The question text
- `choices`: List of choices
- `answer`: The correct answer
- `cot`: AI-generated chain of thought process
- `explanation`: AI-generated explanation of the answer
- `generation_info`: Metadata about the generation (success/failure, number of attempts, prompt method used, etc.)

## Notes

- Access to the Google Gemini API requires a valid API key
- Uploading to Hugging Face requires a valid token
- The `unmatched` split is not created if there is no failed data
- For processing large datasets, using the `--save_interim` option is recommended