# JMLE Gemini Dataset Generator

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

## scripts/jmle_gemini_cot_generator.py

### Usage

```bash
# Use "system prompt + user prompt" method
uv run scripts/jmle_gemini_cot_generator.py --use_system_prompt
```

### Generated Data Format

Each sample includes the following fields:

- `id`: Problem ID from the original dataset
- `question`: The question text
- `choices`: List of choices
- `answer`: The correct answer
- `cot`: AI-generated chain of thought process
- `explanation`: AI-generated explanation of the answer
- `generation_info`: Metadata about the generation (success/failure, number of attempts, prompt method used, etc.)

### Notes

- Access to the Google Gemini API requires a valid API key
- Uploading to Hugging Face requires a valid token
- The `unmatched` split is not created if there is no failed data
- For processing large datasets, using the `--save_interim` option is recommended

## scripts/jmle_gemini_cot_generator.py

```bash
uv run scripts/jmle_gemini_dpo_generator.py \
    --sft_dataset_id "doctorin/JMLE-CoT-gemini-2.5-pro-dataset-combined" \
    --output_dir "JMLE-DPO-gemini-2.5-pro-dataset" \
    --save_interim \
    --interim_interval 100 \
    --upload_to_hub \
    --hub_repo_id "JMLE-DPO-gemini-2.5-pro-dataset" \
    --hub_private \
    --use_system_prompt_for_sft
```