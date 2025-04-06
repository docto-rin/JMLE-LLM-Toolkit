# Med-LLM-Jp

This repository is related to the development of Large Language Models (LLMs) specialized for the Japanese medical domain. It primarily focuses on processing and generating datasets from the Japanese Medical Licensing Examination (JMLE).

## Overview

This repository includes scripts for generating Chain-of-Thought (CoT) style datasets from JMLE data, as well as Jupyter Notebooks for correcting the generated datasets.

## Key Features & Contents

*   **JMLE CoT Dataset Generation:** Uses the Google Gemini API (gemini-2.5-pro) to generate thought processes (CoT) and explanations from JMLE questions (`scripts/jmle_gemini_cot_generator.py`).
    *   Separates successfully and unsuccessfully generated samples into distinct outputs.
    *   Includes functionality for automatic uploading to the Hugging Face Hub.
*   **Dataset Correction:** A Colab Notebook (`colab_notebooks/JMLE-Gemini-2.5-Pro-CoT-Dataset-Correction.ipynb`) for manually reviewing and correcting the generated dataset, especially samples where the generated answer did not match the original (`unmatched` split).
    *   Includes functionality to merge the corrected data and upload it to the Hugging Face Hub.

## Directory Structure

```
├── .gitignore                 # Git ignore list
├── Med-LLM-Jp.code-workspace  # VS Code workspace settings
├── colab_notebooks/           # Notebooks for dataset correction/analysis
│   └── JMLE-Gemini-2.5-Pro-CoT-Dataset-Correction.ipynb
├── pyproject.toml             # Python project configuration (for uv)
├── scripts/                   # Scripts for dataset generation, etc.
│   ├── README.md              # Detailed script description
│   ├── jmle_gemini_cot_generator.py        # CoT data generation script
│   └── jmle_gemini_dpo_generator.py        # DPO data generation script
└── uv.lock                    # Dependency lock file (for uv)
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Med-LLM-Jp
    ```
2.  **Set up environment variables:**
    Create a `.env` file in the project root and provide the necessary API keys and tokens:
    ```env
    # Required API Key for generation
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY

    # Required for uploading datasets to the Hugging Face Hub
    HF_TOKEN=YOUR_HUGGINGFACE_WRITE_TOKEN

    # Optional: For Slack notifications
    # SLACK_OAUTH_TOKEN=YOUR_SLACK_OAUTH_TOKEN
    # SLACK_DEFAULT_CHANNEL=your-slack-channel
    ```
    *   You can obtain a `GEMINI_API_KEY` from Google AI Studio or your Google Cloud project.
    *   A `HF_TOKEN` with write permissions can be created in your Hugging Face account settings (`Settings` -> `Access Tokens`).
    *   For Slack notifications, the script uses [py2slack](https://github.com/docto-rin/py2slack). Refer to its documentation for setting up the `SLACK_OAUTH_TOKEN` and `SLACK_DEFAULT_CHANNEL`.
3.  **Install dependencies:**
    Use [uv](https://github.com/astral-sh/uv) to install the dependencies.
    ```bash
    uv sync
    ```

## Usage

### Generate CoT Dataset

Run the `scripts/jmle_gemini_cot_generator.py` script.
Refer to `scripts/README.md` for detailed options.

**Example Usage (specify sample size):**
```bash
uv run python scripts/jmle_gemini_cot_generator.py --sample_size 10 --use_system_prompt
```

### Correct the Dataset

Open and run `colab_notebooks/JMLE-Gemini-2.5-Pro-CoT-Dataset-Correction.ipynb` in Google Colab.
Follow the instructions within the notebook to manually correct the data where the generated answers didn't match (`unmatched` split) and upload the corrected dataset to the Hugging Face Hub.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/docto-rin/Med-LLM-Jp/blob/main/colab_notebooks/JMLE-Gemini-2.5-Pro-CoT-Dataset-Correction.ipynb)

## Datasets

The primary datasets generated and managed by this repository are (or will be) available on the Hugging Face Hub.

*   **Generated Data:** (e.g.) `doctorin/JMLE-CoT-gemini-2.5-pro-dataset`
*   **Corrected Data:** (e.g.) `doctorin/JMLE-CoT-gemini-2.5-pro-dataset-combined`

(Repository names might vary based on script arguments or notebook settings.)
