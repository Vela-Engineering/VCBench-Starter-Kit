# VCBench-Starter-Kit

A minimal toolkit for evaluating LLMs on venture capital founder success prediction using the VCBench dataset.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your OpenAI API key:
   ```bash
   cp .env.sample .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. Run the sample test:
   ```bash
   python openai_testing_sample.py
   ```

## Key Files

### Core Implementation
- **`openai_testing_sample.py`** - Main script that runs LLM predictions on founder data using multiprocessing
- **`evaluation.py`** - Evaluates model predictions and calculates metrics (precision, accuracy, recall, F0.5)

### Data
- **`vcbench_final_public_sample100.csv`** - Sample dataset (100 founders) for testing
- **`vcbench_final_public.csv`** - Full VCBench dataset

### LLM Integration
- **`llms/`** - LLM provider implementations
  - **`llms/__init__.py`** - Main interface with `get_llm_provider()` function
  - **`llms/openai/_openai.py`** - Minimal OpenAI provider implementation

### Configuration
- **`core/config.py`** - Settings management using pydantic-settings
- **`.env`** - Environment variables (API keys, etc.)
- **`requirements.txt`** - Python dependencies

### Output
- **`vanilla_llm_testing_results/`** - Generated prediction results (gitignored)

## Usage

The toolkit processes founder descriptions and predicts success using the prompt:
- **Success definition**: Companies with >$500M funding or exit/IPO >$500M
- **Input**: Anonymized founder LinkedIn/Crunchbase profiles  
- **Output**: JSON with prediction ("Yes"/"No") and reasoning

Results are saved as CSV files with predictions that can be evaluated using `evaluation.py`.