# VCBench-Starter-Kit

A minimal toolkit for evaluating LLMs on venture capital founder success prediction using the VCBench dataset.

## Objective of the Task

Participants will develop LLM-based systems to predict founder success in venture capital. The task requires analyzing anonymized founder profiles and determining whether their startup will achieve major success.

Success is defined as the founder's most recent startup achieving an IPO/acquisition above $500M or raising $500M+. Startups that raised $100K–$4M at founding but failed to reach a major outcome in eight years are marked unsuccessful.

Participants must design effective prompts and utilize the anonymized prose format to make binary predictions ("Yes"/"No"), optimizing for F0.5 primarily and precision secondarily.

## Dataset

**VCBench Dataset**
9,000 anonymized founder–startup profiles with a 9% success rate. Profiles include founder background (education, jobs, prior IPO/acquisitions) and startup details (industry, outcomes). Data comes from LinkedIn, Crunchbase, and licensed sources, restricted to what was known before founding.

**Training Dataset**
Public training set: 4,500 founders (available in this repository as `vcbench_final_public.csv`)

**Testing Dataset** 
Private test set: 4,500 founders (3 folds) - held by Vela Research for leaderboard evaluation

**Data Format**
- Anonymized prose (natural-language summaries optimized for direct LLM use)
- Structured JSON (for feature-based ML models)

**Evaluation**
Metrics: precision, recall, and F0.5, averaged across folds. Private labels are never released to prevent data leakage into LLM training corpora.
The primary evaluation metric adopted is F0.5; precision is treated as the secondary metric.


## What Participants Should Submit

- Complete code for running LLM predictions
- Prediction results on the training set demonstrating model performance
- An evaluation script that processes results and computes metrics
- Clear documentation of prompt engineering and methodology

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