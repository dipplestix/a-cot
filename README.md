# Adaptive Chain of Thought (A-CoT)

This project aims to optimize the length of reasoning traces in Large Reasoning Models (LRMs) using reinforcement learning techniques, without sacrificing performance.

## Project Overview

Large Reasoning Models (LRMs), such as OpenAI's o1/o3 and DeepSeek's R1, utilize a reasoning phase before producing solutions, which can be considered a form of automated chain of thought (CoT). However, these models may "overthink" and generate more reasoning tokens than necessary, increasing cost and solution time.

Adaptive Chain of Thought (A-CoT) addresses this issue by dynamically determining the optimal length of reasoning traces while maintaining problem-solving capabilities.

## Proposed Technique

Our approach uses a reinforcement learning controller that:

1. Examines the input prompt and current reasoning trace
2. Decides whether to stop reasoning (by inserting a `</think>` token) or continue for n more tokens
3. Optimizes based on a reward signal that considers both answer accuracy and reasoning trace length

## Experiment Pipeline

For each model size (small 1.5B, medium 14B, large 70B), our testing process includes:

1. Running each question through the LRM and saving the response
2. Using the RL controller to decide when to cut off the reasoning trace
3. Querying the LRM with the truncated reasoning trace
4. Comparing the results with full-length reasoning

We will also compare our approach against fixed-length token restrictions.

## Datasets

We're evaluating performance using one or more of these datasets:

- AIME - American Invitational Mathematics Examination problems
- MMLU - Multi-task benchmark requiring strong problem-solving ability
- GPQA - Benchmark for general knowledge and common-sense questions
- MATH - Collection of challenging competition mathematics problems
- GSM8K - Grade school math word problems created by human problem writers

## Current Implementation

This repository currently contains tools for:

- Running LRMs on math reasoning datasets while tracking token-level information
- Analyzing reasoning patterns and token usage
- Evaluating answer accuracy with and without optimized reasoning traces

### Running the Evaluation

```bash
python run_llm_on_gsm8k.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --temperature 0.6
```

### Forcing Explicit Thinking and Answer Generation

To test our approach of separating thinking from answering, we use the `run_with_forced_answer.py` script:

```bash
python run_with_forced_answer.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --max_samples 10 \
  --verbose
```

This script:
- Prompts the model to generate a thinking trace enclosed in `<think>...</think>` tags
- Explicitly stops generation when `</think>` is detected (or after a maximum token count)
- Provides the thinking trace back to the model with a prompt to generate just the final answer
- Records detailed statistics on token usage for both phases

Key options:
- `--verbose`: Displays detailed output including prompts and model responses
- `--max_samples`: Limits the number of problems to process
- `--quantize`: Enables 4-bit quantization for reduced memory usage

### Answer Extraction and Verification

After generating model responses, we extract and evaluate numerical answers using `query_llm_for_answers.py`:

```bash
python query_llm_for_answers.py \
  --input_jsonl forced_answer_results/results_deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B_test.jsonl \
  --model llama3.2 \
  --verbose
```

This script:
- Takes model responses from a JSONL file generated by the evaluation scripts
- Uses an external LLM (default: Llama 3.2 via Ollama) to extract numerical answers
- Normalizes both extracted and reference answers for fair comparison
- Produces a CSV file and JSON summary of extraction results in the same directory as the input

The extracted answers are saved as `extracted_answers.csv` and `extracted_answers.json` in the same directory as the input JSONL file.

## Team

- Chris Mascioli
- Jeffrey Brill
- Minghao Shen

## Setup

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. For HuggingFace models:
   - Have a HuggingFace token if the model requires authentication
   - Sufficient GPU VRAM (or enable quantization with `--quantize`)

3. For answer extraction:
   - Install Ollama from https://ollama.com/
   - Pull the Llama 3.2 model: `ollama pull llama3.2`