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

### Answer Extraction and Verification

We've implemented scripts to:
- Extract numerical answers from model responses
- Compare with reference answers
- Analyze the relationship between reasoning length and accuracy

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