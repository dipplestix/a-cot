# RL Controller for Chain-of-Thought Reasoning

This project implements a reinforcement learning (RL) controller for optimizing when to terminate chain-of-thought (CoT) reasoning. The controller learns to decide whether to continue generating CoT or force an answer based on the current state of reasoning.

## Overview

The RL controller works as follows:

1. Takes the current state of CoT reasoning as input
2. Uses ModernBERT to embed the CoT into a fixed-size representation
3. Makes a decision: continue thinking for 10 more tokens or force an answer
4. Receives rewards based on the outcome:
   - +1000 if it forces an answer and gets it correct
   - -10 if it chooses to continue thinking
   - -100 if it forces an answer and gets it incorrect

The controller learns an optimal policy through reinforcement learning, using policy gradient methods and experience replay.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the RL controller on the GSM8K dataset:

```bash
python rl_cot_controller.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --bert_model "roberta-base" --max_samples 10
```

### Command Line Arguments

- `--model`: LLM model to use for generating CoT and answers
- `--bert_model`: BERT model to use for state embeddings (default: "roberta-base")
- `--max_samples`: Maximum number of samples to process (default: 10)
- `--output_dir`: Directory to save results (default: "rl_cot_results")
- `--split`: Dataset split to use, either "train" or "test" (default: "test")
- `--epochs`: Number of epochs to train (default: 1)
- `--batch_size`: Batch size for policy updates (default: 32)
- `--learning_rate`: Learning rate for the policy optimizer (default: 1e-4)
- `--gamma`: Discount factor for future rewards (default: 0.99)
- `--epsilon`: Exploration rate for epsilon-greedy policy (default: 0.1)
- `--max_steps`: Maximum steps per sample before forcing an answer (default: 50)
- `--continue_tokens`: Number of tokens to generate when continuing (default: 10)
- `--quantize`: Use 4-bit quantization for reduced memory usage
- `--verbose`: Print detailed information during processing

## Output

The controller saves results to the specified output directory, including:

- JSONL files with per-example results
- JSON summary of the experiment
- Saved policy model (PyTorch state dict)

## Example

```bash
# Run with 100 samples, 3 epochs
python rl_cot_controller.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --max_samples 100 --epochs 3 --verbose

# Use 4-bit quantization for reduced memory usage
python rl_cot_controller.py --model "meta-llama/Llama-2-7b-hf" --quantize --max_samples 50

# Specify output directory
python rl_cot_controller.py --model "microsoft/Phi-2" --output_dir "phi2_rl_results"
```

## Visualization

To visualize the training progress and results, you can use TensorBoard:

```bash
tensorboard --logdir rl_cot_results
``` 