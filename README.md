# GSM8K LLM Evaluation with Token Tracking

This project allows you to run any LLM, including open source models like Llama 3.2, on the GSM8K math reasoning dataset while tracking token-level information.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. If you plan to use Llama 3.2 or other HuggingFace models, make sure you have:
   - A HuggingFace token if the model requires authentication
   - Sufficient GPU VRAM (or enable quantization with `--quantize`)

## Running with Llama 3.2 locally

```bash
python run_llm_on_gsm8k.py \
  --model meta-llama/Meta-Llama-3.2-8B-Instruct \
  --api_key YOUR_HUGGINGFACE_TOKEN \
  --quantize \
  --max_samples 10  # Optional: limit number of samples for testing
```

## Options

- `--model`: Model identifier (default: "meta-llama/Meta-Llama-3.2-8B-Instruct")
- `--max_samples`: Maximum number of samples to process
- `--output_dir`: Directory to save results (default: "gsm8k_results")
- `--split`: Dataset split to use: train or test (default: test)
- `--system_prompt`: System prompt for the LLM
- `--temperature`: Temperature for sampling (default: 0)
- `--api_key`: HuggingFace API token for downloading the model
- `--quantize`: Use 4-bit quantization for reduced memory usage
- `--use_openai`: Use OpenAI API instead of local model
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 1024)

## Output Format

For each GSM8K problem, the script tracks:
- Input prompt and tokens
- Generated response and tokens 
- Time taken for generation
- Token counts

Results are saved in JSONL format in the output directory, with a summary JSON file containing aggregate statistics.

## Alternative Models

You can use any model supported by the Transformers library:

```bash
python run_llm_on_gsm8k.py --model mistralai/Mistral-7B-Instruct-v0.2
```

## OpenAI API (Alternative)

You can also use the OpenAI API by adding the `--use_openai` flag:

```bash
export OPENAI_API_KEY=your_api_key
python run_llm_on_gsm8k.py --model gpt-4o --use_openai
```