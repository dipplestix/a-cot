#!/usr/bin/env python3
"""
Script to run an LLM (e.g., Llama 3.2) on the GSM8K dataset and save token-level data.
Tracks both input and output tokens for analysis.
"""

import os
import json
import time
import argparse
from pathlib import Path
import jsonlines
import tiktoken
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from openai import OpenAI

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM on GSM8K dataset")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.2-8B-Instruct",
                        help="Model to use (default: meta-llama/Meta-Llama-3.2-8B-Instruct)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--output_dir", type=str, default="gsm8k_results",
                        help="Directory to save results (default: gsm8k_results)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                        help="Dataset split to use (default: test)")
    parser.add_argument("--system_prompt", type=str, 
                        default="",
                        help="System prompt (default: empty)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for sampling (default: 0.6)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="HuggingFace API token for downloading the model")
    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit quantization for reduced memory usage")
    parser.add_argument("--use_openai", action="store_true",
                        help="Use OpenAI API instead of local model")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of new tokens to generate")
    return parser.parse_args()

def format_prompt_for_model(system_prompt, user_prompt, model_name):
    """Format the prompt based on model type."""
    if "mistral" in model_name.lower():
        return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    elif "deepseek" in model_name.lower():
        # For DeepSeek models, directly using the string "<think>" works better than the character code
        return f"Human: {system_prompt}\n\n{user_prompt}\n\nAssistant: Let me solve this step by step.\n<think>"
    # For Llama models
    return f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"

def get_openai_encoding(model_name):
    """Get the appropriate encoding for the specified OpenAI model."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Warning: model '{model_name}' not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding

def load_local_model(args):
    """Load the model using vLLM."""
    # Set HuggingFace token in environment if provided
    if args.api_key:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.api_key
    
    # Get tokenizer information and special token IDs silently
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Get token IDs for special tokens without printing
    if "deepseek" in args.model.lower():
        # Store these in global variables for use in formatting
        global THINK_TOKEN, THINK_END_TOKEN
        added_tokens = tokenizer.get_added_vocab()
        THINK_TOKEN = added_tokens.get('<think>', 151648)  # Default to known token ID if not found
        THINK_END_TOKEN = added_tokens.get('</think>', 151649)  # Default to known token ID if not found
    
    # Configure vLLM parameters
    kwargs = {
        "trust_remote_code": True,
        "download_dir": None,
    }
    
    if args.quantize:
        kwargs["quantization"] = "awq"  # or "squeezellm" depending on model support
    
    # Initialize vLLM model
    llm = LLM(
        model=args.model,
        **kwargs
    )
    
    return llm

def format_prompt(question):
    """Format the prompt for the LLM."""
    return f"""Do not include units or any other characters except the number corresponding to the answer. For example, if the answer is 10, you should output "FINAL ANSWER: 10" and not "FINAL ANSWER: 10 dollars" or "FINAL ANSWER: $10.
    {question}
    Let's think step by step:



"
"""

def run_local_llm(llm, system_prompt, user_prompt, args):
    """Run local LLM inference using vLLM."""
    # Format the base prompt
    if "deepseek" in args.model.lower():
        # For DeepSeek, add the <think> token at the end
        full_prompt = f"Human: {system_prompt}\n\n{user_prompt}\n\nAssistant: Let me solve this step by step.\n<think>"
    else:
        full_prompt = format_prompt_for_model(system_prompt, user_prompt, args.model)
    
    # Configure sampling parameters
    stop_tokens = ["</s>", "[/INST]", "<|im_end|>"]  # Default stop tokens
    if "deepseek" in args.model.lower():
        stop_tokens = ["Human:", "Assistant:"]  # Don't stop at </think>
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=stop_tokens
    )
    
    # Generate text
    start_time = time.time()
    outputs = llm.generate([full_prompt], sampling_params)
    end_time = time.time()
    
    # Extract response and token counts
    output = outputs[0]
    generated_text = output.outputs[0].text
    generated_token_ids = output.outputs[0].token_ids
    
    # For DeepSeek models, look for token 151649 (</think>) in the generated tokens
    model_thought = None
    model_response = generated_text
    
    if "deepseek" in args.model.lower():
        # Import tokenizer for decoding tokens to text
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Check if the </think> token (151649) is in the output
        think_tokens = []
        response_tokens = []
        
        try:
            if 151649 in generated_token_ids:
                # Find the index of the </think> token
                split_idx = generated_token_ids.index(151649)
                
                # Get the tokens before and after the </think> token
                thought_tokens = generated_token_ids[:split_idx]
                response_tokens = generated_token_ids[split_idx+1:] if split_idx+1 < len(generated_token_ids) else []
                
                # Store token lists for metrics
                think_tokens = thought_tokens
                
                # Convert back to text
                model_thought = "<think>" + tokenizer.decode(thought_tokens) + "</think>"
                model_response = tokenizer.decode(response_tokens)
            else:
                # If </think> token not found, assume it's all thinking
                think_tokens = generated_token_ids
                model_thought = "<think>" + generated_text + "</think>"
                model_response = ""
        except Exception as e:
            think_tokens = generated_token_ids
            model_thought = "<think>" + generated_text + "</think>"
            model_response = ""
    
    result = {
        "model_thought": model_thought,
        "model_response": model_response,
        "prompt_token_count": len(output.prompt_token_ids),
        "completion_token_count": len(output.outputs[0].token_ids),
        "think_token_count": len(think_tokens) if "deepseek" in args.model.lower() else 0,
        "response_token_count": len(response_tokens) if "deepseek" in args.model.lower() else len(output.outputs[0].token_ids),
        "total_token_count": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
        "input_tokens": output.prompt_token_ids,
        "think_tokens": think_tokens if "deepseek" in args.model.lower() else [],
        "response_tokens": response_tokens if "deepseek" in args.model.lower() else output.outputs[0].token_ids,
        "generation_time": end_time - start_time
    }
    
    return result

def run_openai_llm(client, encoding, system_prompt, user_prompt, args):
    """Run OpenAI API and track tokens."""
    # Count tokens for the prompt
    prompt_token_count, prompt_tokens = len(encoding.encode(user_prompt)), encoding.encode(user_prompt)
    system_token_count, system_tokens = len(encoding.encode(system_prompt)), encoding.encode(system_prompt)
    
    # Call the API
    start_time = time.time()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    end_time = time.time()
    
    # Extract response content and token usage
    model_response = response.choices[0].message.content
    
    # Get token counts from the response
    completion_token_count = response.usage.completion_tokens
    total_token_count = response.usage.total_tokens
    
    # Count output tokens
    output_tokens = encoding.encode(model_response)
    
    result = {
        "model_thought": None,  # OpenAI API doesn't expose thinking
        "model_response": model_response,
        "prompt_token_count": prompt_token_count,
        "system_token_count": system_token_count,
        "completion_token_count": completion_token_count,
        "think_token_count": 0,  # OpenAI API doesn't expose thinking
        "response_token_count": completion_token_count,
        "total_token_count": total_token_count,
        "input_tokens": system_tokens + prompt_tokens,
        "think_tokens": [],  # OpenAI API doesn't expose thinking
        "response_tokens": output_tokens,
        "generation_time": end_time - start_time
    }
    
    return result

def run_llm_on_gsm8k(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load GSM8K dataset
    print(f"Loading GSM8K {args.split} dataset...")
    dataset = load_dataset("gsm8k", "main")[args.split]
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples")
    
    if args.use_openai:
        # Initialize OpenAI client
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided either through --api_key argument or OPENAI_API_KEY environment variable")
        
        client = OpenAI(api_key=api_key)
        encoding = get_openai_encoding(args.model)
        llm = None
        print(f"Using OpenAI API with model: {args.model}")
    else:
        # Initialize local model with vLLM
        llm = load_local_model(args)
        client, encoding = None, None
        print(f"Using local model: {args.model}")
    
    results = []
    
    # Setup output files
    model_name_safe = args.model.replace('/', '_').replace('-', '_')
    results_file = output_dir / f"results_{model_name_safe}_{args.split}.jsonl"
    
    # Configure tqdm to avoid nested progress bars
    tqdm.pandas(desc="Processing GSM8K problems", position=0, leave=True)
    
    # Process each example
    for idx, example in enumerate(tqdm(dataset, desc="Processing GSM8K problems", position=0, leave=True)):
        question = example["question"]
        answer = example["answer"]
        
        # Format the prompt
        user_prompt = format_prompt(question)
        
        # Run inference with appropriate model
        try:
            if args.use_openai:
                result = run_openai_llm(client, encoding, args.system_prompt, user_prompt, args)
            else:
                result = run_local_llm(llm, args.system_prompt, user_prompt, args)
            
            # Add metadata to result
            result.update({
                "id": idx,
                "question": question,
                "reference_answer": answer,
                "model": args.model,
                "system_prompt": args.system_prompt,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            
            results.append(result)
            
            # Write to JSONL file
            with jsonlines.open(results_file, mode="a") as writer:
                writer.write(result)
                
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            # Write error to file
            with jsonlines.open(output_dir / "errors.jsonl", mode="a") as writer:
                writer.write({
                    "id": idx,
                    "question": question,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            time.sleep(1)  # Delay after an error
    
    # Save summary
    summary = {
        "model": args.model,
        "split": args.split,
        "samples_processed": len(results),
        "total_prompt_tokens": sum(r["prompt_token_count"] for r in results),
        "total_completion_tokens": sum(r["completion_token_count"] for r in results),
        "total_think_tokens": sum(r["think_token_count"] for r in results),
        "total_response_tokens": sum(r["response_token_count"] for r in results),
        "think_response_ratio": sum(r["think_token_count"] for r in results) / sum(r["response_token_count"] for r in results) if sum(r["response_token_count"] for r in results) > 0 else 0,
        "total_tokens": sum(r["total_token_count"] for r in results),
        "total_generation_time": sum(r["generation_time"] for r in results),
        "average_generation_time": sum(r["generation_time"] for r in results) / len(results) if results else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if args.use_openai:
        summary["total_system_tokens"] = sum(r["system_token_count"] for r in results)
    
    with open(output_dir / f"summary_{model_name_safe}_{args.split}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processed {len(results)} samples. Results saved to {output_dir}")
    print(f"Total tokens used: {summary['total_tokens']}")
    print(f"Total generation time: {summary['total_generation_time']:.2f} seconds")
    print(f"Average generation time per problem: {summary['average_generation_time']:.2f} seconds")

if __name__ == "__main__":
    args = parse_args()
    run_llm_on_gsm8k(args) 