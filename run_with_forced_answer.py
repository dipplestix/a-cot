#!/usr/bin/env python3
"""
Script to run an LLM on GSM8K, force stopping at </think> token,
then prompting again for just the final answer.
"""

import os
import json
import time
import argparse
from pathlib import Path
import jsonlines
import re
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM on GSM8K with forced answer after thinking")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Model to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Maximum number of samples to process (default: 10)")
    parser.add_argument("--output_dir", type=str, default="forced_answer_results",
                        help="Directory to save results (default: forced_answer_results)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                        help="Dataset split to use (default: test)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for sampling (default: 0.6)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="HuggingFace API token for downloading the model")
    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit quantization for reduced memory usage")
    parser.add_argument("--max_new_tokens", type=int, default=16384,
                        help="Maximum number of new tokens to generate (default: 16384)")
    parser.add_argument("--max_think_tokens", type=int, default=16384,
                        help="Maximum number of thinking tokens to generate (default: 16384)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information during processing")
    return parser.parse_args()

def format_prompt(question):
    """Format the prompt for the LLM."""
    #return #f"""Do not include units or any other characters except the number corresponding to the answer. For example, if the answer is 10, you should output "FINAL ANSWER: 10" and not "FINAL ANSWER: 10 dollars" or "FINAL ANSWER: $10.
    return f"""{question}
    Let's think step by step:
    """

def run_model_with_forced_answer(llm, tokenizer, question, args):
    """Run the model by first getting the thinking trace, then asking for just the final answer."""
    # Phase 1: Get the thinking trace
    first_prompt = f"{format_prompt(question)}\n<think>"
    if args.verbose:
        print("\n" + "="*80)
        print("PHASE 1 PROMPT:")
        print("-"*80)
        print(first_prompt)
        print("="*80 + "\n")
    
    think_sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_think_tokens,
        stop=["</think>", "Human:", "Assistant:", "FINAL ANSWER:"]
    )
    
    start_time = time.time()
    think_outputs = llm.generate([first_prompt], think_sampling_params)
    think_output = think_outputs[0]
    think_text = think_output.outputs[0].text
    
    if args.verbose:
        print("\n" + "="*80)
        print("PHASE 1 THINKING OUTPUT:")
        print("-"*80)
        print(think_text)
        print("="*80 + "\n")
    
    # Check if the model actually produced a </think> token
    if "</think>" in think_text:
        # Find where the </think> token appears and truncate
        think_text = think_text[:think_text.find("</think>")]
    
    # Phase 2: Ask for just the final answer
    second_prompt = f"{format_prompt(question)}\n<think>{think_text}</think>\n\n My final answer with no further justiciation, explanation or examples is:"
    
    if args.verbose:
        print("\n" + "="*80)
        print("PHASE 2 PROMPT:")
        print("-"*80)
        print(second_prompt[-200:]) # Only show the last part to avoid flooding the console
        print("..." + "(truncated for display)")
        print("="*80 + "\n")
    
    answer_sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens - len(think_output.outputs[0].token_ids),
        stop=["Human:", "Assistant:"]
    )
    
    answer_outputs = llm.generate([second_prompt], answer_sampling_params)
    answer_output = answer_outputs[0]
    answer_text = answer_output.outputs[0].text
    
    if args.verbose:
        print("\n" + "="*80)
        print("PHASE 2 ANSWER OUTPUT:")
        print("-"*80)
        print(answer_text)
        print("="*80 + "\n")
    
    end_time = time.time()
    
    result = {
        "thinking_trace": f"<think>{think_text}</think>",
        "model_answer": answer_text,
        "prompt_token_count": len(think_output.prompt_token_ids),
        "thinking_token_count": len(think_output.outputs[0].token_ids),
        "answer_token_count": len(answer_output.outputs[0].token_ids),
        "total_token_count": len(think_output.prompt_token_ids) + len(think_output.outputs[0].token_ids) + len(answer_output.outputs[0].token_ids),
        "generation_time": end_time - start_time
    }
    
    return result


def run_forced_answer_experiment(args):
    """Run the experiment with forced answer extraction."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set HuggingFace token in environment if provided
    if args.api_key:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.api_key
    
    # Load GSM8K dataset
    print(f"Loading GSM8K {args.split} dataset...")
    dataset = load_dataset("gsm8k", "main")[args.split]
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize model
    kwargs = {
        "trust_remote_code": True,
        "download_dir": None,
    }
    
    if args.quantize:
        kwargs["quantization"] = "awq"
    
    # Initialize model and tokenizer
    llm = LLM(model=args.model, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print(f"Using model: {args.model}")
    
    results = []
    
    # Setup output files
    model_name_safe = args.model.replace('/', '_').replace('-', '_')
    results_file = output_dir / f"results_{model_name_safe}_{args.split}.jsonl"
    
    # Process each example
    progress_bar = tqdm(dataset, desc="Processing GSM8K problems", position=0, leave=True, disable=not args.verbose)
    for idx, example in enumerate(progress_bar):
        question = example["question"]
        reference_answer_text = example["answer"]
        
        try:
            # Run model with forced answer
            result = run_model_with_forced_answer(llm, tokenizer, question, args)
            
            # Add metadata
            result.update({
                "id": idx,
                "question": question,
                "reference_answer": reference_answer_text,
                "model": args.model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            
            results.append(result)
            
            # Write to JSONL file
            with jsonlines.open(results_file, mode="a") as writer:
                writer.write(result)
                
        except Exception as e:
            if args.verbose:
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
        "total_thinking_tokens": sum(r["thinking_token_count"] for r in results),
        "total_answer_tokens": sum(r["answer_token_count"] for r in results),
        "total_tokens": sum(r["total_token_count"] for r in results),
        "total_generation_time": sum(r["generation_time"] for r in results),
        "average_generation_time": sum(r["generation_time"] for r in results) / len(results) if results else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(output_dir / f"summary_{model_name_safe}_{args.split}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    if args.verbose:
        print(f"Processed {len(results)} samples. Results saved to {output_dir}")
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Total generation time: {summary['total_generation_time']:.2f} seconds")
        print(f"Average generation time per problem: {summary['average_generation_time']:.2f} seconds")

def main():
    args = parse_args()
    run_forced_answer_experiment(args)

if __name__ == "__main__":
    main() 