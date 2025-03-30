import os
import re
import json
import jsonlines
import time
import argparse
from typing import Iterable
import pandas as pd
from pathlib import Path
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
    parser.add_argument("--output_dir", type=str, default="truncated_answer_results",
                        help="Directory to save results (default: truncated_answer_results)")
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
    parser.add_argument("--full_saved_token_file", type=str, default="save_tokens/results_deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B_test.jsonl",
                        help="File containing full chain of thought prompts (default: save_tokens/results_deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B_test.jsonl)")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of the input sequence (default: )")
    return parser.parse_args()

def format_prompt(question):
    """Format the prompt for the LLM."""
    #return #f"""Do not include units or any other characters except the number corresponding to the answer. For example, if the answer is 10, you should output "FINAL ANSWER: 10" and not "FINAL ANSWER: 10 dollars" or "FINAL ANSWER: $10.
    return f"""{question}
    Let's think step by step:
    """

def truncate_cot(llm, tokenizer, saved_prompts, max_length, special_tokens, args):
    # append max_length to output
    output_dir = Path(args.output_dir + f"/maxlen_{max_length}")
    output_dir.mkdir(exist_ok=True, parents=True)
    progress_bar = tqdm(saved_prompts["thinking_trace"], total=len(saved_prompts), desc="Processing samples", unit="sample")
    results = []
    model_name_safe = args.model.replace('/', '_').replace('-', '_')
    results_file = output_dir / f"results_{model_name_safe}_{args.split}.jsonl"

    for idx, thinking_trace in enumerate(progress_bar):
        # tokenize the input, remove the <bos_token>, then truncate the input to max_length
        input_ids = tokenizer.encode(thinking_trace, add_special_tokens=False)
        input_ids = input_ids[:max_length]
        
        if special_tokens["</think>"] in input_ids:
            input_ids = input_ids[ : input_ids.index(special_tokens["</think>"])]
        
        truncated_thinking_trace = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Use the truncated model to get the final answer
        # Phase 2: Ask for just the final answer
        question = saved_prompts.iloc[idx]["question"]
        second_prompt = f"{format_prompt(question)}\n<think>{truncated_thinking_trace}</think>\n\n My final answer with no further justiciation, explanation or examples is:"

        if args.verbose:
            print("\n" + "="*80)
            print("PHASE 2 PROMPT:")
            print("-"*80)
            print(second_prompt[-200:]) # Only show the last part to avoid flooding the console
            print("..." + "(truncated for display)")
            print("="*80 + "\n")

        answer_sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens - len(input_ids),
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
        
    
        result = {
            "id": idx,
            "question": question,
            "reference_answer": saved_prompts.iloc[idx]["reference_answer"],
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "thinking_trace": f"<think>{truncated_thinking_trace}</think>",
            "model_answer": answer_text,
            "thinking_token_count": len(input_ids),
            "answer_token_count": len(answer_output.outputs[0].token_ids),
            # "generation_time": end_time - start_time
        }
        results.append(result)

        with jsonlines.open(results_file, mode="a") as writer:
                writer.write(result)
    
    # Save summary
    summary = {
        "model": args.model,
        "split": args.split,
        "samples_processed": len(results),
        "total_thinking_tokens": sum(r["thinking_token_count"] for r in results),
        "total_answer_tokens": sum(r["answer_token_count"] for r in results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_dir / f"summary_{model_name_safe}_{args.split}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    if args.verbose:
        print(f"Processed {len(results)} samples. Results saved to {output_dir}")


def run_truncate_cot(args):
    file_name = args.full_saved_token_file
    full_saved_prompts = pd.read_json(file_name, lines=True)

    # Initialize model
    kwargs = {
        "trust_remote_code": True,
        "download_dir": None,
    }
    
    if args.quantize:
        kwargs["quantization"] = "awq"
    llm = LLM(model=args.model, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Using model: {args.model}")

    special_tokens = tokenizer.special_tokens_map
    special_tokens["<think>"] = tokenizer.convert_tokens_to_ids("<think>")
    special_tokens["</think>"] = tokenizer.convert_tokens_to_ids("</think>")

    max_length = args.max_length
    truncate_cot(llm, tokenizer, full_saved_prompts, max_length, special_tokens, args)
    

def main():
    args = parse_args()
    run_truncate_cot(args)

if __name__ == "__main__":
    main()