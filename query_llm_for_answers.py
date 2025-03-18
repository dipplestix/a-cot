#!/usr/bin/env python3
"""
Script to extract numerical answers from model responses using an external model.
Uses the OpenAI client format to communicate with Ollama.
"""
import json
import jsonlines
import argparse
import re
import time
from openai import OpenAI
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Extract numerical answers from model responses')
    parser.add_argument('--input_jsonl', type=str, default='gsm8k_results/results_deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B_test.jsonl',
                        help='Path to input JSONL file containing GSM8K problems and model responses')
    parser.add_argument('--output_csv', type=str, default='gsm8k_results/extracted_answers.csv',
                        help='Path to output CSV file for extracted answers')
    parser.add_argument('--ollama_url', type=str, default='http://localhost:11434/v1',
                        help='URL for Ollama API')
    parser.add_argument('--model', type=str, default='llama3.2',
                        help='Model name in Ollama')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    return parser.parse_args()

def extract_reference_answer(answer_text):
    """Extract the numerical answer from the reference answer."""
    # Look for the final part after #### which contains just the answer
    if not answer_text:
        return None
        
    final_match = re.search(r'####\s*(.+?)$', answer_text)
    if final_match:
        return final_match.group(1).strip()
    
    return None

def normalize_answer(answer):
    """Normalize the answer for comparison (remove $, etc.)"""
    if answer is None:
        return None
        
    # Convert to string
    answer = str(answer)
    
    # Strip any leading/trailing whitespace and common punctuation
    answer = answer.strip().rstrip('.,:;')
    
    # Handle percentage answers
    percentage_match = re.search(r'(\d+)\\?%', answer)
    if percentage_match:
        return percentage_match.group(1)
    
    # Handle dollar amounts with commas and decimal points
    dollar_match = re.search(r'\$?([\d,]+(?:\.\d+)?)', answer)
    if dollar_match:
        # Remove commas from number
        return dollar_match.group(1).replace(',', '').replace('!', '')
    
    # Remove currency symbols, commas and other non-numeric characters except decimal points
    normalized = re.sub(r'[^\d\.\-]', '', answer)
    
    return normalized

def extract_answer(client, model_response, model_name="llama3.2"):
    """Extract the numerical answer from a model's response."""
    try:
        prompt = f"""Below is a solution to a math problem. Extract ONLY the final numerical answer with no explanation, no units, no dollar signs, and no other text.

Problem solution:
{model_response}

Final numerical answer:"""
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts numerical answers from math problem solutions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # Extract the answer from the response
        answer_text = response.choices[0].message.content.strip()
        
        # Try to extract just a number from the response
        numeric_match = re.search(r'^[-+]?[0-9]*\.?[0-9]+$', answer_text)
        if numeric_match:
            # If it's already just a number, return it
            return answer_text
        else:
            # Otherwise try to extract a number from the text
            number_match = re.search(r'[-+]?[0-9]*\.?[0-9]+', answer_text)
            if number_match:
                return number_match.group(0)
            else:
                return answer_text
            
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None

def main():
    args = parse_args()
    
    # Initialize OpenAI client with Ollama base URL
    client = OpenAI(
        base_url=args.ollama_url,
        api_key="ollama"  # Ollama doesn't need a real API key
    )
    
    results = []
    
    # Read JSONL file
    with jsonlines.open(args.input_jsonl) as reader:
        problems = list(reader)
        
    # Limit number of samples if specified
    if args.max_samples:
        problems = problems[:min(args.max_samples, len(problems))]
    
    print(f"Processing {len(problems)} problems...")
    
    for idx, item in enumerate(problems):
        print(f"Processing problem {idx+1}/{len(problems)}...")
        
        # Extract question and reference answer
        question = item.get("question", "")
        reference_text = item.get("reference_answer", "")
        reference_answer = extract_reference_answer(reference_text)
        model_response = item.get("model_response", "")
        
        # Extract the answer from the model response
        extracted_answer = extract_answer(client, model_response, args.model)
        
        # Normalize answers for comparison
        normalized_extracted = normalize_answer(extracted_answer)
        normalized_reference = normalize_answer(reference_answer)
        
        # Check if the answer is correct
        correct = normalized_extracted == normalized_reference if normalized_extracted and normalized_reference else False
        
        # Store result
        result = {
            "id": item.get("id", idx),
            "question": question,
            "model_response": model_response[:200] + "..." if len(model_response) > 200 else model_response,
            "extracted_answer": extracted_answer,
            "reference_answer": reference_answer,
            "normalized_extracted": normalized_extracted, 
            "normalized_reference": normalized_reference,
            "correct": correct
        }
        
        results.append(result)
        
        # Small delay to avoid overwhelming Ollama
        time.sleep(0.5)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate accuracy
    extraction_accuracy = df["correct"].mean()
    correct_count = df["correct"].sum()
    total_count = len(df)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Total examples: {total_count}")
    print(f"Extracted correct answers: {correct_count}")
    print(f"Extraction accuracy: {extraction_accuracy:.2%}")
    
    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    
    # Also save a summary file
    summary_file = Path(args.output_csv).with_suffix('.json')
    summary = {
        "extraction_model": args.model,
        "total_examples": int(total_count),
        "extraction_accuracy": float(extraction_accuracy),
        "extracted_correct_answers": int(correct_count),
        "problems": [{
            "id": int(row["id"]) if not pd.isna(row["id"]) else i,
            "question": str(row["question"]),
            "extracted_answer": str(row["extracted_answer"]) if not pd.isna(row["extracted_answer"]) else "",
            "reference_answer": str(row["reference_answer"]) if not pd.isna(row["reference_answer"]) else "",
            "correct": bool(row["correct"])
        } for i, row in df.iterrows()]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main() 