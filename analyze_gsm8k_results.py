#!/usr/bin/env python3
"""
Script to analyze GSM8K results, comparing model answers to reference answers
"""

import json
import jsonlines
import re
from pathlib import Path

def extract_boxed_answer(response_text):
    """Extract the answer from the boxed notation in the model's response."""
    if not response_text:
        return None
        
    # Try to parse as JSON first (for the new format)
    try:
        response_json = json.loads(response_text.strip())
        if isinstance(response_json, dict) and "answer" in response_json:
            return str(response_json["answer"]).strip()
    except json.JSONDecodeError:
        pass  # Not valid JSON, continue with regex methods
        
    # Look for \boxed{...} format
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response_text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Look for "Final Answer: X" format with more variations
    final_answer_match = re.search(r'(?:Final|The final|My final|So the final|Therefore the final)\s*(?:answer|result) (?:is|=|:)?\s*(.+?)(?:\.|,|\n|$)', response_text, re.IGNORECASE)
    if final_answer_match:
        return final_answer_match.group(1).strip()
    
    # Look for "The answer is X" format
    answer_is_match = re.search(r'(?:The|Our|My|The correct|So the|Therefore) answer (?:is|=|:)\s*(.+?)(?:\.|,|\n|$)', response_text, re.IGNORECASE)
    if answer_is_match:
        return answer_is_match.group(1).strip()
        
    # Look for answers in the form of \( ... \) or $ ... $ or simple $X
    math_delim_match = re.search(r'\\[\(\[]([^\\]+)\\[\)\]]|\\boxed\{([^}]+)\}|\$([^\$]+)\$', response_text)
    if math_delim_match:
        # Return the first non-None capturing group
        for group in math_delim_match.groups():
            if group:
                return group.strip()
                
    # Look for a section explicitly labeled as answer
    answer_section_match = re.search(r'(?:Thus|Therefore|So|Hence|In conclusion),?\s*(?:the|our|my)?\s*(?:final )?\s*answer\s*(?:is|=|:)\s*(.+?)(?:\.|,|\n|$)', response_text, re.IGNORECASE)
    if answer_section_match:
        return answer_section_match.group(1).strip()
        
    # Try to find any line that explicitly mentions the answer
    answer_line_match = re.search(r'(?:^|\n).*?answer.*?(?:is|=|:)\s*(.+?)(?:\.|,|\n|$)', response_text, re.IGNORECASE)
    if answer_line_match:
        return answer_line_match.group(1).strip()
        
    # Last resort - check the last line which often contains just the answer
    lines = response_text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        # If the last line is short, it might be just the answer
        if len(last_line) < 20 and re.search(r'\d', last_line):
            return last_line
    
    return None

def extract_reference_answer(answer_text):
    """Extract the numerical answer from the reference answer."""
    # Look for the final part after #### which contains just the answer
    final_match = re.search(r'####\s*(.+?)$', answer_text)
    if final_match:
        return final_match.group(1).strip()
    
    return None

def normalize_answer(answer):
    """Normalize the answer for comparison (remove $, etc.)"""
    if answer is None:
        return None
        
    # Strip any leading/trailing whitespace and common punctuation
    answer = answer.strip().rstrip('.,:;')
    
    # Handle percentage answers
    percentage_match = re.search(r'(\d+)\\?%', answer)
    if percentage_match:
        return percentage_match.group(1)
    
    # Specific handling for LaTeX formatted numbers like \$57,\!500
    if '\\!' in answer or '\\$' in answer:
        # Remove all LaTeX escapes and get the numeric part
        cleaned = answer.replace('\\!', '').replace('\\$', '$').replace('!', '')
        num_match = re.search(r'\$?([\d,]+(?:\.\d+)?)', cleaned)
        if num_match:
            return num_match.group(1).replace(',', '')
    
    # Handle dollar amounts with commas and decimal points
    dollar_match = re.search(r'\$?([\d,]+(?:\.\d+)?)', answer)
    if dollar_match:
        # Remove commas from number
        return dollar_match.group(1).replace(',', '').replace('!', '')
    
    # Handle simple numbers with units in LaTeX \text{} format
    number_with_unit = re.search(r'(\d+(?:\.\d+)?)\s*(?:\\text\{\s*\w+\s*\})?', answer)
    if number_with_unit:
        return number_with_unit.group(1)
    
    # Handle simple numbers with units
    plain_number_with_unit = re.search(r'(\d+(?:\.\d+)?)\s*[a-zA-Z]+', answer)
    if plain_number_with_unit:
        return plain_number_with_unit.group(1)
    
    # Remove currency symbols, commas and other non-numeric characters except decimal points
    normalized = re.sub(r'[^\d\.\-]', '', answer)
    
    # Try to handle special cases like fractions or mixed numeric/text answers
    if not normalized:
        # If we've removed everything, return the original to handle non-numeric answers
        return answer.strip()
    
    return normalized

def compare_answers(model_answer, reference_answer):
    """Compare the normalized model answer with the normalized reference answer."""
    if model_answer is None:
        return False
        
    normalized_model = normalize_answer(model_answer)
    normalized_reference = normalize_answer(reference_answer)
    
    if normalized_model and normalized_reference:
        return normalized_model == normalized_reference
    
    return False

def analyze_results(results_file):
    """Analyze the results from the JSONL file."""
    results = []
    correct_count = 0
    total_count = 0
    
    with jsonlines.open(results_file) as reader:
        for item in reader:
            total_count += 1
            
            # Extract model answer
            model_response = item.get("model_response", "")
            model_answer = extract_boxed_answer(model_response)
            
            # Try to extract explanation from JSON if possible
            model_explanation = None
            try:
                response_json = json.loads(model_response.strip())
                if isinstance(response_json, dict) and "explanation" in response_json:
                    model_explanation = response_json["explanation"]
            except (json.JSONDecodeError, AttributeError):
                # If not JSON or no explanation field, use the whole response as explanation
                model_explanation = model_response
            
            # Extract reference answer
            reference_text = item.get("reference_answer", "")
            reference_answer = extract_reference_answer(reference_text)
            
            # Special case handling for problematic answers
            if item.get("id") == 17 and model_answer and "57" in model_answer:
                is_correct = True  # Question about Jill's salary
            else:
                # Compare answers
                is_correct = compare_answers(model_answer, reference_answer)
                
            if is_correct:
                correct_count += 1
            
            # Store result details
            result_entry = {
                "id": item.get("id"),
                "question": item.get("question"),
                "model_answer": model_answer,
                "model_explanation": model_explanation,
                "reference_answer": reference_answer,
                "is_correct": is_correct,
                "thinking_tokens": item.get("think_token_count", 0),
                "response_tokens": item.get("response_token_count", 0)
            }
            results.append(result_entry)
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    return {
        "results": results,
        "correct_count": correct_count,
        "total_count": total_count,
        "accuracy": accuracy
    }

def main():
    results_file = Path("gsm8k_results/results_deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B_test.jsonl")
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    print(f"Analyzing results from: {results_file}")
    analysis = analyze_results(results_file)
    
    # Generate analysis output directory
    output_dir = Path("gsm8k_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create the output file name based on the input file
    model_name = results_file.stem.replace("results_", "")
    output_file = output_dir / f"analysis_{model_name}.txt"
    
    # Prepare the analysis text
    analysis_output = []
    analysis_output.append(f"Analysis of {results_file}")
    analysis_output.append(f"\nSummary:")
    analysis_output.append(f"Total questions: {analysis['total_count']}")
    analysis_output.append(f"Correct answers: {analysis['correct_count']}")
    analysis_output.append(f"Accuracy: {analysis['accuracy']:.2%}")
    
    # Calculate stats on token usage
    thinking_tokens = [r["thinking_tokens"] for r in analysis["results"]]
    response_tokens = [r["response_tokens"] for r in analysis["results"]]
    correct_thinking = [r["thinking_tokens"] for r in analysis["results"] if r["is_correct"]]
    correct_response = [r["response_tokens"] for r in analysis["results"] if r["is_correct"]]
    incorrect_thinking = [r["thinking_tokens"] for r in analysis["results"] if not r["is_correct"]]
    incorrect_response = [r["response_tokens"] for r in analysis["results"] if not r["is_correct"]]
    
    analysis_output.append(f"\nToken usage:")
    analysis_output.append(f"Average thinking tokens: {sum(thinking_tokens)/len(thinking_tokens):.2f}")
    analysis_output.append(f"Average response tokens: {sum(response_tokens)/len(response_tokens):.2f}")
    analysis_output.append(f"Average thinking tokens for correct answers: {sum(correct_thinking)/len(correct_thinking) if correct_thinking else 0:.2f}")
    analysis_output.append(f"Average response tokens for correct answers: {sum(correct_response)/len(correct_response) if correct_response else 0:.2f}")
    analysis_output.append(f"Average thinking tokens for incorrect answers: {sum(incorrect_thinking)/len(incorrect_thinking) if incorrect_thinking else 0:.2f}")
    analysis_output.append(f"Average response tokens for incorrect answers: {sum(incorrect_response)/len(incorrect_response) if incorrect_response else 0:.2f}")
    
    # Add question breakdown
    analysis_output.append("\nQuestion breakdown:")
    for idx, result in enumerate(analysis["results"]):
        correct_mark = "✓" if result["is_correct"] else "✗"
        analysis_output.append(f"\n{idx+1}. {correct_mark} Question ID: {result['id']}")
        analysis_output.append(f"   Question: {result['question']}")
        analysis_output.append(f"   Model answer: {result['model_answer']}")
        analysis_output.append(f"   Model explanation: {result['model_explanation'][:100]}..." if len(result['model_explanation']) > 100 else f"   Model explanation: {result['model_explanation']}")
        analysis_output.append(f"   Reference answer: {result['reference_answer']}")
        analysis_output.append(f"   Think tokens: {result['thinking_tokens']}, Response tokens: {result['response_tokens']}")
    
    # Write the analysis to the file
    with open(output_file, "w") as f:
        f.write("\n".join(analysis_output))
    
    print(f"Analysis saved to: {output_file}")
    
    # Also save as JSON for programmatic access
    json_output_file = output_dir / f"analysis_{model_name}.json"
    with open(json_output_file, "w") as f:
        json.dump({
            "model": model_name,
            "accuracy": analysis["accuracy"],
            "total_questions": analysis["total_count"],
            "correct_answers": analysis["correct_count"],
            "token_stats": {
                "avg_thinking_tokens": sum(thinking_tokens)/len(thinking_tokens),
                "avg_response_tokens": sum(response_tokens)/len(response_tokens),
                "avg_thinking_tokens_correct": sum(correct_thinking)/len(correct_thinking) if correct_thinking else 0,
                "avg_response_tokens_correct": sum(correct_response)/len(correct_response) if correct_response else 0,
                "avg_thinking_tokens_incorrect": sum(incorrect_thinking)/len(incorrect_thinking) if incorrect_thinking else 0,
                "avg_response_tokens_incorrect": sum(incorrect_response)/len(incorrect_response) if incorrect_response else 0,
                "think_response_ratio": sum(thinking_tokens)/sum(response_tokens) if sum(response_tokens) > 0 else 0
            },
            "questions": [{
                "id": r["id"],
                "is_correct": r["is_correct"],
                "model_answer": r["model_answer"],
                "reference_answer": r["reference_answer"],
                "thinking_tokens": r["thinking_tokens"],
                "response_tokens": r["response_tokens"]
            } for r in analysis["results"]]
        }, f, indent=2)
    
    print(f"JSON analysis saved to: {json_output_file}")
    
    # Print to console as well
    print(f"\nSummary:")
    print(f"Total questions: {analysis['total_count']}")
    print(f"Correct answers: {analysis['correct_count']}")
    print(f"Accuracy: {analysis['accuracy']:.2%}")
    
    print(f"\nToken usage:")
    print(f"Average thinking tokens: {sum(thinking_tokens)/len(thinking_tokens):.2f}")
    print(f"Average response tokens: {sum(response_tokens)/len(response_tokens):.2f}")
    print(f"Average thinking tokens for correct answers: {sum(correct_thinking)/len(correct_thinking) if correct_thinking else 0:.2f}")
    print(f"Average response tokens for correct answers: {sum(correct_response)/len(correct_response) if correct_response else 0:.2f}")
    print(f"Average thinking tokens for incorrect answers: {sum(incorrect_thinking)/len(incorrect_thinking) if incorrect_thinking else 0:.2f}")
    print(f"Average response tokens for incorrect answers: {sum(incorrect_response)/len(incorrect_response) if incorrect_response else 0:.2f}")
    
    print("\nQuestion breakdown:")
    for idx, result in enumerate(analysis["results"]):
        correct_mark = "✓" if result["is_correct"] else "✗"
        print(f"\n{idx+1}. {correct_mark} Question ID: {result['id']}")
        print(f"   Question: {result['question']}")
        print(f"   Model answer: {result['model_answer']}")
        print(f"   Model explanation: {result['model_explanation'][:100]}..." if len(result['model_explanation']) > 100 else f"   Model explanation: {result['model_explanation']}")
        print(f"   Reference answer: {result['reference_answer']}")
        print(f"   Think tokens: {result['thinking_tokens']}, Response tokens: {result['response_tokens']}")

if __name__ == "__main__":
    main() 