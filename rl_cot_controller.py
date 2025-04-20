#!/usr/bin/env python3
"""
RL Controller for optimizing when to terminate chain-of-thought reasoning.
Takes in the current state (CoT), embeds it with ModernBERT, and decides
whether to continue for 10 more tokens or force an answer.

Reward structure:
+1000: Force answer and correct
-10: Continue thinking
-100: Force answer and incorrect
"""

import os
import re
import json
import jsonlines
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel
from collections import deque
from sklearn.metrics import accuracy_score


class ModernBERTEmbedder:
    """Uses ModernBERT to embed chain-of-thought reasoning."""
    
    def __init__(self, model_name="answerdotai/ModernBERT-base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the BERT embedder.
        
        Args:
            model_name: The name of the BERT model to use (default: "answerdotai/ModernBERT-base")
            device: The device to run the model on (default: cuda if available, else cpu)
        """
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size
    
    def embed(self, text):
        """Embed a text sequence using ModernBERT.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedded representation (last layer mean pooling)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling of last hidden state
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings.squeeze(0).cpu().numpy()


class RLPolicy(nn.Module):
    """RL policy network that decides whether to continue thinking or force an answer."""
    
    def __init__(self, input_dim, hidden_dim=128):
        """Initialize the policy network.
        
        Args:
            input_dim: Dimension of the input state (embedding size)
            hidden_dim: Dimension of the hidden layer
        """
        super(RLPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # 2 actions: continue or force answer
        )
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input state embedding
            
        Returns:
            Logits for each action
        """
        return self.network(x)


class CoTController:
    """Reinforcement Learning controller for chain-of-thought reasoning."""
    
    def __init__(self, llm, tokenizer, bert_model="answerdotai/ModernBERT-base", 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate=1e-4, gamma=0.99, buffer_size=10000):
        """Initialize the CoT controller.
        
        Args:
            llm: The language model for generating CoT and answers
            tokenizer: The tokenizer for the language model
            bert_model: The name of the BERT model to use for embeddings
            device: The device to run on
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            buffer_size: Size of the experience replay buffer
        """
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = gamma
        self.special_tokens = tokenizer.special_tokens_map
        
        # Ensure think tokens are in the map
        if "<think>" not in self.special_tokens:
            self.special_tokens["<think>"] = tokenizer.convert_tokens_to_ids("<think>")
        if "</think>" not in self.special_tokens:
            self.special_tokens["</think>"] = tokenizer.convert_tokens_to_ids("</think>")
        
        # Initialize embedder
        self.embedder = ModernBERTEmbedder(model_name=bert_model, device=device)
        
        # Initialize policy network
        self.policy = RLPolicy(input_dim=self.embedder.embedding_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Statistics
        self.episode_rewards = []
        self.episode_decisions = []
        self.correct_answers = 0
        self.total_answers = 0
    
    def extract_answer(self, text):
        """Extract numerical answer from model output using more robust patterns.
        
        Args:
            text: The text containing the answer
            
        Returns:
            The extracted numerical answer
        """
        if not text:
            return None
            
        # Look for "Final Answer: X" format with more variations
        final_answer_match = re.search(r'(?:Final|The final|My final|So the final|Therefore the final)\s*(?:answer|result) (?:is|=|:)?\s*(.+?)(?:\.|,|\n|$)', text, re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        
        # Look for "The answer is X" format
        answer_is_match = re.search(r'(?:The|Our|My|The correct|So the|Therefore) answer (?:is|=|:)\s*(.+?)(?:\.|,|\n|$)', text, re.IGNORECASE)
        if answer_is_match:
            return answer_is_match.group(1).strip()
            
        # Look for a section explicitly labeled as answer
        answer_section_match = re.search(r'(?:Thus|Therefore|So|Hence|In conclusion),?\s*(?:the|our|my)?\s*(?:final )?\s*answer\s*(?:is|=|:)\s*(.+?)(?:\.|,|\n|$)', text, re.IGNORECASE)
        if answer_section_match:
            return answer_section_match.group(1).strip()
            
        # Try to find any line that explicitly mentions the answer
        answer_line_match = re.search(r'(?:^|\n).*?answer.*?(?:is|=|:)\s*(.+?)(?:\.|,|\n|$)', text, re.IGNORECASE)
        if answer_line_match:
            return answer_line_match.group(1).strip()
            
        # Last resort - check the last line which often contains just the answer
        lines = text.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # If the last line is short, it might be just the answer
            if len(last_line) < 20 and re.search(r'\d', last_line):
                return last_line
        
        # Basic number extraction if nothing else worked
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]  # Return the last number found
            
        return text.strip()
        
    def normalize_answer(self, answer):
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
        
        return normalized
        
    def extract_reference_answer(self, answer_text):
        """Extract the numerical answer from the reference answer."""
        if not answer_text:
            return None
            
        # Look for the final part after #### which contains just the answer
        final_match = re.search(r'####\s*(.+?)$', answer_text)
        if final_match:
            return final_match.group(1).strip()
        
        return answer_text

    def is_correct(self, model_answer, reference_answer):
        """Check if the model's answer is correct using more robust comparison.
        
        Args:
            model_answer: The model's generated answer
            reference_answer: The reference answer
            
        Returns:
            True if the answers match, False otherwise
        """
        if model_answer is None:
            return False
            
        # Extract the answer from the model's response
        extracted_answer = self.extract_answer(model_answer)
        
        # Extract the reference answer 
        extracted_reference = self.extract_reference_answer(reference_answer)
        
        # Normalize both answers
        normalized_model = self.normalize_answer(extracted_answer)
        normalized_reference = self.normalize_answer(extracted_reference)
        
        # Check if they match after normalization
        if normalized_model and normalized_reference:
            return normalized_model == normalized_reference
            
        return False
    
    def select_action(self, state_embedding, epsilon=0.1):
        """Select an action using epsilon-greedy policy.
        
        Args:
            state_embedding: The embedding of the current state
            epsilon: Probability of selecting a random action
            
        Returns:
            action: 0 for continue, 1 for force answer
            log_prob: Log probability of the selected action
        """
        if np.random.random() < epsilon:
            # Random action
            action = np.random.randint(0, 2)
            state_tensor = torch.tensor(state_embedding, dtype=torch.float32).to(self.device)
            logits = self.policy(state_tensor)
            log_prob = torch.log_softmax(logits, dim=0)[action]
            return action, log_prob.item()
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.tensor(state_embedding, dtype=torch.float32).to(self.device)
                logits = self.policy(state_tensor)
                probs = torch.softmax(logits, dim=0)
                action = torch.argmax(probs).item()
                log_prob = torch.log(probs[action])
                return action, log_prob.item()
    
    def continue_thinking(self, question, current_thinking, num_tokens=100):
        """Continue the chain of thought for a specified number of tokens.
        
        Args:
            question: The original question
            current_thinking: The current thinking trace
            num_tokens: Number of tokens to continue generating
            
        Returns:
            The updated thinking trace
        """
        prompt = f"{format_prompt(question)}\n<think>{current_thinking}"
        
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=num_tokens,
            stop=["</think>", "Human:", "Assistant:", "FINAL ANSWER:"]
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        continuation = outputs[0].outputs[0].text
        
        # Check if the model produced a </think> token
        if "</think>" in continuation:
            continuation = continuation[:continuation.find("</think>")]
        
        return current_thinking + continuation
    
    def force_answer(self, question, thinking_trace):
        """Force the model to generate a final answer after the thinking trace.
        
        Args:
            question: The original question
            thinking_trace: The thinking trace
            
        Returns:
            The model's answer
        """
        prompt = f"{format_prompt(question)}\n<think>{thinking_trace}</think>\n\nBased on my step-by-step reasoning, the final answer is: "
        
        sampling_params = SamplingParams(
            temperature=0.2,  # Lower temperature for more deterministic answers
            max_tokens=50,    # We only need a short answer
            stop=["Human:", "Assistant:", "\n\n"]  # Stop at newlines to get just the answer
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        answer_text = outputs[0].outputs[0].text.strip()
        
        # If the answer doesn't have a clear structure, try to add one
        if not re.search(r'answer is|final answer|=', answer_text, re.IGNORECASE):
            # If the model just returned a number without context, format it
            if re.match(r'^[\d\.\,\$]+$', answer_text):
                answer_text = f"The final answer is {answer_text}"
        
        return answer_text
    
    def update_policy(self, batch_size=32):
        """Update the policy network using experience replay.
        
        Args:
            batch_size: Number of experiences to sample for each update
            
        Returns:
            The loss value
        """
        if len(self.buffer) < batch_size:
            return None
        
        # Sample batch from buffer
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, log_probs = [], [], [], []
        
        for i in indices:
            s, a, r, lp = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            log_probs.append(lp)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
        
        # Calculate loss using policy gradient
        logits = self.policy(states)
        log_probs = torch.log_softmax(logits, dim=1)
        selected_log_probs = log_probs[range(batch_size), actions]
        
        # Policy gradient loss
        loss = -torch.mean(rewards * selected_log_probs)
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def process_sample(self, question, reference_answer, max_steps=100, epsilon=0.1, continue_tokens=10):
        """Process a single sample through the RL controller.
        
        Args:
            question: The question to answer
            reference_answer: The reference answer
            max_steps: Maximum number of steps before forcing an answer
            epsilon: Exploration rate
            continue_tokens: Number of tokens to generate when continuing
            
        Returns:
            A dictionary containing the results
        """
        # Initialize thinking trace
        current_thinking = ""
        step = 0
        total_reward = 0
        trajectory = []
        prev_state_embedding = None
        
        while step < max_steps:
            # Embed current state
            state_embedding = self.embedder.embed(current_thinking if current_thinking else question)
            
            # Select action
            action, log_prob = self.select_action(state_embedding, epsilon)
            
            if action == 0:  # Continue thinking
                # Store previous thinking state for progress measurement
                prev_thinking = current_thinking
                prev_embedding = state_embedding.copy() if prev_state_embedding is None else prev_state_embedding
                
                # Continue thinking for specified tokens
                current_thinking = self.continue_thinking(question, current_thinking, continue_tokens)
                
                # Calculate thinking progress (cosine similarity between previous and current state)
                if prev_thinking != current_thinking:
                    # Only penalize if no progress was made in thinking
                    new_embedding = self.embedder.embed(current_thinking)
                    
                    # Calculate cosine similarity to measure progress
                    progress = np.dot(prev_embedding, new_embedding) / (np.linalg.norm(prev_embedding) * np.linalg.norm(new_embedding))
                    
                    # Early steps should have minimal penalty or even small reward
                    if step < max_steps / 3:  # First third of steps
                        # Encourage early exploration with minimal penalty
                        reward = -1  # Very small penalty for thinking early on
                    elif step < 2 * max_steps / 3:  # Middle third of steps
                        # Start increasing penalty gradually
                        reward = -3
                    else:  # Final third of steps
                        # Higher penalty for thinking too much
                        reward = -7
                    
                    # If significant progress was made, reduce penalty even more
                    # Cosine similarity < 0.9 indicates more progress (less similar)
                    if progress < 0.9:
                        reward = max(reward + 2, 0)  # Possible small positive reward for significant progress
                    
                    # Store state for next comparison
                    prev_state_embedding = new_embedding
                else:
                    # No progress was made, so we should force an answer
                    action = 1
                    reward = -15  # Higher penalty for no progress
            
            if action == 1 or step == max_steps - 1:  # Force answer or reached max steps
                # Generate final answer
                answer = self.force_answer(question, current_thinking)
                
                # Check correctness
                correct = self.is_correct(answer, reference_answer)
                self.total_answers += 1
                
                if correct:
                    # Base reward for correct answer
                    reward = 1000
                    
                    # Bonus reward for efficiency (using fewer steps)
                    efficiency_bonus = (max_steps - step) * 5  # 5 points per step saved
                    reward += efficiency_bonus
                    
                    # Bonus reward for conciseness if thinking is not too short
                    min_tokens = 100  # Minimum tokens to consider the thinking substantial
                    thinking_tokens = len(self.tokenizer.encode(current_thinking))
                    if thinking_tokens >= min_tokens:
                        conciseness_reward = min(500, int(1000 * (min_tokens / thinking_tokens)))
                        reward += conciseness_reward
                    
                    self.correct_answers += 1
                else:
                    # Penalty for incorrect answer
                    base_penalty = -1000
                    
                    # Reduced penalty if thinking was substantial
                    thinking_tokens = len(self.tokenizer.encode(current_thinking))
                    if thinking_tokens > 300:  # If significant thinking was done
                        penalty_reduction = min(50, thinking_tokens // 10)
                        reward = base_penalty + penalty_reduction
                    else:
                        reward = base_penalty
                
                # Store experience
                trajectory.append((state_embedding, action, reward, log_prob))
                total_reward += reward
                break
            
            # Store experience
            trajectory.append((state_embedding, action, reward, log_prob))
            total_reward += reward
            step += 1
        
        # Add experiences to replay buffer
        for exp in trajectory:
            self.buffer.append(exp)
        
        # Update policy
        loss = self.update_policy()
        
        # Record statistics
        self.episode_rewards.append(total_reward)
        self.episode_decisions.append(step)
        
        # Return result with explicit numeric types for key metrics
        result = {
            "question": question,
            "reference_answer": reference_answer,
            "thinking_trace": f"<think>{current_thinking}</think>",
            "model_answer": answer if 'answer' in locals() else None,
            "steps": int(step),
            "total_reward": float(total_reward),
            "correct": bool(correct) if 'correct' in locals() else False,
            "thinking_token_count": int(len(self.tokenizer.encode(current_thinking)))
        }
        
        return result


def format_prompt(question):
    """Format the prompt for the LLM."""
    return f"""{question}
    Let's think step by step:
    """


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RL controller for chain-of-thought reasoning")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      help="LLM model to use")
    parser.add_argument("--bert_model", type=str, default="answerdotai/ModernBERT-base",
                      help="BERT model for embeddings (default: answerdotai/ModernBERT-base)")
    parser.add_argument("--max_samples", type=int, default=10,
                      help="Maximum number of samples to process")
    parser.add_argument("--output_dir", type=str, default="rl_cot_results",
                      help="Directory to save results")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                      help="Dataset split to use")
    parser.add_argument("--epochs", type=int, default=10,
                      help="Number of epochs to train (default: 10)")
    parser.add_argument("--iterations_per_example", type=int, default=1,
                      help="Number of iterations to run for each example per epoch (default: 1)")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for policy updates")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate for the policy optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
                      help="Discount factor for future rewards")
    parser.add_argument("--epsilon", type=float, default=0.1,
                      help="Exploration rate for epsilon-greedy policy")
    parser.add_argument("--max_steps", type=int, default=50,
                      help="Maximum steps per sample before forcing an answer")
    parser.add_argument("--continue_tokens", type=int, default=10,
                      help="Number of tokens to generate when continuing")
    parser.add_argument("--quantize", action="store_true",
                      help="Use 4-bit quantization for reduced memory usage")
    parser.add_argument("--verbose", action="store_true",
                      help="Print detailed information during processing")
    parser.add_argument("--use_ollama_eval", action="store_true",
                      help="Use Ollama for secondary answer evaluation")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/v1",
                      help="URL for Ollama API (default: http://localhost:11434/v1)")
    parser.add_argument("--ollama_model", type=str, default="llama3.2",
                      help="Model name in Ollama for evaluation (default: llama3.2)")
    parser.add_argument("--save_all_iterations", action="store_true",
                      help="Save results from all iterations (default: only save the last iteration)")
    parser.add_argument("--save_best_per_epoch", action="store_true", 
                      help="Save only the best result for each example in each epoch")
    return parser.parse_args()


def run_rl_controller(args):
    """Run the RL controller on the GSM8K dataset."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
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
    
    print(f"Using LLM: {args.model}")
    print(f"Using BERT: {args.bert_model}")
    
    # Initialize controller
    controller = CoTController(
        llm=llm, 
        tokenizer=tokenizer,
        bert_model=args.bert_model,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )
    
    model_name_safe = args.model.replace('/', '_').replace('-', '_')
    results_file = output_dir / f"results_{model_name_safe}_{args.split}.jsonl"
    
    # Clear previous results file
    if results_file.exists():
        results_file.unlink()
    
    # Track overall best results and history per epoch
    overall_best_results = {}  # Maps example_id -> best result
    epoch_stats = []
    
    # Process each epoch
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        
        # Initialize epoch statistics
        epoch_correct = 0
        epoch_total = 0
        epoch_results = []
        epoch_best_results = {}  # Maps example_id -> best result for this epoch
        
        # Create shuffled dataset for this epoch
        shuffled_indices = np.random.permutation(len(dataset))
        
        # Process each example in random order
        for i, idx in enumerate(shuffled_indices):
            # Convert numpy.int64 to Python int
            idx = int(idx)
            example = dataset[idx]
            question = example["question"]
            reference_answer = example["answer"]
            
            print(f"\nProcessing example {i+1}/{len(dataset)} (id={idx}): {question[:50]}...")
            
            # Run iterations for this example in this epoch
            example_results = []
            
            # Track best result for this example in this epoch
            best_result = None
            best_reward = float('-inf')
            
            # Calculate decaying epsilon for this epoch
            epsilon = max(0.01, args.epsilon * (1.0 - epoch / args.epochs))
            
            # Run iterations for this example
            for iter_idx in range(args.iterations_per_example):
                try:
                    # Process with RL controller
                    result = controller.process_sample(
                        question=question,
                        reference_answer=reference_answer,
                        max_steps=args.max_steps,
                        epsilon=epsilon,
                        continue_tokens=args.continue_tokens
                    )
                    
                    # Add metadata
                    result.update({
                        "id": int(idx),
                        "epoch": int(epoch),
                        "iteration": int(iter_idx),
                        "model": args.model,
                        "bert_model": args.bert_model,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    
                    example_results.append(result)
                    
                    # Track best result for this example in this epoch
                    if result['total_reward'] > best_reward:
                        best_result = result
                        best_reward = result['total_reward']
                    
                    # Write to JSONL file if save_all_iterations is True
                    if args.save_all_iterations:
                        with jsonlines.open(results_file, mode="a") as writer:
                            writer.write(result)
                
                except Exception as e:
                    print(f"Error processing iteration {iter_idx} of example {idx} in epoch {epoch}: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
            
            # After all iterations, track the best result for this example
            if best_result:
                # Add to epoch results
                epoch_results.append(best_result)
                epoch_best_results[idx] = best_result
                
                # Update overall best results
                if idx not in overall_best_results or best_result['total_reward'] > overall_best_results[idx]['total_reward']:
                    overall_best_results[idx] = best_result
                
                # Write epoch's best result if saving per epoch
                if args.save_best_per_epoch and not args.save_all_iterations:
                    with jsonlines.open(results_file, mode="a") as writer:
                        writer.write(best_result)
                
                # Update epoch statistics
                if best_result.get('correct', False):
                    epoch_correct += 1
                epoch_total += 1
                
                # Print statistics for this example
                print(f"Example {i+1} (id={idx}) complete:")
                print(f"  Reward: {best_reward:.1f}")
                print(f"  Steps: {best_result['steps']}")
                print(f"  Thinking tokens: {best_result['thinking_token_count']}")
                print(f"  Correct: {best_result.get('correct', False)}")
                print(f"  Current epoch accuracy: {epoch_correct/epoch_total:.2%}")
        
        # End of epoch - calculate statistics
        epoch_accuracy = epoch_correct / max(1, epoch_total)
        epoch_avg_steps = np.mean([r['steps'] for r in epoch_results])
        epoch_avg_reward = np.mean([r['total_reward'] for r in epoch_results])
        epoch_avg_tokens = np.mean([r['thinking_token_count'] for r in epoch_results])
        
        # Store epoch statistics
        epoch_stats.append({
            "epoch": epoch,
            "accuracy": float(epoch_accuracy),
            "avg_steps": float(epoch_avg_steps),
            "avg_reward": float(epoch_avg_reward),
            "avg_thinking_tokens": float(epoch_avg_tokens),
            "correct_count": int(epoch_correct),
            "total_count": int(epoch_total)
        })
        
        # Print epoch summary
        print(f"\n{'-'*80}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"Accuracy: {epoch_accuracy:.4f} ({epoch_correct}/{epoch_total})")
        print(f"Average steps: {epoch_avg_steps:.2f}")
        print(f"Average reward: {epoch_avg_reward:.2f}")
        print(f"Average thinking tokens: {epoch_avg_tokens:.2f}")
        print(f"{'-'*80}")
    
    # Write overall best results at the end
    if not args.save_all_iterations and not args.save_best_per_epoch:
        with jsonlines.open(results_file, mode="a") as writer:
            for result in overall_best_results.values():
                writer.write(result)
    
    # Calculate final statistics from overall best results
    final_results = list(overall_best_results.values())
    correct_answers = sum(1 for r in final_results if r.get('correct', False))
    total_answers = len(final_results)
    final_accuracy = correct_answers / max(1, total_answers)
    
    # Calculate averages safely with explicit type conversion
    avg_steps = np.mean([float(r['steps']) for r in final_results])
    avg_reward = np.mean([float(r['total_reward']) for r in final_results])
    avg_tokens = np.mean([float(r['thinking_token_count']) for r in final_results])
    
    # Run Ollama evaluation if requested
    ollama_eval_results = None
    if args.use_ollama_eval:
        ollama_eval_results = evaluate_with_ollama(
            results_file,
            ollama_url=args.ollama_url,
            model=args.ollama_model,
            verbose=args.verbose
        )
        # Update accuracy if Ollama evaluation was successful
        if ollama_eval_results and "accuracy" in ollama_eval_results and ollama_eval_results["accuracy"] > 0:
            final_accuracy = ollama_eval_results["accuracy"]
            correct_answers = ollama_eval_results["correct_count"]
    
    # Save summary with properly typed values
    summary = {
        "model": args.model,
        "bert_model": args.bert_model,
        "split": args.split,
        "epochs": int(args.epochs),
        "iterations_per_example": int(args.iterations_per_example),
        "samples_processed": len(final_results),
        "final_accuracy": float(final_accuracy),
        "avg_steps": float(avg_steps),
        "avg_reward": float(avg_reward),
        "avg_thinking_tokens": float(avg_tokens),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch_history": epoch_stats
    }
    
    # Add Ollama evaluation results if available
    if ollama_eval_results:
        summary["ollama_evaluation"] = {
            "accuracy": float(ollama_eval_results.get("accuracy", 0)),
            "correct_count": int(ollama_eval_results.get("correct_count", 0)),
            "total_count": int(ollama_eval_results.get("total_count", 0)),
        }
    
    with open(output_dir / f"summary_{model_name_safe}_{args.split}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save the trained policy
    torch.save(controller.policy.state_dict(), output_dir / f"policy_{model_name_safe}.pt")
    
    # Print learning curve
    print("\nLearning Curve:")
    print("Epoch | Accuracy | Avg Steps | Avg Reward | Avg Tokens")
    print("-" * 60)
    for stat in epoch_stats:
        print(f"{stat['epoch']+1:5d} | {stat['accuracy']:.4f} | {stat['avg_steps']:9.2f} | {stat['avg_reward']:10.2f} | {stat['avg_thinking_tokens']:10.2f}")
    
    print(f"\nTraining complete. Processed {len(final_results)} samples across {args.epochs} epochs.")
    print(f"Final accuracy: {final_accuracy:.4f} ({correct_answers}/{total_answers})")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average thinking tokens: {avg_tokens:.2f}")
    print(f"Results saved to {output_dir}")


def evaluate_with_ollama(results_file, ollama_url="http://localhost:11434/v1", model="llama3.2", verbose=False):
    """Evaluate results using Ollama-based extraction for higher accuracy.
    
    Args:
        results_file: Path to the results JSONL file
        ollama_url: URL for Ollama API
        model: Model name in Ollama
        verbose: Whether to print details
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Check if openai package is available
        import openai
        from openai import OpenAI
        
        client = OpenAI(
            base_url=ollama_url,
            api_key="ollama"  # Ollama doesn't need a real API key
        )
        
        results = []
        correct_count = 0
        total_count = 0
        
        print(f"Evaluating results with Ollama ({model}) for higher accuracy...")
        
        # Read JSONL file
        with jsonlines.open(results_file) as reader:
            items = list(reader)
            
        for item in tqdm(items, desc="Evaluating with Ollama"):
            # Skip items without model answer or reference answer
            if "model_answer" not in item or "reference_answer" not in item:
                continue
                
            # Extract answers
            total_count += 1
            model_answer = item["model_answer"]
            reference_answer = item["reference_answer"]
            
            # Use Ollama to extract the numerical answer
            prompt = f"""Below is a solution to a math problem. Extract ONLY the final numerical answer with no explanation, no units, no dollar signs, and no other text.

Problem solution:
{model_answer}

For example if you see $70,000, you should return 70000. If you see 12 miles per hour, you should return 12. If you see 15 cups, you should return 15. If you see 40%, you should return 40.

Final numerical answer:"""
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that extracts numerical answers from math problem solutions."},
                {"role": "user", "content": prompt}
            ]
            
            if verbose:
                print(f"\nEvaluating item {total_count}:")
                print(f"Model answer: {model_answer}")
                print(f"Reference answer: {reference_answer}")
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1
                )
                
                # Extract the answer from the response
                extracted_answer = response.choices[0].message.content.strip()
                
                # Extract reference answer
                reference_match = re.search(r'####\s*(.+?)$', reference_answer)
                if reference_match:
                    extracted_reference = reference_match.group(1).strip()
                else:
                    extracted_reference = reference_answer
                
                # Normalize for comparison
                def normalize(answer):
                    if not answer:
                        return ""
                    # Remove non-numeric except decimal points and negatives
                    answer = re.sub(r'[^\d\.\-]', '', str(answer))
                    return answer
                
                normalized_extracted = normalize(extracted_answer)
                normalized_reference = normalize(extracted_reference)
                
                # Check if correct
                correct = normalized_extracted == normalized_reference
                if correct:
                    correct_count += 1
                
                result = {
                    "id": item.get("id", total_count),
                    "extracted_answer": extracted_answer,
                    "reference_answer": extracted_reference,
                    "normalized_extracted": normalized_extracted,
                    "normalized_reference": normalized_reference,
                    "correct": correct
                }
                
                results.append(result)
                
                if verbose:
                    print(f"Extracted: {extracted_answer}")
                    print(f"Reference: {extracted_reference}")
                    print(f"Correct: {correct}")
                
                # Add a small delay to avoid overwhelming Ollama
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing item {total_count}: {e}")
                
        # Calculate accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\nOllama Evaluation Results:")
        print(f"Total examples: {total_count}")
        print(f"Correct answers: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        
        return {
            "total_count": total_count,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "results": results
        }
        
    except ImportError:
        print("OpenAI package not available. Skipping Ollama evaluation.")
        return {
            "error": "OpenAI package not available",
            "accuracy": 0
        }
    except Exception as e:
        print(f"Error during Ollama evaluation: {e}")
        return {
            "error": str(e),
            "accuracy": 0
        }


def main():
    args = parse_args()
    run_rl_controller(args)


if __name__ == "__main__":
    main() 