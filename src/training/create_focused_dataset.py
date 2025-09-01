# File: VIZOR/src/training/create_focused_dataset.py

import json
import os

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "squad_format_qa.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "squad_focused_qa.json")
CONTEXT_WINDOW_SIZE = 256 # Characters to include before and after the answer

def create_focused_dataset():
    """
    Reads the SQuAD-formatted data and creates a new dataset where each context
    is a small, focused paragraph around the answer.
    """
    print(f"Loading original SQuAD data from: {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    new_squad_data = []
    
    for item in original_data["data"]:
        full_context = item["context"]
        question = item["question"]
        answer_text = item["answers"]["text"][0]
        answer_start_char = item["answers"]["answer_start"][0]
        
        # --- The Core Logic ---
        # 1. Define the start and end of our desired paragraph
        para_start = max(0, answer_start_char - CONTEXT_WINDOW_SIZE)
        para_end = min(len(full_context), answer_start_char + len(answer_text) + CONTEXT_WINDOW_SIZE)
        
        # 2. Extract the paragraph
        focused_context = full_context[para_start:para_end]
        
        # 3. Recalculate the answer's start position relative to the NEW, shorter context
        new_answer_start = focused_context.find(answer_text)
        
        # 4. Only add the example if the answer was found correctly
        if new_answer_start != -1:
            new_squad_data.append({
                "id": item["id"],
                "title": item["title"],
                "context": focused_context,
                "question": question,
                "answers": {
                    "text": [answer_text],
                    "answer_start": [new_answer_start]
                }
            })
        else:
            print(f"Warning: Could not find answer for question ID {item['id']} after focusing context.")

    print(f"Created {len(new_squad_data)} focused examples.")
    
    # Save the new dataset
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"data": new_squad_data}, f, indent=4)
        
    print(f"✅ Successfully saved focused dataset to: {OUTPUT_PATH}")

if __name__ == '__main__':
    create_focused_dataset()