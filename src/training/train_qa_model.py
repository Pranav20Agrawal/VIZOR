# File: VIZOR/src/training/train_qa_model.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Note: The print_example helper function is no longer needed, so it has been removed.

def preprocess_function(examples):
    """
    This is the core function for tokenizing our dataset.
    It handles both the text and the answer spans.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            
            token_end_index = len(offsets) - 1
            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1

            if (token_start_index > len(offsets) -1 or 
                token_end_index < 0 or
                offsets[token_start_index][0] > start_char or 
                offsets[token_end_index][1] < end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # We subtract 1 from token_start_index because the loop condition stops *after* passing the start char.
                tokenized_examples["start_positions"].append(token_start_index - 1)
                # We don't need to subtract 1 from token_end_index for the same reason.
                tokenized_examples["end_positions"].append(token_end_index)

    return tokenized_examples


# --- Main execution block ---
if __name__ == '__main__':
    # Define constants
    MODEL_NAME = "distilbert-base-uncased"
    DATASET_PATH = "data/processed/squad_format_qa.json"
    OUTPUT_DIR = "models/vizor-qa-distilbert-baseline"

    print(f"Starting QA model training with base model: {MODEL_NAME}")

    # Load dataset
    raw_datasets = load_dataset('json', data_files=DATASET_PATH, field='data')
    split_datasets = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
    print("\nDataset loaded and split:")
    print(split_datasets)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"\nTokenizer for '{MODEL_NAME}' loaded successfully.")

    # Apply the preprocessing function
    tokenized_datasets = split_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=split_datasets["train"].column_names
    )
    
    print("\nDataset tokenized and processed successfully:")
    print(tokenized_datasets)

    print("\nData is fully prepared for training. Moving to the next step.")