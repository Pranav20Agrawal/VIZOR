import os
import json
import uuid
from difflib import SequenceMatcher

# Define project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
QA_POC_PATH = os.path.join(PROCESSED_DATA_DIR, "vtop_qa_poc.json")
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "squad_format_qa.json")

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def find_best_match(answer_text, context, threshold=0.8):
    """
    Find the best matching substring in context for the answer text.
    Returns (start_index, matched_text) or (None, None) if no good match found.
    """
    words = answer_text.split()
    best_match = None
    best_score = 0
    best_start = -1
    
    # Try to find exact match first
    start_exact = context.find(answer_text)
    if start_exact != -1:
        return start_exact, answer_text
    
    # Try case-insensitive exact match
    start_case = context.lower().find(answer_text.lower())
    if start_case != -1:
        # Find the actual text at this position
        matched_text = context[start_case:start_case + len(answer_text)]
        return start_case, matched_text
    
    # Try fuzzy matching for shorter answers (likely exact terms)
    if len(words) <= 5:
        # Split context into sentences and check each
        sentences = context.replace('\n', ' ').split('. ')
        for sentence in sentences:
            if similarity(answer_text.lower(), sentence.lower()) > threshold:
                start_idx = context.find(sentence)
                if start_idx != -1:
                    return start_idx, sentence.strip()
    
    # For longer answers, try finding partial matches
    if len(words) > 3:
        # Try finding the first few words
        partial = ' '.join(words[:3])
        start_partial = context.lower().find(partial.lower())
        if start_partial != -1:
            # Extend to find a reasonable endpoint
            end_pos = min(start_partial + len(answer_text) + 50, len(context))
            matched_text = context[start_partial:end_pos]
            # Clean up the match
            matched_text = matched_text.split('\n')[0].strip()
            return start_partial, matched_text
    
    return None, None

def load_knowledge_base(directory):
    """
    Loads all .txt files from a directory into a dictionary.
    Key: filename, Value: text content.
    """
    knowledge_base = {}
    print(f"Loading knowledge base from: {directory}")
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_base[filename] = f.read()
                
    print(f"✅ Loaded {len(knowledge_base)} documents.")
    return knowledge_base

def create_squad_format_dataset(qa_pairs, knowledge_base):
    """
    Converts our custom Q&A format to the SQuAD format required by Hugging Face.
    Uses improved matching to find answers in context.
    """
    squad_data = []
    not_found_count = 0
    partial_matches = []

    print("Converting data to SQuAD format...")
    
    for qa in qa_pairs:
        question = qa["question"]
        answer_text = qa["answer"]
        
        context_found = False
        best_match_info = None
        
        # Search through all documents
        for doc_name, context in knowledge_base.items():
            start_idx, matched_text = find_best_match(answer_text, context)
            
            if start_idx is not None:
                squad_entry = {
                    "id": str(uuid.uuid4()),
                    "title": doc_name,
                    "context": context,
                    "question": question,
                    "answers": {
                        "text": [matched_text],
                        "answer_start": [start_idx]
                    }
                }
                squad_data.append(squad_entry)
                context_found = True
                
                # Log if we used a different text than the original answer
                if matched_text != answer_text:
                    partial_matches.append({
                        "question": question,
                        "original_answer": answer_text,
                        "matched_answer": matched_text,
                        "document": doc_name
                    })
                break
        
        if not context_found:
            not_found_count += 1
            print(f"⚠️  Answer not found in any document for question: '{question}'")
            # For debugging, let's see what the answer looks like
            print(f"   Looking for: '{answer_text[:100]}{'...' if len(answer_text) > 100 else ''}'")

    print(f"\n✅ Conversion complete. {len(squad_data)} examples created.")
    if partial_matches:
        print(f"📝 {len(partial_matches)} answers were matched with slight modifications.")
    if not_found_count > 0:
        print(f"❌ {not_found_count} answers could not be located in the knowledge base.")

    # Save partial matches for review
    if partial_matches:
        partial_matches_path = os.path.join(PROCESSED_DATA_DIR, "partial_matches_log.json")
        with open(partial_matches_path, 'w', encoding='utf-8') as f:
            json.dump(partial_matches, f, indent=2)
        print(f"📋 Partial matches saved to: {partial_matches_path}")

    return {"data": squad_data}

def debug_missing_answers(qa_pairs, knowledge_base):
    """
    Debug function to help identify why certain answers aren't being found.
    """
    print("\n🔍 Debugging missing answers...")
    
    missing_questions = [
        "Can NRI students have Indian students as their roommates?",
        "What is the punishment for having a mobile phone during a CAT exam?",
        "What happens if a student is caught with written notes during a CAT or MTT?",
        "What happens if a student misbehaves with an invigilator during a CAT exam?",
        "What happens if a student is caught in malpractice in a continuous assessment lab?",
        "What is the consequence of being caught with written chits during a Final Assessment Test (FAT)?",
        "What is the punishment for possessing a mobile phone during a FAT, even if it's switched off?",
        "What is the punishment for impersonation during a final assessment test?",
        "What happens if a student is involved in malpractice for the second time during a FAT?"
    ]
    
    for question in missing_questions:
        qa_item = next((qa for qa in qa_pairs if qa["question"] == question), None)
        if qa_item:
            print(f"\nQ: {question}")
            print(f"Expected Answer: {qa_item['answer']}")
            
            # Search for keywords in documents
            answer_words = qa_item['answer'].lower().split()
            for doc_name, context in knowledge_base.items():
                if any(word in context.lower() for word in answer_words[:3]):
                    print(f"  Found keywords in: {doc_name}")

if __name__ == '__main__':
    # Load the processed text files
    kb = load_knowledge_base(PROCESSED_DATA_DIR)
    
    # Load the question-answer pairs
    try:
        with open(QA_POC_PATH, 'r', encoding='utf-8') as f:
            qa_poc_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Q&A file not found at {QA_POC_PATH}")
        exit()

    # Run debugging for missing answers
    debug_missing_answers(qa_poc_data, kb)
    
    # Create the SQuAD formatted data
    final_squad_data = create_squad_format_dataset(qa_poc_data, kb)
    
    # Save the final dataset
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_squad_data, f, indent=4)
        
    print(f"\nSuccessfully saved SQuAD-formatted dataset to: {OUTPUT_PATH}")