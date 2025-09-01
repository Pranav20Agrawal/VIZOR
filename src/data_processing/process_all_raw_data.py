import os
import shutil
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

# Tesseract's Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Project's Root Directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Source and Destination Folders
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
POPPLER_PATH = r"C:\Users\agpra\Downloads\Softwares\poppler-25.07.0\Library\bin"

def ocr_pdf(pdf_path):
    """
    Performs OCR on an image-based PDF and returns the extracted text.
    """
    try:
        # 1. Convert PDF pages to a list of images
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        
        # 2. Initialize an empty string to hold all text
        full_text = ""
        
        # 3. Loop through each image (page) and run OCR
        for img in images:
            # image_to_string is the core OCR function from pytesseract
            page_text = pytesseract.image_to_string(img, lang = 'eng')    # Specify English language
            full_text += page_text + "\n"
        
        return full_text
    except Exception as e:
        print(f"    OCR Error: {e}")
        return ""

def process_raw_files():
    """
    Scans the raw data directory, processes PDFs and copies TXTs,
    and saves the clean text output to the processed data directory.
    """
    print(f"Starting to process files from: {RAW_DATA_DIR}")
    
    # Ensure the destination folder exists
    # If data/processed doesn't exist, this line will create it.
    os.makedirs(PROCESSED_DATA_DIR, exist_ok = True)
    
    # --- Step 1: Get the list of all files in data/raw ---
    try:
        # os.listdir gives a list like ['file1.pdf', 'file2.txt', ...]
        all_filenames = os.listdir(RAW_DATA_DIR)
    except FileNotFoundError:
        print(f"❌ Error: Raw data directory not found at {RAW_DATA_DIR}")
        return
    
    # --- Step 2: Loop through each file ---
    for filename in all_filenames:
        # Create the full path to the source file
        source_path = os.path.join(RAW_DATA_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
        
        # --- Step 3: Check if the file is a PDF ---
        if filename.lower().endswith('.pdf'):
            print(f"Processing PDF: {filename}")
            text_content = ""
            
            # Try normal extraction first
            try:
                # --- Step 4: Logic for PDF files ---
                reader = PdfReader(source_path)
                for page in reader.pages:
                    text_content += page.extract_text() + '\n'
            except Exception as e:
                print(f"  Standard extraction failed: {e}")
                text_content = ""    # Ensure it's empty if it fails
                
            # Check if extraction failed
            # If the text is very short, it's likely an image PDF. Let's use OCR.
            if len(text_content.strip()) < 100:     # A threshold to detect failure
                print("  Standard extraction yielded little/no text. Switching to OCR...")
                text_content = ocr_pdf(source_path)
            
            # Save the final text content (either from standard or OCR)
            with open(output_path, "w", encoding = "utf-8") as f:
                f.write(text_content)
            print(f"  ✅ Saved processed text to: {output_path}")
        
        # --- Step 3 (cont.): Check if the file is a TXT ---
        elif filename.lower().endswith('.txt'):
            print(f"Copying TXT: {filename}")
            try:
                # --- Step 5: Logic for TXT files ---
                # The destination path is simply the processed directory
                destination_path = os.path.join(PROCESSED_DATA_DIR, filename)
                shutil.copy(source_path, destination_path)
                print(f"  ✅ Copied to: {destination_path}")
            except Exception as e:
                print(f"  ❌ Failed to copy {filename}. Error: {e}")
        
        else:
            # This will ignore any other file types there, like .png or .docx
            print(f"Skipping unsupported file: {filename}")
    
    print("\nProcessing complete!")

# This makes the script runnable directly from the command line
if __name__ == '__main__':
    process_raw_files()