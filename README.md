# VIZOR: VIT Information & Query Answering Resource 📝

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An SLM-Powered Intelligent Student & Administrative Query Resolution System for VIT, built as part of an NLP course project.

## 🚀 Project Overview

VIZOR is a domain-specific NLP system designed to solve information access challenges at Vellore Institute of Technology (VIT). It leverages a custom-trained Small Language Model (SLM) to provide fast, accurate, and trustworthy answers to queries from students and administrators. The system is built with a focus on efficiency and deployability, avoiding the high costs and privacy concerns of large, general-purpose LLMs.

## ✨ Features

* **Question Answering:** Answer queries about the academic calendar, fee receipts, hostel allocation, etc.
* **Document Summarization:** Summarize circulars, handbooks, and meeting minutes.
* **Grievance Classification:** Automatically categorize student grievances.
* **Natural Language Interface:** Access timetables and book seminar rooms using natural language.

## 🏛️ Architecture

VIZOR uses a Retriever-Augmented Generation (RAG-lite) architecture.

1.  **Indexing:** All VIT documents are processed, chunked, and stored in a FAISS vector database.
2.  **Retrieval:** A user query retrieves the most relevant document chunks from the database.
3.  **Generation:** A custom-trained SLM (distilled from a BERT-base teacher) generates a factually grounded answer using the query and the retrieved context.

## 🛠️ Getting Started

### Prerequisites

* Python 3.9+
* Pip & Git

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/VIZOR.git
    cd VIZOR
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## 📜 Project Structure

```
VIZOR/
├── data/         # Datasets (raw, processed, augmented)
├── notebooks/    # Jupyter notebooks for exploration
├── src/          # Main source code
├── tests/        # Tests for the codebase
├── .gitignore    # Files to be ignored by Git
├── LICENSE       # Project License
├── README.md     # You are here!
└── requirements.txt # Project dependencies
```