"""
This script provides a command-line interface (CLI) for summarizing text using the Hugging Face Transformers library 
and the BART model. Users can input text directly or specify a file containing the text to be summarized.

Usage:
- To summarize text directly:
    python cli_transformer.py --text "Your text here"
- To summarize text from a file:
    python cli_transformer.py --file /path/to/your/file.txt
Dependencies:
- argparse: For parsing command-line arguments.
- transformers: For utilizing the pre-trained BART model for text summarization.
Example:
    python cli_transformer.py --text "Artificial intelligence is a branch of computer science that aims to create machines as intelligent as humans."
"""


import argparse
from transformers import pipeline


def create_parser():
    parser = argparse.ArgumentParser(description="Pass text or file input.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--text', type=str, help="Input text directly.")
    group.add_argument('-f', '--file', type=str, help="Path to input file.")
    return parser


def summarise_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=250, min_length=100, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as file:
            text = file.read()
    
    summary = summarise_text(text)
    print(summary)
