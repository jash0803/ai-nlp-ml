# AI-NLP-ML

## Setup Instructions
python3 -m venv myenv
source myenv/bin/activate  
pip install -r requirements.txt 

## Assignment 1 
Use command 'python assignment1.py' to run the code for Assignment 1.
It fetches the dataset from HuggingFace and first saves it in en_hi_dataset.xlsx file. Then it cleans the dataset based on word count criteria and saves the cleaned dataset in cleaned_en_hi_dataset.xlsx file.

## Assignment 2
Use command 'python assignment2.py' to run the code for Assignment 2.
It uses the cleaned dataset from Assignment 1 and then uses Helsinki-NLP/opus-mt-en-hi model to translate English to Hindi. It then calculates BLEU score, CHRF score and TER score and prints it.
