import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from huggingface_hub import login
from sacrebleu import corpus_bleu,corpus_chrf, corpus_ter

df = pd.read_excel("cleaned_en_hi_dataset.xlsx")
login()
english_sentences = df["English"].head(100).tolist()

translator=pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
translations = translator(english_sentences, max_length=512)

translations = [translator(sent)[0]['translation_text'] for sent in english_sentences]

output_df = pd.DataFrame({
    "English": english_sentences,
    "Model Generated Hindi Translations": translations,})
output_df.to_excel("model_generated_translations.xlsx", index=False)

references = df["Hindi"].head(100).tolist()
with open("references.txt", "w", encoding="utf-8") as f:
    f.write(f"BLEU: {corpus_bleu(translations, [references]).score}\n")
    f.write(f"CHRF: {corpus_chrf(translations, [references]).score}\n")
    f.write(f"TER: {corpus_ter(translations, [references]).score}\n")

