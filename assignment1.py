from datasets import load_dataset
import pandas as pd

from huggingface_hub import login
login()

ds = load_dataset("ainlpml/english-hindi")
df = pd.DataFrame(ds["train"])

output_path = "en_hi_dataset.xlsx"
df.to_excel(output_path, index=False)

# Inspect few rows
print(df.head())

paired_df = pd.DataFrame({
    "English" :df.iloc[:10000,0].values,
    "Hindi"   :df.iloc[10000:,0].values
})

def lenofwords(text):
    return len(str(text).split())

paired_df["WordCount_English"] = paired_df["English"].apply(lenofwords)
paired_df["WordCount_Hindi"]   = paired_df["Hindi"].apply(lenofwords)

# Keep only rows where both are 5–50 words
paired_df = paired_df[
    (paired_df["WordCount_English"].between(5, 50)) &
    (paired_df["WordCount_Hindi"].between(5, 50))
]

paired_df["Count_Diff"] = paired_df["WordCount_English"] - paired_df["WordCount_Hindi"]
paired_df = paired_df[paired_df["Count_Diff"].between(-10, 10)]

output_path = "cleaned_en_hi_dataset.xlsx"
paired_df.to_excel(output_path, index=False)

print(f"Cleaned dataset saved at: {output_path}")
print(paired_df.head())
