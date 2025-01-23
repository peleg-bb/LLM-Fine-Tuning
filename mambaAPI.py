from gradio_client import Client

import datasets
from datasets import load_dataset

# dataset = load_dataset("pubmed_qa", "pqa_labeled")
#
#
# dataset = dataset["train"].train_test_split(
#         test_size=0.2,
#     	seed=42)
#
# dataset.get("train").to_csv("train.csv")
# dataset.get("test").to_csv("test.csv")

# read the csv file
import pandas as pd
df = pd.read_csv("train.csv")

# save the answers in a dictionary
answers = {}


client = Client("tiiuae/falcon-mamba-playground")

for i in range(10):
    result = client.predict(
            message=df["question"][i] + df["context"][i],
            temperature=0.3,
            max_new_tokens=1024,
            top_p=1,
            top_k=20,
            penalty=1.2,
            api_name="/chat"
    )
    answers[df["question"][i]] = result
    print(result)

print(answers)