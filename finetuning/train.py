import nltk
import torch
import pandas as pd

from finetuning.LLMs import *
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from evaluate import load
from datasets import load_dataset
from transformers import  AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback, BitsAndBytesConfig


# ------------ Base Init -------------------------------------------------------------------------------

model_checkpoint = t5_large
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ----- CPU ------
#model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# ----- GPU ------
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
   model_checkpoint,
   device_map="auto",  # Automatically map layers to GPUs
   quantization_config = quantization_config,  # Use 8-bit precision for memory efficiency
)

# ------------ Dataset Handling -------------------------------------------------------------------------

# ---- Load ----------------------

dataset = load_dataset("pubmed_qa", "pqa_labeled")  
dataset = dataset.map(lambda x: {"question": x["question"], "answer": x["long_answer"]})

# ---- Process -------------------

def preprocess(examples):
    inputs = tokenizer(examples["question"], max_length=512, truncation=True)
    targets = tokenizer(examples["answer"], max_length=256, truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

# preprocess dataset
tokenized_datasets = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# split dataset
tokenized_datasets = tokenized_datasets["train"].select(range(40)).train_test_split(test_size=0.2)  # 20 % for validation


# ------------ LoRA --------------------------------------------------------------------------------------

model = prepare_model_for_kbit_training(model)  # Prepares the model for 8-bit fine-tuning

# Configure LoRA
lora_config = LoraConfig(
    r=16,                      # Rank of the LoRA update matrices
    lora_alpha=32,             # Scaling factor
    target_modules=["q", "v"],  # Target specific modules
    lora_dropout=0.1,          # Dropout for LoRA layers
    bias="none",               # Whether to tune bias terms
    task_type="SEQ_2_SEQ_LM",  # Task type for the model
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# ------------ Data Collator -----------------------------------------------------------------------------

data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model)

# ------------ Evaluation Metric -------------------------------------------------------------------------

nltk.download("punkt")
rouge_metric = load("rouge")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    # Ensure predictions are valid
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    predictions = [[max(0, min(token, tokenizer.vocab_size - 1)) for token in pred] for pred in predictions]

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {key: value.mid.fmeasure for key, value in result.items()}

    return result



# ------------ training argument --------------------------------------------------------------------------

training_args = Seq2SeqTrainingArguments(
    output_dir="models/seq2seq",
    eval_strategy="steps",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_total_limit=3,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    fp16=True,  
)


# ------------ LoRA trainer --------------------------------------------------------------------------------

# trainer
trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# callback
class Callback_logits(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):

        if "logits" in kwargs:
            logits = kwargs["logits"]
            predictions = torch.argmax(logits, dim=-1)

# integrate callback to trainer
trainer_callback = Callback_logits()
trainer.add_callback(Callback_logits())


# ------------ train the model ----------------------------------------------------------------------------

trainer.train()

# ------------ save the model -----------------------------------------------------------------------------

peft_model.save_pretrained("models/model")
tokenizer.save_pretrained("models/tokenizer")

