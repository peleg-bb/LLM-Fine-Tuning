# Standard imports
import torch
import logging
import sys
import traceback
import os
from datetime import datetime
from datasets import load_dataset
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from LLMs import falcon_mamba_7b

def setup_logging():
    """Configure detailed logging for monitoring training dynamics"""
    os.makedirs('logs', exist_ok=True)
    log_filename = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def filter_quality_samples(examples, min_length=10, max_length=2000):
    """Filter dataset with relaxed constraints"""
    mask = [
        len(q.split()) >= 3 and  # Question has substance
        len(a.split()) >= min_length and  # Answer has substance
        len(a.split()) <= max_length and  # Not too long
        not any(bad_word in q.lower() for bad_word in ['error', '404', 'n/a'])  # Basic error checking
        for q, a in zip(examples["question"], examples["long_answer"])
    ]
    return mask

def preprocess_function(examples, tokenizer, max_length=512):
    """Prepare inputs with careful handling of length and special tokens"""
    # Format text with clear separation tokens
    formatted_texts = [
        f"Question: {q.strip()}\nAnswer: {a.strip()}\n{tokenizer.eos_token}"
        for q, a in zip(examples["question"], examples["long_answer"])
    ]
    
    # Tokenize with dynamic length adjustment
    tokenized = tokenizer(
        formatted_texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    # Create labels with proper masking
    labels = tokenized["input_ids"].clone()
    # Mask padding tokens in labels
    labels[labels == tokenizer.pad_token_id] = -100
    tokenized["labels"] = labels
    
    return tokenized

class SafetyCallback(EarlyStoppingCallback):
    """Extended callback with additional safety checks"""
    def __init__(self, early_stopping_patience=3, max_loss=5.0):
        super().__init__(early_stopping_patience)
        self.max_loss = max_loss
        self.prev_losses = []
        
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """Monitor training dynamics for instability"""
        if logs is None or "loss" not in logs:
            return control
        
        current_loss = logs["loss"]
        self.prev_losses.append(current_loss)
        
        # Stop if loss is too high
        if current_loss > self.max_loss:
            control.should_training_stop = True
        
        # Stop if loss is unstable
        if len(self.prev_losses) > 5:
            variance = np.var(self.prev_losses[-5:])
            if variance > 2.0:  # High variance threshold
                control.should_training_stop = True
        
        return control

def train_model():
    try:
        logger = setup_logging()
        
        # 1. Load tokenizer with safety settings
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(falcon_mamba_7b)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'  # Consistent padding
        
        # 2. Load model with conservative quantization
        logger.info("Loading model...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            falcon_mamba_7b,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        model.config.use_cache = False
        
        # 3. Ultra-conservative LoRA setup
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=8,  # Minimal rank
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["in_proj"],  # Minimal module targeting
            inference_mode=False,
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        
        # 4. Load and filter dataset
        logger.info("Loading dataset...")
        full_dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        
        # Apply quality filters
        quality_mask = filter_quality_samples(full_dataset)
        filtered_dataset = full_dataset.select([i for i, x in enumerate(quality_mask) if x])
        
        # Select small, high-quality sample
        indices = torch.randperm(len(filtered_dataset))[:5].tolist()  # Minimal dataset
        dataset = filtered_dataset.select(indices)
        
        # Split into train/val
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        # 5. Preprocess datasets
        logger.info("Preprocessing dataset...")
        train_dataset = split_dataset["train"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        val_dataset = split_dataset["test"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 6. Conservative training settings
        training_args = TrainingArguments(
            output_dir="models/mamba",
            per_device_train_batch_size=1,  # Minimal batch size
            gradient_accumulation_steps=8,  # Increased for stability
            learning_rate=5e-5,  # Very conservative
            max_steps=10,  # Very minimal training - 2 passes per sample
            warmup_ratio=0.2,  # Extended warmup
            save_steps=10,
            logging_steps=1,  # Frequent monitoring
            fp16=True,
            optim="paged_adamw_32bit",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            evaluation_strategy="steps",
            eval_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Weight decay and clipping
            weight_decay=0.01,
            max_grad_norm=0.5
        )
        
        # 7. Initialize trainer with safety measures
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[SafetyCallback(early_stopping_patience=2, max_loss=5.0)]
        )
        
        # 8. Train with extra monitoring
        logger.info("Starting training...")
        trainer.train()
        
        # 9. Evaluate before saving
        logger.info("Final evaluation...")
        eval_results = trainer.evaluate()
        if eval_results["eval_loss"] > 3.0:
            logger.warning("High final loss - model may be unstable")
            return
        
        # 10. Save model if safe
        logger.info("Saving model...")
        model.save_pretrained("models/mamba-final")
        tokenizer.save_pretrained("models/mamba-tokenizer")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    train_model()