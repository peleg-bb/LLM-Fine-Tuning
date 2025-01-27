import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
import evaluate
import nltk
from concurrent.futures import ThreadPoolExecutor
import bert_score

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

class Evaluator:
    def __init__(self, base_model_path="tiiuae/falcon-mamba-7b", adapter_path="models/mamba-final"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(adapter_path).to(self.device).half()
        self.model.eval()
        
        self.meteor = evaluate.load('meteor')
        self.bertscore = evaluate.load('bertscore')
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])


    def generate_batch(self, questions, batch_size=2, max_length=256):
        torch.cuda.empty_cache()
        generated_answers = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            prompts = [f"Question: {q}\nAnswer:" for q in batch]
            inputs = self.tokenizer(prompts,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True, max_length=max_length).to(self.device)

            # Generate responses for A
            with torch.no_grad():
                outputs_a = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2
                )

            decoded_a = self.tokenizer.batch_decode(outputs_a, skip_special_tokens=True)

            # Generate responses for B (can tweak parameters slightly if needed)
            with torch.no_grad():
                outputs_b = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    temperature=0.8,  # Slightly different randomness setting
                    top_p=0.95,
                    repetition_penalty=1.1
                )

            decoded_b = self.tokenizer.batch_decode(outputs_b, skip_special_tokens=True)

            # Combine A/B responses for each question
            for a, b in zip(decoded_a, decoded_b):
                generated_answers.append({"A": a, "B": b})

        return generated_answers


    def compute_metrics_parallel(self, generated, reference):
        with ThreadPoolExecutor() as executor:
            bleu_future = executor.submit(lambda: sentence_bleu([reference.split()], generated.split()))
            rouge_future = executor.submit(lambda: self.scorer.score(reference, generated))
            meteor_future = executor.submit(
                lambda: self.meteor.compute(predictions=[generated], references=[reference])['meteor'])
            bertscore_future = executor.submit(lambda: np.mean(self.bertscore.compute(
                predictions=[generated],
                references=[reference],
                lang="en"
            )['f1']))

        return {
            'bleu': bleu_future.result(),
            'rouge_scores': rouge_future.result(),
            'meteor': meteor_future.result(),
            'bertscore': bertscore_future.result()
        }


def evaluate_model(n_samples=200, batch_size=4):
    # Load dataset
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    test_samples = dataset.select(range(len(dataset) - n_samples, len(dataset)))

    # Initialize evaluator
    evaluator = Evaluator()

    # Generate answers in batches
    questions = [sample['question'] for sample in test_samples]
    references = [sample['long_answer'] for sample in test_samples]

    print("Generating answers in batches...")
    generated_answers = evaluator.generate_batch(questions, batch_size)

    # Compute metrics
    all_results = []
    print("Computing metrics...")
    for question, generated_pair, reference in tqdm(zip(questions, generated_answers, references),
                                                    total=len(questions)):
        metrics_a = evaluator.compute_metrics_parallel(generated_pair['A'], reference)
        metrics_b = evaluator.compute_metrics_parallel(generated_pair['B'], reference)

        result = {
            'question': question,
            'generated': generated_pair,
            'reference': reference,
            'metrics': {
                'A': metrics_a,
                'B': metrics_b
            }
        }
        all_results.append(result)

    # Calculate averages for A and B
    avg_metrics = {
        'A': {
            'bleu': np.mean([r['metrics']['A']['bleu'] for r in all_results]),
            'rouge1': np.mean([r['metrics']['A']['rouge_scores']['rouge1'].fmeasure for r in all_results]),
            'rouge2': np.mean([r['metrics']['A']['rouge_scores']['rouge2'].fmeasure for r in all_results]),
            'rougeL': np.mean([r['metrics']['A']['rouge_scores']['rougeL'].fmeasure for r in all_results]),
            'meteor': np.mean([r['metrics']['A']['meteor'] for r in all_results]),
            'bertscore': np.mean([r['metrics']['A']['bertscore'] for r in all_results])
        },
        'B': {
            'bleu': np.mean([r['metrics']['B']['bleu'] for r in all_results]),
            'rouge1': np.mean([r['metrics']['B']['rouge_scores']['rouge1'].fmeasure for r in all_results]),
            'rouge2': np.mean([r['metrics']['B']['rouge_scores']['rouge2'].fmeasure for r in all_results]),
            'rougeL': np.mean([r['metrics']['B']['rouge_scores']['rougeL'].fmeasure for r in all_results]),
            'meteor': np.mean([r['metrics']['B']['meteor'] for r in all_results]),
            'bertscore': np.mean([r['metrics']['B']['bertscore'] for r in all_results])
        }
    }

    return all_results, avg_metrics


if __name__ == "__main__":
    print("Starting evaluation...")
    results, avg_metrics = evaluate_model()

    print("\nEVALUATION RESULTS")
    print("-" * 50)

    print("\nAVERAGE METRICS FOR A:")
    for metric, score in avg_metrics['A'].items():
        print(f"{metric:10s}: {score:.4f}")

    print("\nAVERAGE METRICS FOR B:")
    for metric, score in avg_metrics['B'].items():
        print(f"{metric:10s}: {score:.4f}")

    print("\nDETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        print(f"\nSample {i}:")
        print(f"Q: {result['question']}")
        print(f"A: {result['generated']['A']}")
        print(f"B: {result['generated']['B']}")
        print(f"Ref: {result['reference'][:200]}...")
        print("Metrics for A:")
        for metric, value in result['metrics']['A'].items():
            if metric == 'rouge_scores':
                for rouge_type, score in value.items():
                    print(f"{rouge_type}: {score.fmeasure:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
        print("Metrics for B:")
        for metric, value in result['metrics']['B'].items():
            if metric == 'rouge_scores':
                for rouge_type, score in value.items():
                    print(f"{rouge_type}: {score.fmeasure:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
        print("-" * 50)
