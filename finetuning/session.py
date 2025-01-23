import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse
import sys

class QASession:
    def __init__(self, base_model_path="tiiuae/falcon-mamba-7b", adapter_path="models/mamba-model", tokenizer_path="models/mamba-tokenizer"):
        """
        Initialize the QA session with the Mamba model and LoRA adapter
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map="auto"
        )
        
        self.model.eval()
        print("Model loaded successfully!")

    def generate_answer(self, question, max_length=256):
        """Generate an answer for the given question"""
        prompt = f"Question: {question.strip()}\nAnswer:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
            return_attention_mask=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_response.split("Answer:")[-1].strip()
            return answer
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return "Sorry, I encountered an error while generating the answer."

def main():
    parser = argparse.ArgumentParser(description="QA Session with Mamba model")
    parser.add_argument("--base-model", default="tiiuae/falcon-mamba-7b", help="Path to base model")
    parser.add_argument("--adapter-path", default="models/mamba-model", help="Path to LoRA adapter")
    parser.add_argument("--tokenizer-path", default="models/mamba-tokenizer", help="Path to tokenizer")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--input-file", help="File containing questions (one per line)")
    parser.add_argument("--output-file", help="File to write answers")
    
    args = parser.parse_args()
    
    try:
        print("\nInitializing QA Session...")
        session = QASession(args.base_model, args.adapter_path, args.tokenizer_path)
        
        if args.interactive:
            print("\nStarting interactive session (type 'quit' to exit)")
            print("-" * 50)
            
            while True:
                try:
                    question = input("\nYour question: ").strip()
                    
                    if question.lower() in ['quit', 'exit', 'q']:
                        print("\nEnding session...")
                        break
                        
                    if not question:
                        print("Please enter a valid question.")
                        continue
                        
                    print("\nGenerating answer...\n")
                    answer = session.generate_answer(question)
                    print("Answer:", answer)
                    print("-" * 50)
                except EOFError:
                    print("\nDetected non-interactive environment. Exiting...")
                    break
        
        elif args.input_file:
            # Clear the output file if it exists
            if args.output_file:
                open(args.output_file, 'w').close()
            
            with open(args.input_file, 'r') as f:
                questions = f.readlines()
            
            print(f"\nProcessing {len(questions)} questions from {args.input_file}")
            
            for i, question in enumerate(questions, 1):
                question = question.strip()
                if question:
                    print(f"\nProcessing question {i}/{len(questions)}")
                    print(f"Question: {question}")
                    answer = session.generate_answer(question)
                    print(f"Answer: {answer}\n")
                    
                    if args.output_file:
                        with open(args.output_file, 'a') as f:
                            f.write(f"{answer}\n")
            
            if args.output_file:
                print(f"\nAnswers saved to {args.output_file}")
        
        else:
            print("Error: Must specify either --interactive or --input-file")
            sys.exit(1)

    except Exception as e:
        print(f"Error in session: {str(e)}")
        raise

if __name__ == "__main__":
    main()