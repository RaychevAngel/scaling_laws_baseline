from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Upload a model to Hugging Face Hub')
    parser.add_argument('--model_path', required=True, help='Path to the model')
    parser.add_argument('--repo_name', required=True, help='Repository name (format: username/repo-name)')
    
    # Parse arguments
    args = parser.parse_args()
    
    print(f"Loading model and tokenizer from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print(f"Uploading model and tokenizer to {args.repo_name}")
    model.push_to_hub(args.repo_name, commit_message="Upload fine-tuned model")
    tokenizer.push_to_hub(args.repo_name, commit_message="Upload tokenizer")
    
    print(f"Model successfully uploaded to: https://huggingface.co/{args.repo_name}")

if __name__ == "__main__":
    main()