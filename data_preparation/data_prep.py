from datasets import load_dataset

def data_preparation_full():
    alpaca_prompt = """Answer the following question:
### Question:
{}

### Answer:
{}"""
    
    def formatting_prompts_func(examples):
        texts = [alpaca_prompt.format(question, answer) for question, answer in zip(examples["question"], examples["answer"])]
        return {"text": texts}
    
    # Load the dataset
    dataset = load_dataset("locuslab/TOFU", "full", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
