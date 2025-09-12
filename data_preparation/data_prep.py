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


def data_preparation_retain():
    alpaca_prompt = """Answer the following question:
### Question:
{}

### Answer:
{}"""

    def formatting_prompts_func(examples):
        texts = [alpaca_prompt.format(question, answer) for question, answer in
                 zip(examples["question"], examples["answer"])]
        return {"text": texts}

    # Load the dataset
    dataset_retain_90 = load_dataset("locuslab/TOFU", "retain90", split="train")
    dataset_retain_90 = dataset_retain_90.map(formatting_prompts_func, batched=True)
    return dataset_retain_90


def data_preparation_forget():
    alpaca_prompt = """Answer the following question:
### Question:
{}

### Answer:
{}"""

    def formatting_prompts_func(examples):
        texts = [alpaca_prompt.format(question, answer) for question, answer in
                 zip(examples["question"], examples["answer"])]
        return {"text": texts}

    # Load the dataset
    dataset_forget_10 = load_dataset("locuslab/TOFU", "forget10", split="train")
    dataset_forget_10 = dataset_forget_10.map(formatting_prompts_func, batched=True)
    return dataset_forget_10

def data_preparation_holdout():
    alpaca_prompt = """Answer the following question:
### Question:
{}

### Answer:
{}"""

    def formatting_prompts_func(examples):
        texts = [alpaca_prompt.format(question, answer) for question, answer in
                 zip(examples["question"], examples["answer"])]
        return {"text": texts}

    # Load the dataset
    dataset_holdout01 = load_dataset("locuslab/TOFU", "holdout01", split="train")
    dataset_holdout01 = dataset_holdout01.map(formatting_prompts_func, batched=True)
    return dataset_holdout01
