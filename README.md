# Machine Unlearning Across Scales: Evaluation of Optimization Methods on Language Models

A comprehensive framework for machine unlearning in Large Language Models (LLMs), enabling models to selectively forget specific information while maintaining overall performance.

## Overview

This repository implements state-of-the-art machine unlearning techniques for Large Language Models, addressing critical concerns around data privacy, the "right to be forgotten", and ethical AI deployment. Our framework provides efficient alternatives to costly model retraining by selectively removing unwanted information from trained LLMs.

## Features

- **Multiple Unlearning Methods**: Implementation of various unlearning algorithms including gradient ascent, fine-tuning approaches, and representation-based methods
- **Comprehensive Evaluation**: Robust metrics for assessing unlearning efficacy across forget quality, model utility, and privacy preservation
- **Efficient Implementation**: Computationally efficient unlearning that avoids expensive retraining from scratch
- **Flexible Framework**: Support for different model architectures and datasets
- **Privacy-Preserving**: Techniques to ensure sensitive information is properly removed while maintaining model capabilities

## Installation

### Prerequisites
- Python 3.12.11
- PyTorch 2.8
- CUDA 12.6
- CUDA-compatible GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/amartyahatua/LLM_Unlearning.git
cd LLM_Unlearning

# Create conda environment with Python 3.12.11
conda create -n llm_unlearning python=3.12.11
conda activate llm_unlearning

# Install CUDA 12.6
conda install cuda=12.6 -c nvidia

# Install PyTorch 2.8 with CUDA 12.6 support
conda install pytorch=2.8 pytorch-cuda=12.6 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Basic Unlearning Example

```python
from llm_unlearning import UnlearningTrainer, load_model_and_tokenizer

# Load your pre-trained model
model, tokenizer = load_model_and_tokenizer("your-model-path")

# Initialize unlearning trainer
trainer = UnlearningTrainer(
    model=model,
    tokenizer=tokenizer,
    method="gradient_ascent",
    learning_rate=1e-5,
    num_epochs=3
)

# Perform unlearning
trainer.unlearn(
    forget_dataset="path/to/forget_data",
    retain_dataset="path/to/retain_data",
    output_dir="./unlearned_model"
)
```

### Command Line Interface

```bash
# Run unlearning with gradient ascent
python run_unlearning.py \
    --model_name_or_path "facebook/opt-1.3b" \
    --forget_dataset "data/forget_set.json" \
    --retain_dataset "data/retain_set.json" \
    --method "gradient_ascent" \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --output_dir "./outputs/unlearned_model"

# Evaluate unlearned model
python evaluate.py \
    --model_path "./outputs/unlearned_model" \
    --eval_datasets "forget,retain,general" \
    --output_file "evaluation_results.json"
```

## Supported Methods

### Unlearning Algorithms
- **Gradient Ascent**: Maximizes loss on forget set to reduce memorization
- **Fine-tuning with Random Labels**: Replaces target labels with random ones
- **Gradient Ascent + KL Divergence**: Balances forgetting with utility preservation
- **Representation Editing**: Modifies internal representations directly
- **Elastic Weight Consolidation (EWC)**: Protects important weights during unlearning

### Evaluation Metrics
- **Forget Quality**: Measures how well the model has forgotten target information
- **Model Utility**: Assesses performance on retained knowledge and general tasks
- **Privacy Metrics**: Evaluates resistance to membership inference attacks
- **Computational Efficiency**: Tracks unlearning time and resource usage

## Applications

### Data Privacy Compliance
- Remove personal information from trained models
- Comply with GDPR "right to be forgotten" requests
- Handle data withdrawal scenarios

### Content Moderation
- Unlearn toxic or harmful content
- Remove biased information
- Address copyright concerns

### Model Safety


## Contributing


### Development Setup


## Citation


## Related Work

- [TOFU: A Task of Fictitious Unlearning for LLMs](https://locuslab.github.io/tofu/)
- [OpenUnlearning: Unified Benchmarking Framework](https://github.com/locuslab/open-unlearning)
- [Awesome LLM Unlearning Resources](https://github.com/chrisliu298/awesome-llm-unlearning)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments


## Contact

For questions or support, please:
- Open an issue on GitHub
- Email: amartyahatua@gmail.com
---

**Disclaimer**: This tool is for research purposes. Please ensure compliance with applicable laws and regulations when using unlearning techniques on real-world data.
