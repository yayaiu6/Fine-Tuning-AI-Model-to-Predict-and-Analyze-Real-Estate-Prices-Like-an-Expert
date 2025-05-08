# Real Estate Price Evaluation Expert

This repository contains a Jupyter Notebook that implements a pipeline for generating real estate property descriptions, fine-tuning a language model with LoRA to specialize as a real estate price evaluation expert. The project is designed to be simple, modular, and focused on leveraging AI for accurate real estate price assessments.

## Project Overview

The goal is to create a tool that:

1. **Generates Property Descriptions**: Automatically produces engaging property descriptions from real estate data (e.g., price, bedrooms, location).
2. **Fine-Tunes a Model as a Price Expert**: Uses LoRA (Low-Rank Adaptation) to fine-tune Google's Gemma-3-1B-IT model, specializing it in real estate price evaluation and market trend analysis.
3. **Deploys a User Interface**: Provides a Streamlit web app where users can ask about property prices or market trends, powered by the fine-tuned model.

The project is implemented in a Jupyter Notebook (`notebook.ipynb`) with clear, commented cells for readability. The code is streamlined to prioritize LoRA-based fine-tuning and price expertise, making it ideal for educational purposes or small-scale deployment.

## Why This Approach?

- **LoRA for Fine-Tuning**: LoRA enables efficient fine-tuning by adapting only a small subset of model parameters, making it feasible to specialize a large language model on real estate data using modest hardware.
- **Price Evaluation Expertise**: Fine-tuning on real estate descriptions and price-related data ensures the model provides accurate, context-aware price assessments, critical for real estate professionals and buyers.
- **Streamlit UI**: A web-based interface allows non-technical users (e.g., agents, investors) to query the model for price evaluations without needing coding skills.
- **Simplified Code**: The notebook avoids unnecessary complexity (e.g., multi-GPU support, excessive error handling) to focus on LoRA, training, and price specialization.
- **Open-Source**: Hosting on GitHub encourages community contributions and showcases AI applications in real estate.

## Notebook Structure and Rationale

The Jupyter Notebook has seven cells, each contributing to the pipeline. Below, I detail each cell, with extra emphasis on LoRA configuration, model training, and price expertise specialization.

### Cell 1: Generate Real Estate Descriptions

- **What it does**: Loads a CSV (`realtor-data.zip.csv`), generates varied property descriptions (e.g., "For sale in Ponce, Puerto Rico, with 3 spacious bedrooms..."), and saves them to `realtor-data-with-descriptions.csv`.
- **How it works**:
  - Uses `pandas` to load data.
  - Applies randomized phrases (e.g., "modern bathrooms") and city-specific context (e.g., "Ponce, known for historic architecture").
  - Estimates missing house sizes (300 sq ft/bedroom, 150 sq ft/bathroom).
  - Saves key columns (price, bedrooms, etc.) with descriptions.
- **Why this way**:
  - **Data foundation**: Descriptions provide text data for fine-tuning, embedding price and property details to train the model as a price expert.
  - **Simplicity**: The heuristic for house size and randomized phrases ensures varied, usable data without complex logic.

### Cell 2: Load and Split Data for Training

- **What it does**: Loads descriptions from `realtor-data-with-descriptions.csv`, splits them into training (80%), validation (10%), and test (10%) sets, and converts to Hugging Face `Dataset` objects.
- **How it works**:
  - Uses `pandas` and `sklearn` for splitting.
  - Creates `Dataset` objects for compatibility with `transformers`.
- **Why this way**:
  - **Training readiness**: The split ensures robust training and evaluation, critical for fine-tuning the model on price-related text.
  - **Hugging Face integration**: `Dataset` objects streamline tokenization and training.

### Cell 3: Load Model and Tokenizer

- **What it does**: Loads Google’s Gemma-3-1B-IT model and tokenizer, configures for 16-bit precision, and sets padding.
- **How it works**:
  - Uses `transformers.AutoTokenizer` and `AutoModelForCausalLM`.
  - Sets `torch_dtype=torch.float16` and `device_map="auto"` for efficiency.
  - Assigns the end-of-sequence token as the padding token.
- **Why this way**:
  - **Model choice**: Gemma-3-1B-IT is lightweight yet powerful, ideal for fine-tuning as a price evaluation expert.
  - **Efficiency**: 16-bit precision reduces memory needs, supporting LoRA’s resource-efficient fine-tuning.

### Cell 4: Tokenize Datasets

- **What it does**: Tokenizes the datasets for training, setting a max length of 256 tokens.
- **How it works**:
  - Defines a `tokenize_function` for padding, truncation, and label creation.
  - Applies tokenization in batches and sets PyTorch tensor format.
- **Why this way**:
  - **Model compatibility**: Tokenization prepares data for the model, with 256 tokens balancing detail and efficiency for price-focused text.
  - **Batched processing**: Speeds up preparation for training.

### Cell 5: Apply LoRA Configuration

- **What it does**: Configures LoRA to fine-tune the model efficiently, targeting price evaluation expertise.
- **How it works**:
  - Defines a `LoraConfig` with:
    - `r=8`: Low-rank matrix rank for minimal parameter updates.
    - `lora_alpha=32`: Scaling factor to balance adaptation strength.
    - `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`: Focuses on attention layers, key for text generation and price context understanding.
    - `lora_dropout=0.1`: Regularization to prevent overfitting.
    - `task_type="CAUSAL_LM"`: Aligns with causal language modeling for description generation.
  - Applies LoRA using `peft.get_peft_model`.
- **Why this way**:
  - **Efficiency**: LoRA updates only \~1% of parameters, enabling fine-tuning on a single GPU or CPU, critical for accessibility.
  - **Targeted adaptation**: Attention layers are prioritized because they handle contextual relationships, essential for understanding price-related patterns (e.g., location, bedrooms, market trends).
  - **Price expertise**: Fine-tuning with LoRA on real estate descriptions embeds price evaluation knowledge, making the model a specialized expert.
  - **Regularization**: Dropout ensures the model generalizes to new price queries.

### Cell 6: Set Up and Run Training

- **What it does**: Trains the LoRA-adapted model to specialize in price evaluation and saves it to `final_model`.
- **How it works**:
  - Defines a `custom_collate_fn` for batch processing.
  - Sets `TrainingArguments`:
    - 3 epochs: Sufficient to adapt the model without overfitting.
    - Batch size 8: Balances memory usage and training speed.
    - Learning rate 2e-4: Ensures stable convergence.
    - Eval every 1000 steps: Monitors performance to save the best model (based on validation loss).
    - FP16: Reduces memory footprint.
  - Uses `transformers.Trainer` for training and saves the model.
- **Why this way**:
  - **Price specialization**: Training on descriptions with price data (e.g., "Listed at $250,000, it’s a great deal") teaches the model to associate property features with prices, enhancing its expertise.
  - **Efficiency**: FP16 and modest batch size make training feasible on standard hardware, aligning with LoRA’s low-resource approach.
  - **Evaluation**: Regular evaluation ensures the model learns price-relevant patterns without overfitting.
  - **Simplicity**: The streamlined setup focuses on training for price expertise, avoiding complex configurations.

### Cell 7: Streamlit UI for Price Evaluation Queries

- **What it does**: Deploys a Streamlit app where users can ask price-related questions (e.g., "What’s the price for a 3-bedroom house in Ponce?") using the fine-tuned model.
- **How it works**:
  - Loads the LoRA-adapted model (`final_model`) and tokenizer.
  - Sets a text generation pipeline with a system prompt: "You are a U.S. real estate expert specializing in property price evaluation."
  - Implements a chat-style UI with conversation history, generating responses with `temperature=0.7` for balanced creativity.
- **Why this way**:
  - **Price focus**: The system prompt ensures the model prioritizes price evaluation, leveraging its fine-tuned expertise.
  - **Accessibility**: Streamlit’s simple interface lets users query prices without technical knowledge.
  - **Fine-tuned performance**: The LoRA-adapted model provides accurate, context-aware price assessments, trained on real estate data.

## 

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, open an issue on GitHub. Enjoy using the Real Estate Price Evaluation Expert!