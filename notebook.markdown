## Cell 1: Generate real estate descriptions and save to CSV

```python
import pandas as pd
import random

# Load CSV file with real estate data
def load_data(file_path):
    return pd.read_csv(file_path)

# Lists for varied description phrasing
bedroom_phrases = ["spacious bedrooms", "cozy bedrooms", "well-appointed bedrooms"]
bathroom_phrases = ["modern bathrooms", "elegant bathrooms", "functional bathrooms"]
lot_phrases = ["ample outdoor space", "generous lot", "room for activities"]
value_phrases = ["excellent deal", "fantastic value", "priced to sell"]
appeal_phrases = ["perfect for families", "ideal for investors", "great for urban living"]
city_context = {
    "Adjuntas": "lush greenery and mountain views",
    "Juana Diaz": "rich cultural heritage",
    "Ponce": "historic architecture",
    "Mayaguez": "lively arts scene",
    "Richland": "scenic Columbia River"
}

# Estimate house size if missing (based on bedrooms and bathrooms)
def estimate_house_size(row):
    if pd.isna(row['house_size']):
        bed = row['bed'] if not pd.isna(row['bed']) else 2
        bath = row['bath'] if not pd.isna(row['bath']) else 1
        return bed * 300 + bath * 150  # Approx. 300 sq ft/bedroom, 150 sq ft/bathroom
    return row['house_size']

# Generate a descriptive text for each property
def generate_description(row):
    price = row['price'] if not pd.isna(row['price']) else "unknown"
    bed = int(row['bed']) if not pd.isna(row['bed']) else "unknown"
    bath = int(row['bath']) if not pd.isna(row['bath']) else "unknown"
    acre_lot = row['acre_lot'] if not pd.isna(row['acre_lot']) else "unknown"
    house_size = estimate_house_size(row)
    city = row['city'] if not pd.isna(row['city']) else "the city"
    state = row['state'] if not pd.isna(row['state']) else "the state"

    # Randomly select phrases for variety
    bedroom_phrase = random.choice(bedroom_phrases)
    bathroom_phrase = random.choice(bathroom_phrases)
    lot_phrase = random.choice(lot_phrases)
    value_phrase = random.choice(value_phrases) if price != "unknown" and price < 100000 else "competitive"
    appeal_phrase = random.choice(appeal_phrases)

    # Build description
    description = f"For sale in {city}, {state}, this property is a unique opportunity. "
    description += f"Located in {city}, {city_context.get(city, 'a welcoming community')}, it’s a gem. "
    if bed != "unknown":
        description += f"Features {bed} {bedroom_phrase}, "
    if bath != "unknown":
        description += f"and {bath} {bathroom_phrase}. "
    description += f"Spans ~{house_size:,.0f} sq ft. "
    if acre_lot != "unknown":
        description += f"On a {acre_lot:.2f}-acre lot with {lot_phrase}. "
    if price != "unknown":
        description += f"Listed at ${price:,.2f}, it’s {value_phrase}. "
    description += f"{appeal_phrase.capitalize()} in {city}."

    return description

# Process CSV and save with descriptions
def process_data(file_path, output_path):
    data = load_data(file_path)
    data['description'] = data.apply(generate_description, axis=1)
    output_data = data[['status', 'price', 'bed', 'bath', 'acre_lot', 'city', 'state', 'house_size', 'description']]
    output_data.to_csv(output_path, index=False)
    print(f"Descriptions saved to {output_path}")

# Run the processing
input_file = "realtor-data.zip.csv"
output_file = "realtor-data-with-descriptions.csv"
process_data(input_file, output_file)
```

**Output**:
```
Descriptions saved to realtor-data-with-descriptions.csv
```

---

## Cell 2: Load and split data for model training

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load descriptions from CSV
data_path = "realtor-data-with-descriptions.csv"
descriptions = pd.read_csv(data_path)['description'].dropna().tolist()

# Split data into train (80%), validation (10%), and test (10%) sets
train_texts, temp_texts = train_test_split(descriptions, test_size=0.2, random_state=42)
val_texts, test_texts = train_test_split(temp_texts, test_size=0.5, random_state=42)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})
test_dataset = Dataset.from_dict({"text": test_texts})

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")
```

**Output** (example, depends on data size):
```
Training samples: 8000
Validation samples: 1000
Test samples: 1000
```

---

## Cell 3: Load model and tokenizer

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print("Model and tokenizer loaded!")
```

**Output**:
```
Model and tokenizer loaded!
```

---

## Cell 4: Tokenize datasets

```python
from datasets import Dataset

# Tokenize function to prepare data for training
def tokenize_function(examples):
    encodings = tokenizer(examples["text"], padding=True, truncation=True, max_length=256)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set format for PyTorch
train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

print(f"Tokenized datasets ready! Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
```

**Output** (example, depends on data size):
```
Tokenized datasets ready! Train: 8000, Val: 1000, Test: 1000
```

---

## Cell 5: Apply LoRA configuration

```python
from peft import LoraConfig, get_peft_model

# Define LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # Low-rank matrix rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Transformer modules to adapt
    lora_dropout=0.1,  # Dropout for regularization
    task_type="CAUSAL_LM"  # Causal language modeling
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

print("LoRA applied to model!")
```

**Output**:
```
LoRA applied to model!
```

---

## Cell 6: Set up and run training

```python
from transformers import TrainingArguments, Trainer
import torch

# Define custom collate function for batch processing
def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}

# Define training arguments
training_args = TrainingArguments(
    output_dir="model_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=200,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=custom_collate_fn
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model("final_model")
print("Training completed! Model saved to final_model")
```

**Output** (example, simplified):
```
Training completed! Model saved to final_model
```

**Note**: Training may produce additional logs (e.g., loss values), not shown here for brevity.

---

## Cell 7: Streamlit UI for real estate queries

```python
import streamlit as st
from transformers import pipeline
from peft import PeftModel
import torch

# Set Streamlit page configuration
st.set_page_config(page_title="Real Estate Expert", page_icon="🏠")

# Load fine-tuned model
@st.cache_resource
def load_model():
    base_model_name = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")
    lora_model = PeftModel.from_pretrained(model, "final_model", torch_dtype=torch.float16)
    return tokenizer, lora_model

tokenizer, lora_model = load_model()

# Set up text generation pipeline
text_gen = pipeline("text-generation", model=lora_model, tokenizer=tokenizer, device_map="auto")

# Define system prompt for real estate expertise
system_prompt = (
    "You are a U.S. real estate expert specializing in property price evaluation. "
    "You provide detailed, data-driven price assessments based on location, property condition, and market trends."
)

# Streamlit UI
st.title("🏠 Real Estate Expert")
st.markdown("Ask about property price evaluations or market trends.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Enter your real estate question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            full_prompt = f"{system_prompt}\nUser: {prompt}\nAssistant:"
            response = text_gen(full_prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
            answer = response.split("Assistant:")[-1].strip()
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
```

**Output**: No direct text output since this is a Streamlit app. When run, it launches a web interface.

**Note**: To view the Streamlit UI, run `streamlit run this_file.py` in your environment.

---

### Notes on Images
- The notebook doesn’t generate images (e.g., no Matplotlib plots or saved figures). If images were expected (e.g., from a plotting library), please clarify, and I can add placeholders or generate sample plots.
- Placeholder for images would look like:
  ```markdown
  ![Generated Image](path/to/image.png)
  ```
  Let me know if you need specific image handling.

### Key Details
- **Artifact ID**: New UUID (`f4d2a7c9-3e5a-4b2f-9e1d-8b7c4f5e2a1b`) since this is a Markdown file, distinct from the `.ipynb` artifact.
- **Content Type**: Set to `text/markdown` for the Markdown format.
- **Preserved Content**: All code cells and their outputs (where applicable) are included. Outputs are example-based since the notebook doesn’t specify exact data sizes or training logs.
- **Readability**: Used `## Cell X` headings to separate cells and formatted code/output blocks for GitHub compatibility.
- **Dependencies**: Include in `requirements.txt`:
  ```
  pandas
  scikit-learn
  datasets
  transformers
  peft
  torch
  streamlit
  ```
- **File Paths**: Ensure `realtor-data.zip.csv` is available and `final_model` exists for the Streamlit cell. Adjust paths if needed.
- **GitHub**: Save this as `realtor_notebook.md` in your repo. Add a `README.md` with setup instructions (e.g., installing dependencies, running Streamlit).

### How to Use
1. Copy the content within the `<xaiArtifact>` tag into a file named `realtor_notebook.md`.
2. View it on GitHub or a Markdown viewer for proper rendering.
3. To run the code, copy each code block into a Python environment or a `.py` file (especially for the Streamlit app).
4. For the Streamlit cell, save it as `app.py` and run `streamlit run app.py`.

If you need further adjustments (e.g., adding specific outputs, handling images, or reverting to `.ipynb` with fixes), please let me know!