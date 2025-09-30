# debug_train_minimal.py
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import datasets

# tiny toy dataset
data = [
  {"input": "The cat sat on the mat.", "target": "Cat on mat."},
  {"input": "A quick brown fox jumps over the lazy dog.", "target": "Fox jumps over dog."},
  {"input": "Streamlit is an app framework for ML.", "target": "Streamlit for ML apps."}
]

hf = datasets.Dataset.from_list(data)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

print("Loading tokenizer and model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Loaded model OK")

def preprocess(x):
    inpt = tokenizer(x["input"], truncation=True, padding="max_length", max_length=64)
    tgt = tokenizer(x["target"], truncation=True, padding="max_length", max_length=32)
    inpt["labels"] = tgt["input_ids"]
    return inpt

print("Tokenizing dataset...")
tokenized = hf.map(preprocess, batched=True)
print("Tokenized examples:", len(tokenized))
print("Sample tokenized example keys:", tokenized[0].keys())

training_args = Seq2SeqTrainingArguments(
    output_dir="./tmp_debug",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_steps=1,
    report_to="none",
    predict_with_generate=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

print("Starting trainer.train() — this will run on CPU; it may take 10–60s depending on downloads.")
t0 = time.time()
train_result = trainer.train()
t1 = time.time()
print("trainer.train() finished in %.1fs" % (t1 - t0))
print("Train result keys:", train_result.keys())
print("Done.")
