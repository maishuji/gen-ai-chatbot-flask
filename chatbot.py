from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# We will use the `facebook/blenderbot-400M-distill` model because it has an open-source licennse and runs relatively fast
model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)