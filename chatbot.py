from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# We will use the `facebook/blenderbot-400M-distill` model because it has an open-source licennse and runs relatively fast
model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Keeping track of conversation history
conversation_history = []

while True:
    #Press ctrl-c to exit the conversation 
    
    history_string = "\n".join(conversation_history)

    input_text = input("> ")

    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    print(inputs)
    print(tokenizer.pretrained_vocab_files_map)

    outputs = model.generate(**inputs)
    print(outputs) # A dictionary containing the generated tokens ( not text)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)

    # Update conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    print(conversation_history)