import torch
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device being used: {device}")

from transformers import pipeline, AutoTokenizer, set_seed, AutoModelForCausalLM

set_seed(42)

model_name = input("Enter path to lora checkpoint: ") #'microsoft/Phi-3.5-mini-instruct'
# model_name = "G:\\Masterarbeit\\phi-3_5_finetuned_lora\\checkpoint-375"

print(f"Model being used: {model_name}")


model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda",
                                             attn_implementation="flash_attention_2",
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                )
system_message = input("System message: ")
if system_message == "":
    chat = []
else:
    chat = [
        {"role": "system", "content": system_message}, #"You are a helpful AI assistant."
    ]

print(f"System message: {chat}")

while True:
    user_input = input("Dein Input: ")
    print("")

    if user_input.lower() == "exit":
        print("Chat beendet.")
        break

    chat.append({"role": "user", "content": user_input})

    generation_args = {
        "max_new_tokens": 512,
        "return_full_text": False,
    }
    start = datetime.datetime.now()

    outputs = pipe(chat, **generation_args)[0]['generated_text']
    print(f"Phi 3.5: {outputs}")
    print("")
    end = datetime.datetime.now()
    print(f"Diese Antwort hat {str(end-start).split('.')[0]} gedauert.")
    print("")

    # Antwort anhängen
    chat.append({"role": "assistant", "content": outputs[-1]})


