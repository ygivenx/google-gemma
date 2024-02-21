from transformers import AutoTokenizer, pipeline
import torch

model = "google/gemma-2b-it"
device = "mps" # cpu, cuda

torch_dtype = ""
if device == "mps":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch_dtype},
    device=device
)

messages = [
        {"role": "user", "content": "What is a GLP-1 inhibitor?"},
]
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipeline(
    prompt,
    max_new_tokens=256,
    add_special_tokens=True,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
print(outputs[0]["generated_text"][len(prompt):])
