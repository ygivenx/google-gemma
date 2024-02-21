from transformers import AutoTokenizer, pipeline
import torch


def setup_pipeline(model, device):
    torch_dtype = ""
    if device == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch_dtype},
        device=device
    )

def main():
    model = "google/gemma-2b-it"
    device = "mps" # cpu, cuda

    try:
        chat_pipeline = setup_pipeline(model, device)
        while(True):
            print("Enter your text (Press Enter to exit): ", end="")
            content = input()
            if not content.strip():
                print("Ciao!")
                break
            prompt_message = f"""
                You are an expert medical analyst
                and have a knack for creating structured data
                from unstructured medical texts. You always list
                all the disease with their ICD-10 codes in
                the provided text.
            """
            messages = [{"role": "user", "content": prompt_message},
                        {"role": "model", "content": content}]
            prompt = chat_pipeline.tokenizer.apply_chat_template(messages,
                                                            tokenize=False,
                                                            add_generation_prompt=True)
            outputs = chat_pipeline(
                prompt,
                max_new_tokens=2048,
                add_special_tokens=True,
                do_sample=True,
                temperature=0.7,
                top_k=5,
                top_p=0.95
            )
            print(outputs[0]["generated_text"][len(prompt):])
    except KeyboardInterrupt:
        print("\n Process Interrupted. Exiting ...")

if __name__ == "__main__":
    main()