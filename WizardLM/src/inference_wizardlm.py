import sys
import os
import fire
import torch
import transformers
import json

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def evaluate(
        batch_data,
        input=None,
        temperature=1,
        top_p=0.9,
        top_k=40,
        num_beams=1,
        max_new_tokens=2048,
        **kwargs,
):
    prompts = generate_prompt(batch_data, input)
    inputs = tokenizer(prompts, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output


def generate_prompt(instruction, input=None):
    return f"""{instruction}

### Response:
"""


def main(
    load_8bit: bool = False,
    base_model: str = "Model_Path",
    input_data_path = "Input.jsonl",
    output_data_path = "Output.jsonl",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    input_data = open(input_data_path, mode='r', encoding='utf-8')
    output_data = open(output_data_path, mode='w', encoding='utf-8')

    for num, line in enumerate(input_data.readlines()):
        one_data = json.loads(line)
        id = one_data["idx"]
        instruction = one_data["Instruction"]
        _output = evaluate(instruction)
        final_output = _output[0].split("### Response:")[1].strip()
        new_data = {
            "id": id,
            "instruction": instruction,
            "wizardlm": final_output
        }
        output_data.write(json.dumps(new_data) + '\n')


if __name__ == "__main__":
    fire.Fire(main)
