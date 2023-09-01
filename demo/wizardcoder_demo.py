import os
import sys
import fire
import torch
import transformers
import gradio as gr
from vllm import LLM, SamplingParams

def main(
    base_model="WizardLM/WizardCoder-Python-34B-V1.0",
    n_gpus=4,
    port=8080,
):
    llm = LLM(model=base_model, tensor_parallel_size=n_gpus)
    def evaluate_vllm(
        instruction,
        temperature=1,
        max_new_tokens=2048,
    ):
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
        prompt = problem_prompt.format(instruction=instruction)

        problem_instruction = [prompt]
        stop_tokens = ['</s>']
        sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
        completions = llm.generate(problem_instruction, sampling_params)
        for output in completions:
            prompt = output.prompt
            print('==========================question=============================')
            print(prompt)
            generated_text = output.outputs[0].text
            print('===========================answer=============================')
            print(generated_text)
            return generated_text

    gr.Interface(
        fn=evaluate_vllm,
        inputs=[
            gr.components.Textbox(
                lines=3, label="Instruction", placeholder="Anything you want to ask WizardCoder ?"
            ),
            gr.components.Slider(minimum=0, maximum=1, value=0, label="Temperature"),
            gr.components.Slider(
                minimum=1, maximum=2048, step=1, value=1024, label="Max tokens"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=30,
                label="Output",
            )
        ],
        title="WizardCoder",
        description="Empowering Code Large Language Models with Evol-Instruct, github: https://github.com/nlpxucan/WizardLM, huggingface: https://huggingface.co/WizardLM"
    ).queue().launch(share=True, server_port=port)

if __name__ == "__main__":
    fire.Fire(main)
