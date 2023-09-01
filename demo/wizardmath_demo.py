import json
import fire
import gradio as gr
from vllm import LLM, SamplingParams
import os

def main(
        base_model,
        n_gpus=1,
        title="WizardMath-7B-V1.0",
        port=8080,
        load_8bit: bool = False):
    llm = LLM(model=base_model, tensor_parallel_size=n_gpus)
    def evaluate_vllm(
            instruction,
            use_cot=True,
            temperature=1,
            max_new_tokens=2048,):

        cot_problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
        if use_cot == True:
            prompt = cot_problem_prompt.format(instruction=instruction)
        else:
            prompt = problem_prompt.format(instruction=instruction)

        problem_instruction = [prompt]
        stop_tokens = ["Instruction:", "Instruction", "Response:", "Response", '</s>']
        sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
        completions = llm.generate(problem_instruction, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            return generated_text

    gr.Interface(
        fn=evaluate_vllm,
        inputs=[
            gr.components.Textbox(
                lines=3, label="Instruction", placeholder="Anything you want to ask WizardMath ?"
            ),
            gr.inputs.Checkbox(default=True, label='Use CoT', optional=False),
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
        title=title,
        description="Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct, github: https://github.com/nlpxucan/WizardLM, huggingface: https://huggingface.co/WizardLM"
    ).queue().launch(share=False, server_port=port)

if __name__ == "__main__":
    fire.Fire(main)
