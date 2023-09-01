import gradio as gr
import argparse
import os
import json
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)  # model path
    parser.add_argument("--n_gpus", type=int, default=1)  # n_gpu
    return parser.parse_args()

def predict(message, history, system_prompt, temperature, max_tokens):
    instruction = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    for human, assistant in history:
        instruction += 'USER: '+ human + ' ASSISTANT: '+ assistant + '</s>'
    instruction += 'USER: '+ message + ' ASSISTANT:'
    problem = [instruction]
    stop_tokens = ["USER:", "USER", "ASSISTANT:", "ASSISTANT"]
    sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_tokens, stop=stop_tokens)
    completions = llm.generate(problem, sampling_params)
    for output in completions:
        prompt = output.prompt
        print('==========================question=============================')
        print(prompt)
        generated_text = output.outputs[0].text
        print('===========================answer=============================')
        print(generated_text)
        for idx in range(len(generated_text)):
                yield generated_text[:idx+1]


if __name__ == "__main__":
    args = parse_args()
    llm = LLM(model=args.base_model, tensor_parallel_size=args.n_gpus)
    gr.ChatInterface(
        predict,
        title="LLM playground - WizardLM-13B-V1.2",
        description="This is a LLM playground for WizardLM-13B-V1.2, github: https://github.com/nlpxucan/WizardLM, huggingface: https://huggingface.co/WizardLM",
        theme="soft",
        chatbot=gr.Chatbot(height=1400, label="Chat History",),
        textbox=gr.Textbox(placeholder="input", container=False, scale=7),
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
        additional_inputs=[
            gr.Textbox("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", label="System Prompt"),
            gr.Slider(0, 1, 0.9, label="Temperature"),
            gr.Slider(100, 2048, 1024, label="Max Tokens"),
        ],
        additional_inputs_accordion_name="Parameters",
    ).queue().launch(share=False, server_port=7870)
