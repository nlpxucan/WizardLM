## WizardCoder Inference Demo

We provide the inference demo script for **WizardCoder-Family**.

1. According to the instructions of [Llama-X](https://github.com/AetherCortex/Llama-X), install the environment.
2. Install these packages:
```bash
pip install transformers==4.31.0
pip install vllm==0.1.4
```
3. Enjoy your demo:
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python wizardcoder_demo.py \
   --base_model "WizardLM/WizardCoder-Python-34B-V1.0" \
   --n_gpus 4
```

Note: This script supports `WizardLM/WizardCoder-Python-34B/13B/7B-V1.0`. If you want to inference with `WizardLM/WizardCoder-15B/3B/1B-V1.0`, please change the `stop_tokens = ['</s>']` to `stop_tokens = ['<|endoftext|>']` in the script.