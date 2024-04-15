
## Choice-1: API Server
The OpenAI-compatible APIs are provided by vLLM. We advise you to use vLLM=0.2.1.post1 to build OpenAI-compatible API service. 

### Environment

```
conda create -n myenv python=3.8 -y
source activate myenv
pip install vllm==0.2.1.post1
pip install openai==1.17.1
pip install accelerate
pip install fschat
```

### Server

vLLM provides an HTTP server that implements OpenAI’s Completions and Chat API. To deploy the server, you need to use the following command: 

```
### for 7B model, maybe use 1 gpu
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model xxxx/wizard_model_path --dtype float16 --tensor-parallel-size 1 --ip your_IP --port your_PORT --trust-remote-code --max-model-len 24000
```

```
### for 70B model, maybe use 8 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server --model xxxx/wizard_model_path --dtype float16 --tensor-parallel-size 8 --ip your_IP --port your_PORT --trust-remote-code --max-model-len 24000
```

#### API Inference
```
from openai import OpenAI

API_URL = "http://ip:port/v1"
model_path = "xxxx/wizard_model_path"
client = OpenAI(
    base_url=API_URL,
    api_key="EMPTY",
)
system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
stop_tokens = []
completion = client.chat.completions.create(
    model=model_path,
    temperature=0,
    top_p=1,
    max_tokens=4096,
    stop=stop_tokens,
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Hello! What is your name?"},
    {"role": "assistant", "content": "I am WizardLM2!"},
    {"role": "user", "content": "Nice to meet you!"},
  ]
)

print(completion.choices[0].message.content)

```

If you want to learn more about the deployment process and parameter settings, please refer to the [vLLM_API_Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) official documentation 


## Choice-2: WizardLM Inference Demo

We provide the inference demo script for **WizardLM-Family**.

1. According to the instructions of [Llama-X](https://github.com/AetherCortex/Llama-X), install the environment.
2. Install these packages:
```bash
pip install transformers
pip install vllm
```
3. Enjoy your demo:
```bash
CUDA_VISIBLE_DEVICES=0 python wizardLM_demo.py \
   --base_model "xxx/path/to/wizardLM_7b_model" \
   --n_gpus 1
```


## Choice-2: WizardCoder Inference Demo

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


## Choice-2: WizardMath Inference Demo

We provide the inference demo script for **WizardMath-Family**.

1. According to the instructions of [Llama-X](https://github.com/AetherCortex/Llama-X), install the environment.
2. Install these packages:
```bash
pip install transformers==4.31.0
pip install vllm==0.1.4
```
3. Enjoy your demo:
```bash
CUDA_VISIBLE_DEVICES=0 python wizardmath_demo.py \
   --base_model "xxx/path/to/wizardmath_7b_model" \
   --n_gpus 1
```
