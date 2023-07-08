## WizardLM: An Instruction-following LLM Using Evol-Instruct 
Empowering Large Pre-Trained Language Models to Follow Complex Instructions

<p align="center" width="100%">
<a ><img src="imgs/WizardLM.png" alt="WizardLM" style="width: 20%; min-width: 300px; display: block; margin: auto;"></a>
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## News

At present, our core contributors are preparing the **65B** version and we expect to empower WizardLM with the ability to perform instruction evolution itself, aiming to evolve your specific data at a low cost.

- 🔥🔥🔥 [7/7/2023] We released **WizardLM V1.1** models. The **WizardLM-13B-V1.1** is here ([Demo_13B-V1.1](https://33aaac370fece283.gradio.app), [Demo_13B-V1.1_bak-1](https://ab018a23bf55f3c2.gradio.app), [Demo_13B-V1.1_bak-2](https://86861a9e1cf2e4c2.gradio.app), [Full Model Weight](https://huggingface.co/WizardLM/WizardLM-13B-V1.1)). **WizardLM-7B-V1.1**,  **WizardLM-30B-V1.1**, and **WizardLM-65B-V1.1** are coming soon. Please checkout the [Full Model Weights](https://huggingface.co/WizardLM) and [paper](https://arxiv.org/abs/2304.12244).
- 🔥🔥🔥 [7/7/2023] The **WizardLM-13B-V1.1** achieves **6.74** on [MT-Bench Leaderboard](https://chat.lmsys.org/?leaderboard), **86.32%** on [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/), and **99.3%** on [WizardLM Eval](https://github.com/nlpxucan/WizardLM/blob/main/WizardLM/data/WizardLM_testset.jsonl). (**Note: MT-Bench and AlpacaEval are all self-test, will push update and request review. All tests are completed under their official settings.**)



- 🔥 we released the latest optimized version of Evol-Instruct training data of WizardLM model. Please refer to [this HuggingFace Repo](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k) to download. 

- 🔥 We released **WizardCoder-15B-V1.0** (trained with **78k** evolved code instructions), which surpasses **Claude-Plus (+6.8)**, **Bard (+15.3)** and **InstructCodeT5+ (+22.3)** on the [HumanEval Benchmarks](https://github.com/openai/human-eval). For more details ([Paper](https://arxiv.org/abs/2306.08568), [Demo (Only support code-related English instructions now.)](https://41d6fbcd16627c25.gradio.app/), [Backup Demo1](https://cfb18dadc1051cce.gradio.app/), please refer to [WizardCoder](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder).

- 🔥 Our WizardLM-13B-V1.0 model achieves the 1st-rank of the opensource models on the [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/).
- &#x1F4E3; Please refer to our Twitter account https://twitter.com/WizardLM_AI and HuggingFace Repo https://huggingface.co/WizardLM . We will use them to announce any new release at the 1st time. 
- 🔥 We released  **WizardLM-30B-V1.0** ([Demo_30B](https://3a6bbf2cf3f85af1.gradio.app/)) and **WizardLM-13B-V1.0** ([Demo_13B](https://97159fe0d183fa6b.gradio.app/)) trained with **250k** evolved instructions (from ShareGPT), and **WizardLM-7B-V1.0** ([Demo_7B](https://74ecfd89dccaaa08.gradio.app)) trained with **70k** evolved instructions (from Alpaca data). Please checkout the [Delta Weights](https://huggingface.co/WizardLM) and [paper](https://arxiv.org/abs/2304.12244).

- &#x1F4E3; We are looking for highly motivated students to join us as interns to create more intelligent AI together. Please contact caxu@microsoft.com

<!-- Although on our **complexity-balanced test set**, **WizardLM-7B has more cases that are preferred by human labelers than ChatGPT** in the high-complexity instructions (difficulty level >= 8), it still lags behind ChatGPT on the entire test set, and we also consider WizardLM to still be in a **baby state**. This repository will **continue to improve WizardLM**, train on larger scales, add more training data, and innovate more advanced large-model training methods. -->


<b>Note for 30B and 13B model usage:</b>

To obtain results **identical to our demo**, please strictly follow the prompts and invocation methods provided in the **"src/infer_wizardlm13b.py"** to use our 13B model for inference. Unlike the 7B model, the 13B model adopts the prompt format from <b>Vicuna</b> and supports **multi-turn** conversation.

<b>For WizardLM-13B-V1.0, WizardLM-30B-V1.0 </b>, the Prompt should be as following:

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: hello, who are you? ASSISTANT: 
```

<b>For WizardLM-7B-V1.0 </b>, the Prompt should be as following:

```
"{instruction}\n\n### Response:"
```

<b>For WizardCoder-15B-V1.0 </b>, the Prompt should be as following:

```
"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
```

### GPT-4 automatic evaluation

We adopt the automatic evaluation framework based on GPT-4 proposed by FastChat to assess the performance of chatbot models. As shown in the following figure, WizardLM-30B achieved better results than Guanaco-65B. 
<p align="center" width="100%">
<a ><img src="imgs/WizarLM30b-GPT4.png" alt="WizardLM" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a>
</p>

### WizardLM-30B performance on different skills.

The following figure compares WizardLM-30B and ChatGPT’s skill on Evol-Instruct testset. The result indicates that WizardLM-30B achieves 97.8% of ChatGPT’s performance on average, with almost 100% (or more than) capacity on 18 skills, and more than 90% capacity on 24 skills.

<p align="center" width="100%">
<a ><img src="imgs/evol-testset_skills-30b.png" alt="WizardLM" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a>
</p>

### WizardLM performance on NLP foundation tasks.

The following table provides a comparison of WizardLMs and other LLMs on NLP foundation tasks. The results indicate that WizardLMs consistently exhibit superior performance in comparison to the LLaMa models of the same size. Furthermore, our WizardLM-30B model showcases comparable performance to OpenAI's Text-davinci-003 on the MMLU and HellaSwag benchmarks.

| Model            | MMLU 5-shot | ARC 25-shot | TruthfulQA 0-shot | HellaSwag 10-shot | Average    |
|------------------|-------------|-------------|-------------------|-------------------|------------|
| Text-davinci-003 | <u>56.9<u/> | **85.2**    | **59.3**          | <u>82.2<u/>       | **70.9**   |
|Vicuna-13b 1.1   | 51.3        | 53.0        | 51.8              | 80.1              | 59.1       |
|Guanaco 30B   | 57.6        | 63.7        | 50.7              | **85.1**              | 64.3       |   
| WizardLM-7B 1.0      | 42.7        | 51.6        | 44.7              | 77.7              | 54.2       |
| WizardLM-13B 1.0     | 52.3        | 57.2        | 50.5              | 81.0              | 60.2       |
| WizardLM-30B 1.0    | **58.8**    | <u>62.5<u/> | <u>52.4<u/>       | 83.3          | <u>64.2<u/>|

### WizardLM performance on code generation.

The following table provides a comprehensive comparison of WizardLMs and several other LLMs on the code generation task, namely HumanEval. The evaluation metric is pass@1. The results indicate that WizardLMs consistently exhibit superior performance in comparison to the LLaMa models of the same size. Furthermore, our WizardLM-30B model surpasses StarCoder and OpenAI's code-cushman-001. Moreover, our Code LLM, WizardCoder, demonstrates exceptional performance, achieving a pass@1 score of 57.3, surpassing the open-source SOTA by approximately 20 points.


| Model            | HumanEval Pass@1 |
|------------------|------------------|
| LLaMA-7B         | 10.5             |
| LLaMA-13B        | 15.8             |
| CodeGen-16B-Multi| 18.3             |
| CodeGeeX         | 22.9             |
| LLaMA-33B        | 21.7             |
| LLaMA-65B        | 23.7             |
| PaLM-540B        | 26.2             |
| CodeGen-16B-Mono | 29.3             |
| code-cushman-001 | 33.5             |
| StarCoder        | <u>33.6<u/>      |
| WizardLM-7B 1.0      | 19.1             |
| WizardLM-13B 1.0     | 24.0             |
| WizardLM-30B  1.0   | **37.8**         |
| WizardCoder-15B  1.0 | **57.3**     |

## Call for Feedbacks
We welcome everyone to use your professional and difficult instructions to evaluate WizardLM, and show us examples of poor performance and your suggestions in the [issue discussion](https://github.com/nlpxucan/WizardLM/issues) area. We are focusing on improving the Evol-Instruct now and hope to relieve existing weaknesses and issues in the the next version of WizardLM. After that, we will open the code and pipeline of up-to-date Evol-Instruct algorithm and work with you together to improve it.

## Unofficial Video Introductions
Thanks to the enthusiastic friends, their video introductions are more lively and interesting.
1. [GET WizardLM NOW! 7B LLM KING That Can Beat ChatGPT! I'm IMPRESSED!](https://www.youtube.com/watch?v=SaJ8wyKMBds)
2. [WizardLM: Enhancing Large Language Models to Follow Complex Instructions](https://www.youtube.com/watch?v=I6sER-qivYk)
3. [WizardCoder AI Is The NEW ChatGPT's Coding TWIN!](https://www.youtube.com/watch?v=XjsyHrmd3Xo)

## Overview of Evol-Instruct

[Evol-Instruct](https://github.com/nlpxucan/evol-instruct) is a novel method using LLMs instead of humans to automatically mass-produce open-domain instructions of various difficulty levels and skills range, to improve the performance of LLMs.

<p align="center" width="100%">
<a ><img src="imgs/git_overall.png" alt="WizardLM" style="width: 86%; min-width: 300px; display: block; margin: auto;"></a>
</p>

<p align="center" width="100%">
<a ><img src="imgs/git_running.png" alt="WizardLM" style="width: 86%; min-width: 300px; display: block; margin: auto;"></a>
</p>

### Citation

Please cite the paper if you use the data or code from WizardLM.

```
@misc{xu2023wizardlm,
      title={WizardLM: Empowering Large Language Models to Follow Complex Instructions}, 
      author={Can Xu and Qingfeng Sun and Kai Zheng and Xiubo Geng and Pu Zhao and Jiazhan Feng and Chongyang Tao and Daxin Jiang},
      year={2023},
      eprint={2304.12244},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Please cite the paper if you use the data or code from WizardCoder.

```
@misc{luo2023wizardcoder,
      title={WizardCoder: Empowering Code Large Language Models with Evol-Instruct}, 
      author={Ziyang Luo and Can Xu and Pu Zhao and Qingfeng Sun and Xiubo Geng and Wenxiang Hu and Chongyang Tao and Jing Ma and Qingwei Lin and Daxin Jiang},
      year={2023},
      eprint={2306.08568},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## Disclaimer

The resources, including code, data, and model weights, associated with this project are restricted for academic research purposes only and cannot be used for commercial purposes. The content produced by any version of WizardLM is influenced by uncontrollable variables such as randomness, and therefore, the accuracy of the output cannot be guaranteed by this project. This project does not accept any legal liability for the content of the model output, nor does it assume responsibility for any losses incurred due to the use of associated resources and output results.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=nlpxucan/WizardLM&type=Timeline)](https://star-history.com/#nlpxucan/WizardLM&Timeline)

