# HuatuoGPT2, One-stage Training for Medical Adaption of LLMs

<div align="center">
<h3>
  HuatuoGPT-II
</h3>
</div>


<p align="center">
🖥️ <a href="https://www.huatuogpt.cn/" target="_blank">Online Demo (7B)</a> ｜⬇️ <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT2-7B" target="_blank">7B Model</a> ｜⬇️ <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT2-13B" target="_blank">13B Model</a> ｜ ⬇️ <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT2-34B" target="_blank">34B Model</a>   ｜ 📃 <a href="https://arxiv.org/abs/2311.09774" target="_blank">Paper</a> <br>
</p>


### ✨ Latest News
- [01/10/2024] The HuatuoGPT2 model is now available on the [Wisemodel platform](https://www.wisemodel.cn/models/FreedomIntelligence/HuatuoGPT-II).
- [12/04/2023] We released the **code and dataset for our evaluation**.
- [11/24/2023] We released the **quantitative version** of HuatuoGPT-II.
- [11/21/2023] We released HuatuoGPT-II models. The HuatuoGPT-II will be available in **7B**, **13B**, and **34B** versions.
- [11/17/2023] We released the [HuatuoGPT-II paper](https://arxiv.org/abs/2311.09774), achieving a new **state-of-the-art** in Chinese medical applications! Try our [demo](https://www.huatuogpt.cn/)!



## ⚡ Introduction

Hello! Welcome to the repository for [HuatuoGPT2](https://arxiv.org/abs/2311.09774). 

HuatuoGPT2 employs an innovative domain adaptation method to significantly boost its medical knowledge and dialogue proficiency. It showcases state-of-the-art performance in several medical benchmarks, especially surpassing GPT-4 in expert evaluations and the fresh medical licensing exams.

The open-source release of HuatuoGPT-2 includes:

- **HuatuoGPT2 Model**: Open-sourcing of 7B, 13B, and 34B versions.
- **Training Code**: Training code for one-stage adaptation will be provided, enabling better model adaptation across various languages and domains.
- **HuatuoGPT2 Data**: Release of partial pre-training and fine-tuning instructions.
- **Evaluation for Chinese Medical LLM**: Comprehensive automatic evaluation methods for medical response capabilities of LLM and the fresh professional pharmacist exam assessment.

Note that we're still actively organizing our code and data. Please stay tuned for updates coming soon!

<div align=center>
<img src="assets/figure1.png"  width = "90%" alt="HuatuoGPT2" align=center/>
</div>




## 🌟 Performance

Compared with representative open-source models and closed-source models (including GPT-4), HuatuoGPT2 showed impressive performance on medical benchmarks. Here, we present two of the results.

- **Expert Evaluation**: In assessments by medical professionals, HuatuoGPT-II's responses in Chinese medical contexts were favored over counterparts like GPT-4: 

| **HuatuoGPT-II Win Rate**              | **Win** | **Tie** | **Fail** |
| -------------------------------------- | ------- | ------- | -------- |
| **Single-round Medical Response**      |         |         |          |
| HuatuoGPT-II(7B) vs GPT-4              | **38**  | 38      | 24       |
| HuatuoGPT-II(7B) vs ChatGPT            | **52**  | 33      | 15       |
| HuatuoGPT-II(7B) vs Baichuan2-13B-Chat | **63**  | 19      | 18       |
| HuatuoGPT-II(7B) vs HuatuoGPT          | **81**  | 11      | 8        |
| **Multi-round Medical Dialogue**       |         |         |          |
| HuatuoGPT-II(7B) vs GPT-4              | **53**  | 17      | 30       |
| HuatuoGPT-II(7B) vs ChatGPT            | **56**  | 11      | 33       |
| HuatuoGPT-II(7B) vs Baichuan2-13B-Chat | **63**  | 19      | 18       |
| HuatuoGPT-II(7B) vs HuatuoGPT          | **68**  | 6       | 26       |

- **The Fresh Medical Exams**: We collected the fresh 2023 Chinese National Pharmacist Licensure Examination, which started on October 21, 2023. This date is later than our data finalization. HuatuoGPT2 achieved the best results in this exam, as shown below.

<div align=center>
<img src="assets/res1.png"  width = "90%" alt="HuatuoGPT2" align=center/>
</div>



## 👩‍⚕️ Model

### Model Access

Our model is now available on Huggingface. You can Try our model in https://www.huatuogpt.cn/.

| Model          | Backbone           | Checkpoint    |
| -------------- | ------------------ | ------------- |
| HuatuoGPT2-7B  | Baichuan2-7B-Base  | [HF Lnik](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-7B) |
| HuatuoGPT2-13B | Baichuan2-13B-Base | [HF Lnik](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-13B) |
| HuatuoGPT2-34B | Yi-34B             | [HF Lnik](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-34B) |

### Quantization Model

A quantized version of HuatuoGPT2 is also provided, allowing users with constrained memory or computing resources to access our HuatuoGPT2.
| Quantization          | Backbone      | Checkpoint |
| --------------------- | ------------- | ------------- |
| HuatuoGPT2-7B-4bits  | Baichuan2-7B-Base | [HF Lnik](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-7B-4bits) |
| HuatuoGPT2-7B-8bits | Baichuan2-7B-Base | [HF Lnik](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-7B-8bits) |
| HuatuoGPT2-34B-4bits       | Yi-34B        | [HF Lnik](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-34B-4bits) |
| HuatuoGPT2-34B-8bits        | Yi-34B        | [HF Lnik](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-34B-8bits) |

### Model Inference

```bash
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B", use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
messages = []
messages.append({"role": "user", "content": "肚子疼怎么办？"})
response = model.HuatuoChat(tokenizer, messages)
print(response)
```

#### Inference with Command Line

```bash
python cli_demo.py --model_name FreedomIntelligence/HuatuoGPT2-7B
```



## 🌈 One-stage adaption

### Data Unification

<div align=center>
<img src="assets/figure4.png"  width = "50%" alt="HuatuoGPT2" align=center/>
</div>

- HuatuoGPT2 transforms the pre-training corpus into  (instruction, output) pairs using LLM. Utilize the script for Data Unification.

```Bash
python adaption/data_unification/rewrite.py
```

### One-stage training
<div align=center>
<img src="assets/figure3.png"  width = "50%" alt="HuatuoGPT2" align=center/>
</div>

- We introduce a priority sampling approach, pre-processing data with this algorithm:

```bash
python adaption/one_stage_training/data_process.py
```

- Then, training is conducted using one-stage training:

```Bash
bash adaption/one_stage_training/train.sh
```

By adopting the One-stage Adaptation method, you will observe the following loss curve:

<div align=center>
<img src="assets/loss.png"  width = "50%" alt="HuatuoGPT2" align=center/>
</div>


## 📚 Data

We open source part of the training data.

| Data Type                               | #Sample | Link                                                         |
| --------------------------------------- | ------- | ------------------------------------------------------------ |
| Medical Fine-tuning Instruction (GPT-4) | 50,000  | [HF Link](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT2_sft_instruct_GPT4_50K) |
| Medical Pre-training Instruction        | -       | -                                                            |




## 🧐 Evaluation

### Automated Evaluation of Medical Response Quality

- Single-turn response evaluation using **GPT-4**:

```bash
python evaluation/eval_huatuo_inst.py
```

- Multi-turn dialogue evaluation using **GPT-4**:

```bash
python evaluation/eval_huatuo_conv.py
```

### The Fresh Medical Exams

Access our newest medical exam dataset via the link provided. The dataset includes complete exam questions, with exam dates noted to alert for potential leaks. We plan to release more updated exams in the future.

| Examination                                                  | #Question | Exam Time  | Links                                                        |
| ------------------------------------------------------------ | --------- | ---------- | ------------------------------------------------------------ |
| 2023 Chinese National Pharmacist Licensure Examination (Pharmacy) | 480       | 2023.10.22 | [huggingface](https://huggingface.co/datasets/FreedomIntelligence/2023_Pharmacist_Licensure_Examination-Pharmacy_track) |
| 2023 Chinese National Pharmacist Licensure Examination (TCM) | 480       | 2023.10.22 | [huggingface](https://huggingface.co/datasets/FreedomIntelligence/2023_Pharmacist_Licensure_Examination-TCM_track) |
| Other **Fresh** Medical Examinations is in coming            |           |            |                                                              |



## 🩺 HuatuoGPT Series 

The HuatuoGPT series has so far launched two generations:

- [**HuatuoGPT**](https://github.com/FreedomIntelligence/HuatuoGPT): A Doctor-like Medical Large Language Model
- [**HuatuoGPT-II**](https://github.com/FreedomIntelligence/HuatuoGPT-II):  An Domain-enhanced Medical Large Language Model

In the future, we will continue to release new versions of HuatuoGPT. Our goal is to enhance the capabilities of LLM in the Chinese medical field and to adhere to open-source principles (aligned with the ethos of FreedomIntelligence). We hope to work together with everyone to promote the development of medical LLM!

We are from the School of Data Science, the Chinese University of Hong Kong, Shenzhen (CUHKSZ) and the Shenzhen Research Institute of Big Data (SRIBD).


## Citation
```
@misc{chen2023huatuogptii,
      title={HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs}, 
      author={Junying Chen and Xidong Wang and Anningzhe Gao and Feng Jiang and Shunian Chen and Hongbo Zhang and Dingjie Song and Wenya Xie and Chuyi Kong and Jianquan Li and Xiang Wan and Haizhou Li and Benyou Wang},
      year={2023},
      eprint={2311.09774},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{huatuogpt-2023,
  title={HuatuoGPT, Towards Taming Language Models To Be a Doctor},
  author={Hongbo Zhang and Junying Chen and Feng Jiang and Fei Yu and Zhihong Chen and Jianquan Li and Guiming Chen and Xiangbo Wu and Zhiyi Zhang and Qingying Xiao and Xiang Wan and Benyou Wang and Haizhou Li},
  journal={arXiv preprint arXiv:2305.15075},
  year={2023}
}
```

## Star History

<a href="https://star-history.com/#FreedomIntelligence/HuatuoGPT-II&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FreedomIntelligence/HuatuoGPT-II&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FreedomIntelligence/HuatuoGPT-II&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=FreedomIntelligence/HuatuoGPT-II&type=Date" />
  </picture>
</a>
