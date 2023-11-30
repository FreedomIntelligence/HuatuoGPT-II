import os
import platform
import torch
from threading import Thread
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import argparse
from transformers import TextIteratorStreamer
from transformers.generation.utils import GenerationConfig

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype='auto', trust_remote_code=True)
    return model, tokenizer

def generate_prompt(query, history):
    if not history:
        return  f"<问>：{query}\n<答>："
    else:
        prompt = ''
        for i, (old_query, response) in enumerate(history):
            prompt += "<问>：{}\n<答>：{}\n".format(old_query, response)
        prompt += "<问>：{}\n<答>：".format(query)
        return prompt

def remove_overlap(str1, str2):
    for i in range(len(str1), -1, -1): 
        if str1.endswith(str2[:i]): 
            return str2[i:] 
    return str2 

def main(args):
    model, tokenizer = load_model(args.model_name)
    sep = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
    print(sep)
    
    model = model.eval()

    gen_kwargs = {'max_new_tokens': 1024, 'do_sample':True, 'top_p':0.7, 'temperature':0.3, 'repetition_penalty':1.1}

    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    history = []
    print("HuatuoGPT: 你好，我是一个解答医疗健康问题的大模型，目前处于测试阶段，请以医嘱为准。请问有什么可以帮到您？输入 clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            os.system(clear_command)
            print("HuatuoGPT: 你好，我是一个解答医疗健康问题的大模型，目前处于测试阶段，请以医嘱为准。请问有什么可以帮到您？输入 clear 清空对话历史，stop 终止程序")
            continue
        
        print(f"HuatuoGPT: ", end="", flush=True)


        prompt = generate_prompt(query, history)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer,skip_prompt=True)
        generation_kwargs = dict(input_ids=inputs['input_ids'], streamer=streamer, **gen_kwargs)
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ''

        for new_text in streamer:
            if sep in new_text:
                new_text = remove_overlap(generated_text,new_text[:-len(sep)])
                for char in new_text:
                    generated_text += char
                    print(char,end='',flush = True)
                break
            for char in new_text:
                generated_text += char
                print(char,end='',flush = True)
        history = history + [(query, generated_text)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="FreedomIntelligence/HuatuoGPT2-7B")
    args = parser.parse_args()
    main(args)