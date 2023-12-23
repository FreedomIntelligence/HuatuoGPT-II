# %%
import jsonlines as jl
import os
import random
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
# from gpt import GPT
import random
import requests
from retrying import retry
import argparse
import re

class GPT:
    def __init__(self,model_name = 'gpt-3.5-turbo') -> None:
        self.key_ind = 0
        self.init_api_keys()
        self.max_wrong_time = 5
        self.model_name = model_name
        print(f'use model of {self.model_name}')

    def init_api_keys(self):
        self.keys = []
        with open('gpt_key.txt', encoding="utf-8", mode="r") as fr:
            for l in fr:
                cols = l.split('---')
                if len(cols[0]) < 45 or len(cols[0]) > 55:
                    continue
                if len(cols) == 1:
                    cols.append('None')
                self.keys.append((cols[0],cols[1]))
        assert len(self.keys) > 0, 'have no key'
        print(f'keys: {self.keys}')
        self.wrong_time = [0]*len(self.keys)
        random.shuffle(self.keys)
    
    def get_api_key(self):
        self.key_ind =  (self.key_ind + 1) % len(self.keys)
        return self.keys[self.key_ind]

    def call(self, content, args = {}, showkeys = False):
        api_key, organization = self.get_api_key()
        if showkeys:
            print(api_key, organization)
        if organization == 'None':
            organization = ''
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Organization": organization,
        }
        if isinstance(content,str):
            parameters = {
                "model": self.model_name,
                "messages": [{'role': 'user', 'content': content}],
                **args,
            }
        else:
            parameters = {
                "model": self.model_name,
                "messages": content,
                **args,
            }
        response = requests.post(
            url,
            headers=headers,
            json=parameters
            # verify=False
        )
        response = json.loads(response.content.decode("utf-8"))
        if 'error' in response:
            self.wrong_time[self.key_ind] += 1
            if self.wrong_time[self.key_ind] > self.max_wrong_time:
                print(response)
                print(f'del {self.keys[self.key_ind]}')
                # del self.keys[self.key_ind]
                # del self.wrong_time[self.key_ind]
            assert False, str(response)
        return response['choices'][0]['message']['content']
    
    def test(self):
        for _ in range(len(self.keys)):
            try:
                print(self.call('你好',showkeys=True))
            except Exception as e:
                print(e)
    
    @retry(wait_fixed=200, stop_max_attempt_number=20)
    def retry_call(self, content, args = {}):
        return self.call(content, args)



#%%
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="pubmed_test.json")
parser.add_argument("--num_process", type=int, default=200)
args = parser.parse_args()

# load data
with open(args.data_path) as f:
    tmpdata = json.load(f)
data = []
for ii,text in enumerate(tmpdata):
    data.append({'id':ii,'text':text})
    
print(f"read data:{len(data)}")

query_prompt = """Please create a <question> that closely aligns with the provided <text>. Ensure that the <question> is formulated in Chinese and does not explicitly reference the text. You may incorporate specific scenarios or contexts in the <question>, allowing the <text> to serve as a comprehensive and precise answer.

<text>: {}

<question>: """

ans_prompt = """You are HuatuoGPT-II, equipped with in-depth knowledge in medicine. Your task is to directly answer the user's <question> in Chinese. In formulating your response, you must thoughtfully reference the <reference text>, ensuring that your reply does not disclose your reliance on <reference text>. Aim to provide a comprehensive and informative response, incorporating relevant insights from <reference text> to best assist the user. Please be cautious to avoid including any content that might raise ethical concerns.

<question>: {}

<reference text>: {}

<reply>: """

task_name = f'rewrite_{ os.path.split(args.data_path)[-1].replace(".json","")}'
save_dir = f'tmp_data/{task_name}'
query_try_num = 2
ans_try_num = 2
q_max_length = 400
a_max_length = 600
gpt = GPT(model_name='gpt-3.5-turbo')


#%%

def filter_str(wds,input):
    for wd in wds:
        if wd in input:
            return False
    else:
        return True


# Adding query filtering rules
def get_data_query(d):
    query = d['ChatGPT_response_0']
    if len(query) > 180:
        return False,None
    return True,query
    

from nltk import ngrams
def ngram_jaccard_score(str1, str2, ngram):
    str1 = str1.lower()
    str2 = str2.lower()
    if len(str1) < ngram or len(str2) < ngram:
        return 0.0
    ngrams1 = set(ngrams(str1, ngram))
    ngrams2 = set(ngrams(str2, ngram))
    jaccard_score = len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))
    # jaccard_score = len(ngrams1.intersection(ngrams2)) / len(ngrams1)
    return jaccard_score    

# Adding response filtering rules
def get_data_ans(d):
    da = d['ChatGPT_response_0']
    ans_key_wds = ['参考']
    if not filter_str(ans_key_wds,da):
        return False,None  
    sents = re.split('(?<=[。！？])', da)
    sents = [s for s in sents if s]
    
    # you can use ngram_jaccard to filter (only available in Chinese)
    # if ngram_jaccard_score(da,d['text'][:a_max_length],1) <= 0.5:
    #     return False,None

    return True,da


wrongtime = 0
def write_piece_order_data(d):
    global tmp_data
    global wrongtime
    try:
        save_path = os.path.join(save_dir, str(d['id']) + ".json")
        if os.path.exists(save_path):
            return -1
        if 'query' not in d:
            for ii in range(query_try_num):
                chatgpt_query = query_prompt.format(d['text'].replace('\n','\\n')[:q_max_length])
                d['ChatGPT_query'] = chatgpt_query
                query = gpt.retry_call(chatgpt_query)
                d['ChatGPT_response_0'] = query
                flag,query = get_data_query(d)
                if flag:
                    d['query'] = query
                    break

        assert 'query' in d, 'no query'

        if 'response' not in d:
            for ii in range(ans_try_num):
                chatgpt_query = ans_prompt.format(d['query'], d['text'].replace('\n','\\n')[:a_max_length])
                d['ChatGPT_query_a'] = chatgpt_query
                ans = gpt.retry_call(chatgpt_query)
                d['ChatGPT_response_0'] = ans
                flag,newqa = get_data_ans(d)
                if flag:
                    d['response'] = newqa
                    break
        assert 'response' in d, 'no response'
        with open(save_path, mode="w", encoding="utf-8") as fw:
            json.dump(d, fw, ensure_ascii=False,indent=2)
            wrongtime = 0

    except Exception as e:
        # print(d)
        print(str(e),flush=True)
        wrongtime += 1
        if wrongtime > 10:
            assert 1 == 0, 'wrong'
        
    return 1
        
def deduplicate(data,finished):
    idset = set()
    for da in finished:
        idset.add(da['id'])

    dedup_data=[]
    for da in data:
        if da['id'] not in idset:
            dedup_data.append(da)

    return dedup_data

def merge_files(save_dir):
    _, _, filenames = [i for i in os.walk(save_dir)][0]
    json_files = [f for f in filenames if f.endswith('.json')]
    res = []
    for file_path in json_files:
        try:
            with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                da = json.loads(f.read()) 
                res.append(da)
        except Exception as e:
            print(str(e))
    return res


#  start runing
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    print("Path created at", save_dir)

finished_data = merge_files(save_dir)
print(f'finished_data: {len(finished_data)}')


data = deduplicate(data,finished_data)
print(f"{len(data)} to be processed")
random.shuffle(data)

with ThreadPoolExecutor(max_workers=args.num_process) as executor:
    results = list(tqdm(executor.map(write_piece_order_data, data), total=len(data), desc="Processing samples", unit="sample"))

print(f'finish_')
finished_data = merge_files(save_dir)
with open(f'{task_name}.json','w') as fw:
    json.dump(finished_data,fw,ensure_ascii=False,indent=2)

