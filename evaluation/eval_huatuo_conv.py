import os
import random
import openai
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import random
from GPT4 import GPT4
import re

def parse_classification_cot(review):
    try:
        label_content = review.strip()
        label = re.findall(r'Assistant 1 is ([bB]etter than|`[bB]etter than`|[wW]orse than|`[wW]orse than`|[eE]qual to|`[eE]qual to`) Assistant 2', label_content)
        if len(label):
            label = label[-1].strip('`').lower()
            if label == 'better than':
                return [10, 0]
            elif label == 'worse than':
                return [0, 10]
            else:
                return [5, 5]

        label = re.findall(r'Assistant 2 is ([bB]etter than|`[bB]etter than`|[wW]orse than|`[wW]orse than`|[eE]qual to|`[eE]qual to`) Assistant 1', label_content)
        if len(label):
            label = label[-1].strip('`').lower()
            if label == 'better than':
                return [0, 10]
            elif label == 'worse than':
                return [10, 0]
            else:
                return [5, 5]
        
        if re.search(r'are equal in', label_content):
            return [5, 5]
            
        print('error', review)
        return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


#%%
eval_prompt = """[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

[System]
We would like to request your feedback on two multi-turn conversations between the AI assistant and the user displayed above.
Requirements: The response should be to the point and adress the problem of user. The description of symptoms should be comprehensive and accurate, and the provided diagnosis should be the most reasonable inference based on all relevant factors and possibilities. The treatment recommendations should be effective and reliable, taking into account the severity or stages of the illness. The prescriptions should be effective and reliable, considering indications, contraindications, and dosages.
Please compare the performance of the AI assistant in each conversation. You should tell me whether Assistant 1 is `better than`, `worse than`, or `equal to` Assistant 2.
Please first compare their responses and analyze which one is more in line with the given requirements.
In the last line, please output a single line containing only a single label selecting from `Assistant 1 is better than Assistant 2`, `Assistant 1 is worse than Assistant 2`, and `Assistant 1 is equal to Assistant 2`."""

model_a = 'HuatuoGPT2-7B'
model_b = 'GPT4'

task_name = 'huatuo100_conv'
task_dir = f'{task_name}-{model_a}_vs_{model_b}'
save_dir = f'tmp_data/{task_dir}'
gpt = GPT4()
retry_time = 3

def transfer_conv(model_output):
    roles = ['User: ','AI assistant: ']
    res = ''
    for pa,do in model_output:
        res += f'{roles[0]}{pa}\n{roles[1]}{do}\n\n'
    return res.strip()

wrongtime = 0
def write_piece_order_data(d):
    save_path = os.path.join(save_dir, str(d['id']) + ".json")
    if os.path.exists(save_path):
        return -1
    scores = []

    for _ in range(retry_time):
        try:
            query = eval_prompt.format(transfer_conv(d['model_a']),transfer_conv(d['model_b']))
            res = gpt.retry_call(query)
            ascore,bscore = parse_classification_cot(res)
            if ascore != -1:
                scores.append((ascore,bscore))      
                break
        except Exception as e:
            print(e)
            raise Exception('match miss')
    # reverse for position bias
    for _ in range(retry_time):
        try:
            query = eval_prompt.format(transfer_conv(d['model_b']),transfer_conv(d['model_a']))
            res = gpt.retry_call(query)
            bscore,ascore = parse_classification_cot(res)
            if ascore != -1:
                scores.append((ascore,bscore))      
                break
        except Exception as e:
            print(e)
            raise Exception('match miss')
    d['scores'] = scores
    d['model_a_name'] = model_a
    d['model_b_name'] = model_b
    d['ChatGPT_response_0'] = res
    d['ChatGPT_qeury_0'] = query
    with open(save_path, mode="w", encoding="utf-8") as fw:
        json.dump(d, fw, ensure_ascii=False,indent=2)
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
        full_path = os.path.join(save_dir, file_path)
        try:
            with open(full_path, 'r', encoding="utf-8") as f:
                da = json.load(f)
            if 'ChatGPT_response_0' not in da:
                os.remove(full_path)
                continue
            res.append(da)
        except Exception as e:
            print(str(e))
            if 'Expecting value' in str(e):
                os.remove(full_path)
    return res

import copy
def compute_score(data):
    all_score = {'model_a':model_a,'model_b':model_b,'win':0,'tie':0,'loss':0}
    for da in data:
        scores = da['scores']
        ascore = scores[0][0] + scores[1][0] if len(scores) > 1 else scores[0][0]
        bscore = scores[0][1] + scores[1][1] if len(scores) > 1 else scores[0][1]
        if ascore > bscore:
            all_score['win'] += 1
        elif bscore > ascore:
            all_score['loss'] += 1
        else:
            all_score['tie'] += 1

    for k,v in copy.deepcopy(list(all_score.items())):
        if isinstance(v,int):
            all_score[k+'_rate'] = v / len(data)
    print(all_score)
    return all_score

if __name__ == '__main__':
    num_process = 100
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("Path created at", save_dir)
        
    finished_data = merge_files(save_dir)
    print(f'finished_data: {len(finished_data)}')
    def load_data(data_path_a,data_path_b):
        with open(data_path_a) as f:
            data_a = json.load(f)
        with open(data_path_b) as f:
            data_b = json.load(f)
        assert len(data_a) == len(data_b), f'{len(data_a)},{len(data_b)}'
        id2b = {}
        for da in data_b:
            id2b[da['id']] = da
        data = []
        for da in data_a:
            da['model_a'] = da['model_output']
            da['model_b'] = id2b[da['id']]['model_output']
            data.append(da)  
        return data

    data_path_1 = f'data/{task_name}/{model_a}.json'
    
    data = load_data(f'data/{task_name}/{model_a}.json',f'data/{task_name}/{model_b}.json')
    print(f"read data:{len(data)}")

    data = deduplicate(data,finished_data)
    print(f"{len(data)} to be processed")
    random.shuffle(data)
    
    with ThreadPoolExecutor(max_workers=num_process) as executor:
        # Use map to process the data in parallel and return an iterable
        results = list(tqdm(executor.map(write_piece_order_data, data), total=len(data), desc="Processing samples", unit="sample"))

    print(f'finish_')
    finished_data = merge_files(save_dir)
    print(len(finished_data))
    res = compute_score(finished_data)
    with open(f'results/{task_dir}.json','w') as fw:
        json.dump(res,fw,ensure_ascii=False,indent=2)
    