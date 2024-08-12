#%%
import re
import json,os
from collections import defaultdict
import random
import wandb

def match_choice2(text,data = None):
    option = ['A','B','C','D','E']
    res = re.search(r'答案(?:是|为)(.*?)(。|$)',text,re.S)
    if res:
        tmp = ''.join([x for x in res.group(1) if x in option])
        ans = ''
        index_num = -1
        for x in tmp:
            if x in ans or option.index(x) < index_num:
                continue
            index_num = option.index(x)
            ans += x
        return ans
    return ''

def match_choice(text,data = None):
    ans = []
    for k,v in data['option'].items():
        if v in text and len(v) > 0:
            ans.append((k,text.find(v)))
    if len(ans) > 0:
        ans.sort(key=lambda x: x[1])
        ans = ''.join(x[0] for x in ans)
    if len(ans) == 0 or ans != ''.join(sorted(ans)):
        match = re.findall(r'(?<![A-Za-z])[A-E]+(?![A-Za-z])', text)
        for ma in match[::-1]:
            if len(ma) == len(set(ma)) and ma == ''.join(sorted(ma)):
                data['choice'] = ma + '_1'
                return ma
        return ans
    else:
        data['choice'] = ans + '_2'
        return ans


def match_choice3(text,data = None):
    option = ['A','B','C','D','E']
    res = re.search(r'答案(?:是|为)(.*?)(。|$)',text,re.S)
    if res:
        tmp = ''.join([x for x in res.group(1) if x in option])
        ans = ''
        index_num = -1
        for x in tmp:
            if x in ans or option.index(x) < index_num:
                continue
            index_num = option.index(x)
            ans += x
        if len(ans) > 0:
            data['choice'] = ans + '_3'
            return ans
    ans = []
    for k,v in data['option'].items():
        if v in text and len(v) > 0:
            ans.append((k,text.find(v)))
    if len(ans) > 0:
        ans.sort(key=lambda x: x[1])
        ans = ''.join(x[0] for x in ans)
    if len(ans) == 0 or ans != ''.join(sorted(ans)):
        match = re.findall(r'(?<![A-Za-z])[A-E]+(?![A-Za-z])', text)
        for ma in match:
            # data['choice'] = ma + '_1'
            # return ma
            if len(ma) == len(set(ma)) and ma == ''.join(sorted(ma)):
                data['choice'] = ma + '_1'
                return ma
        return ans
    else:
        data['choice'] = ans + '_2'
        return ans

query_prompt = "下面是一道选择题，请分析每个选项，并最后给出答案。\n[{question_type}]{question}\n{option_str}"
def get_query(da):
    da['option_str'] = '\n'.join([f'{k}. {v}' for k,v in da['option'].items() if len(v) > 1])
    da['query'] = query_prompt.format_map(da)
    return da['query']

def score_result(finished_path, iswandb = False, ans_num = 5):
    print(f'{ans_num} vote')
    # process_json(finished_path)
    datas = []
    with open(finished_path) as f:
        for l in f:
            datas.append(json.loads(l))
    q_type = ['最佳选择题','配伍选择题','综合分析选择题','多项选择题']
    type2score = {k:[0,0] for k in q_type}
    wrong_data = []
    out_name = os.path.basename(finished_path).split('.json')[0]
    wrong_data_file_path = f'./output/{out_name}_wrong_choice.json'

    print(f'{os.path.split(finished_path)[-1]}题目总共 {len(datas)}道')
    miss_match_num = 0
    for da in datas:
        ty = da['question_type']
        ans = da['answer']
        ress = defaultdict(int)
        for ind in range(ans_num):
            res = da[f'huatuo_answer_{ind}']
            choice = match_choice2(res)
            if len(choice) > 1 and ty != '多项选择题':
                choice = choice[0]
            if len(choice) > 0:
                ress[choice] += 1
        if len(ress) > 0:
            model_ans = sorted(ress.items(),key=lambda x:x[1],reverse=True)[0][0]
            sort_ress = sorted(ress.items(),key=lambda x:x[1],reverse=True)
            model_ans = sort_ress[0][0]
        else:
            model_ans = 'A'
            miss_match_num += 1
            
        if model_ans == ans:
            type2score[ty][0] += 1
        else:
            da['model_answer'] = model_ans
            wrong_data.append(da)
        type2score[ty][1] += 1
    res = {}

    for k,v in type2score.items():
        print(f'【{k}】准确率：{(v[0]/v[1] if v[0] > 0 else 0) :.3f}   题目总数：{v[1]}')
        res[k] = (v[0]/v[1] if v[0] > 0 else 0)

    print(f'总分：{sum([sc[0] for sc in type2score.values()])/len(datas) :.3f}   题目总数：{len(datas)}')
    res['总分'] = sum([sc[0] for sc in type2score.values()])/len(datas)

    print(f'错误题目：{len(wrong_data)}道，没有匹配答案的题目{miss_match_num}道 已输出到 {wrong_data_file_path}')
    with open(wrong_data_file_path,'w') as fw:
        json.dump(wrong_data,fw,ensure_ascii=False,indent=2)
    
    if iswandb:
        table = wandb.Table(columns=["Input", "Output","Answer"])
        for da in datas[:30]:
            table.add_data(get_query(da),da['huatuo_answer_0'],da['answer'])
        res['InputOutputTable'] = table

    return res

sample_num = 5


debug = False
import copy
def show_latex_data(res):
    if 'medqa_MCMLE' not in res:
        return 
    res = copy.deepcopy(res)
    for k in res:
        res[k] = res[k]*100
    res1 = 'Model_name  & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}  \\\\'
    res1_keys = ['medqa_MCMLE','medqa_USMLE','medmcqa_dev','CMB_test','CMEexma_test','mmlu_med_test','cmmlu_med_test','ceval_med_test','truthful_qa_choice']
    print(res1.format(*[res[k] for k in res1_keys]))
    # print(res.keys())
    res2 = 'Model_name & {:.1f} & {:.1f} & \multicolumn{{1}}{{c|}}{{ {:.1f} }} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & \multicolumn{{1}}{{c|}}{{ {:.1f} }} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & \multicolumn{{1}}{{c|}}{{ {:.1f} }}   & {:.1f} & {:.1f} & {:.1f} \\\\'
    res2_keys = ['kaoshi_cn___中医执业助理医师__2015年真题','kaoshi_cn___中医执业助理医师__2016年真题','kaoshi_cn___中医执业助理医师__2017年真题','kaoshi_cn___中医执业医师__2012年真题','kaoshi_cn___中医执业医师__2013年真题','kaoshi_cn___中医执业医师__2016年真题','kaoshi_cn___临床执业助理医师__2018年真题','kaoshi_cn___临床执业助理医师__2019年真题','kaoshi_cn___临床执业助理医师__2020年真题', \
                 'kaoshi_cn___临床执业医师__2018年真题','kaoshi_cn___临床执业医师__2019年真题','kaoshi_cn___临床执业医师__2020年真题','kaoshi_cn___执业中药师__2017年真题','kaoshi_cn___执业中药师__2018年真题','kaoshi_cn___执业中药师__2019年真题','kaoshi_cn___执业西药师__2021真题','kaoshi_cn___执业西药师__2022真题']
    sum_num = sum(res[k] for k in res2_keys) // len(res2_keys)
    print(res2.format(*([res[k] for k in res2_keys])+[sum_num]))

    res3 = 'Model_name & {:.2f} & {:.2f} & {:.2f} \\\\'
    res3_keys = ['USMLE___step1','USMLE___step2&3','USMLE']
    print(res3.format(*[res[k] for k in res3_keys]))

test_func = match_choice2
def test_choice(choice_data):
    # 用于测每个药剂师每个项
    type2score = {}
    sample_data = {}
    ty_set = set()
    res = {}
    opt_ans = ['A','B','C','D','E']
    miss_match_num = 0

    if len(choice_data) == 0:
        return sample_data,res

    def _test_choice(da,type2score,sample_data,ty,ans,model_ans):
        if ty not in type2score:
            type2score[ty] = [0,0]
        if sample_data is not None:
            sample_data[ty] = []
            if len(sample_data[ty]) < sample_num:
                sample_data[ty].append(da)
        if model_ans == ans:
            type2score[ty][0] += 1
        type2score[ty][1] += 1

    for da in choice_data:
        ty = da['dataset']
        ans = da['answer']
        # model_ans = da['model_ans']
        model_ans = test_func(da[f'huatuo_answer_0'],da)
        if len(model_ans) < 1:
            miss_match_num += 1
            model_ans = 'B'
        if len(ans) == 1:
            model_ans = model_ans[0]
        
        ty_set.add(ty)
        _test_choice(da,type2score,sample_data,ty,ans,model_ans)
        if 'question_type' in da:
            _test_choice(da,type2score,None,ty+'___'+da['question_type'], ans, model_ans)

    print(f'没有匹配答案的题目{miss_match_num}道')

    for k,v in type2score.items():
        print(f'【{k}】准确率：{(v[0]/v[1] if v[0] > 0 else 0) :.4f}   题目总数：{v[1]}')
        res[k] = (v[0]/v[1] if v[0] > 0 else 0)

    print(f'选择题总分：{sum([sc[0] for k,sc in type2score.items() if "___" not in k ])/len(choice_data) :.3f}   选择题总数：{len(choice_data)}')
    res['选择题总分'] = sum([sc[0] for k,sc in type2score.items() if "___" not in k ])/len(choice_data)

    if debug:
        print(json.dumps(random.sample(choice_data,30),ensure_ascii=False,indent=2))
        
        show_latex_data(res)
    return sample_data,res

def test_chat(chat_data,sample_data):
    for da in chat_data:
        ty = da['dataset']
        da['answer'] = da['output']
        if sample_data is not None:
            sample_data[ty] = []
            if len(sample_data[ty]) < sample_num:
                sample_data[ty].append(da)
    return sample_data
        
    

def score_mix(finished_path, datas, iswandb = False, ans_num = 1):
    global test_func
    test_func = match_choice2
    print(f'{ans_num} vote')
    # process_json(finished_path)
    if finished_path is not None:
        with open(finished_path,'w', encoding='utf-8') as fw:
            json.dump(datas,fw,ensure_ascii=False,indent=2)

    choice_data = []
    chat_data = []
    
    print(f'{os.path.split(finished_path)[-1] if finished_path else ""} 题目总共 {len(datas)}道')
    for da in datas:
        if 'option' in da:
            choice_data.append(da)
        else:
            chat_data.append(da)
    sample_data,res = test_choice(choice_data)    
    sample_data = test_chat(chat_data,sample_data)

    if iswandb:
        table = wandb.Table(columns=["Input", "Output","Answer","Dataset"])
        for ty,tydata in sample_data.items():
            for da in tydata:
                table.add_data(da['query'],da['huatuo_answer_0'],da['answer'],ty)
        res['InputOutputTable'] = table
    return res


def score_mix2(finished_path, datas, iswandb = False, ans_num = 1):
    global test_func
    test_func = match_choice3
    print(f'{ans_num} vote')
    # process_json(finished_path)
    if finished_path is not None:
        with open(finished_path,'w', encoding='utf-8') as fw:
            json.dump(datas,fw,ensure_ascii=False,indent=2)

    choice_data = []
    chat_data = []
    
    print(f'{os.path.split(finished_path)[-1] if finished_path else ""} 题目总共 {len(datas)}道')
    for da in datas:
        if 'option' in da:
            choice_data.append(da)
        else:
            chat_data.append(da)
    sample_data,res = test_choice(choice_data)    
    sample_data = test_chat(chat_data,sample_data)

    if iswandb:
        table = wandb.Table(columns=["Input", "Output","Answer","Dataset"])
        for ty,tydata in sample_data.items():
            for da in tydata:
                table.add_data(da['query'],da['huatuo_answer_0'],da['answer'],ty)
        res['InputOutputTable'] = table
    return res

def score_mix3(finished_path, datas, iswandb = False, ans_num = 1):
    global test_func
    test_func = match_choice
    print(f'{ans_num} vote')
    # process_json(finished_path)
    if finished_path is not None:
        with open(finished_path,'w', encoding='utf-8') as fw:
            json.dump(datas,fw,ensure_ascii=False,indent=2)

    choice_data = []
    chat_data = []
    
    print(f'{os.path.split(finished_path)[-1] if finished_path else ""} 题目总共 {len(datas)}道')
    for da in datas:
        if 'option' in da:
            choice_data.append(da)
        else:
            chat_data.append(da)
    sample_data,res = test_choice(choice_data)    
    sample_data = test_chat(chat_data,sample_data)

    if iswandb:
        table = wandb.Table(columns=["Input", "Output","Answer","Dataset"])
        for ty,tydata in sample_data.items():
            for da in tydata:
                table.add_data(da['query'],da['huatuo_answer_0'],da['answer'],ty)
        res['InputOutputTable'] = table
    return res

def score_result_fewshot_prob(finished_path, iswandb = False, ans_num = 5):
    print(f'{ans_num} vote')
    # process_json(finished_path)
    datas = []
    with open(finished_path) as f:
        for l in f:
            datas.append(json.loads(l))
    q_type = ['最佳选择题','配伍选择题','综合分析选择题','多项选择题']
    type2score = {k:[0,0] for k in q_type}
    wrong_data = []
    out_name = os.path.basename(finished_path).split('.json')[0]
    wrong_data_file_path = f'./output/{out_name}_wrong_choice.json'

    print(f'{os.path.split(finished_path)[-1]}题目总共 {len(datas)}道')
    miss_match_num = 0
    for da in datas:
        ty = da['question_type']
        ans = da['answer']
        model_ans = da['model_ans']
        if model_ans == ans:
            type2score[ty][0] += 1
        else:
            wrong_data.append(da)
        type2score[ty][1] += 1

    res = {}

    for k,v in type2score.items():
        print(f'【{k}】准确率：{(v[0]/v[1] if v[0] > 0 else 0) :.3f}   题目总数：{v[1]}')
        res[k] = (v[0]/v[1] if v[0] > 0 else 0)

    print(f'总分：{sum([sc[0] for sc in type2score.values()])/len(datas) :.3f}   题目总数：{len(datas)}')
    res['总分'] = sum([sc[0] for sc in type2score.values()])/len(datas)

    print(f'错误题目：{len(wrong_data)}道，没有匹配答案的题目{miss_match_num}道 已输出到 {wrong_data_file_path}')
    with open(wrong_data_file_path,'w') as fw:
        json.dump(wrong_data,fw,ensure_ascii=False,indent=2)

    if iswandb:
        table = wandb.Table(columns=["Input", "Output","Answer"])
        for da in datas[:30]:
            table.add_data(get_query(da),da['model_ans'],da['answer'])
        res['InputOutputTable'] = table
    return res