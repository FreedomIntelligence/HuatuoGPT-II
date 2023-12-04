#%%
import requests
import json

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import random
from retrying import retry

class GPT4:
    def __init__(self) -> None:
        self.key_ind = 0
        self.init_api_keys()
        self.max_wrong_time = 5

    def init_api_keys(self):
        self.keys = []
        with open('gpt4key.txt', encoding="utf-8", mode="r") as fr:
            for l in fr:
                cols = l.split('---')
                if len(cols[0]) < 45 or len(cols[0]) > 55:
                    continue
                if len(cols) == 1:
                    cols.append('None')
                self.keys.append((cols[0],cols[1]))
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
        parameters = {
            "model": 'gpt-4',
            "messages": [{'role': 'user', 'content': content}],
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
                # print(f'del {self.keys[self.key_ind]}')
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
    
    @retry(wait_fixed=200, stop_max_attempt_number=10)
    def retry_call(self, content, args = {}):
        return self.call(content, args)
        
        
