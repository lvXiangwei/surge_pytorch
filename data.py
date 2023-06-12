# %%
import numpy as np
import pickle
import pandas as pd
import os
import json

# %%
origin_path = './origin'
cur_path = '.'

# %% [markdown]
# ### Vocab
# - user_vocab : {'0': 0}
# - item_vocab:  {problem_id : idx}
# - category_vocab: {'0': 0, '1': 1, '2': 2}

# %%
# user vocab
with open('user_vocab.pkl', 'wb') as fw:
        dic = {'0': 0}
        pickle.dump(dic, fw)

# %%
# pro vocab ==> item_vocab
with open(os.path.join(origin_path, 'problem_id_hashmap.json'), 'rb') as f:
    dic = json.load(f)
    dic = {str(i): i for i in range(len(dic))}
    with open('item_vocab.pkl', 'wb') as fw:
        pickle.dump(dic, fw)
    

# %%
# cat_vocab
with open('category_vocab.pkl', 'wb') as fw:
    dic = {'0': 0, '1': 1, '2': 2}
    pickle.dump(dic, fw)

# %%
# train_data
def read_user_sequence(filename, max_len=200, min_len=3):
    with open(filename, 'r') as f:
        lines = f.readlines()
    y, skill, problem = [], [], []
    index = 0
    while index < len(lines):
        num = eval(lines[index])
        tmp_pro = list(eval(lines[index+1])[:max_len])
        tmp_pro = [ele+1 for ele in tmp_pro]

        tmp_skills = list(eval(lines[index+2])[:max_len])
        tmp_skills = [ele+1 for ele in tmp_skills] 
        
        tmp_ans = list(eval(lines[index+3])[:max_len])
        tmp_ans = [ele for ele in tmp_ans]
        
        for i in range(0, num, max_len):
            pros = tmp_pro[i:min(num, max_len + i)]
            skills = tmp_skills[i:min(num, max_len + i)]
            ans = tmp_ans[i:min(num, max_len + i)]
            cur_len = len(pros)
            if cur_len < min_len: 
                continue
            y.append(ans)
            skill.append(skills)
            problem.append(pros)
        index += 4        
    return problem, skill, y
    

# %%
def get_train_data(inputfile, outputfile, ratio=1):
    problem, skill, y = read_user_sequence(inputfile)
    problem = problem[:int(len(problem)*ratio)]
    y = y[:int(len(y)*ratio)]
    with open(outputfile, 'w') as f:
        for pro, y_ in zip(problem, y):
            n = len(pro)
            for i in range(1, n): # i 为当前，0~i-1 为历史
                f.write(
                    str(y_[i]) + '\t' +     # ans 
                    '0' + '\t'  +           # user   
                    str(pro[i]) + '\t' +    # pro/item
                    '2' + '\t' + 
                    str(i) + '\t' +         # cur_time
                    ','.join(map(str, pro[:i])) + '\t' +  # history pros/items
                    ','.join(map(str, y_[:i])) + '\t' +   # history ans
                    ','.join(map(str, range(i))) + '\n'   # history ts
                )    

# %%
train_input = f'{origin_path}/train.txt'
valid_input = f'{origin_path}/dev.txt'
test_input = f'{origin_path}/test.txt'

# %%
get_train_data(train_input, 'train_data')
get_train_data(valid_input, 'valid_data')
get_train_data(test_input, 'test_data')
# get_train_data(train_input, 'train_data', ratio=0.01)
# get_train_data(valid_input, 'valid_data', ratio=0.01)
# get_train_data(test_input, 'test_data', ratio=0.01)

# %%
def parse_file(self, input_file):
    """Parse the file to a list ready to be used for downstream tasks
    
    Args:
        input_file: One of train, valid or test file which has never been parsed.
    
    Returns: 
        list: A list with parsing result
    """
    with open(input_file, "r") as f:
        lines = f.readlines()
    res = []
    for line in lines:
        if not line:
            continue
        res.append(self.parser_one_line(line))
    return res

def parser_one_line(self, line):
    """Parse one string line into feature values.
        a line was saved as the following format:
        label \t user_hash \t item_hash \t item_cate \t operation_time \t item_history_sequence \t item_cate_history_sequence \t time_history_sequence

    Args:
        line (str): a string indicating one instance

    Returns:
        tuple/list: Parsed results including label, user_id, target_item_id, target_category, item_history, cate_history(, timeinterval_history,
        timelast_history, timenow_history, mid_mask, seq_len, learning_rate)

    """
    # import ipdb; ipdb.set_trace()
    words = line.strip().split(self.col_spliter)
    label = int(words[0])
    user_id = self.userdict[words[1]] if words[1] in self.userdict else 0
    item_id = self.itemdict[words[2]] if words[2] in self.itemdict else 0
    item_cate = self.catedict[words[3]] if words[3] in self.catedict else 0
    current_time = float(words[4])

    time_history_sequence = []

    item_history_words = words[5].strip().split(",")
    cate_history_words = words[6].strip().split(",")
    item_history_sequence, cate_history_sequence = self.get_item_cate_history_sequence(item_history_words, cate_history_words, user_id)
    time_history_words = words[7].strip().split(",")
    time_history_sequence = self.get_time_history_sequence(time_history_words)


# %%
with open('/data/home/hejiansu/KT/xiangwei/rec/SIGIR21-SURGE/my_data/assist09/train_data', "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
                words = line.strip().split('\t')
                assert len(words) == 8, (i, words)

# %%



