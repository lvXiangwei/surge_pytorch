import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import numpy as np


def read_user_sequence(filename, n_questions, max_len=100, min_len=3, ratio=1.0):
    with open(filename, 'r') as f:
        lines = f.readlines()
    his_pro, his_y, his_len, cur_pro, cur_y, his_skill, cur_skill = [], [], [], [], [], [], []
    index = 0
    while index < len(lines):
        num = eval(lines[index])
        tmp_pro = list(eval(lines[index+1]))
        tmp_pro = [ele+1 for ele in tmp_pro]

        tmp_skills = list(eval(lines[index+2]))
        tmp_skills = [ele+1 for ele in tmp_skills]

        tmp_ans = list(eval(lines[index+3]))
        tmp_ans = [ele for ele in tmp_ans]

        for i in range(min_len, num):  # answer, 3 histroy
            history_pro = tmp_pro[max(0, i - max_len):i]
            history_skill = tmp_skills[max(0, i - max_len):i]
            history_ans = tmp_ans[max(0, i - max_len):i]
            history_length = len(history_pro)
            ans = tmp_ans[i]
            pro = tmp_pro[i]
            skill = tmp_skills[i]
            assert 3 <= history_length <= max_len

            his_y.append(torch.tensor(history_ans))
            his_skill.append(torch.tensor(history_skill))
            his_pro.append(torch.tensor(history_pro))
            his_len.append(history_length)
            cur_pro.append(pro)
            cur_skill.append(skill)
            cur_y.append(ans)
        index += 4
    # import ipdb; ipdb.set_trace()
    n = int(len(his_pro) * ratio)
    return {
        "his_pro": his_skill[:n],
        # "his_pro": his_pro[:n],
        "his_y": his_y[:n],
        "his_len": his_len[:n],
        "cur_pro": cur_skill[:n],
        # "cur_pro": cur_pro[:n],
        "cur_y": cur_y[:n]
    }


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.pad_pro_seq = pad_sequence(
            data_dict['his_pro'], batch_first=True, padding_value=0)
        self.pad_ans_seq = pad_sequence(
            data_dict['his_y'], batch_first=True, padding_value=2)

    def __getitem__(self, index):
        ''' history seq, history ans, cur item, cur ans'''
        return self.pad_pro_seq[index], self.pad_ans_seq[index], torch.tensor(self.data_dict["his_len"][index]), torch.tensor(self.data_dict["cur_pro"][index]), torch.tensor(self.data_dict["cur_y"][index])

    def __len__(self):
        return len(self.data_dict["cur_y"])


def get_dataloader(data_folder, max_step=200, batch_size=128, mode='problem', ratio=1.0):
    problem_hashmap_path = os.path.join(data_folder, 'problem_id_hashmap.json')
    pro_hashmap = json.load(open(problem_hashmap_path, 'r'))
    skill_hashmap_path = os.path.join(data_folder, 'skill_id_hashmap.json')
    skill_hashmap = json.load(open(skill_hashmap_path, 'r'))

    n_questions = len(pro_hashmap)

    train_dict = read_user_sequence(
        f'{data_folder}/train.txt', n_questions=n_questions, max_len=max_step, min_len=3, ratio=ratio)
    val_dict = read_user_sequence(
        f'{data_folder}/dev.txt', n_questions=n_questions, max_len=max_step, min_len=3, ratio=ratio)
    test_dict = read_user_sequence(
        f'{data_folder}/test.txt', n_questions=n_questions, max_len=max_step, min_len=3, ratio=ratio)

    assert mode in ['problem', 'skill']

    train_dataset = CustomDataset(train_dict)
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(val_dict)
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = CustomDataset(test_dict)
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, val_data_loader, test_data_loader


if __name__ == '__main__':
    data_folder = '/data/home/hejiansu/KT/xiangwei/surge/my_data/assist09/origin'
    train_data_loader, val_data_loader, test_data_loader = get_dataloader(
        data_folder)
    for data in test_data_loader:
        import ipdb
        ipdb.set_trace()
        print(data)
