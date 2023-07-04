from sklearn.metrics import roc_auc_score
import yaml
import json
import argparse
import torch
import torch.nn as nn
import time 

from data import get_dataloader
from model import Surge
from utils import EarlyStopping, Config, init_random_state
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--config_path', type=str, default='config.yaml')
parser.add_argument('--log_path', type=str, default='assist09.log')
parser.add_argument('--infer', type=bool, default=False)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"
with open(args.config_path, "r", encoding="utf-8") as f:
        config = Config(yaml.load(f, Loader=yaml.FullLoader))
init_random_state(config.seed)
writer = SummaryWriter(f'./runs/assist09_{config.seed}')

def train(dataloader, model, optimizer, loss_func, epoch, config):
    model.train()
    loss_list = []
    auc_list = []
    # for batch in tqdm(dataloader, desc='training ...'):
    for (batch_index, batch) in enumerate(tqdm(dataloader)):
        iteration = (epoch - 1) * len(dataloader) + batch_index
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)
        # if epoch == 1 and batch_index ==0 :
        #     writer.add_graph(model, input_to_model=batch)
        his_pro, his_y, his_len, cur_pro, cur_y = batch
        logit = model(*batch)

        loss = loss_func(logit, cur_y.float())

        loss_list.append(loss.item())
        auc = roc_auc_score(cur_y.cpu().numpy(), logit.detach().cpu().numpy())
        auc_list.append(auc)
        # record loss, auc
        writer.add_scalar('train/loss', loss.item(), iteration)
        writer.add_scalar('train/auc', auc, iteration)
        
        optimizer.zero_grad()
        loss.backward()
        # import ipdb; ipdb.set_trace()
        optimizer.step()
    # for name, param in model.named_parameters():
    #     # import ipdb; ipdb.set_trace()
    #     writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=epoch)
    #     writer.add_histogram(tag=name+'_data', values=param.data, global_step=epoch)
    # import ipdb; ipdb.set_trace()
    return sum(loss_list)/len(loss_list), sum(auc_list)/len(auc_list)


def eval(dataloader, model, loss_func, epoch=None):
    model.eval()
    loss_list = []
    preds = []
    truths = []
    with torch.no_grad():
        for batch in dataloader:
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)
            his_pro, his_y, his_len, cur_pro, cur_y = batch
            logit = model(*batch)
            loss = loss_func(logit, cur_y.float())
            preds.append(logit)
            truths.append(cur_y)
            loss_list.append(loss.item())
    eval_loss = sum(loss_list)/len(loss_list)
    preds = torch.cat(preds).cpu()
    truths = torch.cat(truths).cpu()
    auc = roc_auc_score(truths, preds)
    if epoch:
        writer.add_scalar('eval/loss', eval_loss, epoch)
        writer.add_scalar('eval/auc', auc, epoch)
    return eval_loss, auc


def main():
    
    data_folder = '/data/home/hejiansu/KT/xiangwei/surge/my_data/assist09/origin'

    # load data
    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(data_folder,
                                                                       batch_size=config['batch_size'],
                                                                       ratio=config.ratio)
    # model
    model = Surge(config).to(device)
    #(bs, max_step), (bs, max_step), (bs, ), (bs, ), (bs, )
    
    loss_func = nn.BCEWithLogitsLoss()

    if not args.infer:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['learning_rate'])
        # Early stop
        stopper = EarlyStopping(patience=7, path='surge.ckpt')

        # training...
        start_time = time.time()
        for epoch in range(1, config.epoch):
            train_loss, train_auc = train(train_dataloader, model,
                            optimizer, loss_func, epoch, config)
            eval_loss, val_auc = eval(dev_dataloader, model, loss_func, epoch)

            print(
                f"Epoch {epoch}, train loss: {train_loss:.6f}, train auc: {train_auc:.4f}; eval loss: {eval_loss:.6f}, eval auc: {val_auc:.4f}")
            
            es_flag, es_str = stopper.step(val_auc, model, epoch)
            if es_flag:
                print(
                    f'Early stopped, loading model from epoch-{stopper.best_epoch}')
                break
        end_time = time.time()
        cost_time = end_time - start_time
        print('Time cost for training is {0:.2f} mins'.format((cost_time)/60.0))
    # Test
    model.load_state_dict(torch.load('surge.ckpt'))
    _, test_auc = eval(test_dataloader, model, loss_func)
    print(f"test auc: {test_auc:.4f}")
    writer.close()

if __name__ == '__main__':
    main()
