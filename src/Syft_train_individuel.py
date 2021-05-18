#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import syft as sy  # <-- NEW: import the Pysyft library



from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import BaseDataset
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from config import model_name
from tqdm import tqdm
import os
from pathlib import Path
from evaluate import evaluate
import importlib
import datetime


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "NRMS"
Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
config = getattr(importlib.import_module('config'), f"{model_name}Config")


# In[4]:


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


# In[5]:


def train(fed_num):
    VirtualWorker = []
    hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
    for i in range(fed_num):
        VirtualWorker.append(sy.VirtualWorker(hook, id=str(i)))
    VirtualWorker = tuple(VirtualWorker)
    secure_worker = sy.VirtualWorker(hook, id="secure_worker")
    #bob = sy.VirtualWorker(hook, id="bob")  # <-- NEW: define remote worker bob
    #alice = sy.VirtualWorker(hook, id="alice")  # <-- NEW: and alice
    #celine = sy.VirtualWorker(hook, id="celine")
    #david = sy.VirtualWorker(hook, id="david")
    #elsa = sy.VirtualWorker(hook, id="elsa")
    writer = SummaryWriter(
        log_dir=
        f"./runs/{model_name}/{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    )

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load('./data/train/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    model = Model(config, pretrained_word_embedding)
    print(model)

    dataset = BaseDataset('./data/train/behaviors_parsed.tsv',
                          './data/train/news_parsed.tsv', 
                          './data/train/roberta')

    print(f"Load training dataset with size {len(dataset)}.")
    ###############################################

    ###############################################
    # In the step we need to tranform the dataset in federated manner
    #print(dataset)
    dataloader = sy.FederatedDataLoader(dataset.federate(VirtualWorker),
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=config.num_workers,
                                            drop_last=True,
                                            pin_memory=True
                                           )
    ###############################################
    print(f"The training dataset has been loaded!")
    
    #optimizer = torch.optim.SGD(model.parameters(),lr=config.learning_rate)
        
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping()

    checkpoint_dir = os.path.join('./checkpoint', model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = latest_checkpoint(checkpoint_dir)
    '''
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        early_stopping(checkpoint['early_stop_value'])
        step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
    '''

    #for i in tqdm(range(1,config.num_epochs * len(dataset) // config.batch_size + 1),desc="Training"):
    for _ in range(config.num_epochs):
        models = []
        criterion = nn.CrossEntropyLoss()

        for i in range(fed_num):
            models.append(model.to(device).copy().send(str(i)))
            #criterions.append(nn.CrossEntropyLoss())
        optimizers = []
        for i in range(fed_num):
            optimizers.append(
                torch.optim.Adam(models[i].parameters(),
                                 lr=config.learning_rate)
            )

        for i, (minibatch, target) in enumerate(dataloader):
            step += 1
            minibatch, target = minibatch.to(device), target.to(device)
            location = minibatch.location

            predicts = [0 for _ in range(fed_num)]
            losses = [0 for _ in range(fed_num)]
            for j in range(fed_num):
                if VirtualWorker[j] != location:
                    continue
                else:
                    optimizers[j].zero_grad()
                    predicts[j] = models[j](minibatch)
                    losses[j] = criterion(predicts[j], target)
                    losses[j].backward()
                    optimizers[j].step()
                    losses[j] = losses[j].get().cpu().item()

            print(losses)
            loss = np.sum(losses)
            loss_full.append(loss)


            if i % 10 == 0:
                writer.add_scalar('Train/Loss', loss, step)

            if i % config.num_batches_show_loss == 0:
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, current loss {loss:.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
                )

            if (i % config.num_batches_validate == 0) and (i!=0):
                with torch.no_grad():
                    paraDict = model.state_dict()
                    #model_temp = [0 for _ in range(fed_num)]
                    parasDict = []
                    for k in range(fed_num):
                        #model_temp[k] = models[k].copy().send(secure_worker)
                        models[k].move(secure_worker)
                        parasDict.append(models[k].state_dict())
                    for name in paraDict:
                        paraDict[name] = parasDict[0][name].clone().get()
                        for index in range(1, fed_num):
                            paraDict[name] += parasDict[index][name].clone().get()
                        paraDict[name] /= fed_num
                model.load_state_dict(paraDict)
                    #model = model.to(device)
                models = []
                for index in range(fed_num):
                    models.append(model.to(device).copy().send(str(index)))
                model.eval()
                val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                    model, './data/val',
                    200000)
                model.train()
                writer.add_scalar('Validation/AUC', val_auc, step)
                writer.add_scalar('Validation/MRR', val_mrr, step)
                writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
                writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
                )

                early_stop, get_better = early_stopping(-val_auc)
                if early_stop:
                    tqdm.write('Early stop.')
                    break
                '''
                elif get_better:
                    try:
                        torch.save(
                            {
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'step':
                                step,
                                'early_stop_value':
                                -val_auc
                            }, f"./checkpoint/{model_name}/ckpt-{step}.pth")
                    except OSError as error:
                        print(f"OS error: {error}")'''


def time_since(since):
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


print('Using device:', device)
print(f'Training model {model_name}')
print(torch.cuda.get_device_name(device))
print("lr is:", config.learning_rate)
num_machine = 4
print("Num machine:", num_machine)
train(num_machine)