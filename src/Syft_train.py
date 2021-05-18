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
hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
bob = sy.VirtualWorker(hook, id="bob")  # <-- NEW: define remote worker bob
alice = sy.VirtualWorker(hook, id="alice")  # <-- NEW: and alice
celine = sy.VirtualWorker(hook, id="celine")
david = sy.VirtualWorker(hook, id="david")
elsa = sy.VirtualWorker(hook, id="elsa")

theWorkers = (bob, alice)


# In[2]:


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
config.learning_rate=0.01


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


def train():
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
    
    if model_name == 'DKN':
        try:
            pretrained_entity_embedding = torch.from_numpy(
                np.load(
                    './data/train/pretrained_entity_embedding.npy')).float()
        except FileNotFoundError:
            pretrained_entity_embedding = None

        try:
            pretrained_context_embedding = torch.from_numpy(
                np.load(
                    './data/train/pretrained_context_embedding.npy')).float()
        except FileNotFoundError:
            pretrained_context_embedding = None

        model = Model(config, pretrained_word_embedding,
                      pretrained_entity_embedding,
                      pretrained_context_embedding).to(device)
    else:
        model = Model(config, pretrained_word_embedding).to(device)

    print(model)

    dataset = BaseDataset('./data/train/behaviors_parsed.tsv',
                          './data/train/news_parsed.tsv', 
                          './data/train/roberta')

    print(f"Load training dataset with size {len(dataset)}.")
    ###############################################
    '''
    dataloader = DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=config.num_workers,
                   drop_last=True,
                   pin_memory=True)'''
    ###############################################
    # In the step we need to tranform the dataset in federated manner
    
    '''
    federated_train_loader = sy.FederatedDataLoader(datasets.MNIST(
                                                            '../data', 
                                                            train=True, 
                                                            download=True,
                                                            transform=transforms.Compose(
                                                                            [transforms.ToTensor(),
                                                                             transforms.Normalize((0.1307,), (0.3081,))]
                                                                            )
                                                                    )
    federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader 
                                        dataset.federate((bob, alice)), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
                                        batch_size=args.batch_size, 
                                        shuffle=True, **kwargs)
    dataloader = iter(sy.FederatedDataLoader(dataset.federate((bob, alice)),
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            #num_workers=config.num_workers,
                                            drop_last=True,
                                            #pin_memory=True
                                           ))
                                        '''
    #print(dataset)
    dataloader = sy.FederatedDataLoader(dataset.federate((bob, alice)),
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=config.num_workers,
                                            drop_last=True,
                                            pin_memory=True
                                           )
    ###############################################
    print(f"The training dataset has been loaded!")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                 lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma=0.95, last_epoch=-1)
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
    for i, (minibatch, target) in enumerate(dataloader):
        ##### Get a mini batch of data from federated dataset
        #minibatch ,_ = next(dataloader)
        #print(minibatch)
        #print(minibatch.size())
        #exit()
        #minibatch = next(dataloader)
        step += 1
        if model_name == 'LSTUR':
            y_pred = model(minibatch["user"], minibatch["clicked_news_length"],
                           minibatch["candidate_news"],
                           minibatch["clicked_news"])
        elif model_name == 'HiFiArk':
            y_pred, regularizer_loss = model(minibatch["candidate_news"],
                                             minibatch["clicked_news"])
        elif model_name == 'TANR':
            y_pred, topic_classification_loss = model(
                minibatch["candidate_news"], minibatch["clicked_news"])
        else:
            #################################################
            # Send the model
            model.send(minibatch.location)
            minibatch, target = minibatch.to(device), target.to(device)
            #minibatch = minibatch.to(device)
            #################################################

            y_pred = model(minibatch)
        
        #y = torch.zeros(config.batch_size).long().to(device)
        #print(y_pred.get().size())
        #print(y.size())
        loss = criterion(y_pred, target)

        if model_name == 'HiFiArk':
            if i % 10 == 0:
                writer.add_scalar('Train/BaseLoss', loss.get(), step)
                writer.add_scalar('Train/RegularizerLoss',
                                  regularizer_loss.get(), step)
                writer.add_scalar('Train/RegularizerBaseRatio',
                                  regularizer_loss.get() / loss.get(), step)
            loss += config.regularizer_loss_weight * regularizer_loss
        elif model_name == 'TANR':
            if i % 10 == 0:
                writer.add_scalar('Train/BaseLoss', loss.item(), step)
                writer.add_scalar('Train/TopicClassificationLoss',
                                  topic_classification_loss.item(), step)
                writer.add_scalar(
                    'Train/TopicBaseRatio',
                    topic_classification_loss.item() / loss.item(), step)
            loss += config.topic_classification_loss_weight * topic_classification_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.get()

        loss = loss.get().detach().cpu().item()
        loss_full.append(loss)

        if i % 10 == 0:
            writer.add_scalar('Train/Loss', loss, step)

        if i % config.num_batches_show_loss == 0:
            #print(loss_full)
            #print(type(loss_full))
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, current loss {loss:.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
            )

        if i % config.num_batches_validate == 0:
            (model if model_name != 'Exp1' else models[0]).eval()
            val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                model if model_name != 'Exp1' else models[0], './data/val',
                200000)
            (model if model_name != 'Exp1' else models[0]).train()
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
            elif get_better:
                try:
                    torch.save(
                        {
                            'model_state_dict': (model if model_name != 'Exp1'
                                                 else models[0]).state_dict(),
                            'optimizer_state_dict':
                            (optimizer if model_name != 'Exp1' else
                             optimizefrs[0]).state_dict(),
                            'step':
                            step,
                            'early_stop_value':
                            -val_auc
                        }, f"./checkpoint/{model_name}/ckpt-{step}.pth")
                except OSError as error:
                    print(f"OS error: {error}")


# In[6]:


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


# In[7]:


print('Using device:', device)
print(f'Training model {model_name}')
print(torch.cuda.get_device_name(device))
print("lr is :", config.learning_rate)
train()