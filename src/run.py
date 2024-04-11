import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import argparse
random.seed(0)

import dataset
import model
import trainer
import utils
import london_baseline


argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
argp.add_argument('variant', help="Choose vanilla or perceiver") 
argp.add_argument('--bottleneck_dim', type=int, default=32)
argp.add_argument('pretrain_corpus_path', default=None)
argp.add_argument('--reading_params_path',default=None)
argp.add_argument('--writing_params_path',default=None)
argp.add_argument('--finetune_corpus_path', default=None)
argp.add_argument('--eval_corpus_path', default=None)
argp.add_argument('--outputs_path', default=None)
argp.add_argument('--pretrain_lr', default=6e-3, type=float)
argp.add_argument('--finetune_lr', default=6e-4, type=float)
argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                  default='run')
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# TensorBoard training log
writer = SummaryWriter(log_dir='expt/%s/%s_%s_%d_pt_lr_%f_ft_lr_%f' % (
    args.function,
    args.tb_expt_name,
    args.variant,
    args.bottleneck_dim,
    args.pretrain_lr,
    args.finetune_lr))

# Keep the block size 128
# Why is the pretraining corpus always required (even if we're not pretraining?)
# It's because we're using it as a hack to always have the same vocabulary
# (that is, the same mapping from character to integer, and we build the
# vocab from the pretraining corpus.)
block_size = 128
text = open(args.pretrain_corpus_path, encoding='utf-8').read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

# hyperparameters, for both the vanilla and the perceiver models
mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
    n_layer=4, n_head=8, n_embd=256)


# define models.
if args.variant == 'vanilla':
    model = model.GPT(mconf).to(device)

elif args.variant == 'perceiver':
    mconf.perceiver = True
    mconf.bottleneck_dim = args.bottleneck_dim
    model = model.GPT(mconf).to(device)
else:
    raise ValueError("Unknown model variant")

print('Model on device: ', next(model.parameters()).device)

# Perform pretraining, finetuning, or evaluation
if args.function == 'pretrain':
    assert args.writing_params_path is not None
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path
    tconf = trainer.TrainerConfig(max_epochs=650,batch_size=128,learning_rate=args.pretrain_lr,lr_decay=True,warmup_tokens=512*20,final_tokens=200*len(pretrain_dataset)*block_size,num_workers=4,writer=writer)
    trainer = trainer.Trainer(model, pretrain_dataset, None, tconf)
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)

elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    if args.reading_params_path is not None:
        print("with pretrain")
        #If args.reading_params_path is specified, load these parameters into the model
        model.load_state_dict(torch.load(args.reading_params_path))
        # with pretrain
        tconf = trainer.TrainerConfig(max_epochs=10, batch_size=256, learning_rate=args.finetune_lr,
                                      lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,num_workers=4, writer=writer)
    else:
        # no pretrain
        print("no pretrain")
        tconf = trainer.TrainerConfig(max_epochs=75, batch_size=256, learning_rate=args.finetune_lr,lr_decay=True, warmup_tokens=512 * 20, final_tokens=200*len(pretrain_dataset)*block_size, num_workers=4, writer=writer)

    finetune_dataset = dataset.NameDataset(pretrain_dataset, open(args.finetune_corpus_path, encoding='utf-8').read())
    trainer = trainer.Trainer(model, finetune_dataset, None, tconf)
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)


elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    # model.load_state_dict(torch.load(args.reading_params_path))
    # for running on cpu
    model.load_state_dict(torch.load(args.reading_params_path, map_location=torch.device('cpu')))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions) # original evaluate function

    if total > 0:
      print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('Predictions written to {}; no targets provided'
                .format(args.outputs_path))



elif args.function == 'get_name':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path, map_location=torch.device('cpu')))


    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
            x = line.split('\t')[0]
            x = x + '⁇'
            pred = utils.get_name_prediction(model, pretrain_dataset, x,train_dataset=pretrain_dataset)
            predictions.append(pred)
            print(f"Input: {x}, Prediction: {pred}")  # print input and prediction
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
      print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('Predictions written to {}; no targets provided'
                .format(args.outputs_path))
