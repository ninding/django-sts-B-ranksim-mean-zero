# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
########################################################################################
# Code is based on the LDS and FDS (https://arxiv.org/pdf/2102.09554.pdf) implementation
# from https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir 
# by Yuzhe Yang et al.
########################################################################################
import os
import sys
import time
import random
import shutil
import logging
import argparse
import datetime
# import Merge

import torch
import numpy as np
from allennlp.data.iterators import BasicIterator

from djangoSts.origin.preprocess import build_tasks, myprocess_task
from djangoSts.origin.models import build_model
from djangoSts.origin.trainer import build_trainer
from djangoSts.origin.evaluate import evaluate
from djangoSts.origin.util import device_mapping, query_yes_no, resume_checkpoint
from djangoSts.origin.loss import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main(s,mydata=None):

    args = argparse.Namespace()
    d = vars(args)
    d.update(s)
    if 'cuda' not in d:
        d['cuda'] = 0
    # parser.add_argument('--cuda', help='-1 if no CUDA, else gpu id (single gpu is enough)', type=int, default=0)
    if 'random_seed' not in d:
        d['random_seed'] = 111
    # parser.add_argument('--random_seed', help='random seed to use', type=int, default=111)

    # Paths and logging
    if 'log_file' not in d:
        d['log_file'] = 'training.log'

    # parser.add_argument('--log_file', help='file to log to', type=str, default='training.log')
    if 'store_root' not in d:
        d['store_root'] = './djangoSts/origin/checkpoint'
    # parser.add_argument('--store_root', help='store root path', type=str, default='checkpoint')
    if 'store_name' not in d:
        d['store_name'] = 'sts'
    # parser.add_argument('--store_name', help='store name prefix for current experiment', type=str, default='sts')
    if 'suffix' not in d:
        d['suffix'] = ''
    # parser.add_argument('--suffix', help='store name suffix for current experiment', type=str, default='')
    if 'word_embs_file' not in d:
        d['word_embs_file'] = './shared-data/imb-reg/glove/glove.840B.300d.txt'
    # parser.add_argument('--word_embs_file', help='file containing word embs', type=str, default='./shared-data/imb-reg/glove/glove.840B.300d.txt')

    # Training resuming flag
    if 'resume' not in d:
        d['resume'] = False
    else:
        d['resume'] = True
    # parser.add_argument('--resume', help='whether to resume training', action='store_true', default=False)

    # Tasks
    if 'task' not in d:
        d['task'] = 'sts-b'
    # parser.add_argument('--task', help='training and evaluation task', type=str, default='sts-b')

    # Preprocessing options
    if 'max_seq_len' not in d:
        d['max_seq_len'] = 40
    # parser.add_argument('--max_seq_len', help='max sequence length', type=int, default=40)
    if 'max_word_v_size' not in d:
        d['max_word_v_size'] = 30000
    # parser.add_argument('--max_word_v_size', help='max word vocab size', type=int, default=30000)

    # Embedding options
    if 'dropout_embs' not in d:
        d['dropout_embs'] = .2
    # parser.add_argument('--dropout_embs', help='dropout rate for embeddings', type=float, default=.2)
    if 'd_word' not in d:
        d['d_word'] = 300
    # parser.add_argument('--d_word', help='dimension of word embeddings', type=int, default=300)
    if 'glove' not in d:
        d['glove'] = 1
    # parser.add_argument('--glove', help='1 if use glove, else from scratch', type=int, default=1)
    if 'train_words' not in d:
        d['train_words'] = 0
    # parser.add_argument('--train_words', help='1 if make word embs trainable', type=int, default=0)

    # Model options
    if 'd_hid' not in d:
        d['d_hid'] = 1500
    # parser.add_argument('--d_hid', help='hidden dimension size', type=int, default=1500)
    if 'n_layers_enc' not in d:
        d['n_layers_enc'] = 2
    # parser.add_argument('--n_layers_enc', help='number of RNN layers', type=int, default=2)
    if 'n_layers_highway' not in d:
        d['n_layers_highway'] = 0
    # parser.add_argument('--n_layers_highway', help='number of highway layers', type=int, default=0)
    if 'dropout' not in d:
        d['dropout'] = 0.2
    # parser.add_argument('--dropout', help='dropout rate to use in training', type=float, default=0.2)

    # Training options
    if 'batch_size' not in d:
        d['batch_size'] = 16
    # parser.add_argument('--batch_size', help='batch size', type=int, default=16)
    if 'optimizer' not in d:
        d['optimizer'] = 'adam'
    # parser.add_argument('--optimizer', help='optimizer to use', type=str, default='adam')
    if 'lr' not in d:
        d['lr'] = 2.5e-4
    # parser.add_argument('--lr', help='starting learning rate', type=float, default=2.5e-4)
    if 'loss' not in d:
        d['loss'] = 'mse'
    # parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'])
    if 'huber_beta' not in d:
        d['huber_beta'] = 0.3
    # parser.add_argument('--huber_beta', type=float, default=0.3, help='beta for huber loss')
    if 'max_grad_norm' not in d:
        d['max_grad_norm'] = 5.
    # parser.add_argument('--max_grad_norm', help='max grad norm', type=float, default=5.)
    if 'val_interval' not in d:
        d['val_interval'] = 400
    # parser.add_argument('--val_interval', help='number of iterations between validation checks', type=int, default=400)
    if 'max_vals' not in d:
        d['max_vals'] = 300
    # parser.add_argument('--max_vals', help='maximum number of validation checks', type=int, default=300)
    if 'patience' not in d:
        d['patience'] = 30
    # parser.add_argument('--patience', help='patience for early stopping', type=int, default=30)

    # imbalanced related
    # LDS
    if 'lds' not in d:
        d['lds'] = False
    else:
        d['lds'] = True
    # parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
    if 'lds_kernel' not in d:
        d['lds_kernel'] = 'gaussian'
    # parser.add_argument('--lds_kernel', type=str, default='gaussian',
                        # choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    if 'lds_ks' not in d:
        d['lds_ks'] = 5
    # parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
    if 'lds_sigma' not in d:
        d['lds_sigma'] = 2
    # parser.add_argument('--lds_sigma', type=float, default=2, help='LDS gaussian/laplace kernel sigma')
    # FDS
    if 'fds' not in d:
        d['fds'] = False
    else:
        d['fds'] = True
    # parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
    if 'fds_kernel' not in d:
        d['fds_kernel'] = 'gaussian'
    # parser.add_argument('--fds_kernel', type=str, default='gaussian',
                        # choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
    if 'fds_ks' not in d:
        d['fds_ks'] = 5
    # parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
    if 'fds_sigma' not in d:
        d['fds_sigma'] = 2
    # parser.add_argument('--fds_sigma', type=float, default=2, help='FDS gaussian/laplace kernel sigma')
    if 'start_update' not in d:
        d['start_update'] = 0
    # parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
    if 'start_smooth' not in d:
        d['start_smooth'] = 1
    # parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
    if 'bucket_num' not in d:
        d['bucket_num'] = 50
    # parser.add_argument('--bucket_num', type=int, default=50, help='maximum bucket considered for FDS')
    if 'bucket_start' not in d:
        d['bucket_start'] = 0
    # parser.add_argument('--bucket_start', type=int, default=0, help='minimum(starting) bucket for FDS')
    if 'fds_mmt' not in d:
        d['fds_mmt'] = 0.9
    # parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

    # re-weighting: SQRT_INV / INV
    if 'reweight' not in d:
        d['reweight'] = 'none'
    # parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'],
    #                     help='cost-sensitive reweighting scheme')
    # two-stage training: RRT
    if 'retrain_fc' not in d:
        d['retrain_fc'] = False
    else:
        d['retrain_fc'] = True
    # parser.add_argument('--retrain_fc', action='store_true', default=False,
                        # help='whether to retrain last regression layer (regressor)')
    if 'pretrained' not in d:
        d['pretrained'] = ''
    # parser.add_argument('--pretrained', type=str, default='', help='pretrained checkpoint file path to load backbone weights for RRT')
    # evaluate only
    if 'evaluate' not in d:
        d['evaluate'] = False
    else:
        d['evaluate'] = True
    # parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate only flag')
    if 'eval_model' not in d:
        d['eval_model'] = ''
    # parser.add_argument('--eval_model', type=str, default='', help='the model to evaluate on; if not specified, '
    #                                                                'use the default best model in store_dir')
    # batchwise ranking regularizer
    if 'regularization_weight' not in d:
        d['regularization_weight'] = 0
    # parser.add_argument('--regularization_weight', type=float, default=0, help='weight of the regularization term')
    if 'interpolation_lambda' not in d:
        d['interpolation_lambda'] = 1.0
    # parser.add_argument('--interpolation_lambda', type=float, default=1.0, help='interpolation strength')
    # d = argparse.Namespace()
    # print(evaluate in d)
    # d = parser.parse_d(arguments)
    # print(d)

    now_time = str(datetime.datetime.now())
    now_time = '-'.join(now_time.split(' '))
    args.store_root = args.store_root + '_' + now_time

    os.makedirs(args.store_root, exist_ok=True)

    if not args.lds and args.reweight != 'none':
        args.store_name += f'_{args.reweight}'
    if args.lds:
        args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
        if args.lds_kernel in ['gaussian', 'laplace']:
            args.store_name += f'_{args.lds_sigma}'
    if args.fds:
        args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
        if args.fds_kernel in ['gaussian', 'laplace']:
            args.store_name += f'_{args.fds_sigma}'
        args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
    if args.retrain_fc:
        args.store_name += f'_retrain_fc'

    if args.loss == 'huber':
        args.store_name += f'_{args.loss}_beta_{args.huber_beta}'
    else:
        args.store_name += f'_{args.loss}'
    if args.regularization_weight > 0:
        args.store_name += f'_reg{args.regularization_weight}_il{args.interpolation_lambda}'
    args.store_name += f'_seed_{args.random_seed}_valint_{args.val_interval}_patience_{args.patience}' \
                       f'_{args.optimizer}_{args.lr}_{args.batch_size}'
    args.store_name += f'_{args.suffix}' if len(args.suffix) else ''

    # now_time = str(datetime.datetime.now())
    # now_time = '-'.join(now_time.split(' '))
    # args.store_name = args.store_name + '_' + now_time
    args.store_dir = os.path.join(args.store_root, args.store_name)
    # print('store_dir::::',args.store_dir)

    if not args.evaluate and not args.resume:
        if os.path.exists(args.store_dir):
            if query_yes_no('overwrite previous folder: {} ?'.format(args.store_dir)):
                shutil.rmtree(args.store_dir)
                print(args.store_dir + ' removed.\n')
            else:
                raise RuntimeError('Output folder {} already exists'.format(args.store_dir))
        logging.info(f"===> Creating folder: {args.store_dir}")
        os.makedirs(args.store_dir)

    # Logistics
    logging.root.handlers = []
    if os.path.exists(args.store_dir):
        log_file = os.path.join(args.store_dir, args.log_file)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ])
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[logging.StreamHandler()]
        )
    logging.info(args)

    seed = random.randint(1, 10000) if args.random_seed < 0 else args.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda >= 0:
        logging.info("Using GPU %d", args.cuda)
        torch.cuda.set_device(args.cuda)
        torch.cuda.manual_seed_all(seed)
    logging.info("Using random seed %d", seed)

    # Load tasks
    logging.info("Loading tasks...")
    start_time = time.time()
    tasks, vocab, word_embs = build_tasks(args)
    logging.info('\tFinished loading tasks in %.3fs', time.time() - start_time)

    # Build model
    logging.info('Building model...')
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    # print(model)
    # eixt()
    logging.info('\tFinished building model in %.3fs', time.time() - start_time)

    # Set up trainer
    iterator = BasicIterator(args.batch_size)
    trainer, train_params, opt_params = build_trainer(args, model, iterator)

    # Train
    if tasks and not args.evaluate:
        if args.retrain_fc and len(args.pretrained):
            model_path = args.pretrained
            assert os.path.isfile(model_path), f"No checkpoint found at '{model_path}'"
            model_state = torch.load(model_path, map_location=device_mapping(args.cuda))
            trainer._model = resume_checkpoint(trainer._model, model_state, backbone_only=True)
            logging.info(f'Pre-trained backbone weights loaded: {model_path}')
            logging.info('Retrain last regression layer only!')
            for name, param in trainer._model.named_parameters():
                if "sts-b_pred_layer" not in name:
                    param.requires_grad = False
            logging.info(f'Only optimize parameters: {[n for n, p in trainer._model.named_parameters() if p.requires_grad]}')
            to_train = [(n, p) for n, p in trainer._model.named_parameters() if p.requires_grad]
        else:
            to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        trainer.train(tasks, args.val_interval, to_train, opt_params, args.resume)
    else:
        logging.info("Skipping training...")

    logging.info('Testing on test set...')
    model_path = os.path.join(args.store_dir, "model_state_best.th") if not len(args.eval_model) else args.eval_model
    # print(model_path)
    assert os.path.isfile(model_path), f"No checkpoint found at '{model_path}'"
    logging.info(f'Evaluating {model_path}...')
    # print(model_path)
    # exit()
    model_state = torch.load(model_path, map_location=device_mapping(args.cuda))
    model = resume_checkpoint(model, model_state)

    tasks[0].test_data=myprocess_task(mydata or tasks[0].test_data_text,vocab)

    te_preds, te_labels, _ = evaluate(model, tasks, iterator, cuda_device=args.cuda, split="test")
    if not len(args.eval_model):
        np.savez_compressed(os.path.join(args.store_dir, f"{args.store_name}.npz"), preds=te_preds, labels=te_labels)

    logging.info("Done testing.")
    return _

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
