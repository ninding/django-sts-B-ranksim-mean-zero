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
import logging
import numpy as np

from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

from loss import *
from util import calibrate_mean_var


def build_model(args, vocab, pretrained_embs, tasks):
    '''
    Build model according to arguments
    '''
    d_word, n_layers_highway = args.d_word, args.n_layers_highway

    # Build embedding layers
    if args.glove:
        word_embs = pretrained_embs
        train_embs = bool(args.train_words)
    else:
        logging.info("\tLearning embeddings from scratch!")
        word_embs = None
        train_embs = True
    word_embedder = Embedding(vocab.get_vocab_size('tokens'), d_word, weight=word_embs, trainable=train_embs,
                              padding_index=vocab.get_token_index('@@PADDING@@'))
    d_inp_phrase = 0

    token_embedder = {"words": word_embedder}
    d_inp_phrase += d_word
    text_field_embedder = BasicTextFieldEmbedder(token_embedder)
    d_hid_phrase = args.d_hid

    # Build encoders
    phrase_layer = s2s_e.by_name('lstm').from_params(Params({'input_size': d_inp_phrase,
                                                             'hidden_size': d_hid_phrase,
                                                             'num_layers': args.n_layers_enc,
                                                             'bidirectional': True})).cuda()
    pair_encoder = HeadlessPairEncoder(vocab, text_field_embedder, n_layers_highway,
                                       phrase_layer, dropout=args.dropout)
    d_pair = 2 * d_hid_phrase

    if args.fds:
        _FDS = FDS(feature_dim=d_pair * 4, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                   start_update=args.start_update, start_smooth=args.start_smooth,
                   kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)

    # Build model and classifiers
    model = MultiTaskModel(args, pair_encoder, _FDS if args.fds else None)
    build_regressor(tasks, model, d_pair)

    if args.cuda >= 0:
        model = model.cuda()

    return model

def build_regressor(tasks, model, d_pair):
    '''
    Build the regressor
    '''
    for task in tasks:
        d_task =  d_pair * 4
        model.build_regressor(task, d_task)
    return

class MultiTaskModel(nn.Module):
    def __init__(self, args, pair_encoder, FDS=None):
        super(MultiTaskModel, self).__init__()
        self.args = args
        self.pair_encoder = pair_encoder

        self.FDS = FDS
        self.start_smooth = args.start_smooth

    def build_regressor(self, task, d_inp):
        d_inp = 24000
        layer = nn.Linear(d_inp, 1)
        setattr(self, '%s_pred_layer' % task.name, layer)

    def forward(self, task=None, epoch=None, input1=None, input2=None, mask1=None, mask2=None, label=None, weight=None):
        pred_layer = getattr(self, '%s_pred_layer' % task.name)

        pair_emb = self.pair_encoder(input1, input2, mask1, mask2)
        pair_emb_s = pair_emb
        if self.training and self.FDS is not None:
            if epoch >= self.start_smooth:
                pair_emb_s = self.FDS.smooth(pair_emb_s, label, epoch)
        logits = pred_layer(pair_emb_s)
        if logits is not None:
            logits = logits.cuda()
        if label is not None:
            label = label.cuda()
        if weight is not None:
            weight = weight.cuda() 
        out = {}
        if self.training and (self.FDS is not None or self.args.regularization_weight>0):
            out['embs'] = pair_emb
            out['labels'] = label
        if self.args.loss == 'huber':
            loss = globals()[f"weighted_{self.args.loss}_loss"](
                inputs=logits, targets=label / torch.tensor(5.).cuda(), weights=weight,
                beta=self.args.huber_beta
            )
        else:
            loss = globals()[f"weighted_{self.args.loss}_loss"](
                inputs=logits, targets=label / torch.tensor(5.).cuda(), weights=weight
            )
        out['logits'] = logits
        label = label.squeeze(-1).data.cpu().numpy()
        # print('LABEL:', label)
        # print('LABEL shape: ', label.shape)
        logits = logits.squeeze(-1).data.cpu().numpy()
        # print(out['embs'].shape)
        # exit()
        task.scorer(logits, label)
        out['loss'] = loss

        return out

class HeadlessPairEncoder(Model):
    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 dropout=0.2, mask_lstms=True, initializer=InitializerApplicator()):
        super(HeadlessPairEncoder, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        d_emb = text_field_embedder.get_output_dim()
        self._highway_layer = TimeDistributed(Highway(d_emb, num_highway_layers))

        self._phrase_layer = phrase_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)
        self.output_dim = phrase_layer.get_output_dim()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, s1, s2, m1=None, m2=None):
        s1={key:s1[key].cuda() for key in s1}
        s2={key:s2[key].cuda() for key in s2}
        s1_embs = self._highway_layer(self._text_field_embedder(s1) if m1 is None else s1)
        s2_embs = self._highway_layer(self._text_field_embedder(s2) if m2 is None else s2)

        s1_embs = self._dropout(s1_embs)
        s2_embs = self._dropout(s2_embs)

        # Set up masks
        s1_mask = util.get_text_field_mask(s1) if m1 is None else m1.long()
        s2_mask = util.get_text_field_mask(s2) if m2 is None else m2.long()

        s1_lstm_mask = s1_mask.float() if self._mask_lstms else None
        s2_lstm_mask = s2_mask.float() if self._mask_lstms else None

        # Sentence encodings with LSTMs
        s1_enc = self._phrase_layer(s1_embs, s1_lstm_mask)
        s2_enc = self._phrase_layer(s2_embs, s2_lstm_mask)

        s1_enc = self._dropout(s1_enc)
        s2_enc = self._dropout(s2_enc)

        # Max pooling
        s1_mask = s1_mask.unsqueeze(dim=-1)
        s2_mask = s2_mask.unsqueeze(dim=-1)
        s1_enc.data.masked_fill_(1 - s1_mask.byte().data, -float('inf'))
        s2_enc.data.masked_fill_(1 - s2_mask.byte().data, -float('inf'))
        s1_enc, _ = s1_enc.max(dim=1)
        s2_enc, _ = s2_enc.max(dim=1)
        s1_enc = torch.cat([s1_enc,torch.tensor(np.zeros(s1_enc.shape),dtype=s1_enc.dtype).cuda()],1)
        s2_enc = torch.cat([s2_enc,torch.tensor(np.zeros(s2_enc.shape),dtype=s2_enc.dtype).cuda()],1)

        return torch.cat([s1_enc, s2_enc, torch.abs(s1_enc - s2_enc), s1_enc * s2_enc], 1)

class FDS(nn.Module):

    def __init__(self, feature_dim, bucket_num=50, bucket_start=0, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma)
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(
                map(laplace, np.arange(-half_ks, half_ks + 1)))

        logging.info(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).cuda()

    def _get_bucket_idx(self, label):
        label = np.float32(label)
        _, bins_edges = np.histogram(a=np.array([], dtype=np.float32), bins=self.bucket_num, range=(0., 5.))
        if label == 5.:
            return self.bucket_num - 1
        else:
            return max(np.where(bins_edges > label)[0][0] - 1, self.bucket_start)

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            logging.info(f"Updated smoothed statistics of last epoch on Epoch [{epoch}]!")

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        buckets = np.array([self._get_bucket_idx(label) for label in labels])
        for bucket in np.unique(buckets):
            curr_feats = features[torch.tensor((buckets == bucket).astype(np.uint8))]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[bucket - self.bucket_start] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[bucket - self.bucket_start]))
            factor = 0 if epoch == self.start_update else factor
            self.running_mean[bucket - self.bucket_start] = \
                (1 - factor) * curr_mean + factor * self.running_mean[bucket - self.bucket_start]
            self.running_var[bucket - self.bucket_start] = \
                (1 - factor) * curr_var + factor * self.running_var[bucket - self.bucket_start]

        # make up for zero training samples buckets
        for bucket in range(self.bucket_start, self.bucket_num):
            if bucket not in np.unique(buckets):
                if bucket == self.bucket_start:
                    self.running_mean[0] = self.running_mean[1]
                    self.running_var[0] = self.running_var[1]
                elif bucket == self.bucket_num - 1:
                    self.running_mean[bucket - self.bucket_start] = self.running_mean[bucket - self.bucket_start - 1]
                    self.running_var[bucket - self.bucket_start] = self.running_var[bucket - self.bucket_start - 1]
                else:
                    self.running_mean[bucket - self.bucket_start] = (self.running_mean[bucket - self.bucket_start - 1] +
                                                                     self.running_mean[bucket - self.bucket_start + 1]) / 2.
                    self.running_var[bucket - self.bucket_start] = (self.running_var[bucket - self.bucket_start - 1] +
                                                                    self.running_var[bucket - self.bucket_start + 1]) / 2.
        logging.info(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features

        labels = labels.squeeze(1)
        buckets = np.array([self._get_bucket_idx(label) for label in labels])
        for bucket in np.unique(buckets):
            features[torch.tensor((buckets == bucket).astype(np.uint8))] = calibrate_mean_var(
                features[torch.tensor((buckets == bucket).astype(np.uint8))],
                self.running_mean_last_epoch[bucket - self.bucket_start],
                self.running_var_last_epoch[bucket - self.bucket_start],
                self.smoothed_mean_last_epoch[bucket - self.bucket_start],
                self.smoothed_var_last_epoch[bucket - self.bucket_start]
            )

        return features
