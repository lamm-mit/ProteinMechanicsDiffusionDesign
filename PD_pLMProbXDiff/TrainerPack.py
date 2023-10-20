import os
import time
import copy
from pathlib import Path
from math import ceil
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler

import pytorch_warmup as warmup

import shutil

import esm
from einops import rearrange

from packaging import version
__version__ = '1.9.3'

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

import matplotlib.pyplot as plt

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
from packaging import version

import numpy as np

from ema_pytorch import EMA

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

# # --
# from PD_SpLMxDiff.ModelPack import resize_image_to,ProteinDesigner_B 
# from PD_SpLMxDiff.UtilityPack import get_Model_A_error, convert_into_tokens
# from PD_SpLMxDiff.UtilityPack import decode_one_ems_token_rec,decode_many_ems_token_rec
# from PD_SpLMxDiff.UtilityPack import decode_one_ems_token_rec_for_folding,decode_many_ems_token_rec_for_folding

# from PD_SpLMxDiff.UtilityPack import decode_one_ems_token_rec_for_folding_with_mask,decode_many_ems_token_rec_for_folding_with_mask,read_mask_from_input
# from PD_SpLMxDiff.UtilityPack import get_DSSP_result, string_diff
# from PD_SpLMxDiff.DataSetPack import pad_a_np_arr
# ++
from PD_pLMProbXDiff.ModelPack import (
    resize_image_to, ProteinDesigner_B,
)
from PD_pLMProbXDiff.UtilityPack import (
    get_Model_A_error, convert_into_tokens,convert_into_tokens_using_prob,
    decode_one_ems_token_rec, decode_many_ems_token_rec,
    decode_one_ems_token_rec_for_folding, 
    decode_many_ems_token_rec_for_folding,
    decode_one_ems_token_rec_for_folding_with_mask,
    decode_many_ems_token_rec_for_folding_with_mask,
    read_mask_from_input,
    get_DSSP_result, 
    string_diff,
    load_in_pLM,
)
from PD_pLMProbXDiff.DataSetPack import (
    pad_a_np_arr
)

# loss function 
criterion_MSE_sum =  nn.MSELoss(reduction='sum')
criterion_MAE_sum =  nn.L1Loss(reduction='sum')

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, length = 1):
    if isinstance(val, list):
        val = tuple(val)
    
    return val if isinstance(val, tuple) else ((val,) * length)

def find_first(fn, arr):
    for ind, el in enumerate(arr):
        if fn(el):
            return ind
    return -1

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# url to fs, bucket, path - for checkpointing to cloud

def url_to_bucket(url):
    if '://' not in url:
        return url

    _, suffix = url.split('://')

    if prefix in {'gs', 's3'}:
        return suffix.split('/')[0]
    else:
        raise ValueError(f'storage type prefix "{prefix}" is not supported yet')

# decorators

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def cast_torch_tensor(fn, cast_fp16 = False):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        device = kwargs.pop('_device', model.device)
        cast_device = kwargs.pop('_cast_device', True)

        should_cast_fp16 = cast_fp16 and model.cast_half_at_training

        kwargs_keys = kwargs.keys()
        all_args = (*args, *kwargs.values())
        split_kwargs_index = len(all_args) - len(kwargs_keys)
        all_args = tuple(map(lambda t: torch.from_numpy(t) if exists(t) and isinstance(t, np.ndarray) else t, all_args))

        if cast_device:
            all_args = tuple(map(lambda t: t.to(device) if exists(t) and isinstance(t, torch.Tensor) else t, all_args))

        if should_cast_fp16:
            all_args = tuple(map(lambda t: t.half() if exists(t) and isinstance(t, torch.Tensor) and t.dtype != torch.bool else t, all_args))

        args, kwargs_values = all_args[:split_kwargs_index], all_args[split_kwargs_index:]
        kwargs = dict(tuple(zip(kwargs_keys, kwargs_values)))

        out = fn(model, *args, **kwargs)
        return out
    return inner

# gradient accumulation functions

def split_iterable(it, split_size):
    accum = []
    for ind in range(ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index: (start_index + split_size)])
    return accum

def split(t, split_size = None):
    if not exists(split_size):
        return t

    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim = 0)

    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    return TypeError

def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

def split_args_and_kwargs(*args, split_size = None, **kwargs):
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)
    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
    assert exists(first_tensor)

    batch_size = len(first_tensor)
    split_size = default(split_size, batch_size)
    num_chunks = ceil(batch_size / split_size)

    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len

    split_all_args = [split(arg, split_size = split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else ((arg,) * num_chunks) for arg in all_args]
    chunk_sizes = tuple(map(len, split_all_args[0]))

    for (chunk_size, *chunked_all_args) in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)

# imagen trainer

def imagen_sample_in_chunks(fn):
    @wraps(fn)
    def inner(self, *args, max_batch_size = None, **kwargs):
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)

        if self.imagen.unconditional:
            batch_size = kwargs.get('batch_size')
            batch_sizes = num_to_groups(batch_size, max_batch_size)
            outputs = [fn(self, *args, **{**kwargs, 'batch_size': sub_batch_size}) for sub_batch_size in batch_sizes]
        else:
            outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs)]

        if isinstance(outputs[0], torch.Tensor):
            return torch.cat(outputs, dim = 0)

        return list(map(lambda t: torch.cat(t, dim = 0), list(zip(*outputs))))

    return inner


def restore_parts(state_dict_target, state_dict_from):
    for name, param in state_dict_from.items():

        if name not in state_dict_target:
            continue

        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
        else:
            print(f"layer {name}({param.size()} different than target: {state_dict_target[name].size()}")

    return state_dict_target

class ImagenTrainer(nn.Module):
    locked = False

    def __init__(
        self,
        #imagen = None,
        model = None,
        
        imagen_checkpoint_path = None,
        use_ema = True,
        lr = 1e-4,
        eps = 1e-8,
        beta1 = 0.9,
        beta2 = 0.99,
        max_grad_norm = None,
        group_wd_params = True,
        warmup_steps = None,
        cosine_decay_max_steps = None,
        only_train_unet_number = None,
        fp16 = False,
        precision = None,
        split_batches = True,
        dl_tuple_output_keywords_names = ('images', 'text_embeds', 'text_masks', 'cond_images'),
        verbose = True,
        split_valid_fraction = 0.025,
        split_valid_from_train = False,
        split_random_seed = 42,
        checkpoint_path = None,
        checkpoint_every = None,
        checkpoint_fs = None,
        fs_kwargs: dict = None,
        max_checkpoints_keep = 20,
        # +++++++++++++++++++++
        CKeys=None,
        #
        **kwargs
    ):
        super().__init__()
        assert not ImagenTrainer.locked, 'ImagenTrainer can only be initialized once per process - for the sake of distributed training, you will now have to create a separate script to train each unet (or a script that accepts unet number as an argument)'
        assert exists(model.imagen) ^ exists(imagen_checkpoint_path), 'either imagen instance is passed into the trainer, or a checkpoint path that contains the imagen config'

        # determine filesystem, using fsspec, for saving to local filesystem or cloud

        self.fs = checkpoint_fs

        if not exists(self.fs):
            fs_kwargs = default(fs_kwargs, {})
            self.fs, _ = url_to_fs(default(checkpoint_path, './'), **fs_kwargs)
        
        # # -----------------------------------
        # # from MJB
        # assert isinstance(model.imagen, (ProteinDesigner_B))
        # modified by BN
        # ++: try this trainer for all models
        # assert isinstance(model, (ProteinDesigner_B))
        
        # +++++++++++++++++++++++++
        self.CKeys = CKeys
        
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)

         
        self.imagen = model.imagen
       
        

        self.model=model
        self.is_elucidated = self.model.is_elucidated 
        # create accelerator instance

        accelerate_kwargs, kwargs = groupby_prefix_and_trim('accelerate_', kwargs)

        assert not (fp16 and exists(precision)), 'either set fp16 = True or forward the precision ("fp16", "bf16") to Accelerator'
        accelerator_mixed_precision = default(precision, 'fp16' if fp16 else 'no')

        self.accelerator = Accelerator(**{
            'split_batches': split_batches,
            'mixed_precision': accelerator_mixed_precision,
            'kwargs_handlers': [DistributedDataParallelKwargs(find_unused_parameters = True)]
        , **accelerate_kwargs})

        ImagenTrainer.locked = self.is_distributed

        # cast data to fp16 at training time if needed

        self.cast_half_at_training = accelerator_mixed_precision == 'fp16'

        # grad scaler must be managed outside of accelerator

        grad_scaler_enabled = fp16
   
        self.num_unets = len(self.imagen.unets)

        self.use_ema = use_ema and self.is_main
        self.ema_unets = nn.ModuleList([])

        # keep track of what unet is being trained on
        # only going to allow 1 unet training at a time

        self.ema_unet_being_trained_index = -1 # keeps track of which ema unet is being trained on

        # data related functions

        self.train_dl_iter = None
        self.train_dl = None

        self.valid_dl_iter = None
        self.valid_dl = None

        self.dl_tuple_output_keywords_names = dl_tuple_output_keywords_names

        # auto splitting validation from training, if dataset is passed in

        self.split_valid_from_train = split_valid_from_train

        assert 0 <= split_valid_fraction <= 1, 'split valid fraction must be between 0 and 1'
        self.split_valid_fraction = split_valid_fraction
        self.split_random_seed = split_random_seed

        # be able to finely customize learning rate, weight decay
        # per unet

        lr, eps, warmup_steps, cosine_decay_max_steps = map(partial(cast_tuple, length = self.num_unets), (lr, eps, warmup_steps, cosine_decay_max_steps))

        for ind, (unet, unet_lr, unet_eps, unet_warmup_steps, unet_cosine_decay_max_steps) in enumerate(zip(self.imagen.unets, lr, eps, warmup_steps, cosine_decay_max_steps)):
            optimizer = Adam(
                unet.parameters(),
                lr = unet_lr,
                eps = unet_eps,
                betas = (beta1, beta2),
                **kwargs
            )

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

            scaler = GradScaler(enabled = grad_scaler_enabled)

            scheduler = warmup_scheduler = None

            if exists(unet_cosine_decay_max_steps):
                scheduler = CosineAnnealingLR(optimizer, T_max = unet_cosine_decay_max_steps)

            if exists(unet_warmup_steps):
                warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period = unet_warmup_steps)

                if not exists(scheduler):
                    scheduler = LambdaLR(optimizer, lr_lambda = lambda step: 1.0)

            # set on object

            setattr(self, f'optim{ind}', optimizer) # cannot use pytorch ModuleList for some reason with optimizers
            setattr(self, f'scaler{ind}', scaler)
            setattr(self, f'scheduler{ind}', scheduler)
            setattr(self, f'warmup{ind}', warmup_scheduler)

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        # step tracker and misc

        self.register_buffer('steps', torch.tensor([0] * self.num_unets))

        self.verbose = verbose

        # automatic set devices based on what accelerator decided

        self.imagen.to(self.device)
        self.to(self.device)

        # checkpointing

        assert not (exists(checkpoint_path) ^ exists(checkpoint_every))
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.max_checkpoints_keep = max_checkpoints_keep

        self.can_checkpoint = self.is_local_main if isinstance(checkpoint_fs, LocalFileSystem) else self.is_main

        if exists(checkpoint_path) and self.can_checkpoint:
            bucket = url_to_bucket(checkpoint_path)

            if not self.fs.exists(bucket):
                self.fs.mkdir(bucket)

            self.load_from_checkpoint_folder()

        # only allowing training for unet

        self.only_train_unet_number = only_train_unet_number
        self.validate_and_set_unet_being_trained(only_train_unet_number)

    # computed values

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @property
    def unwrapped_unet(self):
        return self.accelerator.unwrap_model(self.unet_being_trained)

    # optimizer helper functions

    def get_lr(self, unet_number):
        self.validate_unet_number(unet_number)
        unet_index = unet_number - 1

        optim = getattr(self, f'optim{unet_index}')

        return optim.param_groups[0]['lr']

    # function for allowing only one unet from being trained at a time

    def validate_and_set_unet_being_trained(self, unet_number = None):
        if exists(unet_number):
            self.validate_unet_number(unet_number)

        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you cannot only train on one unet at a time. you will need to save the trainer into a checkpoint, and resume training on a new unet'

        self.only_train_unet_number = unet_number
        self.imagen.only_train_unet_number = unet_number

        if not exists(unet_number):
            return

        self.wrap_unet(unet_number)

    def wrap_unet(self, unet_number):
        if hasattr(self, 'one_unet_wrapped'):
            return

        unet = self.imagen.get_unet(unet_number)
        self.unet_being_trained = self.accelerator.prepare(unet)
        unet_index = unet_number - 1

        optimizer = getattr(self, f'optim{unet_index}')
        scheduler = getattr(self, f'scheduler{unet_index}')

        optimizer = self.accelerator.prepare(optimizer)

        if exists(scheduler):
            scheduler = self.accelerator.prepare(scheduler)

        setattr(self, f'optim{unet_index}', optimizer)
        setattr(self, f'scheduler{unet_index}', scheduler)

        self.one_unet_wrapped = True

    # hacking accelerator due to not having separate gradscaler per optimizer

    def set_accelerator_scaler(self, unet_number):
        unet_number = self.validate_unet_number(unet_number)
        scaler = getattr(self, f'scaler{unet_number - 1}')

        self.accelerator.scaler = scaler
        for optimizer in self.accelerator._optimizers:
            optimizer.scaler = scaler

    # helper print

    def print(self, msg):
        if not self.is_main:
            return

        if not self.verbose:
            return

        return self.accelerator.print(msg)

    # validating the unet number

    def validate_unet_number(self, unet_number = None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        assert 0 < unet_number <= self.num_unets, f'unet number should be in between 1 and {self.num_unets}'
        return unet_number

    # number of training steps taken

    def num_steps_taken(self, unet_number = None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        return self.steps[unet_number - 1].item()

    def print_untrained_unets(self):
        print_final_error = False

        for ind, (steps, unet) in enumerate(zip(self.steps.tolist(), self.imagen.unets)):
            if steps > 0 or isinstance(unet, NullUnet):
                continue

            self.print(f'unet {ind + 1} has not been trained')
            print_final_error = True

        if print_final_error:
            self.print('when sampling, you can pass stop_at_unet_number to stop early in the cascade, so it does not try to generate with untrained unets')

    # data related functions

    def add_train_dataloader(self, dl = None):
        if not exists(dl):
            return

        assert not exists(self.train_dl), 'training dataloader was already added'
        self.train_dl = self.accelerator.prepare(dl)

    def add_valid_dataloader(self, dl):
        if not exists(dl):
            return

        assert not exists(self.valid_dl), 'validation dataloader was already added'
        self.valid_dl = self.accelerator.prepare(dl)

    def add_train_dataset(self, ds = None, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        assert not exists(self.train_dl), 'training dataloader was already added'

        valid_ds = None
        if self.split_valid_from_train:
            train_size = int((1 - self.split_valid_fraction) * len(ds))
            valid_size = len(ds) - train_size

            ds, valid_ds = random_split(ds, [train_size, valid_size], generator = torch.Generator().manual_seed(self.split_random_seed))
            self.print(f'training with dataset of {len(ds)} samples and validating with randomly splitted {len(valid_ds)} samples')

        dl = DataLoader(ds, batch_size = batch_size, **dl_kwargs)
        self.train_dl = self.accelerator.prepare(dl)

        if not self.split_valid_from_train:
            return

        self.add_valid_dataset(valid_ds, batch_size = batch_size, **dl_kwargs)

    def add_valid_dataset(self, ds, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        assert not exists(self.valid_dl), 'validation dataloader was already added'

        dl = DataLoader(ds, batch_size = batch_size, **dl_kwargs)
        self.valid_dl = self.accelerator.prepare(dl)

    def create_train_iter(self):
        assert exists(self.train_dl), 'training dataloader has not been registered with the trainer yet'

        if exists(self.train_dl_iter):
            return

        self.train_dl_iter = cycle(self.train_dl)

    def create_valid_iter(self):
        assert exists(self.valid_dl), 'validation dataloader has not been registered with the trainer yet'

        if exists(self.valid_dl_iter):
            return

        self.valid_dl_iter = cycle(self.valid_dl)

    def train_step(self, unet_number = None, **kwargs):
        self.create_train_iter()
        loss = self.step_with_dl_iter(self.train_dl_iter, unet_number = unet_number, **kwargs)
        self.update(unet_number = unet_number)
        return loss

    @torch.no_grad()
    @eval_decorator
    def valid_step(self, **kwargs):
        self.create_valid_iter()

        context = self.use_ema_unets if kwargs.pop('use_ema_unets', False) else nullcontext

        with context():
            loss = self.step_with_dl_iter(self.valid_dl_iter, **kwargs)
        return loss

    def step_with_dl_iter(self, dl_iter, **kwargs):
        dl_tuple_output = cast_tuple(next(dl_iter))
        model_input = dict(list(zip(self.dl_tuple_output_keywords_names, dl_tuple_output)))
        loss = self.forward(**{**kwargs, **model_input})
        return loss

    # checkpointing functions

    @property
    def all_checkpoints_sorted(self):
        glob_pattern = os.path.join(self.checkpoint_path, '*.pt')
        checkpoints = self.fs.glob(glob_pattern)
        sorted_checkpoints = sorted(checkpoints, key = lambda x: int(str(x).split('.')[-2]), reverse = True)
        return sorted_checkpoints

    def load_from_checkpoint_folder(self, last_total_steps = -1):
        if last_total_steps != -1:
            filepath = os.path.join(self.checkpoint_path, f'checkpoint.{last_total_steps}.pt')
            self.load(filepath)
            return

        sorted_checkpoints = self.all_checkpoints_sorted

        if len(sorted_checkpoints) == 0:
            self.print(f'no checkpoints found to load from at {self.checkpoint_path}')
            return

        last_checkpoint = sorted_checkpoints[0]
        self.load(last_checkpoint)

    def save_to_checkpoint_folder(self):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        total_steps = int(self.steps.sum().item())
        filepath = os.path.join(self.checkpoint_path, f'checkpoint.{total_steps}.pt')

        self.save(filepath)

        if self.max_checkpoints_keep <= 0:
            return

        sorted_checkpoints = self.all_checkpoints_sorted
        checkpoints_to_discard = sorted_checkpoints[self.max_checkpoints_keep:]

        for checkpoint in checkpoints_to_discard:
            self.fs.rm(checkpoint)

    # saving and loading functions

    def save(
        self,
        path,
        overwrite = True,
        without_optim_and_sched = False,
        **kwargs
    ):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        fs = self.fs

        assert not (fs.exists(path) and not overwrite)

        self.reset_ema_unets_all_one_device()

        save_obj = dict(
            model = self.imagen.state_dict(),
            version = __version__,
            steps = self.steps.cpu(),
            **kwargs
        )

        save_optim_and_sched_iter = range(0, self.num_unets) if not without_optim_and_sched else tuple()

        for ind in save_optim_and_sched_iter:
            scaler_key = f'scaler{ind}'
            optimizer_key = f'optim{ind}'
            scheduler_key = f'scheduler{ind}'
            warmup_scheduler_key = f'warmup{ind}'

            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)
            warmup_scheduler = getattr(self, warmup_scheduler_key)

            if exists(scheduler):
                save_obj = {**save_obj, scheduler_key: scheduler.state_dict()}

            if exists(warmup_scheduler):
                save_obj = {**save_obj, warmup_scheduler_key: warmup_scheduler.state_dict()}

            save_obj = {**save_obj, scaler_key: scaler.state_dict(), optimizer_key: optimizer.state_dict()}

        if self.use_ema:
            save_obj = {**save_obj, 'ema': self.ema_unets.state_dict()}

        # determine if imagen config is available

        if hasattr(self.imagen, '_config'):
            self.print(f'this checkpoint is commandable from the CLI - "imagen --model {str(path)} \"<prompt>\""')

            save_obj = {
                **save_obj,
                'imagen_type': 'elucidated' if self.is_elucidated else 'original',
                'imagen_params': self.imagen._config
            }

        #save to path

        with fs.open(path, 'wb') as f:
            torch.save(save_obj, f)

        self.print(f'checkpoint saved to {path}')

    def load(self, path, only_model = False, strict = True, noop_if_not_exist = False):
        fs = self.fs

        if noop_if_not_exist and not fs.exists(path):
            self.print(f'trainer checkpoint not found at {str(path)}')
            return

        assert fs.exists(path), f'{path} does not exist'

        self.reset_ema_unets_all_one_device()

        # to avoid extra GPU memory usage in main process when using Accelerate

        with fs.open(path) as f:
            loaded_obj = torch.load(f, map_location='cpu')

        if version.parse(__version__) != version.parse(loaded_obj['version']):
            self.print(f'loading saved imagen at version {loaded_obj["version"]}, but current package version is {__version__}')

        try:
            self.imagen.load_state_dict(loaded_obj['model'], strict = strict)
        except RuntimeError:
            print("Failed loading state dict. Trying partial load")
            self.imagen.load_state_dict(restore_parts(self.imagen.state_dict(),
                                                      loaded_obj['model']))

        if only_model:
            return loaded_obj

        self.steps.copy_(loaded_obj['steps'])

        for ind in range(0, self.num_unets):
            scaler_key = f'scaler{ind}'
            optimizer_key = f'optim{ind}'
            scheduler_key = f'scheduler{ind}'
            warmup_scheduler_key = f'warmup{ind}'

            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)
            warmup_scheduler = getattr(self, warmup_scheduler_key)

            if exists(scheduler) and scheduler_key in loaded_obj:
                scheduler.load_state_dict(loaded_obj[scheduler_key])

            if exists(warmup_scheduler) and warmup_scheduler_key in loaded_obj:
                warmup_scheduler.load_state_dict(loaded_obj[warmup_scheduler_key])

            if exists(optimizer):
                try:
                    optimizer.load_state_dict(loaded_obj[optimizer_key])
                    scaler.load_state_dict(loaded_obj[scaler_key])
                except:
                    self.print('could not load optimizer and scaler, possibly because you have turned on mixed precision training since the last run. resuming with new optimizer and scalers')

        if self.use_ema:
            assert 'ema' in loaded_obj
            try:
                self.ema_unets.load_state_dict(loaded_obj['ema'], strict = strict)
            except RuntimeError:
                print("Failed loading state dict. Trying partial load")
                self.ema_unets.load_state_dict(restore_parts(self.ema_unets.state_dict(),
                                                             loaded_obj['ema']))

        self.print(f'checkpoint loaded from {path}')
        return loaded_obj

    # managing ema unets and their devices

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    def get_ema_unet(self, unet_number = None):
        if not self.use_ema:
            return

        unet_number = self.validate_unet_number(unet_number)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.ema_unets]
            delattr(self, 'ema_unets')
            self.ema_unets = unets_list

        if index != self.ema_unet_being_trained_index:
            for unet_index, unet in enumerate(self.ema_unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.ema_unet_being_trained_index = index
        return self.ema_unets[index]

    def reset_ema_unets_all_one_device(self, device = None):
        if not self.use_ema:
            return

        device = default(device, self.device)
        self.ema_unets = nn.ModuleList([*self.ema_unets])
        self.ema_unets.to(device)

        self.ema_unet_being_trained_index = -1

    @torch.no_grad()
    @contextmanager
    def use_ema_unets(self):
        if not self.use_ema:
            output = yield
            return output

        self.reset_ema_unets_all_one_device()
        self.imagen.reset_unets_all_one_device()

        self.unets.eval()

        trainable_unets = self.imagen.unets
        self.imagen.unets = self.unets                  # swap in exponential moving averaged unets for sampling

        output = yield

        self.imagen.unets = trainable_unets             # restore original training unets

        # cast the ema_model unets back to original device
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        return output

    def print_unet_devices(self):
        self.print('unet devices:')
        for i, unet in enumerate(self.imagen.unets):
            device = next(unet.parameters()).device
            self.print(f'\tunet {i}: {device}')

        if not self.use_ema:
            return

        self.print('\nema unet devices:')
        for i, ema_unet in enumerate(self.ema_unets):
            device = next(ema_unet.parameters()).device
            self.print(f'\tema unet {i}: {device}')

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_ema_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_ema_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # encoding text functions

    def encode_text(self, text, **kwargs):
        return self.imagen.encode_text(text, **kwargs)

    # forwarding functions and gradient step updates

    def update(self, unet_number = None):
        unet_number = self.validate_unet_number(unet_number)
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        index = unet_number - 1
        unet = self.unet_being_trained

        optimizer = getattr(self, f'optim{index}')
        scaler = getattr(self, f'scaler{index}')
        scheduler = getattr(self, f'scheduler{index}')
        warmup_scheduler = getattr(self, f'warmup{index}')

        # set the grad scaler on the accelerator, since we are managing one per u-net

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(unet.parameters(), self.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        if self.use_ema:
            ema_unet = self.get_ema_unet(unet_number)
            ema_unet.update()

        # scheduler, if needed

        maybe_warmup_context = nullcontext() if not exists(warmup_scheduler) else warmup_scheduler.dampening()

        with maybe_warmup_context:
            if exists(scheduler) and not self.accelerator.optimizer_step_was_skipped: # recommended in the docs
                scheduler.step()

        self.steps += F.one_hot(torch.tensor(unet_number - 1, device = self.steps.device), num_classes = len(self.steps))

        if not exists(self.checkpoint_path):
            return

        total_steps = int(self.steps.sum().item())

        if total_steps % self.checkpoint_every:
            return

        self.save_to_checkpoint_folder()

    @torch.no_grad()
    @cast_torch_tensor
    @imagen_sample_in_chunks
    def sample(self, *args, **kwargs):
        context = nullcontext if  kwargs.pop('use_non_ema', False) else self.use_ema_unets

        self.print_untrained_unets()        
        
        if not self.is_main:
            kwargs['use_tqdm'] = False

        with context():
            output = self.imagen.sample(*args, device = self.device, **kwargs)

        return output

    @partial(cast_torch_tensor, cast_fp16 = True)
    def forward(
        self,
        *args,
        unet_number = None,
        max_batch_size = None,
        **kwargs
    ):
        unet_number = self.validate_unet_number(unet_number)
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, f'you can only train unet #{self.only_train_unet_number}'

        total_loss = 0.
        
        
        # + for debug
        if self.CKeys['Debug_TrainerPack']==1:
            print("In Trainer:Forward, check inputs:")
            print('args: ', len(args))
            print('args in:',args[0].shape)
            print('kwargs: ', kwargs.keys())
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            # + for debug
            if self.CKeys['Debug_TrainerPack']==1:
                print("after chunks,...")
                print('chun_frac: ', chunk_size_frac)
                print('chun_args: ', chunked_args)
                print('chun_kwargs: ', chunked_kwargs)
            
            with self.accelerator.autocast():
                loss = self.model(
                    *chunked_args, 
                    unet_number = unet_number, 
                    **chunked_kwargs
                )
                loss = loss * chunk_size_frac
            
            # + for debug
            if self.CKeys['Debug_TrainerPack']==1:
                print('part chun loss: ', loss)

            total_loss += loss#.item()

            if self.training:
                self.accelerator.backward(loss)

        return total_loss
    
# ========================================================
# 
class ImagenTrainer_ModelB(nn.Module):
    locked = False

    def __init__(
        self,
        #imagen = None,
        model = None,
        
        imagen_checkpoint_path = None,
        use_ema = True,
        lr = 1e-4,
        eps = 1e-8,
        beta1 = 0.9,
        beta2 = 0.99,
        max_grad_norm = None,
        group_wd_params = True,
        warmup_steps = None,
        cosine_decay_max_steps = None,
        only_train_unet_number = None,
        fp16 = False,
        precision = None,
        split_batches = True,
        dl_tuple_output_keywords_names = ('images', 'text_embeds', 'text_masks', 'cond_images'),
        verbose = True,
        split_valid_fraction = 0.025,
        split_valid_from_train = False,
        split_random_seed = 42,
        checkpoint_path = None,
        checkpoint_every = None,
        checkpoint_fs = None,
        fs_kwargs: dict = None,
        max_checkpoints_keep = 20,
        # +++++++++++++++++++++
        CKeys=None,
        #
        **kwargs
    ):
        super().__init__()
        assert not ImagenTrainer.locked, 'ImagenTrainer can only be initialized once per process - for the sake of distributed training, you will now have to create a separate script to train each unet (or a script that accepts unet number as an argument)'
        assert exists(model.imagen) ^ exists(imagen_checkpoint_path), 'either imagen instance is passed into the trainer, or a checkpoint path that contains the imagen config'

        # determine filesystem, using fsspec, for saving to local filesystem or cloud

        self.fs = checkpoint_fs

        if not exists(self.fs):
            fs_kwargs = default(fs_kwargs, {})
            self.fs, _ = url_to_fs(default(checkpoint_path, './'), **fs_kwargs)
        
        # # -----------------------------------
        # # from MJB
        # assert isinstance(model.imagen, (ProteinDesigner_B))
        # modified by BN
        # ++
        assert isinstance(model, (ProteinDesigner_B))
        
        # +++++++++++++++++++++++++
        self.CKeys = CKeys
        
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)

         
        self.imagen = model.imagen
       
        

        self.model=model
        self.is_elucidated = self.model.is_elucidated 
        # create accelerator instance

        accelerate_kwargs, kwargs = groupby_prefix_and_trim('accelerate_', kwargs)

        assert not (fp16 and exists(precision)), 'either set fp16 = True or forward the precision ("fp16", "bf16") to Accelerator'
        accelerator_mixed_precision = default(precision, 'fp16' if fp16 else 'no')

        self.accelerator = Accelerator(**{
            'split_batches': split_batches,
            'mixed_precision': accelerator_mixed_precision,
            'kwargs_handlers': [DistributedDataParallelKwargs(find_unused_parameters = True)]
        , **accelerate_kwargs})

        ImagenTrainer.locked = self.is_distributed

        # cast data to fp16 at training time if needed

        self.cast_half_at_training = accelerator_mixed_precision == 'fp16'

        # grad scaler must be managed outside of accelerator

        grad_scaler_enabled = fp16
   
        self.num_unets = len(self.imagen.unets)

        self.use_ema = use_ema and self.is_main
        self.ema_unets = nn.ModuleList([])

        # keep track of what unet is being trained on
        # only going to allow 1 unet training at a time

        self.ema_unet_being_trained_index = -1 # keeps track of which ema unet is being trained on

        # data related functions

        self.train_dl_iter = None
        self.train_dl = None

        self.valid_dl_iter = None
        self.valid_dl = None

        self.dl_tuple_output_keywords_names = dl_tuple_output_keywords_names

        # auto splitting validation from training, if dataset is passed in

        self.split_valid_from_train = split_valid_from_train

        assert 0 <= split_valid_fraction <= 1, 'split valid fraction must be between 0 and 1'
        self.split_valid_fraction = split_valid_fraction
        self.split_random_seed = split_random_seed

        # be able to finely customize learning rate, weight decay
        # per unet

        lr, eps, warmup_steps, cosine_decay_max_steps = map(partial(cast_tuple, length = self.num_unets), (lr, eps, warmup_steps, cosine_decay_max_steps))

        for ind, (unet, unet_lr, unet_eps, unet_warmup_steps, unet_cosine_decay_max_steps) in enumerate(zip(self.imagen.unets, lr, eps, warmup_steps, cosine_decay_max_steps)):
            optimizer = Adam(
                unet.parameters(),
                lr = unet_lr,
                eps = unet_eps,
                betas = (beta1, beta2),
                **kwargs
            )

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

            scaler = GradScaler(enabled = grad_scaler_enabled)

            scheduler = warmup_scheduler = None

            if exists(unet_cosine_decay_max_steps):
                scheduler = CosineAnnealingLR(optimizer, T_max = unet_cosine_decay_max_steps)

            if exists(unet_warmup_steps):
                warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period = unet_warmup_steps)

                if not exists(scheduler):
                    scheduler = LambdaLR(optimizer, lr_lambda = lambda step: 1.0)

            # set on object

            setattr(self, f'optim{ind}', optimizer) # cannot use pytorch ModuleList for some reason with optimizers
            setattr(self, f'scaler{ind}', scaler)
            setattr(self, f'scheduler{ind}', scheduler)
            setattr(self, f'warmup{ind}', warmup_scheduler)

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        # step tracker and misc

        self.register_buffer('steps', torch.tensor([0] * self.num_unets))

        self.verbose = verbose

        # automatic set devices based on what accelerator decided

        self.imagen.to(self.device)
        self.to(self.device)

        # checkpointing

        assert not (exists(checkpoint_path) ^ exists(checkpoint_every))
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.max_checkpoints_keep = max_checkpoints_keep

        self.can_checkpoint = self.is_local_main if isinstance(checkpoint_fs, LocalFileSystem) else self.is_main

        if exists(checkpoint_path) and self.can_checkpoint:
            bucket = url_to_bucket(checkpoint_path)

            if not self.fs.exists(bucket):
                self.fs.mkdir(bucket)

            self.load_from_checkpoint_folder()

        # only allowing training for unet

        self.only_train_unet_number = only_train_unet_number
        self.validate_and_set_unet_being_trained(only_train_unet_number)

    # computed values

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @property
    def unwrapped_unet(self):
        return self.accelerator.unwrap_model(self.unet_being_trained)

    # optimizer helper functions

    def get_lr(self, unet_number):
        self.validate_unet_number(unet_number)
        unet_index = unet_number - 1

        optim = getattr(self, f'optim{unet_index}')

        return optim.param_groups[0]['lr']

    # function for allowing only one unet from being trained at a time

    def validate_and_set_unet_being_trained(self, unet_number = None):
        if exists(unet_number):
            self.validate_unet_number(unet_number)

        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you cannot only train on one unet at a time. you will need to save the trainer into a checkpoint, and resume training on a new unet'

        self.only_train_unet_number = unet_number
        self.imagen.only_train_unet_number = unet_number

        if not exists(unet_number):
            return

        self.wrap_unet(unet_number)

    def wrap_unet(self, unet_number):
        if hasattr(self, 'one_unet_wrapped'):
            return

        unet = self.imagen.get_unet(unet_number)
        self.unet_being_trained = self.accelerator.prepare(unet)
        unet_index = unet_number - 1

        optimizer = getattr(self, f'optim{unet_index}')
        scheduler = getattr(self, f'scheduler{unet_index}')

        optimizer = self.accelerator.prepare(optimizer)

        if exists(scheduler):
            scheduler = self.accelerator.prepare(scheduler)

        setattr(self, f'optim{unet_index}', optimizer)
        setattr(self, f'scheduler{unet_index}', scheduler)

        self.one_unet_wrapped = True

    # hacking accelerator due to not having separate gradscaler per optimizer

    def set_accelerator_scaler(self, unet_number):
        unet_number = self.validate_unet_number(unet_number)
        scaler = getattr(self, f'scaler{unet_number - 1}')

        self.accelerator.scaler = scaler
        for optimizer in self.accelerator._optimizers:
            optimizer.scaler = scaler

    # helper print

    def print(self, msg):
        if not self.is_main:
            return

        if not self.verbose:
            return

        return self.accelerator.print(msg)

    # validating the unet number

    def validate_unet_number(self, unet_number = None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        assert 0 < unet_number <= self.num_unets, f'unet number should be in between 1 and {self.num_unets}'
        return unet_number

    # number of training steps taken

    def num_steps_taken(self, unet_number = None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        return self.steps[unet_number - 1].item()

    def print_untrained_unets(self):
        print_final_error = False

        for ind, (steps, unet) in enumerate(zip(self.steps.tolist(), self.imagen.unets)):
            if steps > 0 or isinstance(unet, NullUnet):
                continue

            self.print(f'unet {ind + 1} has not been trained')
            print_final_error = True

        if print_final_error:
            self.print('when sampling, you can pass stop_at_unet_number to stop early in the cascade, so it does not try to generate with untrained unets')

    # data related functions

    def add_train_dataloader(self, dl = None):
        if not exists(dl):
            return

        assert not exists(self.train_dl), 'training dataloader was already added'
        self.train_dl = self.accelerator.prepare(dl)

    def add_valid_dataloader(self, dl):
        if not exists(dl):
            return

        assert not exists(self.valid_dl), 'validation dataloader was already added'
        self.valid_dl = self.accelerator.prepare(dl)

    def add_train_dataset(self, ds = None, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        assert not exists(self.train_dl), 'training dataloader was already added'

        valid_ds = None
        if self.split_valid_from_train:
            train_size = int((1 - self.split_valid_fraction) * len(ds))
            valid_size = len(ds) - train_size

            ds, valid_ds = random_split(ds, [train_size, valid_size], generator = torch.Generator().manual_seed(self.split_random_seed))
            self.print(f'training with dataset of {len(ds)} samples and validating with randomly splitted {len(valid_ds)} samples')

        dl = DataLoader(ds, batch_size = batch_size, **dl_kwargs)
        self.train_dl = self.accelerator.prepare(dl)

        if not self.split_valid_from_train:
            return

        self.add_valid_dataset(valid_ds, batch_size = batch_size, **dl_kwargs)

    def add_valid_dataset(self, ds, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        assert not exists(self.valid_dl), 'validation dataloader was already added'

        dl = DataLoader(ds, batch_size = batch_size, **dl_kwargs)
        self.valid_dl = self.accelerator.prepare(dl)

    def create_train_iter(self):
        assert exists(self.train_dl), 'training dataloader has not been registered with the trainer yet'

        if exists(self.train_dl_iter):
            return

        self.train_dl_iter = cycle(self.train_dl)

    def create_valid_iter(self):
        assert exists(self.valid_dl), 'validation dataloader has not been registered with the trainer yet'

        if exists(self.valid_dl_iter):
            return

        self.valid_dl_iter = cycle(self.valid_dl)

    def train_step(self, unet_number = None, **kwargs):
        self.create_train_iter()
        loss = self.step_with_dl_iter(self.train_dl_iter, unet_number = unet_number, **kwargs)
        self.update(unet_number = unet_number)
        return loss

    @torch.no_grad()
    @eval_decorator
    def valid_step(self, **kwargs):
        self.create_valid_iter()

        context = self.use_ema_unets if kwargs.pop('use_ema_unets', False) else nullcontext

        with context():
            loss = self.step_with_dl_iter(self.valid_dl_iter, **kwargs)
        return loss

    def step_with_dl_iter(self, dl_iter, **kwargs):
        dl_tuple_output = cast_tuple(next(dl_iter))
        model_input = dict(list(zip(self.dl_tuple_output_keywords_names, dl_tuple_output)))
        loss = self.forward(**{**kwargs, **model_input})
        return loss

    # checkpointing functions

    @property
    def all_checkpoints_sorted(self):
        glob_pattern = os.path.join(self.checkpoint_path, '*.pt')
        checkpoints = self.fs.glob(glob_pattern)
        sorted_checkpoints = sorted(checkpoints, key = lambda x: int(str(x).split('.')[-2]), reverse = True)
        return sorted_checkpoints

    def load_from_checkpoint_folder(self, last_total_steps = -1):
        if last_total_steps != -1:
            filepath = os.path.join(self.checkpoint_path, f'checkpoint.{last_total_steps}.pt')
            self.load(filepath)
            return

        sorted_checkpoints = self.all_checkpoints_sorted

        if len(sorted_checkpoints) == 0:
            self.print(f'no checkpoints found to load from at {self.checkpoint_path}')
            return

        last_checkpoint = sorted_checkpoints[0]
        self.load(last_checkpoint)

    def save_to_checkpoint_folder(self):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        total_steps = int(self.steps.sum().item())
        filepath = os.path.join(self.checkpoint_path, f'checkpoint.{total_steps}.pt')

        self.save(filepath)

        if self.max_checkpoints_keep <= 0:
            return

        sorted_checkpoints = self.all_checkpoints_sorted
        checkpoints_to_discard = sorted_checkpoints[self.max_checkpoints_keep:]

        for checkpoint in checkpoints_to_discard:
            self.fs.rm(checkpoint)

    # saving and loading functions

    def save(
        self,
        path,
        overwrite = True,
        without_optim_and_sched = False,
        **kwargs
    ):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        fs = self.fs

        assert not (fs.exists(path) and not overwrite)

        self.reset_ema_unets_all_one_device()

        save_obj = dict(
            model = self.imagen.state_dict(),
            version = __version__,
            steps = self.steps.cpu(),
            **kwargs
        )

        save_optim_and_sched_iter = range(0, self.num_unets) if not without_optim_and_sched else tuple()

        for ind in save_optim_and_sched_iter:
            scaler_key = f'scaler{ind}'
            optimizer_key = f'optim{ind}'
            scheduler_key = f'scheduler{ind}'
            warmup_scheduler_key = f'warmup{ind}'

            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)
            warmup_scheduler = getattr(self, warmup_scheduler_key)

            if exists(scheduler):
                save_obj = {**save_obj, scheduler_key: scheduler.state_dict()}

            if exists(warmup_scheduler):
                save_obj = {**save_obj, warmup_scheduler_key: warmup_scheduler.state_dict()}

            save_obj = {**save_obj, scaler_key: scaler.state_dict(), optimizer_key: optimizer.state_dict()}

        if self.use_ema:
            save_obj = {**save_obj, 'ema': self.ema_unets.state_dict()}

        # determine if imagen config is available

        if hasattr(self.imagen, '_config'):
            self.print(f'this checkpoint is commandable from the CLI - "imagen --model {str(path)} \"<prompt>\""')

            save_obj = {
                **save_obj,
                'imagen_type': 'elucidated' if self.is_elucidated else 'original',
                'imagen_params': self.imagen._config
            }

        #save to path

        with fs.open(path, 'wb') as f:
            torch.save(save_obj, f)

        self.print(f'checkpoint saved to {path}')

    def load(self, path, only_model = False, strict = True, noop_if_not_exist = False):
        fs = self.fs

        if noop_if_not_exist and not fs.exists(path):
            self.print(f'trainer checkpoint not found at {str(path)}')
            return

        assert fs.exists(path), f'{path} does not exist'

        self.reset_ema_unets_all_one_device()

        # to avoid extra GPU memory usage in main process when using Accelerate

        with fs.open(path) as f:
            loaded_obj = torch.load(f, map_location='cpu')

        if version.parse(__version__) != version.parse(loaded_obj['version']):
            self.print(f'loading saved imagen at version {loaded_obj["version"]}, but current package version is {__version__}')

        try:
            self.imagen.load_state_dict(loaded_obj['model'], strict = strict)
        except RuntimeError:
            print("Failed loading state dict. Trying partial load")
            self.imagen.load_state_dict(restore_parts(self.imagen.state_dict(),
                                                      loaded_obj['model']))

        if only_model:
            return loaded_obj

        self.steps.copy_(loaded_obj['steps'])

        for ind in range(0, self.num_unets):
            scaler_key = f'scaler{ind}'
            optimizer_key = f'optim{ind}'
            scheduler_key = f'scheduler{ind}'
            warmup_scheduler_key = f'warmup{ind}'

            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)
            warmup_scheduler = getattr(self, warmup_scheduler_key)

            if exists(scheduler) and scheduler_key in loaded_obj:
                scheduler.load_state_dict(loaded_obj[scheduler_key])

            if exists(warmup_scheduler) and warmup_scheduler_key in loaded_obj:
                warmup_scheduler.load_state_dict(loaded_obj[warmup_scheduler_key])

            if exists(optimizer):
                try:
                    optimizer.load_state_dict(loaded_obj[optimizer_key])
                    scaler.load_state_dict(loaded_obj[scaler_key])
                except:
                    self.print('could not load optimizer and scaler, possibly because you have turned on mixed precision training since the last run. resuming with new optimizer and scalers')

        if self.use_ema:
            assert 'ema' in loaded_obj
            try:
                self.ema_unets.load_state_dict(loaded_obj['ema'], strict = strict)
            except RuntimeError:
                print("Failed loading state dict. Trying partial load")
                self.ema_unets.load_state_dict(restore_parts(self.ema_unets.state_dict(),
                                                             loaded_obj['ema']))

        self.print(f'checkpoint loaded from {path}')
        return loaded_obj

    # managing ema unets and their devices

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    def get_ema_unet(self, unet_number = None):
        if not self.use_ema:
            return

        unet_number = self.validate_unet_number(unet_number)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.ema_unets]
            delattr(self, 'ema_unets')
            self.ema_unets = unets_list

        if index != self.ema_unet_being_trained_index:
            for unet_index, unet in enumerate(self.ema_unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.ema_unet_being_trained_index = index
        return self.ema_unets[index]

    def reset_ema_unets_all_one_device(self, device = None):
        if not self.use_ema:
            return

        device = default(device, self.device)
        self.ema_unets = nn.ModuleList([*self.ema_unets])
        self.ema_unets.to(device)

        self.ema_unet_being_trained_index = -1

    @torch.no_grad()
    @contextmanager
    def use_ema_unets(self):
        if not self.use_ema:
            output = yield
            return output

        self.reset_ema_unets_all_one_device()
        self.imagen.reset_unets_all_one_device()

        self.unets.eval()

        trainable_unets = self.imagen.unets
        self.imagen.unets = self.unets                  # swap in exponential moving averaged unets for sampling

        output = yield

        self.imagen.unets = trainable_unets             # restore original training unets

        # cast the ema_model unets back to original device
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        return output

    def print_unet_devices(self):
        self.print('unet devices:')
        for i, unet in enumerate(self.imagen.unets):
            device = next(unet.parameters()).device
            self.print(f'\tunet {i}: {device}')

        if not self.use_ema:
            return

        self.print('\nema unet devices:')
        for i, ema_unet in enumerate(self.ema_unets):
            device = next(ema_unet.parameters()).device
            self.print(f'\tema unet {i}: {device}')

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_ema_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_ema_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # encoding text functions

    def encode_text(self, text, **kwargs):
        return self.imagen.encode_text(text, **kwargs)

    # forwarding functions and gradient step updates

    def update(self, unet_number = None):
        unet_number = self.validate_unet_number(unet_number)
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        index = unet_number - 1
        unet = self.unet_being_trained

        optimizer = getattr(self, f'optim{index}')
        scaler = getattr(self, f'scaler{index}')
        scheduler = getattr(self, f'scheduler{index}')
        warmup_scheduler = getattr(self, f'warmup{index}')

        # set the grad scaler on the accelerator, since we are managing one per u-net

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(unet.parameters(), self.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        if self.use_ema:
            ema_unet = self.get_ema_unet(unet_number)
            ema_unet.update()

        # scheduler, if needed

        maybe_warmup_context = nullcontext() if not exists(warmup_scheduler) else warmup_scheduler.dampening()

        with maybe_warmup_context:
            if exists(scheduler) and not self.accelerator.optimizer_step_was_skipped: # recommended in the docs
                scheduler.step()

        self.steps += F.one_hot(torch.tensor(unet_number - 1, device = self.steps.device), num_classes = len(self.steps))

        if not exists(self.checkpoint_path):
            return

        total_steps = int(self.steps.sum().item())

        if total_steps % self.checkpoint_every:
            return

        self.save_to_checkpoint_folder()

    @torch.no_grad()
    @cast_torch_tensor
    @imagen_sample_in_chunks
    def sample(self, *args, **kwargs):
        context = nullcontext if  kwargs.pop('use_non_ema', False) else self.use_ema_unets

        self.print_untrained_unets()        
        
        if not self.is_main:
            kwargs['use_tqdm'] = False

        with context():
            output = self.imagen.sample(*args, device = self.device, **kwargs)

        return output

    @partial(cast_torch_tensor, cast_fp16 = True)
    def forward(
        self,
        *args,
        unet_number = None,
        max_batch_size = None,
        **kwargs
    ):
        unet_number = self.validate_unet_number(unet_number)
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, f'you can only train unet #{self.only_train_unet_number}'

        total_loss = 0.
        
        
        # + for debug
        if self.CKeys['Debug_TrainerPack']==1:
            print("In Trainer:Forward, check inputs:")
            print('args: ', len(args))
            print('args in:',args[0].shape)
            print('kwargs: ', kwargs.keys())
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            # + for debug
            if self.CKeys['Debug_TrainerPack']==1:
                print("after chunks,...")
                print('chun_frac: ', chunk_size_frac)
                print('chun_args: ', chunked_args)
                print('chun_kwargs: ', chunked_kwargs)
            
            with self.accelerator.autocast():
                loss = self.model(
                    *chunked_args, 
                    unet_number = unet_number, 
                    **chunked_kwargs
                )
                loss = loss * chunk_size_frac
            
            # + for debug
            if self.CKeys['Debug_TrainerPack']==1:
                print('part chun loss: ', loss)

            total_loss += loss#.item()

            if self.training:
                self.accelerator.backward(loss)

        return total_loss
    
class ImagenTrainer_Old(nn.Module):
    locked = False

    def __init__(
        self,
        #imagen = None,
        model = None,
        
        imagen_checkpoint_path = None,
        use_ema = True,
        lr = 1e-4,
        eps = 1e-8,
        beta1 = 0.9,
        beta2 = 0.99,
        max_grad_norm = None,
        group_wd_params = True,
        warmup_steps = None,
        cosine_decay_max_steps = None,
        only_train_unet_number = None,
        fp16 = False,
        precision = None,
        split_batches = True,
        dl_tuple_output_keywords_names = ('images', 'text_embeds', 'text_masks', 'cond_images'),
        verbose = True,
        split_valid_fraction = 0.025,
        split_valid_from_train = False,
        split_random_seed = 42,
        checkpoint_path = None,
        checkpoint_every = None,
        checkpoint_fs = None,
        fs_kwargs: dict = None,
        max_checkpoints_keep = 20,
        **kwargs
    ):
        super().__init__()
        assert not ImagenTrainer.locked, 'ImagenTrainer can only be initialized once per process - for the sake of distributed training, you will now have to create a separate script to train each unet (or a script that accepts unet number as an argument)'
        assert exists(model.imagen) ^ exists(imagen_checkpoint_path), 'either imagen instance is passed into the trainer, or a checkpoint path that contains the imagen config'

        # determine filesystem, using fsspec, for saving to local filesystem or cloud

        self.fs = checkpoint_fs

        if not exists(self.fs):
            fs_kwargs = default(fs_kwargs, {})
            self.fs, _ = url_to_fs(default(checkpoint_path, './'), **fs_kwargs)
        
        # # -----------------------------------
        # # from MJB
        # assert isinstance(model.imagen, (ProteinDesigner_B))
        
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)

         
        self.imagen = model.imagen
       
        

        self.model=model
        self.is_elucidated = self.model.is_elucidated 
        # create accelerator instance

        accelerate_kwargs, kwargs = groupby_prefix_and_trim('accelerate_', kwargs)

        assert not (fp16 and exists(precision)), 'either set fp16 = True or forward the precision ("fp16", "bf16") to Accelerator'
        accelerator_mixed_precision = default(precision, 'fp16' if fp16 else 'no')

        self.accelerator = Accelerator(**{
            'split_batches': split_batches,
            'mixed_precision': accelerator_mixed_precision,
            'kwargs_handlers': [DistributedDataParallelKwargs(find_unused_parameters = True)]
        , **accelerate_kwargs})

        ImagenTrainer.locked = self.is_distributed

        # cast data to fp16 at training time if needed

        self.cast_half_at_training = accelerator_mixed_precision == 'fp16'

        # grad scaler must be managed outside of accelerator

        grad_scaler_enabled = fp16
   
        self.num_unets = len(self.imagen.unets)

        self.use_ema = use_ema and self.is_main
        self.ema_unets = nn.ModuleList([])

        # keep track of what unet is being trained on
        # only going to allow 1 unet training at a time

        self.ema_unet_being_trained_index = -1 # keeps track of which ema unet is being trained on

        # data related functions

        self.train_dl_iter = None
        self.train_dl = None

        self.valid_dl_iter = None
        self.valid_dl = None

        self.dl_tuple_output_keywords_names = dl_tuple_output_keywords_names

        # auto splitting validation from training, if dataset is passed in

        self.split_valid_from_train = split_valid_from_train

        assert 0 <= split_valid_fraction <= 1, 'split valid fraction must be between 0 and 1'
        self.split_valid_fraction = split_valid_fraction
        self.split_random_seed = split_random_seed

        # be able to finely customize learning rate, weight decay
        # per unet

        lr, eps, warmup_steps, cosine_decay_max_steps = map(partial(cast_tuple, length = self.num_unets), (lr, eps, warmup_steps, cosine_decay_max_steps))

        for ind, (unet, unet_lr, unet_eps, unet_warmup_steps, unet_cosine_decay_max_steps) in enumerate(zip(self.imagen.unets, lr, eps, warmup_steps, cosine_decay_max_steps)):
            optimizer = Adam(
                unet.parameters(),
                lr = unet_lr,
                eps = unet_eps,
                betas = (beta1, beta2),
                **kwargs
            )

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

            scaler = GradScaler(enabled = grad_scaler_enabled)

            scheduler = warmup_scheduler = None

            if exists(unet_cosine_decay_max_steps):
                scheduler = CosineAnnealingLR(optimizer, T_max = unet_cosine_decay_max_steps)

            if exists(unet_warmup_steps):
                warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period = unet_warmup_steps)

                if not exists(scheduler):
                    scheduler = LambdaLR(optimizer, lr_lambda = lambda step: 1.0)

            # set on object

            setattr(self, f'optim{ind}', optimizer) # cannot use pytorch ModuleList for some reason with optimizers
            setattr(self, f'scaler{ind}', scaler)
            setattr(self, f'scheduler{ind}', scheduler)
            setattr(self, f'warmup{ind}', warmup_scheduler)

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        # step tracker and misc

        self.register_buffer('steps', torch.tensor([0] * self.num_unets))

        self.verbose = verbose

        # automatic set devices based on what accelerator decided

        self.imagen.to(self.device)
        self.to(self.device)

        # checkpointing

        assert not (exists(checkpoint_path) ^ exists(checkpoint_every))
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.max_checkpoints_keep = max_checkpoints_keep

        self.can_checkpoint = self.is_local_main if isinstance(checkpoint_fs, LocalFileSystem) else self.is_main

        if exists(checkpoint_path) and self.can_checkpoint:
            bucket = url_to_bucket(checkpoint_path)

            if not self.fs.exists(bucket):
                self.fs.mkdir(bucket)

            self.load_from_checkpoint_folder()

        # only allowing training for unet

        self.only_train_unet_number = only_train_unet_number
        self.validate_and_set_unet_being_trained(only_train_unet_number)

    # computed values

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @property
    def unwrapped_unet(self):
        return self.accelerator.unwrap_model(self.unet_being_trained)

    # optimizer helper functions

    def get_lr(self, unet_number):
        self.validate_unet_number(unet_number)
        unet_index = unet_number - 1

        optim = getattr(self, f'optim{unet_index}')

        return optim.param_groups[0]['lr']

    # function for allowing only one unet from being trained at a time

    def validate_and_set_unet_being_trained(self, unet_number = None):
        if exists(unet_number):
            self.validate_unet_number(unet_number)

        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you cannot only train on one unet at a time. you will need to save the trainer into a checkpoint, and resume training on a new unet'

        self.only_train_unet_number = unet_number
        self.imagen.only_train_unet_number = unet_number

        if not exists(unet_number):
            return

        self.wrap_unet(unet_number)

    def wrap_unet(self, unet_number):
        if hasattr(self, 'one_unet_wrapped'):
            return

        unet = self.imagen.get_unet(unet_number)
        self.unet_being_trained = self.accelerator.prepare(unet)
        unet_index = unet_number - 1

        optimizer = getattr(self, f'optim{unet_index}')
        scheduler = getattr(self, f'scheduler{unet_index}')

        optimizer = self.accelerator.prepare(optimizer)

        if exists(scheduler):
            scheduler = self.accelerator.prepare(scheduler)

        setattr(self, f'optim{unet_index}', optimizer)
        setattr(self, f'scheduler{unet_index}', scheduler)

        self.one_unet_wrapped = True

    # hacking accelerator due to not having separate gradscaler per optimizer

    def set_accelerator_scaler(self, unet_number):
        unet_number = self.validate_unet_number(unet_number)
        scaler = getattr(self, f'scaler{unet_number - 1}')

        self.accelerator.scaler = scaler
        for optimizer in self.accelerator._optimizers:
            optimizer.scaler = scaler

    # helper print

    def print(self, msg):
        if not self.is_main:
            return

        if not self.verbose:
            return

        return self.accelerator.print(msg)

    # validating the unet number

    def validate_unet_number(self, unet_number = None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        assert 0 < unet_number <= self.num_unets, f'unet number should be in between 1 and {self.num_unets}'
        return unet_number

    # number of training steps taken

    def num_steps_taken(self, unet_number = None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        return self.steps[unet_number - 1].item()

    def print_untrained_unets(self):
        print_final_error = False

        for ind, (steps, unet) in enumerate(zip(self.steps.tolist(), self.imagen.unets)):
            if steps > 0 or isinstance(unet, NullUnet):
                continue

            self.print(f'unet {ind + 1} has not been trained')
            print_final_error = True

        if print_final_error:
            self.print('when sampling, you can pass stop_at_unet_number to stop early in the cascade, so it does not try to generate with untrained unets')

    # data related functions

    def add_train_dataloader(self, dl = None):
        if not exists(dl):
            return

        assert not exists(self.train_dl), 'training dataloader was already added'
        self.train_dl = self.accelerator.prepare(dl)

    def add_valid_dataloader(self, dl):
        if not exists(dl):
            return

        assert not exists(self.valid_dl), 'validation dataloader was already added'
        self.valid_dl = self.accelerator.prepare(dl)

    def add_train_dataset(self, ds = None, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        assert not exists(self.train_dl), 'training dataloader was already added'

        valid_ds = None
        if self.split_valid_from_train:
            train_size = int((1 - self.split_valid_fraction) * len(ds))
            valid_size = len(ds) - train_size

            ds, valid_ds = random_split(ds, [train_size, valid_size], generator = torch.Generator().manual_seed(self.split_random_seed))
            self.print(f'training with dataset of {len(ds)} samples and validating with randomly splitted {len(valid_ds)} samples')

        dl = DataLoader(ds, batch_size = batch_size, **dl_kwargs)
        self.train_dl = self.accelerator.prepare(dl)

        if not self.split_valid_from_train:
            return

        self.add_valid_dataset(valid_ds, batch_size = batch_size, **dl_kwargs)

    def add_valid_dataset(self, ds, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        assert not exists(self.valid_dl), 'validation dataloader was already added'

        dl = DataLoader(ds, batch_size = batch_size, **dl_kwargs)
        self.valid_dl = self.accelerator.prepare(dl)

    def create_train_iter(self):
        assert exists(self.train_dl), 'training dataloader has not been registered with the trainer yet'

        if exists(self.train_dl_iter):
            return

        self.train_dl_iter = cycle(self.train_dl)

    def create_valid_iter(self):
        assert exists(self.valid_dl), 'validation dataloader has not been registered with the trainer yet'

        if exists(self.valid_dl_iter):
            return

        self.valid_dl_iter = cycle(self.valid_dl)

    def train_step(self, unet_number = None, **kwargs):
        self.create_train_iter()
        loss = self.step_with_dl_iter(self.train_dl_iter, unet_number = unet_number, **kwargs)
        self.update(unet_number = unet_number)
        return loss

    @torch.no_grad()
    @eval_decorator
    def valid_step(self, **kwargs):
        self.create_valid_iter()

        context = self.use_ema_unets if kwargs.pop('use_ema_unets', False) else nullcontext

        with context():
            loss = self.step_with_dl_iter(self.valid_dl_iter, **kwargs)
        return loss

    def step_with_dl_iter(self, dl_iter, **kwargs):
        dl_tuple_output = cast_tuple(next(dl_iter))
        model_input = dict(list(zip(self.dl_tuple_output_keywords_names, dl_tuple_output)))
        loss = self.forward(**{**kwargs, **model_input})
        return loss

    # checkpointing functions

    @property
    def all_checkpoints_sorted(self):
        glob_pattern = os.path.join(self.checkpoint_path, '*.pt')
        checkpoints = self.fs.glob(glob_pattern)
        sorted_checkpoints = sorted(checkpoints, key = lambda x: int(str(x).split('.')[-2]), reverse = True)
        return sorted_checkpoints

    def load_from_checkpoint_folder(self, last_total_steps = -1):
        if last_total_steps != -1:
            filepath = os.path.join(self.checkpoint_path, f'checkpoint.{last_total_steps}.pt')
            self.load(filepath)
            return

        sorted_checkpoints = self.all_checkpoints_sorted

        if len(sorted_checkpoints) == 0:
            self.print(f'no checkpoints found to load from at {self.checkpoint_path}')
            return

        last_checkpoint = sorted_checkpoints[0]
        self.load(last_checkpoint)

    def save_to_checkpoint_folder(self):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        total_steps = int(self.steps.sum().item())
        filepath = os.path.join(self.checkpoint_path, f'checkpoint.{total_steps}.pt')

        self.save(filepath)

        if self.max_checkpoints_keep <= 0:
            return

        sorted_checkpoints = self.all_checkpoints_sorted
        checkpoints_to_discard = sorted_checkpoints[self.max_checkpoints_keep:]

        for checkpoint in checkpoints_to_discard:
            self.fs.rm(checkpoint)

    # saving and loading functions

    def save(
        self,
        path,
        overwrite = True,
        without_optim_and_sched = False,
        **kwargs
    ):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        fs = self.fs

        assert not (fs.exists(path) and not overwrite)

        self.reset_ema_unets_all_one_device()

        save_obj = dict(
            model = self.imagen.state_dict(),
            version = __version__,
            steps = self.steps.cpu(),
            **kwargs
        )

        save_optim_and_sched_iter = range(0, self.num_unets) if not without_optim_and_sched else tuple()

        for ind in save_optim_and_sched_iter:
            scaler_key = f'scaler{ind}'
            optimizer_key = f'optim{ind}'
            scheduler_key = f'scheduler{ind}'
            warmup_scheduler_key = f'warmup{ind}'

            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)
            warmup_scheduler = getattr(self, warmup_scheduler_key)

            if exists(scheduler):
                save_obj = {**save_obj, scheduler_key: scheduler.state_dict()}

            if exists(warmup_scheduler):
                save_obj = {**save_obj, warmup_scheduler_key: warmup_scheduler.state_dict()}

            save_obj = {**save_obj, scaler_key: scaler.state_dict(), optimizer_key: optimizer.state_dict()}

        if self.use_ema:
            save_obj = {**save_obj, 'ema': self.ema_unets.state_dict()}

        # determine if imagen config is available

        if hasattr(self.imagen, '_config'):
            self.print(f'this checkpoint is commandable from the CLI - "imagen --model {str(path)} \"<prompt>\""')

            save_obj = {
                **save_obj,
                'imagen_type': 'elucidated' if self.is_elucidated else 'original',
                'imagen_params': self.imagen._config
            }

        #save to path

        with fs.open(path, 'wb') as f:
            torch.save(save_obj, f)

        self.print(f'checkpoint saved to {path}')

    def load(self, path, only_model = False, strict = True, noop_if_not_exist = False):
        fs = self.fs

        if noop_if_not_exist and not fs.exists(path):
            self.print(f'trainer checkpoint not found at {str(path)}')
            return

        assert fs.exists(path), f'{path} does not exist'

        self.reset_ema_unets_all_one_device()

        # to avoid extra GPU memory usage in main process when using Accelerate

        with fs.open(path) as f:
            loaded_obj = torch.load(f, map_location='cpu')

        if version.parse(__version__) != version.parse(loaded_obj['version']):
            self.print(f'loading saved imagen at version {loaded_obj["version"]}, but current package version is {__version__}')

        try:
            self.imagen.load_state_dict(loaded_obj['model'], strict = strict)
        except RuntimeError:
            print("Failed loading state dict. Trying partial load")
            self.imagen.load_state_dict(restore_parts(self.imagen.state_dict(),
                                                      loaded_obj['model']))

        if only_model:
            return loaded_obj

        self.steps.copy_(loaded_obj['steps'])

        for ind in range(0, self.num_unets):
            scaler_key = f'scaler{ind}'
            optimizer_key = f'optim{ind}'
            scheduler_key = f'scheduler{ind}'
            warmup_scheduler_key = f'warmup{ind}'

            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)
            warmup_scheduler = getattr(self, warmup_scheduler_key)

            if exists(scheduler) and scheduler_key in loaded_obj:
                scheduler.load_state_dict(loaded_obj[scheduler_key])

            if exists(warmup_scheduler) and warmup_scheduler_key in loaded_obj:
                warmup_scheduler.load_state_dict(loaded_obj[warmup_scheduler_key])

            if exists(optimizer):
                try:
                    optimizer.load_state_dict(loaded_obj[optimizer_key])
                    scaler.load_state_dict(loaded_obj[scaler_key])
                except:
                    self.print('could not load optimizer and scaler, possibly because you have turned on mixed precision training since the last run. resuming with new optimizer and scalers')

        if self.use_ema:
            assert 'ema' in loaded_obj
            try:
                self.ema_unets.load_state_dict(loaded_obj['ema'], strict = strict)
            except RuntimeError:
                print("Failed loading state dict. Trying partial load")
                self.ema_unets.load_state_dict(restore_parts(self.ema_unets.state_dict(),
                                                             loaded_obj['ema']))

        self.print(f'checkpoint loaded from {path}')
        return loaded_obj

    # managing ema unets and their devices

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    def get_ema_unet(self, unet_number = None):
        if not self.use_ema:
            return

        unet_number = self.validate_unet_number(unet_number)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.ema_unets]
            delattr(self, 'ema_unets')
            self.ema_unets = unets_list

        if index != self.ema_unet_being_trained_index:
            for unet_index, unet in enumerate(self.ema_unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.ema_unet_being_trained_index = index
        return self.ema_unets[index]

    def reset_ema_unets_all_one_device(self, device = None):
        if not self.use_ema:
            return

        device = default(device, self.device)
        self.ema_unets = nn.ModuleList([*self.ema_unets])
        self.ema_unets.to(device)

        self.ema_unet_being_trained_index = -1

    @torch.no_grad()
    @contextmanager
    def use_ema_unets(self):
        if not self.use_ema:
            output = yield
            return output

        self.reset_ema_unets_all_one_device()
        self.imagen.reset_unets_all_one_device()

        self.unets.eval()

        trainable_unets = self.imagen.unets
        self.imagen.unets = self.unets                  # swap in exponential moving averaged unets for sampling

        output = yield

        self.imagen.unets = trainable_unets             # restore original training unets

        # cast the ema_model unets back to original device
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        return output

    def print_unet_devices(self):
        self.print('unet devices:')
        for i, unet in enumerate(self.imagen.unets):
            device = next(unet.parameters()).device
            self.print(f'\tunet {i}: {device}')

        if not self.use_ema:
            return

        self.print('\nema unet devices:')
        for i, ema_unet in enumerate(self.ema_unets):
            device = next(ema_unet.parameters()).device
            self.print(f'\tema unet {i}: {device}')

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_ema_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_ema_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # encoding text functions

    def encode_text(self, text, **kwargs):
        return self.imagen.encode_text(text, **kwargs)

    # forwarding functions and gradient step updates

    def update(self, unet_number = None):
        unet_number = self.validate_unet_number(unet_number)
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        index = unet_number - 1
        unet = self.unet_being_trained

        optimizer = getattr(self, f'optim{index}')
        scaler = getattr(self, f'scaler{index}')
        scheduler = getattr(self, f'scheduler{index}')
        warmup_scheduler = getattr(self, f'warmup{index}')

        # set the grad scaler on the accelerator, since we are managing one per u-net

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(unet.parameters(), self.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        if self.use_ema:
            ema_unet = self.get_ema_unet(unet_number)
            ema_unet.update()

        # scheduler, if needed

        maybe_warmup_context = nullcontext() if not exists(warmup_scheduler) else warmup_scheduler.dampening()

        with maybe_warmup_context:
            if exists(scheduler) and not self.accelerator.optimizer_step_was_skipped: # recommended in the docs
                scheduler.step()

        self.steps += F.one_hot(torch.tensor(unet_number - 1, device = self.steps.device), num_classes = len(self.steps))

        if not exists(self.checkpoint_path):
            return

        total_steps = int(self.steps.sum().item())

        if total_steps % self.checkpoint_every:
            return

        self.save_to_checkpoint_folder()

    @torch.no_grad()
    @cast_torch_tensor
    @imagen_sample_in_chunks
    def sample(self, *args, **kwargs):
        context = nullcontext if  kwargs.pop('use_non_ema', False) else self.use_ema_unets

        self.print_untrained_unets()        
        
        if not self.is_main:
            kwargs['use_tqdm'] = False

        with context():
            output = self.imagen.sample(*args, device = self.device, **kwargs)

        return output

    @partial(cast_torch_tensor, cast_fp16 = True)
    def forward(
        self,
        *args,
        unet_number = None,
        max_batch_size = None,
        **kwargs
    ):
        unet_number = self.validate_unet_number(unet_number)
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, f'you can only train unet #{self.only_train_unet_number}'

        total_loss = 0.
        
        
        # + for debug
        print('args: ', len(args))
        print('args in:',args[0].shape)
        print('kwargs: ', kwargs.keys())
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            # + for debug
            print('chun_frac: ', chunk_size_frac)
            print('chun_args: ', chunked_args)
            print('chun_kwargs: ', chunked_kwargs)
            
            with self.accelerator.autocast():
                loss = self.model(*chunked_args, unet_number = unet_number, **chunked_kwargs)
                loss = loss * chunk_size_frac
            
            print('loss: ', loss)

            total_loss += loss#.item()

            if self.training:
                self.accelerator.backward(loss)

        print('I am here')
        return total_loss

    
def write_fasta (sequence, filename_out):
    
    with open (filename_out, mode ='w') as f:
        f.write (f'>{filename_out}\n')
        f.write (f'{sequence}')


#
def sample_sequence_FromModelB (
    model,
    X=None, #this is the target conventionally when using text embd
    flag=0,
    cond_scales=1.,
    foldproteins=False,
    X_string=None,
    x_data=None,  
    skip_steps=0,
    inpaint_images = None,
    inpaint_masks = None,
    inpaint_resample_times = None,
    init_images = None,
    num_cycle=16,
    # ++++++++++++++++++++++++
    ynormfac=1,
    train_unet_number=1,
    tokenizer_X=None,
    Xnormfac=1.,
    max_length=1.,
    prefix=None,
    tokenizer_y=None,
               ):
    steps=0
    e=flag


    

    #num_samples = min (num_samples,y_train_batch.shape[0] )
    if X!=None:
        print (f"Producing {len(X)} samples...from text conditioning X...")
        lenn_val=len(X)
    if X_string!=None:
        lenn_val=len(X_string)
        print (f"Producing {len(X_string)} samples...from text conditioning X_String (from string)...")
    
    if x_data!=None:
        print (f"Producing {len(x_data)} samples...from image conditingig x_data  ...")
        lenn_val=len(x_data)
        print (x_data)
        
    print ('Device: ', device)


    for iisample in range (lenn_val):
        X_cond=None  
        if X_string==None and X != None: #only do if X provided
            X_cond=torch.Tensor (X[iisample]).to(device).unsqueeze (0)
        if X_string !=None:
            X = tokenizer_X.texts_to_sequences(X_string[iisample])
            X= sequence.pad_sequences(X,  maxlen=max_length, padding='post', truncating='post')  
            X=np.array(X)
            X_cond=torch.from_numpy(X).float()/Xnormfac
            print ('Tokenized and processed: ', X_cond)
        
        print ("X_cond=", X_cond)
        
        result=model.sample ( 
            x=X_cond,
            stop_at_unet_number=train_unet_number ,
            cond_scale=cond_scales ,
            x_data=x_data, skip_steps=skip_steps,
            inpaint_images = inpaint_images,
            inpaint_masks = inpaint_masks,
            inpaint_resample_times = inpaint_resample_times,
            init_images = init_images,device=device,
            # ++++++++++++++++++++++++++
            tokenizer_X=tokenizer_X,
            Xnormfac=Xnormfac,
            max_length=max_length,
                            )
        result=torch.round(result*ynormfac)
        
        plt.plot (result[0,0,:].cpu().detach().numpy(),label= f'Predicted')
        #plt.plot (GT[samples,0,:]*ynormfac,label= f'GT {0}')
        plt.legend()

        outname = prefix+ f"sampled_from_X_{flag}_condscale-{str (cond_scales)}_{e}_{steps}.jpg"
        #plt.title (f"Sample {samples}, cond scale={str (cond_scales[iisample])}")
        plt.savefig(outname, dpi=200)
        plt.show ()

        to_rev=result[:,0,:] 
        to_rev=to_rev.long().cpu().detach().numpy()
        print (to_rev.shape)
        y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

        for iii in range (len(y_data_reversed)):
            y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
        ### reverse second structure input....
        if X_cond != None:
            X_cond=torch.round(X_cond*Xnormfac)

            to_rev=X_cond[:,:] 
            to_rev=to_rev.long().cpu().detach().numpy()
            print (to_rev.shape)
            X_data_reversed=tokenizer_X.sequences_to_texts (to_rev)

            for iii in range (len(y_data_reversed)):
                X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")
        if x_data !=None:
            X_data_reversed=x_data #is already in sequence fromat..
               

        print (f"For {X} or {X_data_reversed[iisample]}, predicted sequence: ", y_data_reversed[iisample])
        if foldproteins:
            
            if X_cond != None:
                xbc=X_cond[iisample,:].cpu().detach().numpy()
                out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})+f'_{flag}_{steps}'
            if x_data !=None:
                #xbc=x_data[iisample] 
                out_nam=x_data[iisample] 
             
            
            tempname='temp'
            pdb_file=foldandsavePDB (
                sequence=y_data_reversed[0], 
                filename_out=tempname, 
                num_cycle=num_cycle, 
                flag=flag,
                # +++++++++++++++++++
                prefix=prefix
            )

            out_nam_fasta=f'{prefix}{out_nam}_{flag}_{steps}.fasta'

            write_fasta (y_data_reversed[0], out_nam_fasta)            
            
        
            out_nam=f'{prefix}{X_data_reversed[iisample]}_{flag}_{steps}.pdb'
            # print('Debug 1: out: ', out_nam)
            # print('Debug 2: in: ', pdb_file)
            shutil.copy (pdb_file, out_nam) #source, dest
            # cmd_line = 'cp ' + pdb_file + ' ' + out_nam
            # print(cmd_line)
            # os.popen(cmd_line)
            # print('Debug 3')
            pdb_file=out_nam
            
            
            
            
            
            print (f"Properly named PDB file produced: {pdb_file}")
            #flag=1000
            view=show_pdb(pdb_file=pdb_file, flag=flag,
                          show_sidechains=show_sidechains, show_mainchains=show_mainchains, color=color)
            view.show()


        steps=steps+1
        
        return pdb_file 
    
def sample_loop_FromModelB (model,
                train_loader,
                cond_scales=[7.5], #list of cond scales - each sampled...
                num_samples=2, #how many samples produced every time tested.....
                timesteps=100,
                 flag=0,foldproteins=False,
                 use_text_embedd=True,skip_steps=0,
                 # +++++++++++++++++++
                 train_unet_number=1,
                 ynormfac=1,
                 prefix=None,
                 tokenizer_y=None,
                 Xnormfac=1,
                 tokenizer_X=None,
                 
               ):
    steps=0
    e=flag
    for item  in train_loader:

            X_train_batch= item[0].to(device)
            y_train_batch=item[1].to(device)

            GT=y_train_batch.cpu().detach() 
                    
            GT= GT.unsqueeze(1)
            num_samples = min (num_samples,y_train_batch.shape[0] )
            print (f"Producing {num_samples} samples...")
            
            print ('X_train_batch shape: ', X_train_batch.shape)

            for iisample in range (len (cond_scales)):
                
                if use_text_embedd:
                    result=model.sample (x= X_train_batch,stop_at_unet_number=train_unet_number ,
                                         cond_scale=cond_scales[iisample], device=device, skip_steps=skip_steps)
                else:
                    result=model.sample (x= None, x_data_tokenized= X_train_batch,
                                         stop_at_unet_number=train_unet_number ,
                                         cond_scale=cond_scales[iisample],device=device,skip_steps=skip_steps)
                    
                result=torch.round(result*ynormfac)
                GT=torch.round (GT*ynormfac)

                for samples in range  (num_samples):
                    print ("sample ", samples, "out of ", num_samples)
                    
                    plt.plot (result[samples,0,:].cpu().detach().numpy(),label= f'Predicted')
                    plt.plot (GT[samples,0,:],label= f'GT {0}')
                    plt.legend()

                    outname = prefix+ f"sample-{samples}_condscale-{str (cond_scales[iisample])}_{e}_{steps}.jpg"
                   
                    plt.savefig(outname, dpi=200)
                    plt.show ()
                    
                    #reverse y sequence
                    to_rev=result[:,0,:]
                    to_rev=to_rev.long().cpu().detach().numpy()
                    
                    y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

                    for iii in range (len(y_data_reversed)):
                        y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
                        
                    #reverse GT_y sequence
                    to_rev=GT[:,0,:]
                    to_rev=to_rev.long().cpu().detach().numpy()
                    
                    GT_y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

                    for iii in range (len(y_data_reversed)):
                        GT_y_data_reversed[iii]=GT_y_data_reversed[iii].upper().strip().replace(" ", "")
                    
                    ### reverse second structure input....
                    to_rev=torch.round (X_train_batch[:,:]*Xnormfac)
                    to_rev=to_rev.long().cpu().detach().numpy()
                   
                    X_data_reversed=tokenizer_X.sequences_to_texts (to_rev)

                    for iii in range (len(y_data_reversed)):
                        X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")

                    print (f"For {X_train_batch[samples,:].cpu().detach().numpy()} or {X_data_reversed[samples]}, predicted sequence: ", y_data_reversed[samples])
                    print (f"Ground truth: {GT_y_data_reversed[samples]}")
                   
                    if foldproteins:
                        xbc=X_train_batch[samples,:].cpu().detach().numpy()
                        out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})
                        tempname='temp'
                        pdb_file=foldandsavePDB (
                            sequence=y_data_reversed[samples], 
                            filename_out=tempname, 
                            num_cycle=16, flag=flag,
                            # +++++++++++++++++++
                            prefix=prefix
                        )
                        
                        #out_nam=f'{prefix}{out_nam}.pdb'
                        out_nam=f'{prefix}{X_data_reversed[samples]}.pdb'
                        print (f'Original PDB: {pdb_file} OUT: {out_nam}')
                        shutil.copy (pdb_file, out_nam) #source, dest
                        pdb_file=out_nam
                        print (f"Properly named PDB file produced: {pdb_file}")
                        
                        view=show_pdb(pdb_file=pdb_file, flag=flag, show_sidechains=show_sidechains,  show_mainchains=show_mainchains, color=color)
                        view.show()

                    steps=steps+1
            if steps>num_samples:
                break
# ++
def sample_sequence_omegafold_pLM_ModelB (
    model,
    X=None, #this is the target conventionally when using text embd
    flag=0,
    cond_scales=1.,
    foldproteins=False,
    X_string=None,
    x_data=None,  
    skip_steps=0,
    inpaint_images = None,
    inpaint_masks = None,
    inpaint_resample_times = None,
    init_images = None,
    num_cycle=16,
    # ++++++++++++++++++++++++
    ynormfac=1,
    train_unet_number=1,
    tokenizer_X=None,
    Xnormfac=1.,
    max_length=1.,
    prefix=None,
    tokenizer_y=None,
    # ++
    CKeys=None,
    sample_dir=None,
    steps=None,
    e=None,
    IF_showfig=True, # effective only after foldproteins=True
    # ++
    pLM_Model=None, # pLM_Model,
    pLM_Model_Name=None, # pLM_Model_Name,
    image_channels=None, # image_channels,
    pLM_alphabet=None, # esm_alphabet,
):
    
    # steps=0
    # e=flag

    #num_samples = min (num_samples,y_train_batch.shape[0] )
    if X!=None:
        print (f"Producing {len(X)} samples...from text conditioning X...")
        lenn_val=len(X)
    if X_string!=None:
        lenn_val=len(X_string)
        print (f"Producing {len(X_string)} samples...from text conditioning X_String (from string)...")
    
    if x_data!=None:
        print (f"Producing {len(x_data)} samples...from image conditingig x_data  ...")
        lenn_val=len(x_data)
        print (x_data)
        
    print ('Device: ', device)
    
    pdb_file_list=[]
    fasta_file_list=[]

    # + for debug
    print('tot ', lenn_val)
    for iisample in range (lenn_val):
        print("Working on ", iisample)
        X_cond=None  # this is for text-conditioning
        if X_string==None and X != None: #only do if X provided
            X_cond=torch.Tensor (X[iisample]).to(device).unsqueeze (0)
        if X_string !=None:
            XX = tokenizer_X.texts_to_sequences(X_string[iisample])
            XX= sequence.pad_sequences(XX,  maxlen=max_length, padding='post', truncating='post')  
            XX=np.array(XX)
            X_cond=torch.from_numpy(XX).float()/Xnormfac
            print ('Tokenized and processed: ', X_cond)
        
        print ("X_cond=", X_cond)
        
        # # --
        # result=model.sample ( 
        #     x=X_cond,
        #     stop_at_unet_number=train_unet_number ,
        #     cond_scale=cond_scales ,
        #     x_data=x_data[iisample], 
        #     # ++
        #     x_data_tokenized=
        #     skip_steps=skip_steps,
        #     inpaint_images = inpaint_images,
        #     inpaint_masks = inpaint_masks,
        #     inpaint_resample_times = inpaint_resample_times,
        #     init_images = init_images,device=device,
        #     # ++++++++++++++++++++++++++
        #     tokenizer_X=tokenizer_X,
        #     Xnormfac=Xnormfac,
        #     max_length=max_length,
        #                     )
        # ++
        # use cond_image as the conditioning, via x_data_tokenized channel
        
        # -----------------------------------------------------------------
        # for below, two branches are all for cond_img, not for text_cond
        if tokenizer_X!=None:
            # for SecStr+ModelB
            result_embedding=model.sample ( 
                x=X_cond,
                stop_at_unet_number=train_unet_number ,
                cond_scale=cond_scales ,
                x_data=x_data[iisample],  # will pass through tokenizer_X in this sample(), channels will be matched with self.pred_dim
                # ++
                x_data_tokenized=None,
                skip_steps=skip_steps,
                inpaint_images = inpaint_images,
                inpaint_masks = inpaint_masks,
                inpaint_resample_times = inpaint_resample_times,
                init_images = init_images,device=device,
                # ++++++++++++++++++++++++++
                tokenizer_X=tokenizer_X,
                Xnormfac=Xnormfac,
                max_length=max_length,
            )
        else:
            # for ForcPath+ModelB:
            # for model.sample() here using x_data_tokenized channel
            #
            x_data_tokenized=torch.from_numpy(x_data[iisample]/Xnormfac)
            x_data_tokenized=x_data_tokenized.to(torch.float)
            # here, only one input list is read in
            x_data_tokenized=x_data_tokenized.unsqueeze(0) # [batch=1, seq_len]
            # leave channel expansion for the self.sample() to handle
            
            # + for debug:
            if CKeys['Debug_TrainerPack']==3:
                print("x_data_tokenized dim: ", x_data_tokenized.shape)
                print("x_data_tokenized dtype: ", x_data_tokenized.dtype)
                print("test x_data_tokenized!=None: ", x_data_tokenized!=None)
            
            result_embedding=model.sample ( 
                x=X_cond,
                stop_at_unet_number=train_unet_number ,
                cond_scale=cond_scales ,
                x_data=None, 
                # ++
                x_data_tokenized=x_data_tokenized,
                #
                skip_steps=skip_steps,
                inpaint_images = inpaint_images,
                inpaint_masks = inpaint_masks,
                inpaint_resample_times = inpaint_resample_times,
                init_images = init_images,device=device,
                # ++++++++++++++++++++++++++
                tokenizer_X=tokenizer_X,
                Xnormfac=Xnormfac,
                max_length=max_length,
            )
            
        # # --
        # result=torch.round(result*ynormfac) # (batch=1, channel=1, seq_len)
        
        # ++ for pLM
        # full record
        # result_embedding as image.dim: [batch, channels, seq_len]
        # result_tokens.dim: [batch, seq_len]
        if image_channels==33:
            result_tokens,result_logits = convert_into_tokens_using_prob(
                result_embedding,
                pLM_Model_Name,
            )
        else:
            result_tokens,result_logits = convert_into_tokens(
                pLM_Model, 
                result_embedding,
                pLM_Model_Name,
            )
        # +++++++++++++++++++++++++++++++++
        result=result_tokens.unsqueeze(1) # dim: [batch, 1, seq_len]
        
        # + for debug
        print('result dim: ', result.shape)
        
        # plot sequence token code: esm (33 tokens)
        fig=plt.figure()
        plt.plot (
            result[0,0,:].cpu().detach().numpy(),
            label= f'Predicted'
        )
        #plt.plot (GT[samples,0,:]*ynormfac,label= f'GT {0}')
        plt.legend()
        outname = sample_dir+ f"sampled_from_X_{iisample}_condscale-{str (cond_scales)}_{e}_{steps}.jpg"
        #plt.title (f"Sample {samples}, cond scale={str (cond_scales[iisample])}")
        if IF_showfig==1:
            plt.show ()
        else:
            plt.savefig(outname, dpi=200)
        plt.close()
        
        # 
        # # --
        # to_rev=result[:,0,:] 
        # to_rev=to_rev.long().cpu().detach().numpy()
        # print (to_rev.shape)
        # y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)
        # 
        # for iii in range (len(y_data_reversed)):
        #     y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
        # # ++: for model A: no mask is provided from input
        # # reverse the PREDICTED y into a foldable sequence
        # # save this block for Model A 
        # to_rev=result[:,0,:] # token (batch,seq_len)
        # y_data_reversed=decode_many_ems_token_rec_for_folding(
        #     to_rev,
        #     result_logits,
        #     pLM_alphabet,
        #     pLM_Model,
        # )
        
        # if CKeys['Debug_TrainerPack']==3:
        #     print("on foldable result: ", to_rev[0])
        #     print("on result_logits: ", result_logits[0])
        #     a = decode_one_ems_token_rec_for_folding(
        #         to_rev[0],
        #         result_logits[0],
        #         pLM_alphabet,
        #         pLM_Model,
        #     )
        #     print('One resu: ', a)
        #     print("on y_data_reversed: ", y_data_reversed[0])
        #    print("y_data_reversed.type", y_data_reversed.dtype)
        #
        
        # ++: for model B: using mask from the input
        # extract the mask/seq_len from input if possible
        if tokenizer_X!=None:
            # for SecStr+ModelB
            result_mask = read_mask_from_input(
                tokenized_data=None, 
                mask_value=None,
                seq_data=x_data[iisample],
                max_seq_length=max_length,
            )
        else:
            # for ForcPath+ModelB
            result_mask = read_mask_from_input(
                tokenized_data=x_data_tokenized, # None, 
                mask_value=0, # None,
                seq_data=None, # x_data[iisample],
                max_seq_length=None, # max_length,
            )
        
        to_rev=result[:,0,:] # token (batch,seq_len)
        if CKeys['Debug_TrainerPack']==3:
            print("on foldable result: ", to_rev[0])
            print("on result_logits: ", result_logits[0])
            print("on mask: ", result_mask[0])
            a = decode_one_ems_token_rec_for_folding_with_mask(
                to_rev[0],
                result_logits[0],
                pLM_alphabet,
                pLM_Model,
                result_mask[0],
            )
            print('One resu: ', a)

        y_data_reversed=decode_many_ems_token_rec_for_folding_with_mask(
            to_rev,
            result_logits,
            pLM_alphabet,
            pLM_Model,
            result_mask,
        )
        if CKeys['Debug_TrainerPack']==3:
            print("on y_data_reversed[0]: ", y_data_reversed[0])
            
            
        
        ### reverse second structure input....
        if X_cond != None:
            X_cond=torch.round(X_cond*Xnormfac)

            to_rev=X_cond[:,:] 
            to_rev=to_rev.long().cpu().detach().numpy()
            print (to_rev.shape)
            X_data_reversed=tokenizer_X.sequences_to_texts (to_rev)

            for iii in range (len(y_data_reversed)):
                X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")
        if x_data !=None:
            X_data_reversed=x_data #is already in sequence fromat..
               

        # print (f"For {X} or {X_data_reversed[iisample]}, predicted sequence", y_data_reversed[iisample])
        print (f"For {X} or {X_data_reversed[iisample]}, predicted sequence: ", y_data_reversed)
        
        # + for debug
        print("================================================")
        print("foldproteins: ", foldproteins)
        
        if not foldproteins:
            pdb_file=None
            
        else:
            
            if X_cond != None:
                xbc=X_cond[iisample,:].cpu().detach().numpy()
                out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})+f'_{flag}_{steps}'
            if x_data !=None:
                #xbc=x_data[iisample] 
                # ----------------------------------
                # this one can be too long for a name
                out_nam=x_data[iisample] 
                # ++++++++++++++++++++++++++++++++++
                # 
                out_nam=iisample
             
            
            tempname='temp'
            pdb_file, fasta_file=foldandsavePDB_pdb_fasta (
                sequence=y_data_reversed[0], 
                filename_out=tempname, 
                num_cycle=num_cycle, 
                flag=flag,
                # +++++++++++++++++++
                # prefix=prefix,
                prefix=sample_dir,
            )

            # out_nam_fasta=f'{prefix}{out_nam}_{flag}_{steps}.fasta'
            # ------------------------------------------
            # this one can be too long for a name
            # out_nam_fasta=f'{sample_dir}{out_nam}_{flag}_{e}_{iisample}.fasta'
            # ++++++++++++++++++++++++++++++++++++++++++
            out_nam_fasta=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.fasta'

            write_fasta (y_data_reversed[0], out_nam_fasta)            
            
        
            # out_nam=f'{prefix}{X_data_reversed[iisample]}_{flag}_{steps}.pdb'
            # out_nam=f'{sample_dir}{X_data_reversed[iisample]}_{flag}_{steps}.pdb'
            # -------------------------------------------
            # this one can be too long for a name
            # However, the input X is recorded in the code
            # out_nam=f'{sample_dir}{X_data_reversed[iisample]}_{flag}_{iisample}.pdb'
            # +++++++++++++++++++++++++++++++++++++++++++
            out_nam=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.pdb'
            out_nam_fasta=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.fasta'
            
            # print('Debug 1: out: ', out_nam)
            # print('Debug 2: in: ', pdb_file)
            shutil.copy (pdb_file, out_nam) #source, dest
            shutil.copy (fasta_file, out_nam_fasta)
            # cmd_line = 'cp ' + pdb_file + ' ' + out_nam
            # print(cmd_line)
            # os.popen(cmd_line)
            # print('Debug 3')
            # clean the slade to avoid mistakenly using the previous fasta file
            os.remove (pdb_file)
            os.remove (fasta_file)
            
            pdb_file=out_nam
            fasta_file=out_nam_fasta
            pdb_file_list.append(pdb_file)
            fasta_file_list.append(fasta_file)
            
            
            print (f"Properly named PDB file produced: {pdb_file}")
            if IF_showfig==1:
                #flag=1000
                view=show_pdb(
                    pdb_file=pdb_file, 
                    flag=flag,
                    show_sidechains=show_sidechains, 
                    show_mainchains=show_mainchains, 
                    color=color
                )
                view.show()


        # steps=steps+1
        
    return pdb_file_list, fasta_file_list

#
def sample_sequence_omegafold_ModelB (
    model,
    X=None, #this is the target conventionally when using text embd
    flag=0,
    cond_scales=1.,
    foldproteins=False,
    X_string=None,
    x_data=None,  
    skip_steps=0,
    inpaint_images = None,
    inpaint_masks = None,
    inpaint_resample_times = None,
    init_images = None,
    num_cycle=16,
    # ++++++++++++++++++++++++
    ynormfac=1,
    train_unet_number=1,
    tokenizer_X=None,
    Xnormfac=1.,
    max_length=1.,
    prefix=None,
    tokenizer_y=None,
    # ++
    CKeys=None,
    sample_dir=None,
    steps=None,
    e=None,
    IF_showfig=True, # effective only after foldproteins=True
):
    
    # steps=0
    # e=flag

    #num_samples = min (num_samples,y_train_batch.shape[0] )
    if X!=None:
        print (f"Producing {len(X)} samples...from text conditioning X...")
        lenn_val=len(X)
    if X_string!=None:
        lenn_val=len(X_string)
        print (f"Producing {len(X_string)} samples...from text conditioning X_String (from string)...")
    
    if x_data!=None:
        print (f"Producing {len(x_data)} samples...from image conditingig x_data  ...")
        lenn_val=len(x_data)
        print (x_data)
        
    print ('Device: ', device)

    # + for debug
    print('tot ', lenn_val)
    for iisample in range (lenn_val):
        print("Working on ", iisample)
        X_cond=None  
        if X_string==None and X != None: #only do if X provided
            X_cond=torch.Tensor (X[iisample]).to(device).unsqueeze (0)
        if X_string !=None:
            XX = tokenizer_X.texts_to_sequences(X_string[iisample])
            XX= sequence.pad_sequences(XX,  maxlen=max_length, padding='post', truncating='post')  
            XX=np.array(XX)
            X_cond=torch.from_numpy(XX).float()/Xnormfac
            print ('Tokenized and processed: ', X_cond)
        
        print ("X_cond=", X_cond)
        
        # # --
        # result=model.sample ( 
        #     x=X_cond,
        #     stop_at_unet_number=train_unet_number ,
        #     cond_scale=cond_scales ,
        #     x_data=x_data[iisample], 
        #     # ++
        #     x_data_tokenized=
        #     skip_steps=skip_steps,
        #     inpaint_images = inpaint_images,
        #     inpaint_masks = inpaint_masks,
        #     inpaint_resample_times = inpaint_resample_times,
        #     init_images = init_images,device=device,
        #     # ++++++++++++++++++++++++++
        #     tokenizer_X=tokenizer_X,
        #     Xnormfac=Xnormfac,
        #     max_length=max_length,
        #                     )
        # ++
        if tokenizer_X!=None:
            result=model.sample ( 
                x=X_cond,
                stop_at_unet_number=train_unet_number ,
                cond_scale=cond_scales ,
                x_data=x_data[iisample], 
                # ++
                x_data_tokenized=None,
                skip_steps=skip_steps,
                inpaint_images = inpaint_images,
                inpaint_masks = inpaint_masks,
                inpaint_resample_times = inpaint_resample_times,
                init_images = init_images,device=device,
                # ++++++++++++++++++++++++++
                tokenizer_X=tokenizer_X,
                Xnormfac=Xnormfac,
                max_length=max_length,
            )
        else:
            x_data_tokenized=torch.from_numpy(x_data[iisample]/Xnormfac)
            x_data_tokenized=x_data_tokenized.to(torch.float)
            # + for debug:
            if CKeys['Debug_TrainerPack']==1:
                print("x_data_tokenized dim: ", x_data_tokenized.shape)
                print("x_data_tokenized dtype: ", x_data_tokenized.dtype)
                print("test: ", x_data_tokenized!=None)
            result=model.sample ( 
                x=X_cond,
                stop_at_unet_number=train_unet_number ,
                cond_scale=cond_scales ,
                x_data=None, 
                # ++
                x_data_tokenized=x_data_tokenized,
                #
                skip_steps=skip_steps,
                inpaint_images = inpaint_images,
                inpaint_masks = inpaint_masks,
                inpaint_resample_times = inpaint_resample_times,
                init_images = init_images,device=device,
                # ++++++++++++++++++++++++++
                tokenizer_X=tokenizer_X,
                Xnormfac=Xnormfac,
                max_length=max_length,
            )
          
            
        result=torch.round(result*ynormfac)
        # + for debug
        print('result dim: ', result.shape)
        
        fig=plt.figure()
        plt.plot (
            result[0,0,:].cpu().detach().numpy(),
            label= f'Predicted'
        )
        #plt.plot (GT[samples,0,:]*ynormfac,label= f'GT {0}')
        plt.legend()
        outname = sample_dir+ f"sampled_from_X_{iisample}_condscale-{str (cond_scales)}_{e}_{steps}.jpg"
        #plt.title (f"Sample {samples}, cond scale={str (cond_scales[iisample])}")
        if IF_showfig==1:
            plt.show ()
        else:
            plt.savefig(outname, dpi=200)
        plt.close()
            

        to_rev=result[:,0,:] 
        to_rev=to_rev.long().cpu().detach().numpy()
        print (to_rev.shape)
        y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

        for iii in range (len(y_data_reversed)):
            y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
        ### reverse second structure input....
        if X_cond != None:
            X_cond=torch.round(X_cond*Xnormfac)

            to_rev=X_cond[:,:] 
            to_rev=to_rev.long().cpu().detach().numpy()
            print (to_rev.shape)
            X_data_reversed=tokenizer_X.sequences_to_texts (to_rev)

            for iii in range (len(y_data_reversed)):
                X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")
        if x_data !=None:
            X_data_reversed=x_data #is already in sequence fromat..
               

        # print (f"For {X} or {X_data_reversed[iisample]}, predicted sequence", y_data_reversed[iisample])
        print (f"For {X} or {X_data_reversed[iisample]}, predicted sequence: ", y_data_reversed)
        
        # + for debug
        print("================================================")
        print("foldproteins: ", foldproteins)
        
        if not foldproteins:
            pdb_file=None
            
        else:
            
            if X_cond != None:
                xbc=X_cond[iisample,:].cpu().detach().numpy()
                out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})+f'_{flag}_{steps}'
            if x_data !=None:
                #xbc=x_data[iisample] 
                # ----------------------------------
                # this one can be too long for a name
                out_nam=x_data[iisample] 
                # ++++++++++++++++++++++++++++++++++
                # 
                out_nam=iisample
             
            
            tempname='temp'
            pdb_file=foldandsavePDB (
                sequence=y_data_reversed[0], 
                filename_out=tempname, 
                num_cycle=num_cycle, 
                flag=flag,
                # +++++++++++++++++++
                # prefix=prefix,
                prefix=sample_dir,
            )

            # out_nam_fasta=f'{prefix}{out_nam}_{flag}_{steps}.fasta'
            # ------------------------------------------
            # this one can be too long for a name
            # out_nam_fasta=f'{sample_dir}{out_nam}_{flag}_{e}_{iisample}.fasta'
            # ++++++++++++++++++++++++++++++++++++++++++
            out_nam_fasta=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.fasta'

            write_fasta (y_data_reversed[0], out_nam_fasta)            
            
        
            # out_nam=f'{prefix}{X_data_reversed[iisample]}_{flag}_{steps}.pdb'
            # out_nam=f'{sample_dir}{X_data_reversed[iisample]}_{flag}_{steps}.pdb'
            # -------------------------------------------
            # this one can be too long for a name
            # However, the input X is recorded in the code
            # out_nam=f'{sample_dir}{X_data_reversed[iisample]}_{flag}_{iisample}.pdb'
            # +++++++++++++++++++++++++++++++++++++++++++
            out_nam=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.pdb'
            
            # print('Debug 1: out: ', out_nam)
            # print('Debug 2: in: ', pdb_file)
            shutil.copy (pdb_file, out_nam) #source, dest
            # cmd_line = 'cp ' + pdb_file + ' ' + out_nam
            # print(cmd_line)
            # os.popen(cmd_line)
            # print('Debug 3')
            pdb_file=out_nam           
            
            
            
            print (f"Properly named PDB file produced: {pdb_file}")
            if IF_showfig==1:
                #flag=1000
                view=show_pdb(
                    pdb_file=pdb_file, 
                    flag=flag,
                    show_sidechains=show_sidechains, 
                    show_mainchains=show_mainchains, 
                    color=color
                )
                view.show()


        # steps=steps+1
        
    return pdb_file 

# ++ for de novo input of ForcPath
# ++
def sample_sequence_omegafold_pLM_ModelB_For_ForcPath (
    model,
    X=None, #this is the target conventionally when using text embd
    flag=0,
    cond_scales=[1.], # 1.,
    foldproteins=False,
    X_string=None,
    x_data=None,  
    skip_steps=0,
    inpaint_images = None,
    inpaint_masks = None,
    inpaint_resample_times = None,
    init_images = None,
    num_cycle=16,
    # ++++++++++++++++++++++++
    ynormfac=1,
    train_unet_number=1,
    tokenizer_X=None,
    Xnormfac=1.,
    max_length=1.,
    prefix=None,
    tokenizer_y=None,
    # ++
    CKeys=None,
    sample_dir=None,
    steps=None,
    e=None,
    IF_showfig=True, # effective only after foldproteins=True
    # ++
    pLM_Model=None, # pLM_Model,
    pLM_Model_Name=None, # pLM_Model_Name,
    image_channels=None, # image_channels,
    pLM_alphabet=None, # esm_alphabet,
):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # prepare input in different channels
    # 
    X_cond=None  # this is for text-conditioning
    if X_string==None and X != None: #only do if X provided
        print (f"Producing {len(X)} samples...from text conditioning X...")
        lenn_val=len(X)
        # shape of X: [[..],[..]]: double bracket
        X_cond=torch.Tensor(X).to(device)
        # --
        # X_cond=torch.Tensor (X[iisample]).to(device).unsqueeze (0)
    if X_string !=None:
        print (f"Producing {len(X_string)} samples...from text conditioning X_String (from string)...")
        lenn_val=len(X_string)
        # --
        XX = tokenizer_X.texts_to_sequences(X_string[iisample])
        # ++
        XX = tokenizer_X.texts_to_sequences(X_string)
        XX= sequence.pad_sequences(XX,  maxlen=max_length, padding='post', truncating='post')  
        XX=np.array(XX)
        X_cond=torch.from_numpy(XX).float()/Xnormfac
        print ('Tokenized and processed: ', X_cond)
        
    if x_data!=None:
        print (f"Producing {len(x_data)} samples...from image conditingig x_data  ...")
        lenn_val=len(x_data)
        if tokenizer_X==None: # for ForcPath,
            # need to do Padding and Normalization
            # and then put into tokenized data channel
            x_data_tokenized=[]
            for ii in range(lenn_val):
                x_data_one_line=pad_a_np_arr(x_data[ii], 0.0, max_length)
                x_data_tokenized.append(x_data_one_line)
            x_data_tokenized=np.array(x_data_tokenized)
            x_data_tokenized=torch.from_numpy(x_data_tokenized/Xnormfac)
        else:
            # leave for SecStr case: TBA
            pass
        # print (x_data)
        # ++ for result_mask based on input: x_data or x_data_tokenized
        # ++: for model B: using mask from the input
        # extract the mask/seq_len from input if possible
        if tokenizer_X!=None:
            # for SecStr+ModelB
            result_mask = read_mask_from_input(
                tokenized_data=None, 
                mask_value=None,
                seq_data=x_data, # x_data[iisample],
                max_seq_length=max_length,
            )
        else:
            # for ForcPath+ModelB
            result_mask = read_mask_from_input(
                tokenized_data=x_data_tokenized, # None, 
                mask_value=0, # None,
                seq_data=None, # x_data[iisample],
                max_seq_length=None, # max_length,
            )
            
        
    print ("Input contents:")    
    print ("cond_img condition: x_data=\n", x_data)
    print ("Text condition: X_cond=\n", X_cond)
    
    # store the results
    pdb_file_list=[]
    fasta_file_list=[]
    
    # loop over cond_scales
    for idx_cond, this_cond_scale in enumerate(cond_scales):
        print(f"Working on cond_scale {str(this_cond_scale)}")
        # do sampling
        # -----------------------------------------------------------------
        # for below, two branches are all for cond_img, not for text_cond
        if tokenizer_X!=None:
            # for SecStr+ModelB, not test here
            result_embedding=model.sample ( 
                x=X_cond,
                stop_at_unet_number=train_unet_number ,
                cond_scale=this_cond_scale, # cond_scales ,
                x_data=x_data, # x_data[iisample],  # will pass through tokenizer_X in this sample(), channels will be matched with self.pred_dim
                # ++
                x_data_tokenized=None,
                skip_steps=skip_steps,
                inpaint_images = inpaint_images,
                inpaint_masks = inpaint_masks,
                inpaint_resample_times = inpaint_resample_times,
                init_images = init_images,device=device,
                # ++++++++++++++++++++++++++
                tokenizer_X=tokenizer_X,
                Xnormfac=Xnormfac,
                max_length=max_length,
            )
        else:
            # for ForcPath+ModelB:
            # for model.sample() here using x_data_tokenized channel
            x_data_tokenized=x_data_tokenized.to(torch.float) # shape [batch, max_seq_len]
            # leave channel expansion for the self.sample() to handle
            
            # + for debug:
            if CKeys['Debug_TrainerPack']==3:
                print("x_data_tokenized dim: ", x_data_tokenized.shape)
                print("x_data_tokenized dtype: ", x_data_tokenized.dtype)
                print("test x_data_tokenized!=None: ", x_data_tokenized!=None)
            
            result_embedding=model.sample ( 
                x=X_cond,
                stop_at_unet_number=train_unet_number ,
                cond_scale=this_cond_scale, # cond_scales ,
                x_data=None, 
                # ++
                x_data_tokenized=x_data_tokenized,
                #
                skip_steps=skip_steps,
                inpaint_images = inpaint_images,
                inpaint_masks = inpaint_masks,
                inpaint_resample_times = inpaint_resample_times,
                init_images = init_images,device=device,
                # ++++++++++++++++++++++++++
                tokenizer_X=tokenizer_X,
                Xnormfac=Xnormfac,
                max_length=max_length,
            )
            
        # handle the results: from embedding into AA  
        # ++ for pLM
        if image_channels==33:
            # pass
            result_tokens,result_logits = convert_into_tokens_using_prob(
                result_embedding,
                pLM_Model_Name,
            )
        else:
            # full record
            # result_embedding as image.dim: [batch, channels, seq_len]
            # result_tokens.dim: [batch, seq_len]
            result_tokens,result_logits = convert_into_tokens(
                pLM_Model, 
                result_embedding,
                pLM_Model_Name,
            )
        # +++++++++++++++++++++++++++++++++
        result=result_tokens.unsqueeze(1) # dim: [batch, 1, seq_len]
        
        # + for debug
        print('result dim: ', result.shape)
        
        # plot sequence token code: esm (33 tokens), for one batch
        fig=plt.figure()
        for ii in range(lenn_val):
            plt.plot (
                result[ii,0,:].cpu().detach().numpy(),
                label= f'Predicted for Input#{str(ii)}'
            )
        #plt.plot (GT[samples,0,:]*ynormfac,label= f'GT {0}')
        plt.legend()
        outname = sample_dir+ f"DenovoInputXs_CondScale_No{str(idx_cond)}_Val_{str(this_cond_scale)}_{e}_{steps}.jpg"
        #plt.title (f"Sample {samples}, cond scale={str (cond_scales[iisample])}")
        if IF_showfig==1:
            plt.show ()
        else:
            plt.savefig(outname, dpi=200)
        plt.close()
        
        # translate result into AA
        to_rev=result[:,0,:] # token (batch,seq_len)
        if CKeys['Debug_TrainerPack']==3:
            print("on foldable result: ", to_rev[0])
            print("on result_logits: ", result_logits[0])
            print("on mask: ", result_mask[0])
            a = decode_one_ems_token_rec_for_folding_with_mask(
                to_rev[0],
                result_logits[0],
                pLM_alphabet,
                pLM_Model,
                result_mask[0],
            )
            print('One resu: ', a)

        y_data_reversed=decode_many_ems_token_rec_for_folding_with_mask(
            to_rev,
            result_logits,
            pLM_alphabet,
            pLM_Model,
            result_mask,
        )
        if CKeys['Debug_TrainerPack']==3:
            print("on y_data_reversed[0]: ", y_data_reversed[0])
            
        ### reverse second structure input....
        if X_cond != None:
            X_cond=torch.round(X_cond*Xnormfac)

            to_rev=X_cond[:,:] 
            to_rev=to_rev.long().cpu().detach().numpy()
            print (to_rev.shape)
            X_data_reversed=tokenizer_X.sequences_to_texts (to_rev)

            for iii in range (len(y_data_reversed)):
                X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")
        if x_data !=None:
            # work for second structure input....
            # work for ForcPath input...
            X_data_reversed=x_data #is already in sequence fromat..
        
        # sections for each one result
        for iisample in range(lenn_val):
            print (f"For {X} or {X_data_reversed[iisample]}, predicted sequence: ", y_data_reversed[iisample])
            
            out_nam_fasta=f'{sample_dir}DN_{iisample}_CondS_No_{idx_cond}_Val_{this_cond_scale}_epo_{e}_step_{steps}.fasta'
            write_fasta (y_data_reversed[iisample], out_nam_fasta) 
            fasta_file_list.append(out_nam_fasta)
        
            # + for debug
            print("================================================")
            print("foldproteins: ", foldproteins)
            
            if not foldproteins:
                pdb_file=None

            else:

                if X_cond != None:
                    # not maintained
                    xbc=X_cond[iisample,:].cpu().detach().numpy()
                    out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.4f" % xbc})+f'_{flag}_{steps}'
                if x_data !=None:
                    pass
                    # #xbc=x_data[iisample] 
                    # # ----------------------------------
                    # # this one can be too long for a name
                    # out_nam=x_data[iisample] 
                    # # ++++++++++++++++++++++++++++++++++
                    # # 
                    # out_nam=iisample

                tempname='temp'
                pdb_file, fasta_file=foldandsavePDB_pdb_fasta (
                    sequence=y_data_reversed[iisample], 
                    filename_out=tempname, 
                    num_cycle=num_cycle, 
                    flag=flag,
                    # +++++++++++++++++++
                    # prefix=prefix,
                    prefix=sample_dir,
                )         


                # out_nam=f'{prefix}{X_data_reversed[iisample]}_{flag}_{steps}.pdb'
                # out_nam=f'{sample_dir}{X_data_reversed[iisample]}_{flag}_{steps}.pdb'
                # -------------------------------------------
                # this one can be too long for a name
                # However, the input X is recorded in the code
                # out_nam=f'{sample_dir}{X_data_reversed[iisample]}_{flag}_{iisample}.pdb'
                # +++++++++++++++++++++++++++++++++++++++++++
                out_nam=f'{sample_dir}DN_{iisample}_CondS_No_{idx_cond}_Val_{this_cond_scale}_epo_{e}_step_{steps}.pdb'
                # out_nam_fasta=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.fasta'

                # print('Debug 1: out: ', out_nam)
                # print('Debug 2: in: ', pdb_file)
                shutil.copy (pdb_file, out_nam) #source, dest
                # shutil.copy (fasta_file, out_nam_fasta)
                # cmd_line = 'cp ' + pdb_file + ' ' + out_nam
                # print(cmd_line)
                # os.popen(cmd_line)
                # print('Debug 3')
                # clean the slade to avoid mistakenly using the previous fasta file
                os.remove (pdb_file)
                os.remove (fasta_file)

                pdb_file=out_nam
                # fasta_file=out_nam_fasta
                pdb_file_list.append(pdb_file)
                
                # ++ write the input condtion as a reference: for ForcPath
                out_nam_inX=f'{sample_dir}DN_{iisample}_CondS_No_{idx_cond}_Val_{this_cond_scale}_epo_{e}_step_{steps}_input.txt'
                if torch.is_tensor(X_data_reversed[iisample]):
                    # for safety, not used usually
                    xbc=X_data_reversed[iisample].cpu().detach().numpy()
                else:
                    xbc=X_data_reversed[iisample]
                if tokenizer_X==None:
                    # for ForcPath case
                    out_inX=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.4f" % xbc})
                else:
                    # for SecStr case
                    out_inX=xbc
                with open(out_nam_inX, "w") as inX_file:
                    inX_file.write(out_inX)


                print (f"Properly named PDB file produced: {pdb_file}")
                if IF_showfig==1:
                    #flag=1000
                    view=show_pdb(
                        pdb_file=pdb_file, 
                        flag=flag,
                        show_sidechains=show_sidechains, 
                        show_mainchains=show_mainchains, 
                        color=color
                    )
                    view.show()            
    
        
    return pdb_file_list, fasta_file_list

# ++
def sample_sequence_pLM_ModelB_For_ForcPath_Predictor (
    model,
    X=None, #this is the target conventionally when using text embd
    flag=0,
    cond_scales=[1.], # 1.,
    foldproteins=False,
    X_string=None,
    x_data=None,  
    skip_steps=0,
    inpaint_images = None,
    inpaint_masks = None,
    inpaint_resample_times = None,
    init_images = None,
    num_cycle=16,
    # ++++++++++++++++++++++++
    ynormfac=1,
    train_unet_number=1,
    tokenizer_X=None,
    Xnormfac=1.,
    max_length=1.,
    prefix=None,
    tokenizer_y=None,
    # ++
    CKeys=None,
    sample_dir=None,
    steps=None,
    e=None,
    IF_showfig=True, # effective only after foldproteins=True
    # ++
    pLM_Model=None, # pLM_Model,
    pLM_Model_Name=None, # pLM_Model_Name,
    image_channels=None, # image_channels,
    pLM_alphabet=None, # esm_alphabet,
    # ++
    esm_layer=None,
):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # input: a list of AA sequence in string format
    # output: ForcPath prediction
    #
    # 1. decide input channel: text_cond or img_cond
    X_cond=None # this is for text-conditioning
    if X_string==None and X != None: #only do if X provided
        print (f"Producing {len(X)} samples...from text conditioning X...")
        lenn_val=len(X)
        # shape of X: [[..],[..]]: double bracket
        X_cond=torch.Tensor(X).to(device)
        # --
        # X_cond=torch.Tensor (X[iisample]).to(device).unsqueeze (0)
    if X_string !=None:
        print (f"Producing {len(X_string)} samples...from text conditioning X_String (from string)...")
        lenn_val=len(X_string)
        # --
        XX = tokenizer_X.texts_to_sequences(X_string[iisample])
        # ++
        XX = tokenizer_X.texts_to_sequences(X_string)
        XX= sequence.pad_sequences(XX,  maxlen=max_length, padding='post', truncating='post')  
        XX=np.array(XX)
        X_cond=torch.from_numpy(XX).float()/Xnormfac
        print ('Tokenized and processed: ', X_cond)
        
        
    if x_data!=None: # this is for img_conditioning channel
        # Will use this channel
        print (f"Producing {len(x_data)} samples...from image conditingig x_data  ...")
        lenn_val=len(x_data)
        seq_len_list=[]
        for this_AA in x_data:
            seq_len_list.append(len(this_AA))
            
        
    print ("Input contents:")    
    print ("cond_img condition: x_data=\n", x_data)
    print ("Text condition: X_cond=\n", X_cond)
    
    # 2. perform sampling
    # loop over cond_scales
    resu_prediction={}
    
    for idx_cond, this_cond_scale in enumerate(cond_scales):
        print(f"Working on cond_scale {str(this_cond_scale)}")
        # leave the translation from seq to tokenized 
        # in the model.sample function using x_data channel
        # Need to pass on the esm model part or tokenizer_X
        # 
        result_embedding=model.sample ( 
            x=X_cond,
            stop_at_unet_number=train_unet_number ,
            cond_scale=this_cond_scale, # cond_scales ,
            x_data=x_data, # x_data[iisample],  # will pass through tokenizer_X in this sample(), channels will be matched with self.pred_dim
            # ++
            x_data_tokenized=None,
            skip_steps=skip_steps,
            inpaint_images = inpaint_images,
            inpaint_masks = inpaint_masks,
            inpaint_resample_times = inpaint_resample_times,
            init_images = init_images,device=device,
            # ++++++++++++++++++++++++++
            tokenizer_X=tokenizer_X,
            Xnormfac=Xnormfac,
            max_length=max_length,
            # ++ for esm
            pLM_Model=pLM_Model,
            pLM_alphabet=pLM_alphabet,
            esm_layer=esm_layer,
            pLM_Model_Name=pLM_Model_Name,
            # image_channels=image_channels,
        )
        
        # convert into prediction
        # 3. translate prediction into something meaningful
        #    consider channel average and masking
        # average across channels
        result_embedding=torch.mean(result_embedding, 1) # (batch, seq_len)
        # read mask from input: X_train_batch_picked (batch, seq_len)
        # result_mask looks like, 0,1,1,...,1,0,0
        # will fill 0th component be zero
        result_mask = read_mask_from_input(
            tokenized_data=None, # X_train_batch[:num_samples], 
            mask_value=0.0,
            seq_data=x_data, # None,
            max_seq_length=max_length, # None,
        )
        # apply mask to result: keep true and zero all false
        # this also make sure 0th components are zero ACCIDENTLY
        result = result_embedding.cpu()*result_mask # (batch, seq_len)
        # result = result.cpu()
        y_data_reversed = result*ynormfac
        # 4. translate the results into a list
        prediction_list = []
        for ii in range(len(x_data)):
            prediction_list.append(
                y_data_reversed[ii, :seq_len_list[ii]+1]
            )
        if CKeys['Debug_TrainerPack']==3:
            print("check prediction dim:")
            print(f"model output: ", y_data_reversed[0])
            print(f"prediction output: ", prediction_list[0])
        # 
        # store the results
        resu_prediction[str(this_cond_scale)]=prediction_list
    
    return resu_prediction,seq_len_list
       
        
        

        
    
    
    
    
#     # --------------------------------------------------------------------------
#     # prepare input in different channels
#     # 
#     X_cond=None  # this is for text-conditioning
#     if X_string==None and X != None: #only do if X provided
#         print (f"Producing {len(X)} samples...from text conditioning X...")
#         lenn_val=len(X)
#         # shape of X: [[..],[..]]: double bracket
#         X_cond=torch.Tensor(X).to(device)
#         # --
#         # X_cond=torch.Tensor (X[iisample]).to(device).unsqueeze (0)
#     if X_string !=None:
#         print (f"Producing {len(X_string)} samples...from text conditioning X_String (from string)...")
#         lenn_val=len(X_string)
#         # --
#         XX = tokenizer_X.texts_to_sequences(X_string[iisample])
#         # ++
#         XX = tokenizer_X.texts_to_sequences(X_string)
#         XX= sequence.pad_sequences(XX,  maxlen=max_length, padding='post', truncating='post')  
#         XX=np.array(XX)
#         X_cond=torch.from_numpy(XX).float()/Xnormfac
#         print ('Tokenized and processed: ', X_cond)
        
#     if x_data!=None:
#         print (f"Producing {len(x_data)} samples...from image conditingig x_data  ...")
#         lenn_val=len(x_data)
#         if tokenizer_X==None: # for ForcPath,
#             # need to do Padding and Normalization
#             # and then put into tokenized data channel
#             x_data_tokenized=[]
#             for ii in range(lenn_val):
#                 x_data_one_line=pad_a_np_arr(x_data[ii], 0.0, max_length)
#                 x_data_tokenized.append(x_data_one_line)
#             x_data_tokenized=np.array(x_data_tokenized)
#             x_data_tokenized=torch.from_numpy(x_data_tokenized/Xnormfac)
#         else:
#             # leave for SecStr case: TBA
#             pass
#         # print (x_data)
#         # ++ for result_mask based on input: x_data or x_data_tokenized
#         # ++: for model B: using mask from the input
#         # extract the mask/seq_len from input if possible
#         if tokenizer_X!=None:
#             # for SecStr+ModelB
#             result_mask = read_mask_from_input(
#                 tokenized_data=None, 
#                 mask_value=None,
#                 seq_data=x_data, # x_data[iisample],
#                 max_seq_length=max_length,
#             )
#         else:
#             # for ForcPath+ModelB
#             result_mask = read_mask_from_input(
#                 tokenized_data=x_data_tokenized, # None, 
#                 mask_value=0, # None,
#                 seq_data=None, # x_data[iisample],
#                 max_seq_length=None, # max_length,
#             )
            
        
#     print ("Input contents:")    
#     print ("cond_img condition: x_data=\n", x_data)
#     print ("Text condition: X_cond=\n", X_cond)
    
#     # store the results
#     pdb_file_list=[]
#     fasta_file_list=[]
    
#     # loop over cond_scales
#     for idx_cond, this_cond_scale in enumerate(cond_scales):
#         print(f"Working on cond_scale {str(this_cond_scale)}")
#         # do sampling
#         # -----------------------------------------------------------------
#         # for below, two branches are all for cond_img, not for text_cond
#         if tokenizer_X!=None:
#             # for SecStr+ModelB, not test here
#             result_embedding=model.sample ( 
#                 x=X_cond,
#                 stop_at_unet_number=train_unet_number ,
#                 cond_scale=this_cond_scale, # cond_scales ,
#                 x_data=x_data, # x_data[iisample],  # will pass through tokenizer_X in this sample(), channels will be matched with self.pred_dim
#                 # ++
#                 x_data_tokenized=None,
#                 skip_steps=skip_steps,
#                 inpaint_images = inpaint_images,
#                 inpaint_masks = inpaint_masks,
#                 inpaint_resample_times = inpaint_resample_times,
#                 init_images = init_images,device=device,
#                 # ++++++++++++++++++++++++++
#                 tokenizer_X=tokenizer_X,
#                 Xnormfac=Xnormfac,
#                 max_length=max_length,
#             )
#         else:
#             # for ForcPath+ModelB:
#             # for model.sample() here using x_data_tokenized channel
#             x_data_tokenized=x_data_tokenized.to(torch.float) # shape [batch, max_seq_len]
#             # leave channel expansion for the self.sample() to handle
            
#             # + for debug:
#             if CKeys['Debug_TrainerPack']==3:
#                 print("x_data_tokenized dim: ", x_data_tokenized.shape)
#                 print("x_data_tokenized dtype: ", x_data_tokenized.dtype)
#                 print("test x_data_tokenized!=None: ", x_data_tokenized!=None)
            
#             result_embedding=model.sample ( 
#                 x=X_cond,
#                 stop_at_unet_number=train_unet_number ,
#                 cond_scale=this_cond_scale, # cond_scales ,
#                 x_data=None, 
#                 # ++
#                 x_data_tokenized=x_data_tokenized,
#                 #
#                 skip_steps=skip_steps,
#                 inpaint_images = inpaint_images,
#                 inpaint_masks = inpaint_masks,
#                 inpaint_resample_times = inpaint_resample_times,
#                 init_images = init_images,device=device,
#                 # ++++++++++++++++++++++++++
#                 tokenizer_X=tokenizer_X,
#                 Xnormfac=Xnormfac,
#                 max_length=max_length,
#             )
            
#         # handle the results: from embedding into AA    
#         # ++ for pLM
#         # full record
#         # result_embedding as image.dim: [batch, channels, seq_len]
#         # result_tokens.dim: [batch, seq_len]
#         result_tokens,result_logits = convert_into_tokens(
#             pLM_Model, 
#             result_embedding,
#             pLM_Model_Name,
#         )
#         # +++++++++++++++++++++++++++++++++
#         result=result_tokens.unsqueeze(1) # dim: [batch, 1, seq_len]
        
#         # + for debug
#         print('result dim: ', result.shape)
        
#         # plot sequence token code: esm (33 tokens), for one batch
#         fig=plt.figure()
#         for ii in range(lenn_val):
#             plt.plot (
#                 result[ii,0,:].cpu().detach().numpy(),
#                 label= f'Predicted for Input#{str(ii)}'
#             )
#         #plt.plot (GT[samples,0,:]*ynormfac,label= f'GT {0}')
#         plt.legend()
#         outname = sample_dir+ f"DenovoInputXs_CondScale_No{str(idx_cond)}_Val_{str(this_cond_scale)}_{e}_{steps}.jpg"
#         #plt.title (f"Sample {samples}, cond scale={str (cond_scales[iisample])}")
#         if IF_showfig==1:
#             plt.show ()
#         else:
#             plt.savefig(outname, dpi=200)
#         plt.close()
        
#         # translate result into AA
#         to_rev=result[:,0,:] # token (batch,seq_len)
#         if CKeys['Debug_TrainerPack']==3:
#             print("on foldable result: ", to_rev[0])
#             print("on result_logits: ", result_logits[0])
#             print("on mask: ", result_mask[0])
#             a = decode_one_ems_token_rec_for_folding_with_mask(
#                 to_rev[0],
#                 result_logits[0],
#                 pLM_alphabet,
#                 pLM_Model,
#                 result_mask[0],
#             )
#             print('One resu: ', a)

#         y_data_reversed=decode_many_ems_token_rec_for_folding_with_mask(
#             to_rev,
#             result_logits,
#             pLM_alphabet,
#             pLM_Model,
#             result_mask,
#         )
#         if CKeys['Debug_TrainerPack']==3:
#             print("on y_data_reversed[0]: ", y_data_reversed[0])
            
#         ### reverse second structure input....
#         if X_cond != None:
#             X_cond=torch.round(X_cond*Xnormfac)

#             to_rev=X_cond[:,:] 
#             to_rev=to_rev.long().cpu().detach().numpy()
#             print (to_rev.shape)
#             X_data_reversed=tokenizer_X.sequences_to_texts (to_rev)

#             for iii in range (len(y_data_reversed)):
#                 X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")
#         if x_data !=None:
#             # work for second structure input....
#             # work for ForcPath input...
#             X_data_reversed=x_data #is already in sequence fromat..
        
#         # sections for each one result
#         for iisample in range(lenn_val):
#             print (f"For {X} or {X_data_reversed[iisample]}, predicted sequence: ", y_data_reversed[iisample])
            
#             out_nam_fasta=f'{sample_dir}DN_{iisample}_CondS_No_{idx_cond}_Val_{this_cond_scale}_epo_{e}_step_{steps}.fasta'
#             write_fasta (y_data_reversed[iisample], out_nam_fasta) 
#             fasta_file_list.append(out_nam_fasta)
        
#             # + for debug
#             print("================================================")
#             print("foldproteins: ", foldproteins)
            
#             if not foldproteins:
#                 pdb_file=None

#             else:

#                 if X_cond != None:
#                     # not maintained
#                     xbc=X_cond[iisample,:].cpu().detach().numpy()
#                     out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.4f" % xbc})+f'_{flag}_{steps}'
#                 if x_data !=None:
#                     pass
#                     # #xbc=x_data[iisample] 
#                     # # ----------------------------------
#                     # # this one can be too long for a name
#                     # out_nam=x_data[iisample] 
#                     # # ++++++++++++++++++++++++++++++++++
#                     # # 
#                     # out_nam=iisample

#                 tempname='temp'
#                 pdb_file, fasta_file=foldandsavePDB_pdb_fasta (
#                     sequence=y_data_reversed[iisample], 
#                     filename_out=tempname, 
#                     num_cycle=num_cycle, 
#                     flag=flag,
#                     # +++++++++++++++++++
#                     # prefix=prefix,
#                     prefix=sample_dir,
#                 )         


#                 # out_nam=f'{prefix}{X_data_reversed[iisample]}_{flag}_{steps}.pdb'
#                 # out_nam=f'{sample_dir}{X_data_reversed[iisample]}_{flag}_{steps}.pdb'
#                 # -------------------------------------------
#                 # this one can be too long for a name
#                 # However, the input X is recorded in the code
#                 # out_nam=f'{sample_dir}{X_data_reversed[iisample]}_{flag}_{iisample}.pdb'
#                 # +++++++++++++++++++++++++++++++++++++++++++
#                 out_nam=f'{sample_dir}DN_{iisample}_CondS_No_{idx_cond}_Val_{this_cond_scale}_epo_{e}_step_{steps}.pdb'
#                 # out_nam_fasta=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.fasta'

#                 # print('Debug 1: out: ', out_nam)
#                 # print('Debug 2: in: ', pdb_file)
#                 shutil.copy (pdb_file, out_nam) #source, dest
#                 # shutil.copy (fasta_file, out_nam_fasta)
#                 # cmd_line = 'cp ' + pdb_file + ' ' + out_nam
#                 # print(cmd_line)
#                 # os.popen(cmd_line)
#                 # print('Debug 3')
#                 # clean the slade to avoid mistakenly using the previous fasta file
#                 os.remove (pdb_file)
#                 os.remove (fasta_file)

#                 pdb_file=out_nam
#                 # fasta_file=out_nam_fasta
#                 pdb_file_list.append(pdb_file)
                
#                 # ++ write the input condtion as a reference: for ForcPath
#                 out_nam_inX=f'{sample_dir}DN_{iisample}_CondS_No_{idx_cond}_Val_{this_cond_scale}_epo_{e}_step_{steps}_input.txt'
#                 if torch.is_tensor(X_data_reversed[iisample]):
#                     # for safety, not used usually
#                     xbc=X_data_reversed[iisample].cpu().detach().numpy()
#                 else:
#                     xbc=X_data_reversed[iisample]
#                 if tokenizer_X==None:
#                     # for ForcPath case
#                     out_inX=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.4f" % xbc})
#                 else:
#                     # for SecStr case
#                     out_inX=xbc
#                 with open(out_nam_inX, "w") as inX_file:
#                     inX_file.write(out_inX)


#                 print (f"Properly named PDB file produced: {pdb_file}")
#                 if IF_showfig==1:
#                     #flag=1000
#                     view=show_pdb(
#                         pdb_file=pdb_file, 
#                         flag=flag,
#                         show_sidechains=show_sidechains, 
#                         show_mainchains=show_mainchains, 
#                         color=color
#                     )
#                     view.show()            
    
        
#     return pdb_file_list, fasta_file_list

# ++
# for ProteinDesigner
def sample_loop_omegafold_pLM_ModelB (
    model,
    train_loader,
    cond_scales=[7.5], #list of cond scales - each sampled...
    num_samples=2, #how many samples produced every time tested.....
    timesteps=100, # not used
    flag=0,
    foldproteins=False,
    use_text_embedd=True,
    skip_steps=0,
    # +++++++++++++++++++
    train_unet_number=1,
    ynormfac=1,
    prefix=None,
    tokenizer_y=None,
    Xnormfac=1,
    tokenizer_X=None,
    # ++
    CKeys=None,
    sample_dir=None,
    steps=None,
    e=None,
    IF_showfig=True, # effective only after foldproteins=True
    # ++
    pLM_Model=None,
    pLM_Model_Name=None,
    image_channels=None,
    pLM_alphabet=None,
):
    # =====================================================
    # sample # = num_samples*(# of mini-batches)
    # =====================================================
    # steps=0
    # e=flag
    # for item  in train_loader:
    for idx, item  in enumerate(train_loader):

        X_train_batch= item[0].to(device)
        y_train_batch=item[1].to(device)
        
        # --
        # # ++ for pLM case:
        # if pLM_Model_Name=='None':
        #     # just use the encoded sequence
        #     # y_train_batch_in = y_train_batch.unsqueeze(1)
        #     X_train_batch_in = X_train_batch.unsqueeze(1)
        #     # pass
        # elif pLM_Model_Name=='esm2_t33_650M_UR50D':
        #     # with torch.no_grad():
        #     #     results = pLM_Model(
        #     #         y_train_batch,
        #     #         repr_layers=[33],
        #     #         return_contacts=False,
        #     #     )
        #     # y_train_batch_in = results["representations"][33]
        #     # y_train_batch_in = rearrange(
        #     #     y_train_batch_in, 
        #     #     'b l c -> b c l'
        #     # )
        #     X_train_batch_in = X_train_batch.unsqueeze(1).repeat(1,image_channels,1)


        GT=y_train_batch.cpu().detach() 

        GT= GT.unsqueeze(1)
        if num_samples>y_train_batch.shape[0]:
            print("Warning: sampling # > len(mini_batch)")

        num_samples = min (num_samples,y_train_batch.shape[0] )
        print (f"Producing {num_samples} samples...")
        X_train_batch_picked = X_train_batch[:num_samples,:] # X_train_batch_in[:num_samples ] # 
        print ('After pLM, (TEST) X_batch shape: ', X_train_batch_picked.shape)
                        
        for iisample in range (len (cond_scales)):

            if use_text_embedd:
                result_embedding=model.sample (
                    # x= X_train_batch,
                    x= X_train_batch_picked,
                    stop_at_unet_number=train_unet_number ,
                    cond_scale=cond_scales[iisample], 
                    device=device, 
                    skip_steps=skip_steps
                )
            else:
                result_embedding=model.sample (
                    x= None, 
                    # x_data_tokenized= X_train_batch,
                    x_data_tokenized= X_train_batch_picked, # dim=(batch, seq_len), will extend channels inside .sample(),
                    stop_at_unet_number=train_unet_number ,
                    cond_scale=cond_scales[iisample],
                    device=device,
                    skip_steps=skip_steps
                )
            # ++ for pLM:
            if image_channels==33:
                result_tokens,result_logits = convert_into_tokens_using_prob(
                    result_embedding,
                    pLM_Model_Name,
                )
            else:
                # full record
                # result_embedding as image.dim: [batch, channels, seq_len]
                # result_tokens.dim: [batch, seq_len]
                result_tokens,result_logits = convert_into_tokens(
                    pLM_Model, 
                    result_embedding,
                    pLM_Model_Name,
                )

            # # ---------------------------------
            # result=torch.round(result*ynormfac)
            # GT=torch.round (GT*ynormfac)
            # +++++++++++++++++++++++++++++++++
            result=result_tokens.unsqueeze(1) # dim: [batch, 1, seq_len]
            
            # +
                        # # -------------------------------------------
            # #reverse y sequence
            # to_rev=result[:,0,:]
            # to_rev=to_rev.long().cpu().detach().numpy()
            # 
            # y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)
            # 
            # for iii in range (len(y_data_reversed)):
            #     y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
            # ++++++++++++++++++++++++++++++++++++++++++++
            # extract the mask/seq_len from input if possible
            # here, from dataloader, we only use tokenized_data for mask generation
            result_mask = read_mask_from_input(
                tokenized_data=X_train_batch[:num_samples], 
                mask_value=0,
                seq_data=None,
                max_seq_length=None,
            )
            to_rev=result[:,0,:] # token (batch,seq_len)
            if CKeys['Debug_TrainerPack']==3:
                print("on foldable result: ", to_rev[0])
                print("on result_logits: ", result_logits[0])
                print("on mask: ", result_mask[0])
                a = decode_one_ems_token_rec_for_folding_with_mask(
                    to_rev[0],
                    result_logits[0],
                    pLM_alphabet,
                    pLM_Model,
                    result_mask[0],
                )
                print('One resu: ', a)

            y_data_reversed=decode_many_ems_token_rec_for_folding_with_mask(
                to_rev,
                result_logits,
                pLM_alphabet,
                pLM_Model,
                result_mask,
            )
            if CKeys['Debug_TrainerPack']==3:
                print("on y_data_reversed[0]: ", y_data_reversed[0])

            # # ++++++++++++++++++++++++++++++++++++++++++++
            # # reverse the PREDICTED y into a foldable sequence
            # # save this block for Model A 
            # to_rev=result[:,0,:] # token (batch,seq_len)
            # y_data_reversed=decode_many_ems_token_rec_for_folding(
            #     to_rev,
            #     result_logits,
            #     pLM_alphabet,
            #     pLM_Model,
            # )
            # if CKeys['Debug_TrainerPack']==3:
            #     print("on foldable result: ", to_rev[0])
            #     print("on result_logits: ", result_logits[0])
            #     a = decode_one_ems_token_rec_for_folding(
            #         to_rev[0],
            #         result_logits[0],
            #         pLM_alphabet,
            #         pLM_Model,
            #     )
            #     print('One resu: ', a)
            #     print("on y_data_reversed: ", y_data_reversed[0])


            # # -----------------------------------------------------
            # #reverse GT_y sequence
            # to_rev=GT[:,0,:]
            # to_rev=to_rev.long().cpu().detach().numpy()
            # 
            # GT_y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)
            # 
            # for iii in range (len(y_data_reversed)):
            #     GT_y_data_reversed[iii]=GT_y_data_reversed[iii].upper().strip().replace(" ", "")
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #reverse GT_y sequence
            # GT should be SAFE to reverse
            to_rev=GT[:,0,:] # (batch,1,seq_len)->(batch, seq_len)
            GT_y_data_reversed=decode_many_ems_token_rec(
                to_rev,
                pLM_alphabet,
            )


            # -- not for SecStr anymore
            # ### reverse second structure input....
            # to_rev=torch.round (X_train_batch[:,:]*Xnormfac)
            # to_rev=to_rev.long().cpu().detach().numpy()
            # ++ 
            ### reverse general float input...
            to_rev=X_train_batch[:,:]*Xnormfac
            to_rev=to_rev.cpu().detach().numpy()
            # here, assume X_train_batch is for cond_img: there are padding at both beginning and ending part
            # so, first move the 0th padding to the end:
            # Note:
            # 1. this is good for SecStr case: (not maintained here)
            # 2. this is not good for ForcPath, but can be cued in MD postprocess since the first component will always be 0
            n_batch=to_rev.shape[0]
            n_embed=to_rev.shape[1]
            to_rev_1 = np.zeros(to_rev.shape)
            to_rev_1[:,0:n_embed-1]=to_rev[:,1:n_embed]

            # ++ different input
            if tokenizer_X!=None:
                # change into int
                to_rev_1 = np.round(to_rev_1)
                X_data_reversed=tokenizer_X.sequences_to_texts (to_rev_1)
                for iii in range (len(y_data_reversed)):
                    X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")
            else:
                X_data_reversed=to_rev_1.copy()
            # + for debug
            if CKeys['Debug_TrainerPack']==1:
                print("X_data_reversed: ", X_data_reversed)
                    

            for samples in range  (num_samples):
                print ("sample ", samples+1, "out of ", num_samples)

                fig=plt.figure()
                plt.plot (
                    result[samples,0,:].cpu().detach().numpy(),
                    label= f'Predicted'
                )
                plt.plot (
                    GT[samples,0,:],
                    label= f'GT {0}'
                )
                plt.legend()
                outname = sample_dir+ f"Batch_{idx}_sample_{samples}_condscale-{str (cond_scales[iisample])}_{e}_{steps}.jpg"
                if IF_showfig==1:
                    plt.show()
                else:
                    plt.savefig(outname, dpi=200)
                plt.close ()
                
#                 # # -------------------------------------------
#                 # #reverse y sequence
#                 # to_rev=result[:,0,:]
#                 # to_rev=to_rev.long().cpu().detach().numpy()
#                 # 
#                 # y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)
#                 # 
#                 # for iii in range (len(y_data_reversed)):
#                 #     y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
#                 # ++++++++++++++++++++++++++++++++++++++++++++
#                 # extract the mask/seq_len from input if possible
#                 # here, from dataloader, we only use tokenized_data for mask generation
#                 result_mask = read_mask_from_input(
#                     tokenized_data=X_train_batch[:num_samples], 
#                     mask_value=0,
#                     seq_data=None,
#                     max_seq_length=None,
#                 )
#                 to_rev=result[:,0,:] # token (batch,seq_len)
#                 if CKeys['Debug_TrainerPack']==3:
#                     print("on foldable result: ", to_rev[0])
#                     print("on result_logits: ", result_logits[0])
#                     print("on mask: ", result_mask[0])
#                     a = decode_one_ems_token_rec_for_folding_with_mask(
#                         to_rev[0],
#                         result_logits[0],
#                         pLM_alphabet,
#                         pLM_Model,
#                         result_mask[0],
#                     )
#                     print('One resu: ', a)
                    
#                 y_data_reversed=decode_many_ems_token_rec_for_folding_with_mask(
#                     to_rev,
#                     result_logits,
#                     pLM_alphabet,
#                     pLM_Model,
#                     result_mask,
#                 )
#                 if CKeys['Debug_TrainerPack']==3:
#                     print("on y_data_reversed[0]: ", y_data_reversed[0])
                    
#                 # # ++++++++++++++++++++++++++++++++++++++++++++
#                 # # reverse the PREDICTED y into a foldable sequence
#                 # # save this block for Model A 
#                 # to_rev=result[:,0,:] # token (batch,seq_len)
#                 # y_data_reversed=decode_many_ems_token_rec_for_folding(
#                 #     to_rev,
#                 #     result_logits,
#                 #     pLM_alphabet,
#                 #     pLM_Model,
#                 # )
#                 # if CKeys['Debug_TrainerPack']==3:
#                 #     print("on foldable result: ", to_rev[0])
#                 #     print("on result_logits: ", result_logits[0])
#                 #     a = decode_one_ems_token_rec_for_folding(
#                 #         to_rev[0],
#                 #         result_logits[0],
#                 #         pLM_alphabet,
#                 #         pLM_Model,
#                 #     )
#                 #     print('One resu: ', a)
#                 #     print("on y_data_reversed: ", y_data_reversed[0])
                
                
#                 # # -----------------------------------------------------
#                 # #reverse GT_y sequence
#                 # to_rev=GT[:,0,:]
#                 # to_rev=to_rev.long().cpu().detach().numpy()
#                 # 
#                 # GT_y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)
#                 # 
#                 # for iii in range (len(y_data_reversed)):
#                 #     GT_y_data_reversed[iii]=GT_y_data_reversed[iii].upper().strip().replace(" ", "")
#                 # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                 #reverse GT_y sequence
#                 # GT should be SAFE to reverse
#                 to_rev=GT[:,0,:]
#                 GT_y_data_reversed=decode_many_ems_token_rec(
#                     to_rev,
#                     pLM_alphabet,
#                 )
                

#                 ### reverse second structure input....
#                 to_rev=torch.round (X_train_batch[:,:]*Xnormfac)
#                 to_rev=to_rev.long().cpu().detach().numpy()
#                 # here, assume X_train_batch is for cond_img: there are padding at both beginning and ending part
#                 # so, first move the 0th padding to the end
#                 n_batch=to_rev.shape[0]
#                 n_embed=to_rev.shape[1]
#                 to_rev_1 = np.zeros(to_rev.shape)
#                 to_rev_1[:,0:n_embed-1]=to_rev[:,1:n_embed]
                
#                 # ++ different input
#                 if tokenizer_X!=None:
#                     X_data_reversed=tokenizer_X.sequences_to_texts (to_rev_1)
#                     for iii in range (len(y_data_reversed)):
#                         X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")
#                 else:
#                     X_data_reversed=to_rev_1.copy()
#                 # + for debug
#                 if CKeys['Debug_TrainerPack']==1:
#                     print("X_data_reversed: ", X_data_reversed)
                

                # print (f"For {X_train_batch[samples,:].cpu().detach().numpy()} or {X_data_reversed[samples]}, \npredicted sequence: ", y_data_reversed[samples])
                print (f"For {X_train_batch[samples,:].cpu().detach().numpy()} \nor\n {X_data_reversed[samples]}, ")
                print (f"predicted sequence: {y_data_reversed[samples]}")
                print (f"Ground truth:       {GT_y_data_reversed[samples]}")
                error=string_diff (y_data_reversed[samples], GT_y_data_reversed[samples])/len (GT_y_data_reversed[samples])
                print(f"Recovery ratio(Ref): {1.-error}")
                
                # move some
                # # -- X_train_batch is normalized
                # xbc=X_train_batch[samples,:].cpu().detach().numpy()
                # # out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})
                # out_nam_content=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})
                # ++
                xbc = X_data_reversed[samples]
                if type(xbc)==str:
                    out_nam_content=xbc
                else:
                    out_nam_content=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.4f" % xbc})
                # 1. write out the input X in the dataloder
                out_nam_inX=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}.txt'
                # + write the condition clearly
                # X_data_reversed: an array
                with open(out_nam_inX, "w") as inX_file:
                    # inX_file.write(f'{X_data_reversed[samples]}\n')
                    inX_file.write(out_nam_content)
                # 2. write out the predictions
                out_nam_OuY_PR=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}_predict.fasta'
                with open(out_nam_OuY_PR, "w") as ouY_fasta:
                    ouY_fasta.write(f">Predicted\n")
                    ouY_fasta.write(y_data_reversed[samples])
                # 3. Only for dataloader: write out the recovered ground truth
                out_nam_OuY_GT=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}_recGT.fasta'
                with open(out_nam_OuY_GT, "w") as ouY_fasta:
                    ouY_fasta.write(f">reconstructed GT, recoverabliblity: {1.-error}\n")
                    ouY_fasta.write(GT_y_data_reversed[samples])
                
                

                if foldproteins:
                    
                    tempname='temp'
                    pdb_file,fasta_file=foldandsavePDB_pdb_fasta (
                        sequence=y_data_reversed[samples], 
                        filename_out=tempname, 
                        num_cycle=16, flag=flag,
                        # +++++++++++++++++++
                        prefix=prefix
                    )

                    # #out_nam=f'{prefix}{out_nam}.pdb'
                    # out_nam=f'{prefix}{X_data_reversed[samples]}.pdb'
                    # ------------------------------------------------------
                    # sometime, this name below can get too long to fit
                    # out_nam=f'{sample_dir}{X_data_reversed[samples]}.pdb'
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # add a way to save the sampling name and results
                    # ref: outname = sample_dir+ f"sample-{samples}_condscale-{str (cond_scales[iisample])}_{e}_{steps}.jpg"
                    out_nam=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}.pdb'
                    out_nam_seq=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}.fasta'
                    
                    
                    if CKeys['Debug_TrainerPack']==1:
                        print("pdb_file: ", pdb_file)
                        print("out_nam: ", out_nam)
                        
                    print (f'Original PDB: {pdb_file} OUT: {out_nam}')
                    shutil.copy (pdb_file, out_nam) #source, dest
                    shutil.copy (fasta_file, out_nam_seq)
                    
                   
                    # clean the slade to avoid mistakenly using the previous fasta file
                    os.remove (pdb_file)
                    os.remove (fasta_file)
                    
                    
                    pdb_file=out_nam
                    print (f"Properly named PDB file produced: {pdb_file}")
                    print (f"input X for sampling stored: {pdb_file}")
                    
                    if IF_showfig==1:
                        view=show_pdb(
                            pdb_file=pdb_file, 
                            flag=flag, 
                            show_sidechains=show_sidechains,  
                            show_mainchains=show_mainchains, 
                            color=color
                        )
                        view.show()
# ++
# For ProteinPredictor
def sample_loop_omegafold_pLM_ModelB_Predictor (
    model,
    train_loader,
    cond_scales=[7.5], #list of cond scales - each sampled...
    num_samples=2, #how many samples produced every time tested.....
    timesteps=100, # not used
    flag=0,
    foldproteins=False,
    use_text_embedd=True,
    skip_steps=0,
    # +++++++++++++++++++
    train_unet_number=1,
    ynormfac=1,
    prefix=None,
    tokenizer_y=None,
    Xnormfac=1,
    tokenizer_X=None,
    # ++
    CKeys=None,
    sample_dir=None,
    steps=None,
    e=None,
    IF_showfig=True, # effective only after foldproteins=True
    # ++
    pLM_Model=None,
    pLM_Model_Name=None,
    image_channels=None,
    pLM_alphabet=None,
    # ++
    esm_layer=None,
):
    # =====================================================
    # sample # = num_samples*(# of mini-batches)
    # =====================================================
    # steps=0
    # e=flag
    # for item  in train_loader:
    val_epoch_MSE_list=[]
    resu_pred = {}
    resu_grou = {}
    # 
    for iisample in range (len (cond_scales)):
        # calculate loss for one selected cond_scales
        # 
        val_epoch_MSE=0.
        num_rec=0
        this_prediction = []
        this_groundtruth = []
        # 
        for idx, item  in enumerate(train_loader):

            X_train_batch= item[0].to(device)
            y_train_batch=item[1].to(device)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 1. adjust the number of sample to collect in each batch
            if num_samples>y_train_batch.shape[0]:
                print("Warning: sampling # > len(mini_batch)")
            num_samples = min (num_samples,y_train_batch.shape[0])
            print (f"Producing {num_samples} samples...")
            X_train_batch_picked = X_train_batch[:num_samples,:] # X_train_batch_in[:num_samples ] # 
            GT=y_train_batch.cpu().detach()
            GT_picked = GT[:num_samples,:]
            # GT_picked = GT_picked.unsqueeze(1)
            
            # 2. prepare if pLM is used at the input end: 
            #    this is done inised model.sample fun via x_data_tokenized channel
            #
            # 3. sample inside the loop of cond_scales
            if use_text_embedd:
                result_embedding=model.sample (
                    # x= X_train_batch,
                    x= X_train_batch_picked,
                    stop_at_unet_number=train_unet_number ,
                    cond_scale=cond_scales[iisample], 
                    device=device, 
                    skip_steps=skip_steps,
                    # ++
                    pLM_Model_Name=pLM_Model_Name,
                    pLM_Model=pLM_Model,
                    pLM_alphabet=pLM_alphabet,
                    esm_layer=esm_layer,
                )
            else:
                result_embedding=model.sample (
                    x= None, 
                    # x_data_tokenized= X_train_batch,
                    x_data_tokenized= X_train_batch_picked, # dim=(batch, seq_len), will extend channels inside .sample(),
                    stop_at_unet_number=train_unet_number ,
                    cond_scale=cond_scales[iisample],
                    device=device,
                    skip_steps=skip_steps,
                    # ++
                    pLM_Model_Name=pLM_Model_Name,
                    pLM_Model=pLM_Model,
                    pLM_alphabet=pLM_alphabet,
                    esm_layer=esm_layer,
                )
            # result_embedding as image.dim: [batch, channels, seq_len]
            #
            # 4. translate prediction into something meaningful
            #    consider channel average and masking
            # average across channels
            result_embedding=torch.mean(result_embedding, 1) # (batch, seq_len)
            # read mask from input: X_train_batch_picked (batch, seq_len)
            # result_mask looks like, 0,1,1,...,1,0,0
            # will fill 0th component be zero
            result_mask = read_mask_from_input(
                tokenized_data=X_train_batch[:num_samples], 
                mask_value=0.0,
                seq_data=None,
                max_seq_length=None,
            )
            # apply mask to result: keep true and zero all false
            result = result_embedding*result_mask # (batch, seq_len)
            result = result.cpu()
            # result = result.unsqueeze(1) # (batch, 1, seq_len)
            # this is ONLY the result from the model, not predictio yet
            # 
            # 5. calculate loss
            with torch.no_grad():
                val_loss_MSE = criterion_MSE_sum(
                    result,
                    GT_picked,
                )
            val_epoch_MSE += val_loss_MSE.item()/GT_picked.shape[1]
            num_rec += len(GT_picked)
            # 
            # 6. convert into prediction
            y_data_reversed = result*ynormfac
            # prepare GT
            GT_y_data_reversed = GT_picked*ynormfac
            # accumulate the results
            for ibat in range (GT_picked.shape[0]):
                this_prediction.append (np.array( y_data_reversed[ibat,:].cpu() ))
                this_groundtruth.append (np.array( GT_y_data_reversed[ibat,:].cpu() ))
            
            # 
            # 5. reverse input to AA sequence... if needed
            # TBA
        
        # for one scal_cond
        # summarize the loss
        TestSet_MSE = val_epoch_MSE/num_rec
        resu_pred[str(cond_scales[iisample])] = this_prediction
        resu_grou[str(cond_scales[iisample])] = this_groundtruth
        
    # store the MSE along cond_scales
    val_epoch_MSE_list.append(TestSet_MSE)
    
    return val_epoch_MSE_list, resu_pred, resu_grou
        

                        
# ++++++++++++++++++++++++++++++++++++++++++++++++
def sample_loop_omegafold_ModelB (
    model,
    train_loader,
    cond_scales=[7.5], #list of cond scales - each sampled...
    num_samples=2, #how many samples produced every time tested.....
    timesteps=100, # not used
    flag=0,
    foldproteins=False,
    use_text_embedd=True,
    skip_steps=0,
    # +++++++++++++++++++
    train_unet_number=1,
    ynormfac=1,
    prefix=None,
    tokenizer_y=None,
    Xnormfac=1,
    tokenizer_X=None,
    # ++
    CKeys=None,
    sample_dir=None,
    steps=None,
    e=None,
    IF_showfig=True, # effective only after foldproteins=True
):
    # =====================================================
    # sample # = num_samples*(# of mini-batches)
    # =====================================================
    # steps=0
    # e=flag
    # for item  in train_loader:
    for idx, item  in enumerate(train_loader):

        X_train_batch= item[0].to(device)
        y_train_batch=item[1].to(device)

        GT=y_train_batch.cpu().detach() 

        GT= GT.unsqueeze(1)
        if num_samples>y_train_batch.shape[0]:
            print("Warning: sampling # > len(mini_batch)")

        num_samples = min (num_samples,y_train_batch.shape[0] )
        print (f"Producing {num_samples} samples...")
        X_train_batch_picked = X_train_batch[:num_samples,:]
        print ('(TEST) X_batch shape: ', X_train_batch_picked.shape)

        for iisample in range (len (cond_scales)):

            if use_text_embedd:
                result=model.sample (
                    # x= X_train_batch,
                    x= X_train_batch_picked,
                    stop_at_unet_number=train_unet_number ,
                    cond_scale=cond_scales[iisample], 
                    device=device, 
                    skip_steps=skip_steps
                )
            else:
                result=model.sample (
                    x= None, 
                    # x_data_tokenized= X_train_batch,
                    x_data_tokenized= X_train_batch_picked,
                    stop_at_unet_number=train_unet_number ,
                    cond_scale=cond_scales[iisample],
                    device=device,
                    skip_steps=skip_steps
                )
        
            result=torch.round(result*ynormfac)
            GT=torch.round (GT*ynormfac)

            for samples in range  (num_samples):
                print ("sample ", samples+1, "out of ", num_samples)

                fig=plt.figure()
                plt.plot (
                    result[samples,0,:].cpu().detach().numpy(),
                    label= f'Predicted'
                )
                plt.plot (
                    GT[samples,0,:],
                    label= f'GT {0}'
                )
                plt.legend()
                outname = sample_dir+ f"Batch_{idx}_sample_{samples}_condscale-{str (cond_scales[iisample])}_{e}_{steps}.jpg"
                if IF_showfig==1:
                    plt.show()
                else:
                    plt.savefig(outname, dpi=200)
                plt.close ()

                #reverse y sequence
                to_rev=result[:,0,:]
                to_rev=to_rev.long().cpu().detach().numpy()

                y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

                for iii in range (len(y_data_reversed)):
                    y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")

                #reverse GT_y sequence
                to_rev=GT[:,0,:]
                to_rev=to_rev.long().cpu().detach().numpy()

                GT_y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

                for iii in range (len(y_data_reversed)):
                    GT_y_data_reversed[iii]=GT_y_data_reversed[iii].upper().strip().replace(" ", "")

                ### reverse second structure input....
                to_rev=torch.round (X_train_batch[:,:]*Xnormfac)
                to_rev=to_rev.long().cpu().detach().numpy()
                
                # ++ different input
                if tokenizer_X!=None:
                    X_data_reversed=tokenizer_X.sequences_to_texts (to_rev)
                    for iii in range (len(y_data_reversed)):
                        X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")
                else:
                    X_data_reversed=to_rev.copy()
                # + for debug
                if CKeys['Debug_TrainerPack']==1:
                    print("X_data_reversed: ", X_data_reversed)
                

                print (f"For {X_train_batch[samples,:].cpu().detach().numpy()} or {X_data_reversed[samples]}, \npredicted sequence: ", y_data_reversed[samples])
                print (f"Ground truth: {GT_y_data_reversed[samples]}")

                if foldproteins:
                    xbc=X_train_batch[samples,:].cpu().detach().numpy()
                    out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})
                    tempname='temp'
                    pdb_file=foldandsavePDB (
                        sequence=y_data_reversed[samples], 
                        filename_out=tempname, 
                        num_cycle=16, flag=flag,
                        # +++++++++++++++++++
                        prefix=prefix
                    )

                    # #out_nam=f'{prefix}{out_nam}.pdb'
                    # out_nam=f'{prefix}{X_data_reversed[samples]}.pdb'
                    # ------------------------------------------------------
                    # sometime, this name below can get too long to fit
                    # out_nam=f'{sample_dir}{X_data_reversed[samples]}.pdb'
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # add a way to save the sampling name and results
                    # ref: outname = sample_dir+ f"sample-{samples}_condscale-{str (cond_scales[iisample])}_{e}_{steps}.jpg"
                    out_nam=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}.pdb'
                    out_nam_inX=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}.txt'
                    
                    if CKeys['Debug_TrainerPack']==1:
                        print("pdb_file: ", pdb_file)
                        print("out_nam: ", out_nam)
                        
                    print (f'Original PDB: {pdb_file} OUT: {out_nam}')
                    shutil.copy (pdb_file, out_nam) #source, dest
                    # +
                    with open(out_nam_inX, "w") as inX_file:
                        inX_file.write(f'{X_data_reversed[samples]}\n')
                        
                    pdb_file=out_nam
                    print (f"Properly named PDB file produced: {pdb_file}")
                    print (f"input X for sampling stored: {pdb_file}")
                    
                    if IF_showfig==1:
                        view=show_pdb(
                            pdb_file=pdb_file, 
                            flag=flag, 
                            show_sidechains=show_sidechains,  
                            show_mainchains=show_mainchains, 
                            color=color
                        )
                        view.show()

#                 steps=steps+1
                
#         if steps>num_samples:
#             break

# 
#
def sample_loop_FromModelB (model,
                train_loader,
                cond_scales=[7.5], #list of cond scales - each sampled...
                num_samples=2, #how many samples produced every time tested.....
                timesteps=100,
                 flag=0,foldproteins=False,
                 use_text_embedd=True,skip_steps=0,
                 # +++++++++++++++++++
                 train_unet_number=1,
                 ynormfac=1,
                 prefix=None,
                 tokenizer_y=None,
                 Xnormfac=1,
                 tokenizer_X=None,
                 
               ):
    steps=0
    e=flag
    for item  in train_loader:

            X_train_batch= item[0].to(device)
            y_train_batch=item[1].to(device)

            GT=y_train_batch.cpu().detach() 
                    
            GT= GT.unsqueeze(1)
            num_samples = min (num_samples,y_train_batch.shape[0] )
            print (f"Producing {num_samples} samples...")
            
            print ('X_train_batch shape: ', X_train_batch.shape)

            for iisample in range (len (cond_scales)):
                
                if use_text_embedd:
                    result=model.sample (x= X_train_batch,stop_at_unet_number=train_unet_number ,
                                         cond_scale=cond_scales[iisample], device=device, skip_steps=skip_steps)
                else:
                    result=model.sample (x= None, x_data_tokenized= X_train_batch,
                                         stop_at_unet_number=train_unet_number ,
                                         cond_scale=cond_scales[iisample],device=device,skip_steps=skip_steps)
                    
                result=torch.round(result*ynormfac)
                GT=torch.round (GT*ynormfac)

                for samples in range  (num_samples):
                    print ("sample ", samples, "out of ", num_samples)
                    
                    plt.plot (result[samples,0,:].cpu().detach().numpy(),label= f'Predicted')
                    plt.plot (GT[samples,0,:],label= f'GT {0}')
                    plt.legend()

                    outname = prefix+ f"sample-{samples}_condscale-{str (cond_scales[iisample])}_{e}_{steps}.jpg"
                   
                    plt.savefig(outname, dpi=200)
                    plt.show ()
                    
                    #reverse y sequence
                    to_rev=result[:,0,:]
                    to_rev=to_rev.long().cpu().detach().numpy()
                    
                    y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

                    for iii in range (len(y_data_reversed)):
                        y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
                        
                    #reverse GT_y sequence
                    to_rev=GT[:,0,:]
                    to_rev=to_rev.long().cpu().detach().numpy()
                    
                    GT_y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

                    for iii in range (len(y_data_reversed)):
                        GT_y_data_reversed[iii]=GT_y_data_reversed[iii].upper().strip().replace(" ", "")
                    
                    ### reverse second structure input....
                    to_rev=torch.round (X_train_batch[:,:]*Xnormfac)
                    to_rev=to_rev.long().cpu().detach().numpy()
                   
                    X_data_reversed=tokenizer_X.sequences_to_texts (to_rev)

                    for iii in range (len(y_data_reversed)):
                        X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")

                    print (f"For {X_train_batch[samples,:].cpu().detach().numpy()} or {X_data_reversed[samples]}, predicted sequence: ", y_data_reversed[samples])
                    print (f"Ground truth: {GT_y_data_reversed[samples]}")
                   
                    if foldproteins:
                        xbc=X_train_batch[samples,:].cpu().detach().numpy()
                        out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})
                        tempname='temp'
                        pdb_file=foldandsavePDB (
                            sequence=y_data_reversed[samples], 
                            filename_out=tempname, 
                            num_cycle=16, flag=flag,
                            # +++++++++++++++++++
                            prefix=prefix
                        )
                        
                        #out_nam=f'{prefix}{out_nam}.pdb'
                        out_nam=f'{prefix}{X_data_reversed[samples]}.pdb'
                        print (f'Original PDB: {pdb_file} OUT: {out_nam}')
                        shutil.copy (pdb_file, out_nam) #source, dest
                        pdb_file=out_nam
                        print (f"Properly named PDB file produced: {pdb_file}")
                        
                        view=show_pdb(pdb_file=pdb_file, flag=flag, show_sidechains=show_sidechains,  show_mainchains=show_mainchains, color=color)
                        view.show()

                    steps=steps+1
            if steps>num_samples:
                break
        
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# train_loop tasks:
# 1. calculate loss for one batch
# 2. call sample loop
# 3. call sample sequence
# 4. print records and save model
# ===============================================================
# for ProteinDesigner_B
# 1. expanded for Probability case
# ++
cal_norm_prob = nn.Softmax(dim=2)

def train_loop_Model_B (
    model,
    train_loader,
    test_loader,
    #
    optimizer=None,
    print_every=10,
    epochs= 300,
    start_ep=0,
    start_step=0,
    train_unet_number=1,
    print_loss_every_steps=1000,
    #
    trainer=None,
    plot_unscaled=False,
    max_batch_size=4,
    save_model=False,
    cond_scales=[7.5], #list of cond scales - each sampled...
    num_samples=2, #how many samples produced every time tested.....
    foldproteins=False,
    cond_image=False, #use cond_images...
    # add some
    # +++++++++++++++++++++++++++
    device=None,
    loss_list=[],
    epoch_list=[],
    train_hist_file=None,
    train_hist_file_full=None,
    prefix=None,
    Xnormfac=1.,
    ynormfac=1.,
    tokenizer_X=None,
    tokenizer_y=None,
    test_condition_list=[],
    max_length=1,
    CKeys=None,
    sample_steps=1,
    sample_dir=None,
    save_every_epoch=1,
    save_point_info_file=None,
    store_dir=None,
    # ++
    pLM_Model_Name=None,
    image_channels=None,
    print_error=False,
):
    
    if not exists (trainer):
        if not exists (optimizer):
            print ("ERROR: If trainer not used, need to provide optimizer.")
    if exists (trainer):
        print ("Trainer provided... will be used")
        
    # steps=start_step+1
    # # +
    # added_steps=0+1
    #
    steps=start_step
    added_steps=0
    
    loss_total=0
    
    # ++ for pLM
    if pLM_Model_Name=='None':
        pLM_Model=None
        
    elif pLM_Model_Name=='esm2_t33_650M_UR50D':
        # dim: 1280
        esm_layer=33
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)
        
    elif pLM_Model_Name=='esm2_t36_3B_UR50D':
        # dim: 2560
        esm_layer=36
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)
        
    elif pLM_Model_Name=='esm2_t30_150M_UR50D':
        # dim: 640
        esm_layer=30
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)
    
    elif pLM_Model_Name=='esm2_t12_35M_UR50D':
        # dim: 480
        esm_layer=12
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)
        
    else:
        print("pLM model is missing...")

        
    for e in range(1, epochs+1):
            
        # start = time.time()

        torch.cuda.empty_cache()
        print ("######################################################################################")
        start = time.time()
        print ("NOW: Training epoch: ", e+start_ep)

        # TRAINING
        train_epoch_loss = 0
        model.train()

        print ("Loop over ", len(train_loader), " batches (print . every ", print_every, " steps)")

        for item  in train_loader:
            steps += 1
            added_steps += 1

            X_train_batch= item[0].to(device)
            y_train_batch= item[1].to(device)
            # project y_ into embedding space
            if CKeys["Debug_TrainerPack"]==1:
                print("Initial unload the dataloader items: ...")
                print("X_train_batch.dim: ", X_train_batch.shape)
                print("y_train_batch.dim: ", y_train_batch.shape)
                
            # ---------------------------------------------------------
            # prepare for model.forward() to calculate loss
            # ---------------------------------------------------------
            # # --
            # if pLM_Model_Name=='None':
            #     # just use the encoded sequence
            #     y_train_batch_in = y_train_batch.unsqueeze(1)
            #     X_train_batch_in = X_train_batch.unsqueeze(1)
            #     # pass
            # elif pLM_Model_Name=='esm2_t33_650M_UR50D':
            #     with torch.no_grad():
            #         results = pLM_Model(
            #             y_train_batch,
            #             repr_layers=[33],
            #             return_contacts=False,
            #         )
            #     y_train_batch_in = results["representations"][33] # (batch, seq_len, channels)
            #     y_train_batch_in = rearrange(
            #         y_train_batch_in, 
            #         'b l c -> b c l'
            #     )
            #     X_train_batch_in = X_train_batch.unsqueeze(1).repeat(1,image_channels,1)
            # else:
            #     print(f"Required pLM name is not defined!!")
            # ++
            if pLM_Model_Name=='None':
                # just use the encoded sequence
                y_train_batch_in = y_train_batch.unsqueeze(1)
                X_train_batch_in = X_train_batch.unsqueeze(1)
                # pass
            else: # assume ESM models
                with torch.no_grad():
                    results = pLM_Model(
                        y_train_batch,
                        repr_layers=[esm_layer],
                        return_contacts=False,
                    )
                    y_train_batch_in = results["representations"][esm_layer] # (batch, seq_len, channels)
                # ++ for Probability case
                if image_channels==33:
                    with torch.no_grad():
                        # calculate logits: (batch, seq_len, 33)
                        y_train_batch_in = pLM_Model.lm_head(
                            y_train_batch_in
                        )
                        # normalize to get (0,1) probability
                        y_train_batch_in = cal_norm_prob(y_train_batch_in)
                
                # switch the dimension -> (batch, channel, seq_len)
                y_train_batch_in = rearrange(
                    y_train_batch_in, 
                    'b l c -> b c l'
                )
                
                    
                X_train_batch_in = X_train_batch.unsqueeze(1).repeat(1,image_channels,1)

                
            # + for debug
            if CKeys["Debug_TrainerPack"]==1:
                print("After pLM model, the shape of X and y for training:")
                print("X_train_batch_in.dim: ", X_train_batch_in.shape)
                print("y_train_batch_in.dim: ", y_train_batch_in.shape)
                
                
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if exists (trainer):
                
                if cond_image==False:
                    loss = trainer(
                        y_train_batch.unsqueeze(1) , # true image (batch, channels, seq_len)
                        x=X_train_batch,             # tokenized text (batch, )
                        unet_number=train_unet_number,
                        max_batch_size = max_batch_size,    # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                    )
                # # ----------------------------------------------------------
                # if cond_image==True:
                #     loss = trainer(
                #         y_train_batch.unsqueeze(1) ,            # true image
                #         x=None,                                 # tokenized text
                #         cond_images=X_train_batch.unsqueeze(1), # cond_image
                #         unet_number=train_unet_number,
                #         max_batch_size = max_batch_size, # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                #         )
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if cond_image==True:
                    loss = trainer(
                        y_train_batch_in,                          # true image
                        x=None,                                 # tokenized text
                        cond_images=X_train_batch_in,              # cond_image
                        unet_number=train_unet_number,
                        max_batch_size = max_batch_size, # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                        )
                    
                trainer.update(unet_number = train_unet_number)

            else:
                optimizer.zero_grad()
                if cond_image==False:
                    loss=model (
                        y_train_batch.unsqueeze(1) , 
                        x=X_train_batch, 
                        unet_number=train_unet_number
                    )
                # # ------------------------------------------------------
                # if cond_image==True:
                #     loss=model (
                #         y_train_batch.unsqueeze(1) ,
                #         x=None, 
                #         cond_images=X_train_batch.unsqueeze(1), 
                #         unet_number=train_unet_number
                #     )
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if cond_image==True:
                    loss=model (
                        y_train_batch_in ,
                        x=None, 
                        cond_images=X_train_batch_in, 
                        unet_number=train_unet_number
                    )
                #
                loss.backward( )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            loss_total=loss_total+loss.item()
            # +
            train_epoch_loss=train_epoch_loss+loss.item()

            if steps % print_every == 0:
                # for progress bar
                print(".", end="")

            # if steps>0:
            if added_steps>0:
                
                if steps % print_loss_every_steps == 0:
                    # + for debug
                    if CKeys['Debug_TrainerPack']==2:
                        print('I am here')
                        print("Here is steps: ", steps)
                    
                    norm_loss=loss_total/print_loss_every_steps
                    print (f"\nTOTAL LOSS at epoch={e+start_ep}, step={steps}: {norm_loss}")
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # add a line to the hist file
                    add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss)+'\n'
                    with open(train_hist_file,'a') as f:
                        f.write(add_line)


                    loss_list.append (norm_loss)
                    loss_total=0
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    epoch_list.append(e+start_ep)
                    
                    # fig = plt.figure(figsize=(12,8),dpi=200)
                    fig = plt.figure()
                    plt.plot (epoch_list, loss_list, label='Loss')
                    plt.legend()

                    # outname = prefix+ f"loss_{e+start_ep}_{steps}.jpg"
                    outname = sample_dir+ f"loss_{e+start_ep}_{steps}.jpg"
                    # 
                    # the order, save then show, matters
                    if CKeys['SlientRun']==1:
                        plt.savefig(outname, dpi=200)
                    else:
                        plt.show()
                    plt.close(fig)
                    # plt.close()
                    
            if added_steps>0:
                # if steps>0:
                if steps % sample_steps == 0:
                    # + for debug
                    if CKeys['Debug_TrainerPack']==2:
                        print('I am here')
                        print("Here is steps: ", steps)
                    
                    if plot_unscaled:
                        #test before scaling...
                        plt.plot (
                            y_train_batch.unsqueeze(1)[0,0,:].cpu().detach().numpy(),
                            label= 'Unscaled GT'
                        )
                        plt.legend()
                        plt.show()

#                     # --------------------------------------------------
#                     GT=y_train_batch.cpu().detach() 

#                     GT=resize_image_to(
#                         GT.unsqueeze(1),
#                         model.imagen.image_sizes[train_unet_number-1],
#                     )

                    
                    ####
                    print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    print ("I. SAMPLING IN TEST SET: ")
                    print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    ####
                    # num_samples = min (num_samples,y_train_batch.shape[0] )
                    print (f"Producing {num_samples} samples...")

                    
                    if cond_image == True:
                        use_text_embedd=False
                        # -
                        # cond_scales_extended=[1. for i in range(num_samples)]
                        # +
                        cond_scales_extended=cond_scales
                    else:
                        use_text_embedd=True
                        cond_scales_extended=cond_scales

                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    sample_loop_omegafold_pLM_ModelB (
                        model,
                        test_loader,
                        cond_scales=cond_scales_extended, # cond_scales,# #list of cond scales - each sampled...
                        num_samples=num_samples, #how many samples produced every time tested.....
                        timesteps=64,
                        flag=steps,
                        #reverse=False,
                        foldproteins=foldproteins,
                        use_text_embedd= use_text_embedd,
                        # ++++++++++++++++++++
                        train_unet_number=train_unet_number,
                        ynormfac=ynormfac,
                        prefix=prefix,
                        tokenizer_y=tokenizer_y,
                        Xnormfac=Xnormfac,
                        tokenizer_X=tokenizer_X,
                        # ++
                        # ++
                        CKeys=CKeys,
                        sample_dir=sample_dir,
                        steps=steps,
                        e=e+start_ep,
                        IF_showfig= CKeys['SlientRun']!=1,
                        # ++
                        pLM_Model=pLM_Model,
                        pLM_Model_Name=pLM_Model_Name,
                        image_channels=image_channels,
                        pLM_alphabet=esm_alphabet,
                    )   
                    
                    
                    #---------------------------------------------------------
                    # sample_loop (
                    #     model,
                    #     test_loader,
                    #     cond_scales=cond_scales,# #list of cond scales - each sampled...
                    #     num_samples=num_samples, #how many samples produced every time tested.....
                    #     timesteps=64,
                    #     flag=steps,
                    #     #reverse=False,
                    #     foldproteins=foldproteins,
                    #     use_text_embedd= use_text_embedd,
                    #     # ++++++++++++++++++++
                    #     train_unet_number=train_unet_number,
                    #     ynormfac=ynormfac,
                    #     prefix=prefix,
                    #     tokenizer_y=tokenizer_y,
                    #     Xnormfac=Xnormfac,
                    #     tokenizer_X=tokenizer_X,
                    # )   

                    #index_word': '{"1": "~", "2": "h", "3": "e", "4": "s", "5": "t", "6": "g", "7": "b", "8": "i"}', 
                    #'word_index': '{"~": 1, "h": 2, "e": 3, "s": 4, "t": 5, "g": 6, "b": 7, "i": 8}'}

                    AH_code=2/Xnormfac
                    BS_code=3/Xnormfac
                    unstr_code= 1/Xnormfac

                    print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    print ("II. SAMPLING FOR DE NOVO:")
                    print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    
                    # +++++++++++++++++++++++++++++++++++++++++
                    DeNovoSam_pdbs, fasta_file_list=\
                    sample_sequence_omegafold_pLM_ModelB (
                        model,
                        x_data=test_condition_list,
                        flag=steps, # flag="DeNovo", # ,
                        cond_scales=1.,
                        foldproteins=foldproteins,
                        # ++++++++++
                        ynormfac=ynormfac,
                        train_unet_number=train_unet_number,
                        tokenizer_X=tokenizer_X,
                        Xnormfac=Xnormfac,
                        max_length=max_length,
                        prefix=prefix,
                        tokenizer_y=tokenizer_y,
                        # ++
                        CKeys=CKeys,
                        sample_dir=sample_dir,
                        steps=steps,
                        e=e+start_ep,
                        IF_showfig= CKeys['SlientRun']!=1,
                        # ++
                        pLM_Model=pLM_Model,
                        pLM_Model_Name=pLM_Model_Name,
                        image_channels=image_channels,
                        pLM_alphabet=esm_alphabet,
                       )
                    
                    if print_error and len(DeNovoSam_pdbs)>0:
                        print("Calculate SecStr and design error:")
                        #
                        for ii in range(len(test_condition_list)):
                            seq=test_condition_list[ii][0]
                            DSSPresult,_,sequence_res=get_DSSP_result(DeNovoSam_pdbs[ii]) 
                            print (f"INPUT:        {seq}\nRESULT:       {DSSPresult}\nAA sequence:  {sequence_res}")
                            error=string_diff (DSSPresult, seq)/len (seq)
                            print ("Error: ", error)
                        
                    
                    
                    # # +++++++++++++++++++++++++++++++++++++++++
                    # sample_sequence_omegafold_ModelB (
                    #     model,
                    #     x_data=test_condition_list,
                    #     flag=steps, # flag="DeNovo", # ,
                    #     cond_scales=1.,
                    #     foldproteins=foldproteins,
                    #     # ++++++++++
                    #     ynormfac=ynormfac,
                    #     train_unet_number=train_unet_number,
                    #     tokenizer_X=tokenizer_X,
                    #     Xnormfac=Xnormfac,
                    #     max_length=max_length,
                    #     prefix=prefix,
                    #     tokenizer_y=tokenizer_y,
                    #     # ++
                    #     CKeys=CKeys,
                    #     sample_dir=sample_dir,
                    #     steps=steps,
                    #     e=e+start_ep,
                    #     IF_showfig= CKeys['SlientRun']!=1,
                    #    )
                        
                    # for this_x_data in test_condition_list:
                    #     sample_sequence_omegafold (
                    #         model,
                    #         x_data=this_x_data,
                    #         flag=steps, # flag="DeNovo", # ,
                    #         cond_scales=1.,
                    #         foldproteins=True,
                    #         # ++++++++++
                    #         ynormfac=ynormfac,
                    #         train_unet_number=train_unet_number,
                    #         tokenizer_X=tokenizer_X,
                    #         Xnormfac=Xnormfac,
                    #         max_length=max_length,
                    #         prefix=prefix,
                    #         tokenizer_y=tokenizer_y,
                    #         # ++
                    #         CKeys=CKeys,
                    #         sample_dir=sample_dir,
                    #         steps=steps,
                    #         e=e+start_ep,
                    #         IF_showfig= CKeys['SlientRun']!=1,
                    #        )
                    #
    # model,
    # X=None, #this is the target conventionally when using text embd
    # flag=0,
    # cond_scales=1.,
    # foldproteins=False,
    # X_string=None,
    # x_data=None,  
    # skip_steps=0,
    # inpaint_images = None,
    # inpaint_masks = None,
    # inpaint_resample_times = None,
    # init_images = None,
    # num_cycle=16,
    # # ++++++++++++++++++++++++
    # ynormfac=1,
    # train_unet_number=1,
    # tokenizer_X=None,
    # Xnormfac=1.,
    # max_length=1.,
    # prefix=None,
    # tokenizer_y=None,
    # # ++
    # CKeys=None,
    # sample_dir=None,
                    
                    # -----------------------------------------    
                    # sample_sequence (
                    #     model,
                    #     x_data=['~~~HHHHHHHHHHHHHHH~~'],
                    #     flag=steps,cond_scales=1.,
                    #     foldproteins=True,
                    #     # ++++++++++
                    #     ynormfac=ynormfac,
                    #    )
                    # sample_sequence (
                    #     model,
                    #     x_data=['~~~HHHHHHHHHHHHHHH~~~~HHHHHHHHHHHHHH~~~'],
                    #     flag=steps,cond_scales=1.,
                    #     foldproteins=True,
                    #     # ++++++++++
                    #     ynormfac=ynormfac,
                    #    )
                    # sample_sequence (
                    #     model,
                    #     x_data=['~~EEESSTTS~SEEEEEEEEE~SBS~EEEEEE~~'],
                    #     flag=steps,cond_scales=1.,
                    #     foldproteins=True,
                    #     # ++++++++++++
                    #     ynormfac=ynormfac,
                    #    )

            # if steps>0:
            # # --------------------------------------------------------------------
            # if added_steps>0:
            #     if save_model and steps % print_loss_every_steps==0:
            #         fname=f"{prefix}trainer_save-model-epoch_{e+start_ep}.pt"
            #         trainer.save(fname)
            #         print (f"Model saved: ", fname)
            #         fname=f"{prefix}statedict_save-model-epoch_{e+start_ep}.pt"
            #         torch.save(model.state_dict(), fname)
            #         print (f"Statedict model saved: ", fname)
            
            # steps=steps+1
            # added_steps += 1
            
        # every epoch:
        norm_loss_over_e = train_epoch_loss/len(train_loader)
        print("\nnorm_loss over 1 epoch: ", norm_loss_over_e)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # write this into "train_hist_file_full"
        add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss_over_e)+'\n'
        with open(train_hist_file_full,'a') as f:
            f.write(add_line)
            
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # save model every this epoches
        if save_model and (e+start_ep) % save_every_epoch==0 and e>1:
            # fname=f"{prefix}trainer_save-model-epoch_{e+start_ep}.pt"
            fname=f"{store_dir}trainer_save-model-epoch_{e+start_ep}.pt"
            trainer.save(fname)
            print (f"Model saved: ", fname)
            # fname=f"{prefix}statedict_save-model-epoch_{e+start_ep}.pt"
            fname=f"{store_dir}statedict_save-model-epoch_{e+start_ep}.pt"
            torch.save(model.state_dict(), fname)
            print (f"Statedict model saved: ", fname)
            # add a saving point file
            top_line='epoch,steps,norm_loss'+'\n'
            add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss)+'\n'
            with open(save_point_info_file, "w") as f:
                f.write(top_line)
                f.write(add_line)
        


        print (f"\n\n-------------------\nTime for epoch {e+start_ep}={(time.time()-start)/60}\n-------------------")
        
# ===============================================================
# for ProteinPredictor_B
#
def train_loop_Model_B_Predictor (
    model,
    train_loader,
    test_loader,
    #
    optimizer=None,
    print_every=10,
    epochs= 300,
    start_ep=0,
    start_step=0,
    train_unet_number=1,
    print_loss_every_steps=1000,
    #
    trainer=None,
    plot_unscaled=False,
    max_batch_size=4,
    save_model=False,
    cond_scales=[1.], #list of cond scales - each sampled...
    num_samples=2, #how many samples produced every time tested.....
    foldproteins=False,
    cond_image=False, #use cond_images...
    # add some
    # +++++++++++++++++++++++++++
    device=None,
    loss_list=[],
    epoch_list=[],
    train_hist_file=None,
    train_hist_file_full=None,
    prefix=None,
    Xnormfac=1.,
    ynormfac=1.,
    tokenizer_X=None,
    tokenizer_y=None,
    test_condition_list=[],
    max_length=1,
    CKeys=None,
    sample_steps=1,
    sample_dir=None,
    save_every_epoch=1,
    save_point_info_file=None,
    store_dir=None,
    # ++
    pLM_Model_Name=None,
    image_channels=None,
    print_error=False,
    # ++
    train_hist_file_on_testset=None,
):
    
    if not exists (trainer):
        if not exists (optimizer):
            print ("ERROR: If trainer not used, need to provide optimizer.")
    if exists (trainer):
        print ("Trainer provided... will be used")
        
    # steps=start_step+1
    # # +
    # added_steps=0+1
    #
    steps=start_step
    added_steps=0
    
    loss_total=0
    
    # ++ for pLM
#     # --
#     if pLM_Model_Name=='trivial':
#         pLM_Model=None
        
#     elif pLM_Model_Name=='esm2_t33_650M_UR50D':
#         # dim: 1280
#         esm_layer=33
#         pLM_Model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#         len_toks=len(esm_alphabet.all_toks)
#         pLM_Model.eval()
#         pLM_Model. to(device)
        
#     elif pLM_Model_Name=='esm2_t36_3B_UR50D':
#         # dim: 2560
#         esm_layer=36
#         pLM_Model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
#         len_toks=len(esm_alphabet.all_toks)
#         pLM_Model.eval()
#         pLM_Model. to(device)
        
#     elif pLM_Model_Name=='esm2_t30_150M_UR50D':
#         # dim: 640
#         esm_layer=30
#         pLM_Model, esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
#         len_toks=len(esm_alphabet.all_toks)
#         pLM_Model.eval()
#         pLM_Model. to(device)
    
#     elif pLM_Model_Name=='esm2_t12_35M_UR50D':
#         # dim: 480
#         esm_layer=12
#         pLM_Model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
#         len_toks=len(esm_alphabet.all_toks)
#         pLM_Model.eval()
#         pLM_Model. to(device)
        
#     else:
#         print("pLM model is missing...")
    # ++
    pLM_Model, esm_alphabet, \
    esm_layer, len_toks = load_in_pLM(
        pLM_Model_Name,
        device,
    )

        
    for e in range(1, epochs+1):
            
        # start = time.time()

        torch.cuda.empty_cache()
        print ("######################################################################################")
        start = time.time()
        print ("NOW: Training epoch: ", e+start_ep)

        # TRAINING
        train_epoch_loss = 0
        model.train()

        print ("Loop over ", len(train_loader), " batches (print . every ", print_every, " steps)")

        for item  in train_loader:
            steps += 1
            added_steps += 1

            X_train_batch= item[0].to(device)
            y_train_batch= item[1].to(device)
            # project y_ into embedding space
            if CKeys["Debug_TrainerPack"]==1:
                print("Initial unload the dataloader items: ...")
                print("X_train_batch.dim: ", X_train_batch.shape)
                print("y_train_batch.dim: ", y_train_batch.shape)
                
            # ---------------------------------------------------------
            # prepare for model.forward() to calculate loss
            # ---------------------------------------------------------
            # # --
            # if pLM_Model_Name=='None':
            #     # just use the encoded sequence
            #     y_train_batch_in = y_train_batch.unsqueeze(1)
            #     X_train_batch_in = X_train_batch.unsqueeze(1)
            #     # pass
            # elif pLM_Model_Name=='esm2_t33_650M_UR50D':
            #     with torch.no_grad():
            #         results = pLM_Model(
            #             y_train_batch,
            #             repr_layers=[33],
            #             return_contacts=False,
            #         )
            #     y_train_batch_in = results["representations"][33] # (batch, seq_len, channels)
            #     y_train_batch_in = rearrange(
            #         y_train_batch_in, 
            #         'b l c -> b c l'
            #     )
            #     X_train_batch_in = X_train_batch.unsqueeze(1).repeat(1,image_channels,1)
            # else:
            #     print(f"Required pLM name is not defined!!")
            # ++
            if pLM_Model_Name=='trivial':
                # just use the encoded sequence
                y_train_batch_in = y_train_batch.unsqueeze(1)
                X_train_batch_in = X_train_batch.unsqueeze(1)
                # pass
            else: 
                # assume ESM models
                # --
                # # for ProteinDesigner
                # with torch.no_grad():
                #     results = pLM_Model(
                #         y_train_batch,
                #         repr_layers=[esm_layer],
                #         return_contacts=False,
                #     )
                # y_train_batch_in = results["representations"][esm_layer] # (batch, seq_len, channels)
                # y_train_batch_in = rearrange(
                #     y_train_batch_in, 
                #     'b l c -> b c l'
                # )
                # X_train_batch_in = X_train_batch.unsqueeze(1).repeat(1,image_channels,1)
                #
                # ++
                # for ProteinPredictor
                with torch.no_grad():
                    results = pLM_Model(
                        X_train_batch,
                        repr_layers=[esm_layer],
                        return_contacts=False,
                    )
                X_train_batch_in = results["representations"][esm_layer] # (batch, seq_len, channels)
                X_train_batch_in = rearrange(
                    X_train_batch_in, 
                    'b l c -> b c l'
                )
                y_train_batch_in = y_train_batch.unsqueeze(1).repeat(1,image_channels,1)
                

                
            # + for debug
            if CKeys["Debug_TrainerPack"]==1:
                print("After pLM model, the shape of X and y for training:")
                print("X_train_batch_in.dim: ", X_train_batch_in.shape)
                print("y_train_batch_in.dim: ", y_train_batch_in.shape)
                
                
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if exists (trainer):
                
                if cond_image==False:
                    loss = trainer(
                        y_train_batch.unsqueeze(1) , # true image (batch, channels, seq_len)
                        x=X_train_batch,             # tokenized text (batch, )
                        unet_number=train_unet_number,
                        max_batch_size = max_batch_size,    # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                    )
                # # ----------------------------------------------------------
                # if cond_image==True:
                #     loss = trainer(
                #         y_train_batch.unsqueeze(1) ,            # true image
                #         x=None,                                 # tokenized text
                #         cond_images=X_train_batch.unsqueeze(1), # cond_image
                #         unet_number=train_unet_number,
                #         max_batch_size = max_batch_size, # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                #         )
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if cond_image==True:
                    loss = trainer(
                        y_train_batch_in,                          # true image
                        x=None,                                 # tokenized text
                        cond_images=X_train_batch_in,              # cond_image
                        unet_number=train_unet_number,
                        max_batch_size = max_batch_size, # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                        )
                    
                trainer.update(unet_number = train_unet_number)

            else:
                optimizer.zero_grad()
                if cond_image==False:
                    loss=model (
                        y_train_batch.unsqueeze(1) , 
                        x=X_train_batch, 
                        unet_number=train_unet_number
                    )
                # # ------------------------------------------------------
                # if cond_image==True:
                #     loss=model (
                #         y_train_batch.unsqueeze(1) ,
                #         x=None, 
                #         cond_images=X_train_batch.unsqueeze(1), 
                #         unet_number=train_unet_number
                #     )
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if cond_image==True:
                    loss=model (
                        y_train_batch_in ,
                        x=None, 
                        cond_images=X_train_batch_in, 
                        unet_number=train_unet_number
                    )
                #
                loss.backward( )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            loss_total=loss_total+loss.item()
            # +
            train_epoch_loss=train_epoch_loss+loss.item()

            if steps % print_every == 0:
                # for progress bar
                print(".", end="")

            # if steps>0:
            if added_steps>0:
                
                if steps % print_loss_every_steps == 0:
                    # + for debug
                    if CKeys['Debug_TrainerPack']==2:
                        print('I am here')
                        print("Here is steps: ", steps)
                    
                    norm_loss=loss_total/print_loss_every_steps
                    print (f"\nTOTAL LOSS at epoch={e+start_ep}, step={steps}: {norm_loss}")
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # add a line to the hist file
                    add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss)+'\n'
                    with open(train_hist_file,'a') as f:
                        f.write(add_line)


                    loss_list.append (norm_loss)
                    loss_total=0
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    epoch_list.append(e+start_ep)
                    
                    # fig = plt.figure(figsize=(12,8),dpi=200)
                    fig = plt.figure()
                    plt.plot (epoch_list, loss_list, label='Loss')
                    plt.legend()

                    # outname = prefix+ f"loss_{e+start_ep}_{steps}.jpg"
                    outname = sample_dir+ f"loss_{e+start_ep}_{steps}.jpg"
                    # 
                    # the order, save then show, matters
                    if CKeys['SlientRun']==1:
                        plt.savefig(outname, dpi=200)
                    else:
                        plt.show()
                    plt.close(fig)
                    # plt.close()
                    
            if added_steps>0:
                # if steps>0:
                if steps % sample_steps == 0:
                    # + for debug
                    if CKeys['Debug_TrainerPack']==2:
                        print('I am here')
                        print("Here is steps: ", steps)
                    
                    if plot_unscaled:
                        #test before scaling...
                        plt.plot (
                            y_train_batch.unsqueeze(1)[0,0,:].cpu().detach().numpy(),
                            label= 'Unscaled GT'
                        )
                        plt.legend()
                        plt.show()

#                     # --------------------------------------------------
#                     GT=y_train_batch.cpu().detach() 

#                     GT=resize_image_to(
#                         GT.unsqueeze(1),
#                         model.imagen.image_sizes[train_unet_number-1],
#                     )

                    
                    ####
                    print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    print ("I. SAMPLING IN TEST SET: ")
                    print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    ####
                    # num_samples = min (num_samples,y_train_batch.shape[0] )
                    print (f"Producing {num_samples} samples...")

                    
                    if cond_image == True:
                        use_text_embedd=False
                        # -
                        # cond_scales_extended=[1. for i in range(num_samples)]
                        # +
                        cond_scales_extended=cond_scales
                    else:
                        use_text_embedd=True
                        cond_scales_extended=cond_scales

                    # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # # For ProteinDesigner
                    # sample_loop_omegafold_pLM_ModelB (
                    #     model,
                    #     test_loader,
                    #     cond_scales=cond_scales_extended, # cond_scales,# #list of cond scales - each sampled...
                    #     num_samples=num_samples, #how many samples produced every time tested.....
                    #     timesteps=64,
                    #     flag=steps,
                    #     #reverse=False,
                    #     foldproteins=foldproteins,
                    #     use_text_embedd= use_text_embedd,
                    #     # ++++++++++++++++++++
                    #     train_unet_number=train_unet_number,
                    #     ynormfac=ynormfac,
                    #     prefix=prefix,
                    #     tokenizer_y=tokenizer_y,
                    #     Xnormfac=Xnormfac,
                    #     tokenizer_X=tokenizer_X,
                    #     # ++
                    #     # ++
                    #     CKeys=CKeys,
                    #     sample_dir=sample_dir,
                    #     steps=steps,
                    #     e=e+start_ep,
                    #     IF_showfig= CKeys['SlientRun']!=1,
                    #     # ++
                    #     pLM_Model=pLM_Model,
                    #     pLM_Model_Name=pLM_Model_Name,
                    #     image_channels=image_channels,
                    #     pLM_alphabet=esm_alphabet,
                    # )
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # For ProteinPredictor:
                    val_epoch_MSE_list, \
                    resu_pred, resu_grou = \
                    sample_loop_omegafold_pLM_ModelB_Predictor (
                        model,
                        test_loader,
                        cond_scales=[1.], # cond_scales_extended, # #list of cond scales - each sampled...
                        num_samples=num_samples, #how many samples produced every time tested.....
                        timesteps=64,
                        flag=steps,
                        #reverse=False,
                        foldproteins=foldproteins,
                        use_text_embedd= use_text_embedd,
                        # ++++++++++++++++++++
                        train_unet_number=train_unet_number,
                        ynormfac=ynormfac,
                        prefix=prefix,
                        tokenizer_y=tokenizer_y,
                        Xnormfac=Xnormfac,
                        tokenizer_X=tokenizer_X,
                        # ++
                        # ++
                        CKeys=CKeys,
                        sample_dir=sample_dir,
                        steps=steps,
                        e=e+start_ep,
                        IF_showfig= CKeys['SlientRun']!=1,
                        # ++
                        pLM_Model=pLM_Model,
                        pLM_Model_Name=pLM_Model_Name,
                        image_channels=image_channels,
                        pLM_alphabet=esm_alphabet,
                        # ++
                        esm_layer=esm_layer,
                    )
                    # record the ERROR on the test set
                    print(f"Epo {str(e+start_ep)}, on TestSet, MSE: {val_epoch_MSE_list[0]}")
                    # only write the 0th case of MSE
                    add_line = str(e+start_ep)+','+str(steps)+','+str(val_epoch_MSE_list[0])+'\n'
                    with open(train_hist_file_on_testset,'a') as f:
                        f.write(add_line)


                    print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    print ("II. SAMPLING FOR DE NOVO: NOT USED in predictor mode")
                    print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    
#                     # +++++++++++++++++++++++++++++++++++++++++
#                     DeNovoSam_pdbs, fasta_file_list=\
#                     sample_sequence_omegafold_pLM_ModelB (
#                         model,
#                         x_data=test_condition_list,
#                         flag=steps, # flag="DeNovo", # ,
#                         cond_scales=1.,
#                         foldproteins=foldproteins,
#                         # ++++++++++
#                         ynormfac=ynormfac,
#                         train_unet_number=train_unet_number,
#                         tokenizer_X=tokenizer_X,
#                         Xnormfac=Xnormfac,
#                         max_length=max_length,
#                         prefix=prefix,
#                         tokenizer_y=tokenizer_y,
#                         # ++
#                         CKeys=CKeys,
#                         sample_dir=sample_dir,
#                         steps=steps,
#                         e=e+start_ep,
#                         IF_showfig= CKeys['SlientRun']!=1,
#                         # ++
#                         pLM_Model=pLM_Model,
#                         pLM_Model_Name=pLM_Model_Name,
#                         image_channels=image_channels,
#                         pLM_alphabet=esm_alphabet,
#                        )
                    
#                     if print_error and len(DeNovoSam_pdbs)>0:
#                         print("Calculate SecStr and design error:")
#                         #
#                         for ii in range(len(test_condition_list)):
#                             seq=test_condition_list[ii][0]
#                             DSSPresult,_,sequence_res=get_DSSP_result(DeNovoSam_pdbs[ii]) 
#                             print (f"INPUT:        {seq}\nRESULT:       {DSSPresult}\nAA sequence:  {sequence_res}")
#                             error=string_diff (DSSPresult, seq)/len (seq)
#                             print ("Error: ", error)
                        
                    
            
        # every epoch:
        norm_loss_over_e = train_epoch_loss/len(train_loader)
        print("\nnorm_loss over 1 epoch: ", norm_loss_over_e)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # write this into "train_hist_file_full"
        add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss_over_e)+'\n'
        with open(train_hist_file_full,'a') as f:
            f.write(add_line)
            
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # save model every this epoches
        if save_model and (e+start_ep) % save_every_epoch==0 and e>1:
            # fname=f"{prefix}trainer_save-model-epoch_{e+start_ep}.pt"
            fname=f"{store_dir}trainer_save-model-epoch_{e+start_ep}.pt"
            trainer.save(fname)
            print (f"Model saved: ", fname)
            # fname=f"{prefix}statedict_save-model-epoch_{e+start_ep}.pt"
            fname=f"{store_dir}statedict_save-model-epoch_{e+start_ep}.pt"
            torch.save(model.state_dict(), fname)
            print (f"Statedict model saved: ", fname)
            # add a saving point file
            top_line='epoch,steps,norm_loss'+'\n'
            add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss)+'\n'
            with open(save_point_info_file, "w") as f:
                f.write(top_line)
                f.write(add_line)
        


        print (f"\n\n-------------------\nTime for epoch {e+start_ep}={(time.time()-start)/60}\n-------------------")
            
            
def train_loop_Old_FromModelB (model,
                train_loader,
                test_loader,
                optimizer=None,
                print_every=10,
                epochs= 300,
                start_ep=0,
                start_step=0,
                train_unet_number=1,
                print_loss=1000,
                trainer=None,
                plot_unscaled=False,
                max_batch_size=4,
                save_model=False,
                cond_scales=[7.5], #list of cond scales - each sampled...
                num_samples=2, #how many samples produced every time tested.....
                foldproteins=False,
                cond_image=False, #use cond_images...
                # add some
                # +++++++++++++++++++++++++++
                device=None,
                loss_list=[],
                prefix=None,
                ynormfac=1,
                test_condition_list=[],
                tokenizer_y=None,
                Xnormfac=1,
                tokenizer_X=None,
                max_length=1,
                
               ):
    
    if not exists (trainer):
        if not exists (optimizer):
            print ("ERROR: If trainer not used, need to provide optimizer.")
    if exists (trainer):
        print ("Trainer provided... will be used")
        
    steps=start_step
    
    loss_total=0
    for e in range(1, epochs+1):
        
        start = time.time()

        torch.cuda.empty_cache()
        print ("######################################################################################")
        start = time.time()
        print ("NOW: Training epoch: ", e+start_ep)

        train_epoch_loss = 0
        model.train()

        print ("Loop over ", len(train_loader), " batches (print . every ", print_every, " steps)")

        for item  in train_loader:

            X_train_batch= item[0].to(device)

            y_train_batch=item[1].to(device)

            if exists (trainer):
                if cond_image==False:
                    loss = trainer(
                        y_train_batch.unsqueeze(1) ,
                        x=X_train_batch,  
                        unet_number=train_unet_number,
                        max_batch_size = max_batch_size,    # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                    )
                if cond_image==True:

                    loss = trainer(
                        y_train_batch.unsqueeze(1) ,x=None,
                        cond_images=X_train_batch.unsqueeze(1), 
                        unet_number=train_unet_number,
                        max_batch_size = max_batch_size, # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                        )
                trainer.update(unet_number = train_unet_number)

            else:
                optimizer.zero_grad()
                if cond_image==False:
                    loss=model (y_train_batch.unsqueeze(1) , x=X_train_batch, unet_number=train_unet_number)
                if cond_image==True:
                    loss=model (y_train_batch.unsqueeze(1) ,x=None, cond_images=X_train_batch.unsqueeze(1), unet_number=train_unet_number)

                loss.backward( )

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                optimizer.step()

            loss_total=loss_total+loss.item()

            if steps % print_every == 0:
                print(".", end="")

            if steps>0:
                if steps % print_loss == 0:

                    if plot_unscaled:
                        #test before scaling...
                        plt.plot (y_train_batch.unsqueeze(1)[0,0,:].cpu().detach().numpy(),label= 'Unscaled GT')
                        plt.legend()
                        plt.show()


                    GT=y_train_batch.cpu().detach() 

                    GT=resize_image_to(
                        GT.unsqueeze(1),
                        model.imagen.image_sizes[train_unet_number-1],

                    )

                    norm_loss=loss_total/print_loss
                    print (f"\nTOTAL LOSS at epoch={e}, step={steps}: {norm_loss}")

                    loss_list.append (norm_loss)
                    loss_total=0

                    plt.plot (loss_list, label='Loss')
                    plt.legend()

                    outname = prefix+ f"loss_{e}_{steps}.jpg"
                    plt.savefig(outname, dpi=200)
                    plt.show()

                    ####
                    num_samples = min (num_samples,y_train_batch.shape[0] )
                    print (f"Producing {num_samples} samples...")


                    if cond_image == True:
                        use_text_embedd=False
                    else:
                        use_text_embedd=True

                    sample_loop_FromModelB (
                        model,
                        test_loader,
                        cond_scales=cond_scales,# #list of cond scales - each sampled...
                        num_samples=num_samples, #how many samples produced every time tested.....
                        timesteps=64,
                        flag=steps,
                        #reverse=False,
                        foldproteins=foldproteins,
                        use_text_embedd= use_text_embedd,
                        # ++++++++++++++++++++
                        train_unet_number=train_unet_number,
                        ynormfac=ynormfac,
                        prefix=prefix,
                        tokenizer_y=tokenizer_y,
                        Xnormfac=Xnormfac,
                        tokenizer_X=tokenizer_X,
                    )   

                    #index_word': '{"1": "~", "2": "h", "3": "e", "4": "s", "5": "t", "6": "g", "7": "b", "8": "i"}', 
                    #'word_index': '{"~": 1, "h": 2, "e": 3, "s": 4, "t": 5, "g": 6, "b": 7, "i": 8}'}

                    AH_code=2/Xnormfac
                    BS_code=3/Xnormfac
                    unstr_code= 1/Xnormfac

                    print ("SAMPLING FOR DE NOVO:")
                    
                    # +++++++++++++++++++++++++++++++++++++++++
                    for this_x_data in test_condition_list:
                        sample_sequence_FromModelB (
                            model,
                            x_data=this_x_data,
                            flag=steps,cond_scales=1.,
                            foldproteins=True,
                            # ++++++++++
                            ynormfac=ynormfac,
                            train_unet_number=train_unet_number,
                            tokenizer_X=tokenizer_X,
                            Xnormfac=Xnormfac,
                            max_length=max_length,
                            prefix=prefix,
                            tokenizer_y=tokenizer_y,
                           )
                    # -----------------------------------------    
                    # sample_sequence (
                    #     model,
                    #     x_data=['~~~HHHHHHHHHHHHHHH~~'],
                    #     flag=steps,cond_scales=1.,
                    #     foldproteins=True,
                    #     # ++++++++++
                    #     ynormfac=ynormfac,
                    #    )
                    # sample_sequence (
                    #     model,
                    #     x_data=['~~~HHHHHHHHHHHHHHH~~~~HHHHHHHHHHHHHH~~~'],
                    #     flag=steps,cond_scales=1.,
                    #     foldproteins=True,
                    #     # ++++++++++
                    #     ynormfac=ynormfac,
                    #    )
                    # sample_sequence (
                    #     model,
                    #     x_data=['~~EEESSTTS~SEEEEEEEEE~SBS~EEEEEE~~'],
                    #     flag=steps,cond_scales=1.,
                    #     foldproteins=True,
                    #     # ++++++++++++
                    #     ynormfac=ynormfac,
                    #    )

            if steps>0:
                if save_model and steps % print_loss==0:
                    fname=f"{prefix}trainer_save-model-epoch_{e}.pt"
                    trainer.save(fname)
                    print (f"Model saved: ", fname)
                    fname=f"{prefix}statedict_save-model-epoch_{e}.pt"
                    torch.save(model.state_dict(), fname)
                    print (f"Statedict model saved: ", fname)

            steps=steps+1

        print (f"\n\n-------------------\nTime for epoch {e}={(time.time()-start)/60}\n-------------------")
            

# ++++++++++++++++++++++++++++++++++++++++++++++
def foldandsavePDB_pdb_fasta (
    sequence, 
    filename_out, 
    num_cycle=16, 
    flag=0,
    # ++++++++++++
    prefix=None,
):
    
    filename=f"{prefix}fasta_in_{flag}.fasta"
    print ("Writing FASTA file: ", filename)
    OUTFILE=f"{filename_out}_{flag}"
    with open (filename, mode ='w') as f:
        f.write (f'>{OUTFILE}\n')
        f.write (f'{sequence}')
        
    print (f"Now run OmegaFold.... on device={device}")    
    # !omegafold $filename $prefix --num_cycle $num_cycle --device=$device
    cmd_line=F"omegafold {filename} {prefix} --num_cycle {num_cycle} --device={device}"
    print(os.popen(cmd_line).read())
    
    print ("Done OmegaFold")
    
    # PDB_result=f"{prefix}{OUTFILE}.PDB"
    PDB_result=f"{prefix}{OUTFILE}.pdb"
    print (f"Resulting PDB file...:  {PDB_result}")
    
    return PDB_result, filename



def foldandsavePDB (
    sequence, 
    filename_out, 
    num_cycle=16, 
    flag=0,
    # ++++++++++++
    prefix=None,
):
    
    filename=f"{prefix}fasta_in_{flag}.fasta"
    print ("Writing FASTA file: ", filename)
    OUTFILE=f"{filename_out}_{flag}"
    with open (filename, mode ='w') as f:
        f.write (f'>{OUTFILE}\n')
        f.write (f'{sequence}')
        
    print (f"Now run OmegaFold.... on device={device}")    
    # !omegafold $filename $prefix --num_cycle $num_cycle --device=$device
    cmd_line=F"omegafold {filename} {prefix} --num_cycle {num_cycle} --device={device}"
    print(os.popen(cmd_line).read())
    
    print ("Done OmegaFold")
    
    # PDB_result=f"{prefix}{OUTFILE}.PDB"
    PDB_result=f"{prefix}{OUTFILE}.pdb"
    print (f"Resulting PDB file...:  {PDB_result}")
    
    return PDB_result

import py3Dmol
def plot_plddt_legend(dpi=100):
  thresh = ['plDDT:','Very low (<50)','Low (60)','OK (70)','Confident (80)','Very high (>90)']
  plt.figure(figsize=(1,0.1),dpi=dpi)
  ########################################
  for c in ["#FFFFFF","#FF0000","#FFFF00","#00FF00","#00FFFF","#0000FF"]:
    plt.bar(0, 0, color=c)
  plt.legend(thresh, frameon=False,
             loc='center', ncol=6,
             handletextpad=1,
             columnspacing=1,
             markerscale=0.5,)
  plt.axis(False)
  return plt
color = "lDDT" # choose from ["chain", "lDDT", "rainbow"]
show_sidechains = False #choose from {type:"boolean"}
show_mainchains = False #choose from {type:"boolean"}

def show_pdb(pdb_file, flag=0,   show_sidechains=False, show_mainchains=False, color="lDDT"):
  model_name = f"Flag_{flag}"
  view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js',)
  view.addModel(open(pdb_file,'r').read(),'pdb')

  if color == "lDDT":
    view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
  elif color == "rainbow":
    view.setStyle({'cartoon': {'color':'spectrum'}})
  elif color == "chain":
    chains = len(queries[0][1]) + 1 if is_complex else 1
    for n,chain,color in zip(range(chains),list("ABCDEFGH"),
                     ["lime","cyan","magenta","yellow","salmon","white","blue","orange"]):
      view.setStyle({'chain':chain},{'cartoon': {'color':color}})
  if show_sidechains:
    BB = ['C','O','N']
    view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
    view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                        {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
    view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})  
  if show_mainchains:
    BB = ['C','O','N','CA']
    view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

  view.zoomTo()
  if color == "lDDT":
      plot_plddt_legend().show() 
  return view

def get_avg_Bfac (file='./output_v3/[0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0].pdb'):
    p = PDBParser()
    avg_B=0
    bfac_list=[]
    
    structure = p.get_structure("X", file)
    for PDBmodel in structure:
        for chain in PDBmodel:
             for residue in chain:
                     for atom in residue:
                       
                        Bfac=atom.get_bfactor()
                        bfac_list.append(Bfac)
                        avg_B=avg_B+Bfac
                       
    avg_B=avg_B/len (bfac_list)
    print (f"For {file}, average B-factor={avg_B}")
    plt.plot (bfac_list, label='lDDT')
    plt.xlabel ('Atom #'   )
    plt.ylabel ('iDDT')
    plt.legend()
    plt.show()
    return avg_B, bfac_list

def sample_sequence_normalized_Bfac (seccs=[0.3, 0.3, 0.1, 0., 0., 0., 0., 0. ]):
    sample_numbers=torch.tensor([seccs])
    sample_numbers=torch.nn.functional.normalize (sample_numbers, dim=1)
    sample_numbers=sample_numbers/torch.sum(sample_numbers)

    PDB=sample_sequence (model,
                    X=sample_numbers,
                     flag=0,cond_scales=1, foldproteins=True,
                   )

    avg,_ = get_avg_Bfac (file=PDB[0])

    return PDB, avg

# ======================================================
# blocks for Model A
# ======================================================
def train_loop_Old_FromModelA (
    model,
    train_loader,
    test_loader,
    #
    optimizer=None,
    print_every=1,
    epochs= 300,
    start_ep=0,
    start_step=0,
    train_unet_number=1,
    print_loss_every_steps=1000,
    #
    trainer=None,
    plot_unscaled=False,
    max_batch_size=4,
    save_model=False,
    cond_scales=[1.0], #list of cond scales
    num_samples=2, #how many samples produced every time tested.....
    foldproteins=False,
    # ++
    cond_image=False, # not use cond_images... for model A
    cond_text=True,   # use condi_text...      for model A
    # +
    device=None,
    loss_list=[],
    epoch_list=[],
    train_hist_file=None,
    train_hist_file_full=None,
    prefix=None, # not used in this function
    Xnormfac=None,
    ynormfac=1.,
    tokenizer_X=None,
    tokenizer_y=None,
    test_condition_list=[],
    max_length_Y=1,
    max_text_len_X=1,
    CKeys=None,
    sample_steps=1,
    sample_dir=None,
    save_every_epoch=1,
    save_point_info_file=None,
    store_dir=None,
):
    # #+
    # Xnormfac=Xnormfac.to(model.device)
    
    if not exists (trainer):
        if not exists (optimizer):
            print ("ERROR: If trainer not used, need to provide optimizer.")
    if exists (trainer):
        print ("Trainer provided... will be used")
    # --------------------------------
    # steps=start_step
    # ++++++++++++++++++++++++++++++++
    steps=start_step
    added_steps=0

    loss_total=0
    for e in range(1, epochs+1):
        # start = time.time()

        torch.cuda.empty_cache()
        print ("######################################################################################")
        start = time.time()
        print ("NOW: Training epoch: ", e+start_ep)

        # TRAINING
        train_epoch_loss = 0
        model.train()

        print ("Loop over ", len(train_loader), " batches (print . every ", print_every, " steps)")

        for item  in train_loader:
            # ++
            steps += 1
            added_steps += 1

            X_train_batch= item[0].to(device)
            y_train_batch=item[1].to(device)

            if exists (trainer):
                if cond_image==False:
                    # ========================================
                    # Model A: condition via text
                    # ========================================
                    # this block depends on the model:forward
                    loss = trainer(
                        # # --------------------------------
                        # X_train_batch, 
                        # y_train_batch.unsqueeze(1) ,
                        # ++++++++++++++++++++++++++++++++
                        y_train_batch.unsqueeze(1) ,
                        x=X_train_batch, 
                        # 
                        unet_number=train_unet_number,
                        max_batch_size = max_batch_size,    # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                    )
                if cond_image==True:
                    # ========================================
                    # Model B: condition via image/sequence
                    # ========================================
                    # added for future: Train_loop B
                    loss = trainer(
                        y_train_batch.unsqueeze(1) ,
                        x=None,
                        cond_images=X_train_batch.unsqueeze(1), 
                        unet_number=train_unet_number,
                        max_batch_size = max_batch_size, # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                    )
                    # pass
                #
                trainer.update(unet_number = train_unet_number)

            else:
                optimizer.zero_grad()
                if cond_image==False:
                    # this block depends on the model:forward
                    loss=model ( 
                        # # --------------------------------
                        # X_train_batch, 
                        # y_train_batch.unsqueeze(1) ,
                        # ++++++++++++++++++++++++++++++++
                        y_train_batch.unsqueeze(1) ,
                        x=X_train_batch,
                        #
                        unet_number=train_unet_number
                    )
                if cond_image==True:
                    # added for future: Train_loop B
                    loss=model (
                        y_train_batch.unsqueeze(1) ,
                        x=None, 
                        cond_images=X_train_batch.unsqueeze(1), 
                        unet_number=train_unet_number
                    )
                #
                loss.backward( )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            loss_total=loss_total+loss.item()
            # +
            train_epoch_loss=train_epoch_loss+loss.item()

            if steps % print_every == 0:
                # for progress bar
                print(".", end="")

            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # record loss block
            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # if steps>0:
            if added_steps>0:

                if steps % print_loss_every_steps == 0:
                    # + for debug
                    if CKeys['Debug_TrainerPack']==2:
                        print("Here is step: ", steps)

                    norm_loss=loss_total/print_loss_every_steps
                    print (f"\nTOTAL LOSS at epoch={e+start_ep}, step={steps}: {norm_loss}")
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # add a line to the hist file
                    add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss)+'\n'
                    with open(train_hist_file,'a') as f:
                        f.write(add_line)

                    loss_list.append (norm_loss)
                    loss_total=0
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    epoch_list.append(e+start_ep)

                    fig = plt.figure()
                    plt.plot (epoch_list, loss_list, label='Loss')
                    plt.legend()
                    # outname = prefix+ f"loss_{e}_{steps}.jpg"
                    outname = sample_dir+ f"loss_{e+start_ep}_{steps}.jpg"
                    # 
                    # the order, save then show, matters
                    if CKeys['SlientRun']==1:
                        plt.savefig(outname, dpi=200)
                    else:
                        plt.show()
                    plt.close(fig)

            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # sample in test set block
            # set sample_steps < 0 to switch off this block
            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # if steps>0:
            if added_steps>0:
                if steps % sample_steps == 0 and sample_steps > 0:
                    # + for debug
                    if CKeys['Debug_TrainerPack']==2:
                        print("Here is steps: ", steps)

                    if plot_unscaled:
                        # test before scaling...
                        plt.plot (
                            y_train_batch.unsqueeze(1)[0,0,:].cpu().detach().numpy(),
                            label= 'Unscaled GT'
                        )
                        plt.legend()
                        plt.show()

                    #rescale GT to properly plot
                    GT=y_train_batch.cpu().detach() 

                    GT=resize_image_to(
                        GT.unsqueeze(1),
                        model.imagen.image_sizes[train_unet_number-1],

                    )
                    ####
                    print ("I. SAMPLING IN TEST SET: ")
                    ####

                    num_samples = min (num_samples,y_train_batch.shape[0] )
                    print (f"Producing {num_samples} samples...")

                    sample_loop_omegafold_ModelA (
                        model,
                        test_loader,
                        cond_scales=cond_scales,
                        num_samples=num_samples, #how many samples produced every time tested.....
                        timesteps=None,
                        flag=e+start_ep, # steps,
                        foldproteins=foldproteins,
                        # add condi_key
                        cond_image=cond_image, # Not used for now
                        cond_text=cond_text,   # Not used for now
                        skip_steps=0,
                        #
                        max_text_len=max_text_len_X,
                        max_length=max_length_Y,
                        # ++++++++++++++++++++
                        train_unet_number=train_unet_number,
                        ynormfac=ynormfac,
                        prefix=prefix,   #
                        tokenizer_y=tokenizer_y,
                        Xnormfac_CondiText=Xnormfac,
                        tokenizer_X_CondiText=tokenizer_X,
                        # ++
                        CKeys=CKeys,
                        sample_dir=sample_dir,
                        steps=steps,
                        e=e+start_ep,
                        IF_showfig= CKeys['SlientRun']!=1 ,
                    )   

                    print ("II. SAMPLING FOR DE NOVO:")

                    sample_sequence_omegafold_ModelA (
                        # # ----------------------------------------------
                        # model,
                        # X=[[0, 0.7, 0.07, 0.1, 0.01, 0.02, 0.01, 0.11]],
                        # foldproteins=foldproteins,
                        # flag=steps,cond_scales=1.,
                        # ++++++++++++++++++++++++++++++++++++++++++++++
                        model,
                        X=test_condition_list, # [[0.92, 0., 0.04, 0.04, 0., 0., 0., 0., ]], # from text conditioning X
                        flag=e+start_ep, # steps, # 0,
                        cond_scales=cond_scales, # 1.,
                        foldproteins=True, # False,
                        X_string=None,                                # from text conditioning X_string
                        x_data=None,                                  # from image conditioning x_data   
                        skip_steps=0,
                        inpaint_images=None, # in formation Y data
                        inpaint_masks = None,
                        inpaint_resample_times = None,
                        init_images = None,
                        num_cycle=16,          # for omegafolding
                        calc_error=False,      # for check on folded results, not used for every case
                        # ++++++++++++++++++++++++++
                        # tokenizers
                        tokenizer_X_forImageCondi=None, # for x_data
                        Xnormfac_forImageCondi=1.,
                        tokenizer_X_forTextCondi=None,  # for X if NEEDED only
                        Xnormfac_forTextCondi=1.,
                        tokenizer_y=tokenizer_y, # None, # for output Y
                        ynormfac=ynormfac,
                        # length
                        train_unet_number=1,
                        max_length_Y=max_length_Y,                 # for Y, X_forImageCondi
                        max_text_len=max_text_len_X,                 # for    X_forTextCondi
                        # other info
                        steps=steps, # None,
                        e=e, # None,
                        sample_dir=sample_dir, # None,
                        prefix=prefix, # None,
                        IF_showfig= CKeys['SlientRun']!=1, # True,
                        CKeys=CKeys,
                        # TBA to Model B
                        normalize_X_cond_to_one=False,
                    )

                    # sample_sequence (model,
                    #     X=[[0., 0.0, 0.0, 0.0, 0., 0., 0., 0., ]],foldproteins=foldproteins,
                    #      flag=steps,cond_scales=1.,
                    #    )

        # summerize loss over every epoch:
        norm_loss_over_e = train_epoch_loss/len(train_loader)
        print("\nnorm_loss over 1 epoch: ", norm_loss_over_e)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # write this into "train_hist_file_full"
        add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss_over_e)+'\n'
        with open(train_hist_file_full,'a') as f:
            f.write(add_line)
        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # save model every this epoches
        if save_model and (e+start_ep) % save_every_epoch==0 and e>1:
            # fname=f"{prefix}trainer_save-model-epoch_{e+start_ep}.pt"
            fname=f"{store_dir}trainer_save-model-epoch_{e+start_ep}.pt"
            trainer.save(fname)
            print (f"Model saved: ", fname)
            # fname=f"{prefix}statedict_save-model-epoch_{e+start_ep}.pt"
            fname=f"{store_dir}statedict_save-model-epoch_{e+start_ep}.pt"
            torch.save(model.state_dict(), fname)
            print (f"Statedict model saved: ", fname)
            # add a saving point file
            top_line='epoch,steps,norm_loss'+'\n'
            add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss)+'\n'
            with open(save_point_info_file, "w") as f:
                f.write(top_line)
                f.write(add_line)
                
            # if steps>0:
            #     if save_model and steps % print_loss_every_steps==0: 
            #         fname=f"{prefix}trainer_save-model-epoch_{e}.pt"
            #         trainer.save(fname)
            #         fname=f"{prefix}statedict_save-model-epoch_{e}.pt"
            #         torch.save(model.state_dict(), fname)
            #         print (f"Model saved: ")

            # steps=steps+1

        print (f"\n\n-------------------\nTime for epoch {e+start_ep}={(time.time()-start)/60}\n-------------------")
        
def train_loop_ForModelA_II (
    model,
    train_loader,
    test_loader,
    #
    optimizer=None,
    print_every=1,
    epochs= 300,
    start_ep=0,
    start_step=0,
    train_unet_number=1,
    print_loss_every_steps=1000,
    #
    trainer=None,
    plot_unscaled=False,
    max_batch_size=4,
    save_model=False,
    cond_scales=[1.0], #list of cond scales
    num_samples=2, #how many samples produced every time tested.....
    foldproteins=False,
    # ++
    cond_image=False, # not use cond_images... for model A
    cond_text=True,   # use condi_text...      for model A
    # +
    device=None,
    loss_list=[],
    epoch_list=[],
    train_hist_file=None,
    train_hist_file_full=None,
    prefix=None, # not used in this function
    Xnormfac=None,
    ynormfac=1.,
    tokenizer_X=None,
    tokenizer_y=None,
    test_condition_list=[],
    max_length_Y=1,
    max_text_len_X=1,
    CKeys=None,
    sample_steps=1,
    sample_dir=None,
    save_every_epoch=1,
    save_point_info_file=None,
    store_dir=None,
    # ++ for pLM
    pLM_Model_Name=None,
    image_channels=None,
    print_error=False, # not defined for Problem6 # True,
):
    # #+
    # Xnormfac=Xnormfac.to(model.device)
    
    if not exists (trainer):
        if not exists (optimizer):
            print ("ERROR: If trainer not used, need to provide optimizer.")
    if exists (trainer):
        print ("Trainer provided... will be used")
    # --------------------------------
    # steps=start_step
    # ++++++++++++++++++++++++++++++++
    steps=start_step
    added_steps=0

    loss_total=0
    
    # ++ for pLM
    if pLM_Model_Name=='None':
        pLM_Model=None
        
    elif pLM_Model_Name=='esm2_t33_650M_UR50D':
        # dim: 1280
        esm_layer=33
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)
        
    elif pLM_Model_Name=='esm2_t36_3B_UR50D':
        # dim: 2560
        esm_layer=36
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)
        
    elif pLM_Model_Name=='esm2_t30_150M_UR50D':
        # dim: 640
        esm_layer=30
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)
    
    elif pLM_Model_Name=='esm2_t12_35M_UR50D':
        # dim: 480
        esm_layer=12
        pLM_Model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
        pLM_Model.eval()
        pLM_Model. to(device)
        
    else:
        print("pLM model is missing...")
        
        
    for e in range(1, epochs+1):
        # start = time.time()

        torch.cuda.empty_cache()
        print ("######################################################################################")
        start = time.time()
        print ("NOW: Training epoch: ", e+start_ep)

        # TRAINING
        train_epoch_loss = 0
        model.train()

        print ("Loop over ", len(train_loader), " batches (print . every ", print_every, " steps)")

        for item  in train_loader:
            # ++
            steps += 1
            added_steps += 1

            X_train_batch= item[0].to(device)
            y_train_batch=item[1].to(device)
            # project y_ into embedding space
            if CKeys["Debug_TrainerPack"]==1:
                print("Initial unload the dataloader items: ...")
                print(X_train_batch.shape)
                print(y_train_batch.shape)
            # ++
            # project the AA seq into embedding space
            # for output, it is shared between ModelA and ModelB
            # # --
            # if pLM_Model_Name=='None':
            #     # just use the encoded sequence
            #     y_train_batch_in = y_train_batch.unsqueeze(1)
            #     # pass
            # elif pLM_Model_Name=='esm2_t33_650M_UR50D':
            #     with torch.no_grad():
            #         results = pLM_Model(
            #             y_train_batch,
            #             repr_layers=[33],
            #             return_contacts=False,
            #         )
            #     y_train_batch_in = results["representations"][33]
            #     y_train_batch_in = rearrange(
            #         y_train_batch_in, 
            #         'b l c -> b c l'
            #     )
            # else:
            #     print(f"Required pLM name is not defined!!")
            # ++
            if pLM_Model_Name=='None':
                # just use the encoded sequence
                y_train_batch_in = y_train_batch.unsqueeze(1)
                # pass
            else: # for ESM models # pLM_Model_Name=='esm2_t33_650M_UR50D':
                with torch.no_grad():
                    results = pLM_Model(
                        y_train_batch,
                        repr_layers=[esm_layer],
                        return_contacts=False,
                    )
                y_train_batch_in = results["representations"][esm_layer]
                y_train_batch_in = rearrange(
                    y_train_batch_in, 
                    'b l c -> b c l'
                )
            
                
            #
            # For input part, this block is different for ModelA and ModelB
            if cond_image==False:
                # model A: X: text_condi, not affected by pLM
                X_train_batch_in = X_train_batch
            else:
                # model B: X: cond_img, will be affected by pLM
                X_train_batch_in = X_train_batch.unsqueeze(1).repeat(1,image_channels,1)
            #
            # + for debug
            if CKeys["Debug_TrainerPack"]==1:
                print("After pLM model, the shape of X and y for training:")
                print("X_train_batch_in.dim: ", X_train_batch_in.shape)
                print("y_train_batch_in.dim: ", y_train_batch_in.shape)
                    
            
            

            if exists (trainer):
                if cond_image==False:
                    # ========================================
                    # Model A: condition via text
                    # ========================================
                    # this block depends on the model:forward
                    loss = trainer(
                        # # --------------------------------
                        # X_train_batch, 
                        # y_train_batch.unsqueeze(1) ,
                        # # ++++++++++++++++++++++++++++++++
                        # y_train_batch.unsqueeze(1) ,
                        # x=X_train_batch, 
                        # ++ pLM
                        y_train_batch_in,
                        x=X_train_batch_in,
                        # 
                        unet_number=train_unet_number,
                        max_batch_size = max_batch_size,    # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                    )
                if cond_image==True:
                    # ========================================
                    # Model B: condition via image/sequence
                    # ========================================
                    # # --
                    # # added for future: Train_loop B
                    # loss = trainer(
                    #     y_train_batch.unsqueeze(1) ,
                    #     x=None,
                    #     cond_images=X_train_batch.unsqueeze(1), 
                    #     unet_number=train_unet_number,
                    #     max_batch_size = max_batch_size, # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                    # )
                    # ++ from pLM+ModelB
                    loss = trainer(
                        y_train_batch_in,                          # true image
                        x=None,                                 # tokenized text
                        cond_images=X_train_batch_in,              # cond_image
                        unet_number=train_unet_number,
                        max_batch_size = max_batch_size, # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                        )
                    # pass
                #
                trainer.update(unet_number = train_unet_number)

            else:
                optimizer.zero_grad()
                if cond_image==False:
                    # this block depends on the model:forward
                    loss=model ( 
                        # # --------------------------------
                        # X_train_batch, 
                        # y_train_batch.unsqueeze(1) ,
                        # # ++++++++++++++++++++++++++++++++
                        # y_train_batch.unsqueeze(1) ,
                        # x=X_train_batch,
                        # ++ pLM
                        y_train_batch_in,
                        x=X_train_batch_in,
                        #
                        unet_number=train_unet_number
                    )
                if cond_image==True:
                    # added for future: Train_loop B
                    # # --
                    # loss=model (
                    #     y_train_batch.unsqueeze(1) ,
                    #     x=None, 
                    #     cond_images=X_train_batch.unsqueeze(1), 
                    #     unet_number=train_unet_number
                    # )
                    # ++ from pLM
                    loss=model (
                        y_train_batch_in ,
                        x=None, 
                        cond_images=X_train_batch_in, 
                        unet_number=train_unet_number
                    )
                #
                loss.backward( )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            loss_total=loss_total+loss.item()
            # +
            train_epoch_loss=train_epoch_loss+loss.item()

            if steps % print_every == 0:
                # for progress bar
                print(".", end="")

            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # record loss block
            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # if steps>0:
            if added_steps>0:

                if steps % print_loss_every_steps == 0:
                    # + for debug
                    if CKeys['Debug_TrainerPack']==2:
                        print("Here is step: ", steps)

                    norm_loss=loss_total/print_loss_every_steps
                    print (f"\nTOTAL LOSS at epoch={e+start_ep}, step={steps}: {norm_loss}")
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # add a line to the hist file
                    add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss)+'\n'
                    with open(train_hist_file,'a') as f:
                        f.write(add_line)

                    loss_list.append (norm_loss)
                    loss_total=0
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    epoch_list.append(e+start_ep)

                    fig = plt.figure()
                    plt.plot (epoch_list, loss_list, label='Loss')
                    plt.legend()
                    # outname = prefix+ f"loss_{e}_{steps}.jpg"
                    outname = sample_dir+ f"loss_{e+start_ep}_{steps}.jpg"
                    # 
                    # the order, save then show, matters
                    if CKeys['SlientRun']==1:
                        plt.savefig(outname, dpi=200)
                    else:
                        plt.show()
                    plt.close(fig)

            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # sample in test set block
            # set sample_steps < 0 to switch off this block
            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            # if steps>0:
            if added_steps>0:
                if steps % sample_steps == 0 and sample_steps > 0:
                    # + for debug
                    if CKeys['Debug_TrainerPack']==2:
                        print("Here is steps: ", steps)

                    if plot_unscaled:
                        # test before scaling...
                        plt.plot (
                            y_train_batch.unsqueeze(1)[0,0,:].cpu().detach().numpy(),
                            label= 'Unscaled GT'
                        )
                        plt.legend()
                        plt.show()

#                     # -- look like not used
#                     #rescale GT to properly plot
#                     GT=y_train_batch.cpu().detach() 

#                     GT=resize_image_to(
#                         GT.unsqueeze(1),
#                         model.imagen.image_sizes[train_unet_number-1],

#                     )
                    ####
                    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ")
                    print ("I. SAMPLING IN TEST SET: ")
                    print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    ####

                    num_samples = min (num_samples,y_train_batch.shape[0] )
                    print (f"Producing {num_samples} samples...")

                    sample_loop_omegafold_pLM_ModelA (
                        model,
                        test_loader,
                        cond_scales=cond_scales,
                        num_samples=num_samples, #how many samples produced every time tested.....
                        timesteps=None,
                        flag=e+start_ep, # steps,
                        foldproteins=foldproteins,
                        # add condi_key
                        cond_image=cond_image, # Not used for now
                        cond_text=cond_text,   # Not used for now
                        skip_steps=0,
                        #
                        max_text_len=max_text_len_X,
                        max_length=max_length_Y,
                        # ++++++++++++++++++++
                        train_unet_number=train_unet_number,
                        ynormfac=ynormfac,
                        prefix=prefix,   #
                        tokenizer_y=tokenizer_y,
                        Xnormfac_CondiText=Xnormfac,
                        tokenizer_X_CondiText=tokenizer_X,
                        # ++
                        CKeys=CKeys,
                        sample_dir=sample_dir,
                        steps=steps,
                        e=e+start_ep,
                        IF_showfig= CKeys['SlientRun']!=1 ,
                        # ++ for pLM
                        pLM_Model=pLM_Model,
                        pLM_Model_Name=pLM_Model_Name,
                        image_channels=image_channels,
                        pLM_alphabet=esm_alphabet,
                    )   

                    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ")
                    print ("II. SAMPLING FOR DE NOVO:")
                    print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")

                    DeNovoSam_pdbs, fasta_file_list=\
                    sample_sequence_omegafold_pLM_ModelA (
                        # # ----------------------------------------------
                        # model,
                        # X=[[0, 0.7, 0.07, 0.1, 0.01, 0.02, 0.01, 0.11]],
                        # foldproteins=foldproteins,
                        # flag=steps,cond_scales=1.,
                        # ++++++++++++++++++++++++++++++++++++++++++++++
                        model,
                        X=test_condition_list, # [[0.92, 0., 0.04, 0.04, 0., 0., 0., 0., ]], # from text conditioning X
                        flag=e+start_ep, # steps, # 0,
                        cond_scales=cond_scales, # 1.,
                        foldproteins=True, # False,
                        X_string=None,                                # from text conditioning X_string
                        x_data=None,                                  # from image conditioning x_data   
                        skip_steps=0,
                        inpaint_images=None, # in formation Y data
                        inpaint_masks = None,
                        inpaint_resample_times = None,
                        init_images = None,
                        num_cycle=16,          # for omegafolding
                        calc_error=False,      # for check on folded results, not used for every case
                        # ++++++++++++++++++++++++++
                        # tokenizers
                        tokenizer_X_forImageCondi=None, # for x_data
                        Xnormfac_forImageCondi=1.,
                        tokenizer_X_forTextCondi=None,  # for X if NEEDED only
                        Xnormfac_forTextCondi=1.,
                        tokenizer_y=tokenizer_y, # None, # for output Y
                        ynormfac=ynormfac,
                        # length
                        train_unet_number=1,
                        max_length_Y=max_length_Y,                 # for Y, X_forImageCondi
                        max_text_len=max_text_len_X,                 # for    X_forTextCondi
                        # other info
                        steps=steps, # None,
                        e=e, # None,
                        sample_dir=sample_dir, # None,
                        prefix=prefix, # None,
                        IF_showfig= CKeys['SlientRun']!=1, # True,
                        CKeys=CKeys,
                        # TBA to Model B
                        normalize_X_cond_to_one=False,
                        # ++ for pLM
                        pLM_Model=pLM_Model,
                        pLM_Model_Name=pLM_Model_Name,
                        image_channels=image_channels,
                        pLM_alphabet=esm_alphabet,
                    )

                    # sample_sequence (model,
                    #     X=[[0., 0.0, 0.0, 0.0, 0., 0., 0., 0., ]],foldproteins=foldproteins,
                    #      flag=steps,cond_scales=1.,
                    #    )

        # summerize loss over every epoch:
        norm_loss_over_e = train_epoch_loss/len(train_loader)
        print("\nnorm_loss over 1 epoch: ", norm_loss_over_e)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # write this into "train_hist_file_full"
        add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss_over_e)+'\n'
        with open(train_hist_file_full,'a') as f:
            f.write(add_line)
        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # save model every this epoches
        if save_model and (e+start_ep) % save_every_epoch==0 and e>1:
            # fname=f"{prefix}trainer_save-model-epoch_{e+start_ep}.pt"
            fname=f"{store_dir}trainer_save-model-epoch_{e+start_ep}.pt"
            trainer.save(fname)
            print (f"Model saved: ", fname)
            # fname=f"{prefix}statedict_save-model-epoch_{e+start_ep}.pt"
            fname=f"{store_dir}statedict_save-model-epoch_{e+start_ep}.pt"
            torch.save(model.state_dict(), fname)
            print (f"Statedict model saved: ", fname)
            # add a saving point file
            top_line='epoch,steps,norm_loss'+'\n'
            add_line = str(e+start_ep)+','+str(steps)+','+str(norm_loss)+'\n'
            with open(save_point_info_file, "w") as f:
                f.write(top_line)
                f.write(add_line)
                
            # if steps>0:
            #     if save_model and steps % print_loss_every_steps==0: 
            #         fname=f"{prefix}trainer_save-model-epoch_{e}.pt"
            #         trainer.save(fname)
            #         fname=f"{prefix}statedict_save-model-epoch_{e}.pt"
            #         torch.save(model.state_dict(), fname)
            #         print (f"Model saved: ")

            # steps=steps+1

        print (f"\n\n-------------------\nTime for epoch {e+start_ep}={(time.time()-start)/60}\n-------------------")

# from original, not used any more
def train_loop_Model_A (
    model,
    train_loader,
    test_loader,
    optimizer=None,
    print_every=10,
    epochs= 300,
    start_ep=0,
    start_step=0,
    train_unet_number=1,
    print_loss=1000,
    trainer=None,
    plot_unscaled=False,
    max_batch_size=4,
    save_model=False,
    cond_scales=[1.0], #list of cond scales
    num_samples=2, #how many samples produced every time tested.....
    foldproteins=False,
):
    
    
    if not exists (trainer):
        if not exists (optimizer):
            print ("ERROR: If trainer not used, need to provide optimizer.")
    if exists (trainer):
        print ("Trainer provided... will be used")
    steps=start_step

    loss_total=0
    for e in range(1, epochs+1):
            start = time.time()

            torch.cuda.empty_cache()
            print ("######################################################################################")
            start = time.time()
            print ("NOW: Training epoch: ", e+start_ep)

            # TRAINING
            train_epoch_loss = 0
            model.train()
            
            print ("Loop over ", len(train_loader), " batches (print . every ", print_every, " steps)")
            for item  in train_loader:
                X_train_batch= item[0].to(device)
                y_train_batch=item[1].to(device)

                if exists (trainer):
                    loss = trainer(
                            X_train_batch, y_train_batch.unsqueeze(1) ,
                            unet_number=train_unet_number,
                            max_batch_size = max_batch_size,    # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                        )
                    trainer.update(unet_number = train_unet_number)

                else:
                    optimizer.zero_grad()
                    loss=model ( X_train_batch, y_train_batch.unsqueeze(1) ,unet_number=train_unet_number)
                    loss.backward( )
                   
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                    optimizer.step()

                loss_total=loss_total+loss.item()
                
                if steps % print_every == 0:
                    print(".", end="")

                if steps>0:
                    if steps % print_loss == 0:

                        if plot_unscaled:
                            
                            plt.plot (y_train_batch.unsqueeze(1)[0,0,:].cpu().detach().numpy(),label= 'Unscaled GT')
                            plt.legend()
                            plt.show()
 
                        #rescale GT to properly plot
                        GT=y_train_batch.cpu().detach() 
                        
                        GT=resize_image_to(
                            GT.unsqueeze(1),
                            model.imagen.image_sizes[train_unet_number-1],

                        )
                        
                        norm_loss=loss_total/print_loss
                        print (f"\nTOTAL LOSS at epoch={e}, step={steps}: {norm_loss}")

                        loss_list.append (norm_loss)
                        loss_total=0

                        plt.plot (loss_list, label='Loss')
                        plt.legend()

                        outname = prefix+ f"loss_{e}_{steps}.jpg"
                        plt.savefig(outname, dpi=200)
                        plt.show()
                        
                        num_samples = min (num_samples,y_train_batch.shape[0] )
                        print (f"Producing {num_samples} samples...")
                        
                        sample_loop (model,
                            test_loader,
                            cond_scales=cond_scales,
                            num_samples=1, #how many samples produced every time tested.....
                            timesteps=64,
                                    flag=steps,foldproteins=foldproteins,
                                    )   
                        
                        print ("SAMPLING FOR DE NOVO:")
                        sample_sequence (model,
                            X=[[0, 0.7, 0.07, 0.1, 0.01, 0.02, 0.01, 0.11]],foldproteins=foldproteins,
                             flag=steps,cond_scales=1.,
                           )
                        sample_sequence (model,
                            X=[[0., 0.0, 0.0, 0.0, 0., 0., 0., 0., ]],foldproteins=foldproteins,
                             flag=steps,cond_scales=1.,
                           )

                if steps>0:
                    if save_model and steps % print_loss==0: 
                        fname=f"{prefix}trainer_save-model-epoch_{e}.pt"
                        trainer.save(fname)
                        fname=f"{prefix}statedict_save-model-epoch_{e}.pt"
                        torch.save(model.state_dict(), fname)
                        print (f"Model saved: ")
                    
                steps=steps+1
                                         
            print (f"\n\n-------------------\nTime for epoch {e}={(time.time()-start)/60}\n-------------------")

# +++
def sample_sequence_omegafold_ModelA (
    model,
    X=[[0.92, 0., 0.04, 0.04, 0., 0., 0., 0., ]], # from text conditioning X
    flag=0,
    cond_scales=1.,
    foldproteins=False,
    X_string=None,                                # from text conditioning X_string
    x_data=None,                                  # from image conditioning x_data   
    skip_steps=0,
    inpaint_images=None, # in formation Y data
    inpaint_masks = None,
    inpaint_resample_times = None,
    init_images = None,
    num_cycle=16,          # for omegafolding
    calc_error=False,      # for check on folded results, not used for every case
    # ++++++++++++++++++++++++++
    # tokenizers
    tokenizer_X_forImageCondi=None, # for x_data
    Xnormfac_forImageCondi=1.,
    tokenizer_X_forTextCondi=None,  # for X if NEEDED only
    Xnormfac_forTextCondi=1.,
    tokenizer_y=None,               # for output Y
    ynormfac=1,
    # length
    train_unet_number=1,
    max_length_Y=1,                 # for Y, X_forImageCondi
    max_text_len=1,                 # for    X_forTextCondi
    # other info
    steps=None,
    e=None,
    sample_dir=None,
    prefix=None,
    IF_showfig=True,
    CKeys=None,
    # TBA to Model B
    normalize_X_cond_to_one=False,
):
    # -----------
    # steps=0
    # e=flag

    # --
    # print (f"Producing {len(X)} samples...")
    # ++
    if X!=None:
        print (f"Producing {len(X)} samples...from text conditioning X...")
        lenn_val=len(X)
    if X_string!=None:
        lenn_val=len(X_string)
        print (f"Producing {len(X_string)} samples...from text conditioning X_String (from string)...")
    if x_data!=None:
        print (f"Producing {len(x_data)} samples...from image conditingig x_data  ...")
        lenn_val=len(x_data)
        # print (x_data)
    
    print ('Device: ', model.device)
    
    
    for iisample in range (lenn_val):
        print(f"Working on {iisample}")
        X_cond=None
        
        if X_string==None and X!=None: # for X channel
            X_cond=torch.Tensor (X[iisample]).to(device).unsqueeze (0)
        if X_string!=None: # from raw text, ie., X_string: need tokenizer_X and Xnormfac
            # -
            # X = tokenizer_X.texts_to_sequences(X_string[iisample])
            # X= sequence.pad_sequences(X,  maxlen=max_length, padding='post', truncating='post')  
            # X=np.array(X)
            # X_cond=torch.from_numpy(X).float()/Xnormfac
            # +
            XX = tokenizer_X_forTextCondi.texts_to_sequences(X_string[iisample])
            XX = sequence.pad_sequences(XX,  maxlen=max_text_len, padding='post', truncating='post')  
            XX = np.array(XX)
            X_cond = torch.from_numpy(XX).float()/Xnormfac_forTextCondi
            
            print ('Tokenized and processed: ', X_cond)
        
        if X_cond!=None:
            if normalize_X_cond_to_one: # used when there is constrain on X_cond.sum()
                X_cond=X_cond/X_cond.sum()
        
            print ("Text conditoning used: ", X_cond, "...sum: ", X_cond.sum(), "cond scale: ", cond_scales)
        else:
            print ("Text conditioning used: None")
        
        # for now, assume image_condi and text_condi can be used at the same time
        if tokenizer_X_forImageCondi==None:
            # ===========================================================
            # condi_image/seq needs no tokenization, like numbers: force_path
            # only normalization needed
            # Based on ModelB:Force_Path
            if x_data!=None:
                x_data_tokenized=torch.from_numpy(x_data[iisample]/Xnormfac_forImageCondi)
                x_data_tokenized=x_data_tokenized.to(torch.float)
                # + for debug:
                if CKeys['Debug_TrainerPack']==1:
                    print("x_data_tokenized dim: ", x_data_tokenized.shape)
                    print("x_data_tokenized dtype: ", x_data_tokenized.dtype)
                    print("test: ", x_data_tokenized!=None)
            else:
                x_data_tokenized=None
                # + for debug:
                if CKeys['Debug_TrainerPack']==1:
                    print("x_data_tokenized and x_data: None")
            
            # model.sample:full arguments
        # self, 
        # x=None, 
        # stop_at_unet_number=1,
        # cond_scale=7.5,
        # # ++
        # x_data=None, # image_condi data
        # skip_steps=None,
        # inpaint_images = None,
        # inpaint_masks = None,
        # inpaint_resample_times = 5,
        # init_images = None,
        # x_data_tokenized=None,
        # tokenizer_X=None,
        # Xnormfac=1.,
        # # -+
        # device=None,
        # max_length=1., # for XandY data, in image/sequence format; NOT for text condition
        # max_text_len=1., # for X data, in text format
            #
            result=model.sample ( 
                x=X_cond,
                stop_at_unet_number=train_unet_number ,
                cond_scale=cond_scales ,
                x_data=None, 
                # ++
                x_data_tokenized=x_data_tokenized,
                #
                skip_steps=skip_steps,
                inpaint_images = inpaint_images,
                inpaint_masks = inpaint_masks,
                inpaint_resample_times = inpaint_resample_times,
                init_images = init_images,
                device=model.device,
                # ++++++++++++++++++++++++++
                tokenizer_X=tokenizer_X_forImageCondi, # tokenizer_X,
                Xnormfac=Xnormfac_forImageCondi, # Xnormfac,
                # ynormfac=ynormfac, 
                max_length=max_length_Y, # for ImageCondi, max_length,
                max_text_len=max_text_len,
            )
        else:
            #
            result=model.sample ( 
                x=X_cond,
                stop_at_unet_number=train_unet_number ,
                cond_scale=cond_scales ,
                x_data=x_data[iisample], 
                # ++
                x_data_tokenized=None,
                #
                skip_steps=skip_steps,
                inpaint_images = inpaint_images,
                inpaint_masks = inpaint_masks,
                inpaint_resample_times = inpaint_resample_times,
                init_images = init_images,
                device=model.device,
                # ++++++++++++++++++++++++++
                tokenizer_X=tokenizer_X_forImageCondi, # tokenizer_X,
                Xnormfac=Xnormfac_forImageCondi, # Xnormfac,
                # ynormfac=ynormfac, 
                max_length=max_length_Y, # max_length,
                max_text_len=max_text_len,
            )
            
        # # ------------------------------------------    
        # result=model.sample ( 
        #     X_cond,
        #     stop_at_unet_number=train_unet_number,
        #     cond_scale=cond_scales 
        # )
            
        result=torch.round(result*ynormfac)
        # + for debug
        print("result.dim: ", result.shape)
        
        fig=plt.figure()
        plt.plot (
            result[0,0,:].cpu().detach().numpy(),
            label= f'Predicted'
        )
        #plt.plot (GT[samples,0,:]*ynormfac,label= f'GT {0}')
        plt.legend()
        outname = sample_dir+ f"sampled_from_X_{iisample}_condscale-{str (cond_scales)}_{e}_{steps}.jpg"
        #plt.title (f"Sample {samples}, cond scale={str (cond_scales[iisample])}")
        if IF_showfig==1:
            plt.show ()
        else:
            plt.savefig(outname, dpi=200)
        plt.close()
        
        # # ----------------------------------------
        # plt.plot (result[0,0,:].cpu().detach().numpy(),label= f'Predicted')
        # plt.legend()
        # outname = prefix+ f"sampld_from_X_{flag}_condscale-{str (cond_scales)}_{e}_{steps}.jpg"
        # plt.savefig(outname, dpi=200)
        # plt.show ()

        to_rev=result[:,0,:]
        to_rev=to_rev.long().cpu().detach().numpy()
        print("to_rev.dim: ", to_rev.shape)
        y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

        for iii in range (len(y_data_reversed)):
            y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
        # + from Model B
        ### reverse second structure input....
        pdb_list=[]
        if X_cond != None: 
            # there is condi_text
            if X_string!=None:
                X_cond=torch.round(X_cond*Xnormfac_forTextCondi)

                to_rev=X_cond[:,:] 
                to_rev=to_rev.long().cpu().detach().numpy()
                print ("to_rev.dim: ", to_rev.shape)
                # --
                # X_data_reversed=tokenizer_X.sequences_to_texts (to_rev)
                # ++
                X_text_reversed=tokenizer_X_forTextCondi.sequences_to_texts (to_rev)
                for iii in range (len(y_text_reversed)):
                    X_text_reversed[iii]=X_text_reversed[iii].upper().strip().replace(" ", "")
                    
            if X_string==None:
                # reverse this: X_cond=torch.Tensor (X[iisample]).to(device).unsqueeze (0)
                X_text_reversed=X_cond
        else:
            X_text_reversed=None
                
        if x_data !=None: # there is condi_image
            x_data_reversed=x_data #is already in sequence fromat..
        else:
            x_data_reversed=None
        
        # summary
        # print (f"For {X_text_reversed} or {X[iisample]} on Text_Condi,\n and {x_data_reversed} on Image_Condi,\n predicted sequence: ", y_data_reversed)
        print (f"For {X_text_reversed} or {X[iisample]} on Text_Condi,\n and {x_data_reversed} on Image_Condi,")
        print (f"predicted sequence full: {y_data_reversed}")
        # add just for incase check
        print (f"predicted sequence:      {y_data_reversed[0]}")
        
        # + for debug
        print("================================================")
        print("foldproteins: ", foldproteins)
        
        if not foldproteins:
            pdb_file=None
        else:
        # if foldproteins:
            
            if X_cond != None:
                pass
            
            tempname='temp'
            pdb_file=foldandsavePDB (
                sequence=y_data_reversed[0], 
                filename_out=tempname, 
                num_cycle=num_cycle, 
                flag=flag,
                # +++++++++++++++++++
                # prefix=prefix,
                prefix=sample_dir,
            )
            #
            out_nam=iisample
            #
            out_nam_fasta=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.fasta'
            write_fasta (y_data_reversed[0], out_nam_fasta) 
            #
            out_nam=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.pdb'
            shutil.copy (pdb_file, out_nam) #source, dest
            pdb_file=out_nam
            #
            print (f"Properly named PDB file produced: {pdb_file}")
            if IF_showfig==1:
                #flag=1000
                view=show_pdb(
                    pdb_file=pdb_file, 
                    flag=flag,
                    show_sidechains=show_sidechains, 
                    show_mainchains=show_mainchains, 
                    color=color
                )
                view.show()
                
            if calc_error:
                # only work for ModelA:SecStr
                if CKeys['Problem_ID']==7:
                    get_Model_A_error (pdb_file, X[iisample], plotit=True)
                else:
                    print ("Error calculation on the predicted results is not applicable")
                
    pdb_list.append(pdb_file)
    
    return pdb_list
                
            
#             xbc=X_cond[iisample,:].cpu().detach().numpy()
#             out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.2f" % xbc})+f'_{flag}_{steps}'
#             tempname='temp'
#             pdb_file=foldandsavePDB (sequence=y_data_reversed[0], 
#                                                  filename_out=tempname, 
#                                                  num_cycle=16, flag=flag)
#             out_nam_fasta=f'{prefix}{out_nam}.fasta'
            
#             out_nam=f'{prefix}{out_nam}.pdb'
            
#             write_fasta (y_data_reversed[0], out_nam_fasta)
            
#             shutil.copy (pdb_file, out_nam) #source, dest
#             pdb_file=out_nam
#             print (f"Properly named PDB file produced: {pdb_file}")
          
#             view=show_pdb(pdb_file=pdb_file, flag=flag,
#                           show_sidechains=show_sidechains, show_mainchains=show_mainchains, color=color)
#             view.show()

#         if calc_error:
#             get_Model_A_error (pdb_file, X[iisample], plotit=True)
#         return pdb_file

# +++
def sample_sequence_omegafold_pLM_ModelA (
    model,
    X=[[0.92, 0., 0.04, 0.04, 0., 0., 0., 0., ]], # from text conditioning X
    flag=0,
    cond_scales=1.,
    foldproteins=False,
    X_string=None,                                # from text conditioning X_string
    x_data=None,                                  # from image conditioning x_data   
    skip_steps=0,
    inpaint_images=None, # in formation Y data
    inpaint_masks = None,
    inpaint_resample_times = None,
    init_images = None,
    num_cycle=16,          # for omegafolding
    calc_error=False,      # for check on folded results, not used for every case
    # ++++++++++++++++++++++++++
    # tokenizers
    tokenizer_X_forImageCondi=None, # for x_data
    Xnormfac_forImageCondi=1.,
    tokenizer_X_forTextCondi=None,  # for X if NEEDED only
    Xnormfac_forTextCondi=1.,
    tokenizer_y=None,               # for output Y
    ynormfac=1,
    # length
    train_unet_number=1,
    max_length_Y=1,                 # for Y, X_forImageCondi
    max_text_len=1,                 # for    X_forTextCondi
    # other info
    steps=None,
    e=None,
    sample_dir=None,
    prefix=None,
    IF_showfig=True,
    CKeys=None,
    # TBA to Model B
    normalize_X_cond_to_one=False,
    # ++
    pLM_Model=None, # pLM_Model,
    pLM_Model_Name=None, # pLM_Model_Name,
    image_channels=None, # image_channels,
    pLM_alphabet=None, # esm_alphabet,
):
    # -----------
    # steps=0
    # e=flag

    # --
    # print (f"Producing {len(X)} samples...")
    # ++
    if X!=None:
        print (f"Producing {len(X)} samples...from text conditioning X...")
        lenn_val=len(X)
    if X_string!=None:
        lenn_val=len(X_string)
        print (f"Producing {len(X_string)} samples...from text conditioning X_String (from string)...")
    if x_data!=None:
        print (f"Producing {len(x_data)} samples...from image conditingig x_data  ...")
        lenn_val=len(x_data)
        # print (x_data)
    
    print ('Device: ', model.device)
    
    pdb_list=[]
    fasta_list=[]
    
    for iisample in range (lenn_val):
        print(f"Working on {iisample}")
        X_cond=None
        
        if X_string==None and X!=None: # for X channel
            X_cond=torch.Tensor (X[iisample]).to(device).unsqueeze (0)
        if X_string!=None: # from raw text, ie., X_string: need tokenizer_X and Xnormfac
            # -
            # X = tokenizer_X.texts_to_sequences(X_string[iisample])
            # X= sequence.pad_sequences(X,  maxlen=max_length, padding='post', truncating='post')  
            # X=np.array(X)
            # X_cond=torch.from_numpy(X).float()/Xnormfac
            # +
            XX = tokenizer_X_forTextCondi.texts_to_sequences(X_string[iisample])
            XX = sequence.pad_sequences(XX,  maxlen=max_text_len, padding='post', truncating='post')  
            XX = np.array(XX)
            X_cond = torch.from_numpy(XX).float()/Xnormfac_forTextCondi
            
            print ('Tokenized and processed: ', X_cond)
        
        if X_cond!=None:
            if normalize_X_cond_to_one: # used when there is constrain on X_cond.sum()
                X_cond=X_cond/X_cond.sum()
        
            print ("Text conditoning used: ", X_cond, "...sum: ", X_cond.sum(), "cond scale: ", cond_scales)
        else:
            print ("Text conditioning used: None")
        
        # for now, assume image_condi and text_condi can be used at the same time
        if tokenizer_X_forImageCondi==None:
            # ===========================================================
            # condi_image/seq needs no tokenization, like numbers: force_path
            # only normalization needed
            # Based on ModelB:Force_Path
            if x_data!=None:
                x_data_tokenized=torch.from_numpy(x_data[iisample]/Xnormfac_forImageCondi)
                x_data_tokenized=x_data_tokenized.to(torch.float)
                # + for debug:
                if CKeys['Debug_TrainerPack']==1:
                    print("x_data_tokenized dim: ", x_data_tokenized.shape)
                    print("x_data_tokenized dtype: ", x_data_tokenized.dtype)
                    print("test: ", x_data_tokenized!=None)
            else:
                x_data_tokenized=None
                # + for debug:
                if CKeys['Debug_TrainerPack']==1:
                    print("x_data_tokenized and x_data: None")
            
            # model.sample:full arguments
        # self, 
        # x=None, 
        # stop_at_unet_number=1,
        # cond_scale=7.5,
        # # ++
        # x_data=None, # image_condi data
        # skip_steps=None,
        # inpaint_images = None,
        # inpaint_masks = None,
        # inpaint_resample_times = 5,
        # init_images = None,
        # x_data_tokenized=None,
        # tokenizer_X=None,
        # Xnormfac=1.,
        # # -+
        # device=None,
        # max_length=1., # for XandY data, in image/sequence format; NOT for text condition
        # max_text_len=1., # for X data, in text format
            #
            result_embedding=model.sample ( 
                x=X_cond,
                stop_at_unet_number=train_unet_number ,
                cond_scale=cond_scales ,
                x_data=None, 
                # ++
                x_data_tokenized=x_data_tokenized,
                #
                skip_steps=skip_steps,
                inpaint_images = inpaint_images,
                inpaint_masks = inpaint_masks,
                inpaint_resample_times = inpaint_resample_times,
                init_images = init_images,
                device=model.device,
                # ++++++++++++++++++++++++++
                tokenizer_X=tokenizer_X_forImageCondi, # tokenizer_X,
                Xnormfac=Xnormfac_forImageCondi, # Xnormfac,
                # ynormfac=ynormfac, 
                max_length=max_length_Y, # for ImageCondi, max_length,
                max_text_len=max_text_len,
            )
        else:
            # this is for model B in the future
            # two channels should be provided: raw cond_img+img_tokenizer or tokenized_cond_img
            # need to BE UPDATE and merge with code from 
            # fun.sample_sequence_omegafold_pLM_ModelB
            # one branch is currently missing
            result_embedding=model.sample ( 
                x=X_cond,
                stop_at_unet_number=train_unet_number ,
                cond_scale=cond_scales ,
                x_data=x_data[iisample],  
                # ++
                x_data_tokenized=None,
                #
                skip_steps=skip_steps,
                inpaint_images = inpaint_images,
                inpaint_masks = inpaint_masks,
                inpaint_resample_times = inpaint_resample_times,
                init_images = init_images,
                device=model.device,
                # ++++++++++++++++++++++++++
                tokenizer_X=tokenizer_X_forImageCondi, # tokenizer_X,
                Xnormfac=Xnormfac_forImageCondi, # Xnormfac,
                # ynormfac=ynormfac, 
                max_length=max_length_Y, # max_length,
                max_text_len=max_text_len,
            )
            
        # # ------------------------------------------    
        # result=model.sample ( 
        #     X_cond,
        #     stop_at_unet_number=train_unet_number,
        #     cond_scale=cond_scales 
        # )
       
    
        # # -----------------------------------------------
        # result=torch.round(result*ynormfac)
        # +++++++++++++++++++++++++++++++++++++++++++++++
        # ++ for pLM
        # full record
        # result_embedding as image.dim: [batch, channels, seq_len]
        # result_tokens.dim: [batch, seq_len]
        result_tokens,result_logits = convert_into_tokens(
            pLM_Model, 
            result_embedding,
            pLM_Model_Name,
        )
        result=result_tokens.unsqueeze(1) # dim: [batch, 1, seq_len]
        
        # + for debug
        print("result.dim: ", result.shape)
        
        fig=plt.figure()
        plt.plot (
            result[0,0,:].cpu().detach().numpy(),
            label= f'Predicted'
        )
        #plt.plot (GT[samples,0,:]*ynormfac,label= f'GT {0}')
        plt.legend()
        outname = sample_dir+ f"sampled_from_X_{iisample}_condscale-{str (cond_scales)}_{e}_{steps}.jpg"
        #plt.title (f"Sample {samples}, cond scale={str (cond_scales[iisample])}")
        if IF_showfig==1:
            plt.show ()
        else:
            plt.savefig(outname, dpi=200)
        plt.close()
        
        # # ----------------------------------------
        # plt.plot (result[0,0,:].cpu().detach().numpy(),label= f'Predicted')
        # plt.legend()
        # outname = prefix+ f"sampld_from_X_{flag}_condscale-{str (cond_scales)}_{e}_{steps}.jpg"
        # plt.savefig(outname, dpi=200)
        # plt.show ()
        
#         # ---------------------------------------------------------
#         to_rev=result[:,0,:]
#         to_rev=to_rev.long().cpu().detach().numpy()
#         print("to_rev.dim: ", to_rev.shape)
#         y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

#         for iii in range (len(y_data_reversed)):
#             y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        to_rev=result[:,0,:]
        # the following fun decides ending automatically
        y_data_reversed=decode_many_ems_token_rec_for_folding(
            to_rev,
            result_logits,
            pLM_alphabet,
            pLM_Model,
        )
        if CKeys['Debug_TrainerPack']==3:
            print("on y_data_reversed[0]: ", y_data_reversed[0])
        
        
        # + from Model B
        ### reverse second structure input....
        
        #
        if X_cond != None: 
            # there is condi_text
            if X_string!=None:
                X_cond=torch.round(X_cond*Xnormfac_forTextCondi)

                to_rev=X_cond[:,:] 
                to_rev=to_rev.long().cpu().detach().numpy()
                print ("to_rev.dim: ", to_rev.shape)
                # --
                # X_data_reversed=tokenizer_X.sequences_to_texts (to_rev)
                # ++
                X_text_reversed=tokenizer_X_forTextCondi.sequences_to_texts (to_rev)
                for iii in range (len(y_text_reversed)):
                    X_text_reversed[iii]=X_text_reversed[iii].upper().strip().replace(" ", "")
                    
            if X_string==None:
                # reverse this: X_cond=torch.Tensor (X[iisample]).to(device).unsqueeze (0)
                X_text_reversed=X_cond
        else:
            X_text_reversed=None
                
        if x_data !=None: # there is condi_image
            x_data_reversed=x_data #is already in sequence fromat..
        else:
            x_data_reversed=None
        
        # summary
        print (f"For {X_text_reversed} or {X[iisample]} on Text_Condi,\n and {x_data_reversed} on Image_Condi,\n predicted sequence: ", y_data_reversed)
        
        # + for debug
        print("================================================")
        print("foldproteins: ", foldproteins)
        
        if not foldproteins:
            pdb_file=None
        else:
        # if foldproteins:
            
            if X_cond != None:
                pass
            
            tempname='temp'
            pdb_file, fasta_file=foldandsavePDB_pdb_fasta (
                sequence=y_data_reversed[0], 
                filename_out=tempname, 
                num_cycle=num_cycle, 
                flag=flag,
                # +++++++++++++++++++
                # prefix=prefix,
                prefix=sample_dir,
            )
            #
            out_nam=iisample
            #
            # out_nam_fasta=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.fasta'
            # write_fasta (y_data_reversed[0], out_nam_fasta) 
            #
            out_nam=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.pdb'
            out_nam_fasta=f'{sample_dir}DeNovoSampling_{iisample}_epo_{e}_step_{steps}.fasta'
            shutil.copy (pdb_file, out_nam) #source, dest
            shutil.copy (fasta_file, out_nam_fasta)
            # clean the slade to avoid mistakenly using the previous fasta file
            os.remove (pdb_file)
            os.remove (fasta_file)
            #
            pdb_file=out_nam
            fasta_file=out_nam_fasta
            #
            pdb_list.append(pdb_file)
            fasta_list.append(fasta_file)
            #
            print (f"Properly named PDB file produced: {pdb_file}")
            if IF_showfig==1:
                #flag=1000
                view=show_pdb(
                    pdb_file=pdb_file, 
                    flag=flag,
                    show_sidechains=show_sidechains, 
                    show_mainchains=show_mainchains, 
                    color=color
                )
                view.show()
                
            if calc_error:
                if CKeys['Problem_ID']==7:
                    # only work for ModelA:SecStr
                    get_Model_A_error (pdb_file, X[iisample], plotit=True)
                else:
                    print("Error calculation on the predicted results is not applicable...")
                
        
    
    return pdb_list, fasta_list

    
# + TBU
# ++++++++++++++++++++++++++++++++++++++++++++++++
def sample_loop_omegafold_ModelA (
    model,
    train_loader,
    cond_scales=None, # [7.5], #list of cond scales - each sampled...
    num_samples=None, # 2, #how many samples produced every time tested.....
    timesteps=None, # 100, # not used
    flag=None, # 0,
    foldproteins=False,
    #
    cond_image=False, # use_text_embedd=True,
    cond_text=True, 
    skip_steps=0,
    #
    max_text_len=None,
    max_length=None,
    # +++++++++++++++++++
    train_unet_number=1,
    ynormfac=None,
    prefix=None,
    tokenizer_y=None,
    Xnormfac_CondiText=1,
    tokenizer_X_CondiText=None,
    # ++
    CKeys=None,
    sample_dir=None,
    steps=None,
    e=None,
    IF_showfig=True, # effective only after foldproteins=True
):
    # =====================================================
    # sample # = num_samples*(# of mini-batches)
    # =====================================================
    # steps=0
    # e=flag
    # for item  in train_loader:
    for idx, item  in enumerate(train_loader):

        X_train_batch= item[0].to(device)
        y_train_batch=item[1].to(device)

        GT=y_train_batch.cpu().detach() 

        GT= GT.unsqueeze(1)
        if num_samples>y_train_batch.shape[0]:
            print("Warning: sampling # > len(mini_batch)")

        num_samples = min (num_samples,y_train_batch.shape[0] )
        print (f"Producing {num_samples} samples...")
        X_train_batch_picked = X_train_batch[:num_samples,:]
        print ('(TEST) X_batch shape: ', X_train_batch_picked.shape)

        # loop over cond_scales:list
        for iisample in range (len (cond_scales)):

            # ++ for model A
            result=model.sample (
                x=X_train_batch_picked,
                stop_at_unet_number=train_unet_number,
                cond_scale=cond_scales[iisample],
                #
                skip_steps=skip_steps,
                device=model.device,
                #
                max_length=max_length,
                max_text_len=max_text_len,
            )
            # # ++ for model B
            # if use_text_embedd:
            #     result=model.sample (
            #         # x= X_train_batch,
            #         x= X_train_batch_picked,
            #         stop_at_unet_number=train_unet_number ,
            #         cond_scale=cond_scales[iisample], 
            #         device=device, 
            #         skip_steps=skip_steps
            #     )
            # else:
            #     result=model.sample (
            #         x= None, 
            #         # x_data_tokenized= X_train_batch,
            #         x_data_tokenized= X_train_batch_picked,
            #         stop_at_unet_number=train_unet_number ,
            #         cond_scale=cond_scales[iisample],
            #         device=device,
            #         skip_steps=skip_steps
            #     )
        
            result=torch.round(result*ynormfac)
            GT=torch.round (GT*ynormfac)

            for samples in range  (num_samples):
                print ("sample ", samples+1, "out of ", num_samples)

                fig=plt.figure()
                plt.plot (
                    result[samples,0,:].cpu().detach().numpy(),
                    label= f'Predicted'
                )
                plt.plot (
                    GT[samples,0,:],
                    label= f'GT {0}'
                )
                plt.legend()
                outname = sample_dir+ f"Batch_{idx}_sample_{samples}_condscale-{str (cond_scales[iisample])}_{e}_{steps}.jpg"
                if IF_showfig==1:
                    plt.show()
                else:
                    plt.savefig(outname, dpi=200)
                plt.close ()

                #reverse y sequence
                to_rev=result[:,0,:]
                to_rev=to_rev.long().cpu().detach().numpy()

                y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

                for iii in range (len(y_data_reversed)):
                    y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")

                #reverse GT_y sequence
                to_rev=GT[:,0,:]
                to_rev=to_rev.long().cpu().detach().numpy()

                GT_y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

                for iii in range (len(y_data_reversed)):
                    GT_y_data_reversed[iii]=GT_y_data_reversed[iii].upper().strip().replace(" ", "")

                ### reverse second structure input....
                # pay attension to the shape of Xnormfac
                # -
                # to_rev=torch.round (X_train_batch[:,:]*Xnormfac_CondiText)
                # +
                to_rev=torch.round (X_train_batch[:,:]*torch.FloatTensor(Xnormfac_CondiText).to(model.device))
                to_rev=to_rev.long().cpu().detach().numpy()
                
                # ++ different input
                if CKeys['Debug_TrainerPack']==1:
                    print("tokenizer_X_CondiText: ", tokenizer_X_CondiText)
                    print("Xnormfac_CondiText: ", Xnormfac_CondiText)
                    
                if tokenizer_X_CondiText!=None:
                    X_data_reversed=tokenizer_X_CondiText.sequences_to_texts (to_rev)
                    for iii in range (len(y_data_reversed)):
                        X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")
                else:
                    X_data_reversed=to_rev.copy()
                # + for debug
                if CKeys['Debug_TrainerPack']==1:
                    print("X_data_reversed: ", X_data_reversed)
                

                print (f"For {X_train_batch[samples,:].cpu().detach().numpy()} or {X_data_reversed[samples]}, \npredicted sequence: ", y_data_reversed[samples])
                print (f"Ground truth: {GT_y_data_reversed[samples]}")

                if foldproteins:
                    xbc=X_train_batch[samples,:].cpu().detach().numpy()
                    out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})
                    tempname='temp'
                    pdb_file=foldandsavePDB (
                        sequence=y_data_reversed[samples], 
                        filename_out=tempname, 
                        num_cycle=16, flag=flag,
                        # +++++++++++++++++++
                        prefix=prefix
                    )

                    # #out_nam=f'{prefix}{out_nam}.pdb'
                    # out_nam=f'{prefix}{X_data_reversed[samples]}.pdb'
                    # ------------------------------------------------------
                    # sometime, this name below can get too long to fit
                    # out_nam=f'{sample_dir}{X_data_reversed[samples]}.pdb'
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # add a way to save the sampling name and results
                    # ref: outname = sample_dir+ f"sample-{samples}_condscale-{str (cond_scales[iisample])}_{e}_{steps}.jpg"
                    out_nam=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}.pdb'
                    out_nam_inX=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}.txt'
                    
                    if CKeys['Debug_TrainerPack']==1:
                        print("pdb_file: ", pdb_file)
                        print("out_nam: ", out_nam)
                        
                    print (f'Original PDB: {pdb_file} OUT: {out_nam}')
                    shutil.copy (pdb_file, out_nam) #source, dest
                    # +
                    with open(out_nam_inX, "w") as inX_file:
                        inX_file.write(f'{X_data_reversed[samples]}\n')
                        
                    pdb_file=out_nam
                    print (f"Properly named PDB file produced: {pdb_file}")
                    print (f"input X for sampling stored: {pdb_file}")
                    
                    if IF_showfig==1:
                        view=show_pdb(
                            pdb_file=pdb_file, 
                            flag=flag, 
                            show_sidechains=show_sidechains,  
                            show_mainchains=show_mainchains, 
                            color=color
                        )
                        view.show()

#                 steps=steps+1
                
#         if steps>num_samples:
#             break

# + TBU
# ++++++++++++++++++++++++++++++++++++++++++++++++
def sample_loop_omegafold_pLM_ModelA (
    model,
    train_loader,
    cond_scales=None, # [7.5], #list of cond scales - each sampled...
    num_samples=None, # 2, #how many samples produced every time tested.....
    timesteps=None, # 100, # not used
    flag=None, # 0,
    foldproteins=False,
    #
    cond_image=False, # use_text_embedd=True,
    cond_text=True, 
    skip_steps=0,
    #
    max_text_len=None,
    max_length=None,
    # +++++++++++++++++++
    train_unet_number=1,
    ynormfac=None,
    prefix=None,
    tokenizer_y=None,
    Xnormfac_CondiText=1,
    tokenizer_X_CondiText=None,
    # ++
    CKeys=None,
    sample_dir=None,
    steps=None,
    e=None,
    IF_showfig=True, # effective only after foldproteins=True
    # ++ for pLM
    pLM_Model=None,
    pLM_Model_Name=None,
    image_channels=None,
    pLM_alphabet=None,
    # ++ for on-fly check: for SecStr only
    calc_error=False,      # for check on folded results, not used for every case
):
    # =====================================================
    # sample # = num_samples*(# of mini-batches)
    # =====================================================
    # steps=0
    # e=flag
    # for item  in train_loader:
    for idx, item  in enumerate(train_loader):

        X_train_batch= item[0].to(device)
        y_train_batch=item[1].to(device)

        GT=y_train_batch.cpu().detach() 

        GT= GT.unsqueeze(1)
        if num_samples>y_train_batch.shape[0]:
            print("Warning: sampling # > len(mini_batch)")

        num_samples = min (num_samples,y_train_batch.shape[0] )
        print (f"Producing {num_samples} samples...")
        X_train_batch_picked = X_train_batch[:num_samples,:]
        print ('(TEST) X_batch shape: ', X_train_batch_picked.shape)

        # loop over cond_scales:list
        for iisample in range (len (cond_scales)):

            # ++ for model A
            result_embedding = model.sample (
                x=X_train_batch_picked,
                stop_at_unet_number=train_unet_number,
                cond_scale=cond_scales[iisample],
                #
                skip_steps=skip_steps,
                device=model.device,
                #
                max_length=max_length,
                max_text_len=max_text_len,
                #
                x_data=None,
                x_data_tokenized=None,
                #
                tokenizer_X=tokenizer_X_CondiText,
                Xnormfac=Xnormfac_CondiText,
            )
            # # ++ for model B
            # if use_text_embedd:
            #     result=model.sample (
            #         # x= X_train_batch,
            #         x= X_train_batch_picked,
            #         stop_at_unet_number=train_unet_number ,
            #         cond_scale=cond_scales[iisample], 
            #         device=device, 
            #         skip_steps=skip_steps
            #     )
            # else:
            #     result=model.sample (
            #         x= None, 
            #         # x_data_tokenized= X_train_batch,
            #         x_data_tokenized= X_train_batch_picked,
            #         stop_at_unet_number=train_unet_number ,
            #         cond_scale=cond_scales[iisample],
            #         device=device,
            #         skip_steps=skip_steps
            #     )
            
            # ++ for pLM:
            # full record
            # result_embedding as image.dim: [batch, channels, seq_len]
            # result_tokens.dim: [batch, seq_len]
            result_tokens,result_logits = convert_into_tokens(
                pLM_Model, 
                result_embedding,
                pLM_Model_Name,
            )
        
            # # --------------------------------------------
            # result=torch.round(result*ynormfac)
            # GT=torch.round (GT*ynormfac)
            # ++++++++++++++++++++++++++++++++++++++++++++
            result=result_tokens.unsqueeze(1) # dim: [batch, 1, seq_len]
            
#                 # ---------------------------------------------------------
#                 #reverse y sequence
#                 to_rev=result[:,0,:]
#                 to_rev=to_rev.long().cpu().detach().numpy()

#                 y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

#                 for iii in range (len(y_data_reversed)):
#                     y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            to_rev=result[:,0,:] # token (batch,seq_len)
            y_data_reversed=decode_many_ems_token_rec_for_folding(
                to_rev,
                result_logits,
                pLM_alphabet,
                pLM_Model,
            )
            if CKeys['Debug_TrainerPack']==3:
                print("on y_data_reversed[0]: ", y_data_reversed[0])
                

#                 # -----------------------------------------------------------
#                 #reverse GT_y sequence
#                 to_rev=GT[:,0,:]
#                 to_rev=to_rev.long().cpu().detach().numpy()

#                 GT_y_data_reversed=tokenizer_y.sequences_to_texts (to_rev)

#                 for iii in range (len(y_data_reversed)):
#                     GT_y_data_reversed[iii]=GT_y_data_reversed[iii].upper().strip().replace(" ", "")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #reverse GT_y sequence
            # GT should be SAFE to reverse
            to_rev=GT[:,0,:]
            GT_y_data_reversed=decode_many_ems_token_rec(
                to_rev,
                pLM_alphabet,
            )

            ### reverse second structure input....
            # pay attension to the shape of Xnormfac
            # -
            # to_rev=torch.round (X_train_batch[:,:]*Xnormfac_CondiText)
            # +
            # print("X_train_batch", X_train_batch)
            # print("Xnormfac_CondiText: ", Xnormfac_CondiText)

            # to_rev=torch.round (X_train_batch[:,:]*torch.FloatTensor(Xnormfac_CondiText).to(model.device))
            # print("X_train_batch: ", X_train_batch[:,:])
            # print("torch.tensor(Xnormfac_CondiText): ", torch.tensor(Xnormfac_CondiText))
            to_rev=X_train_batch[:,:]*torch.tensor(Xnormfac_CondiText).to(model.device)
            # print("to_rev ", to_rev)
            # # -: convert into int64
            # to_rev=to_rev.long().cpu().detach().numpy()
            # +: just float
            to_rev=to_rev.cpu().detach().numpy()
            # print("to_rev 2", to_rev)

            # ++ different input
            if CKeys['Debug_TrainerPack']==1:
                print("tokenizer_X_CondiText: ", tokenizer_X_CondiText)
                print("Xnormfac_CondiText: ", Xnormfac_CondiText)

            if tokenizer_X_CondiText!=None:
                # round the number into tokens
                to_rev = np.round(to_rev)
                X_data_reversed=tokenizer_X_CondiText.sequences_to_texts (to_rev)
                for iii in range (len(y_data_reversed)):
                    X_data_reversed[iii]=X_data_reversed[iii].upper().strip().replace(" ", "")
            else:
                X_data_reversed=to_rev.copy()
            # + for debug
            if CKeys['Debug_TrainerPack']==1:
                print("X_data_reversed: ", X_data_reversed)
                print("X_data_reversed.dim: ", X_data_reversed.shape)

            for samples in range  (num_samples):
                print ("sample ", samples+1, "out of ", num_samples)

                fig=plt.figure()
                plt.plot (
                    result[samples,0,:].cpu().detach().numpy(),
                    label= f'Predicted'
                )
                plt.plot (
                    GT[samples,0,:],
                    label= f'GT {0}'
                )
                plt.legend()
                outname = sample_dir+ f"Batch_{idx}_sample_{samples}_condscale-{str (cond_scales[iisample])}_{e}_{steps}.jpg"
                if IF_showfig==1:
                    plt.show()
                else:
                    plt.savefig(outname, dpi=200)
                plt.close ()
                

                print (f"For input in dataloader: {X_train_batch[samples,:].cpu().detach().numpy()} or \n recovered input {X_data_reversed[samples]}")
                print (f"predicted sequence: {y_data_reversed[samples]}")
                print (f"Ground truth:       {GT_y_data_reversed[samples]}")

                if foldproteins:
                    # check whether the predicted sequence is valid
                    if len(y_data_reversed[samples])>0:
                        # # --
                        # xbc=X_train_batch[samples,:].cpu().detach().numpy()
                        # # out_nam=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})
                        # out_nam_content=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.1f" % xbc})
                        # ++
                        xbc=X_data_reversed[samples]
                        out_nam_content=np.array2string(xbc, formatter={'float_kind':lambda xbc: "%.4f" % xbc})
                        #
                        tempname='temp'
                        pdb_file,fasta_file=foldandsavePDB_pdb_fasta (
                            sequence=y_data_reversed[samples], 
                            filename_out=tempname, 
                            num_cycle=16, flag=flag,
                            # +++++++++++++++++++
                            prefix=prefix
                        )

                        # #out_nam=f'{prefix}{out_nam}.pdb'
                        # out_nam=f'{prefix}{X_data_reversed[samples]}.pdb'
                        # ------------------------------------------------------
                        # sometime, this name below can get too long to fit
                        # out_nam=f'{sample_dir}{X_data_reversed[samples]}.pdb'
                        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        # add a way to save the sampling name and results
                        # ref: outname = sample_dir+ f"sample-{samples}_condscale-{str (cond_scales[iisample])}_{e}_{steps}.jpg"
                        out_nam=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}.pdb'
                        out_nam_seq=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}.fasta'
                        out_nam_inX=f'{sample_dir}SamplingLoop_B_{idx}_Sample_{samples}_condscale-{str (cond_scales[iisample])}_epo_{e}_step_{steps}.txt'

                        if CKeys['Debug_TrainerPack']==1:
                            print("pdb_file: ", pdb_file)
                            print("out_nam: ", out_nam)

                        print (f'Original PDB: {pdb_file} OUT: {out_nam}')
                        shutil.copy (pdb_file, out_nam) #source, dest
                        shutil.copy (fasta_file, out_nam_seq)
                        # +
                        with open(out_nam_inX, "w") as inX_file:
                            # inX_file.write(f'{X_data_reversed[samples]}\n')
                            inX_file.write(out_nam_content)
                        # clean the slade to avoid mistakenly using the previous fasta file
                        os.remove (pdb_file)
                        os.remove (fasta_file)


                        pdb_file=out_nam
                        print (f"Properly named PDB file produced: {pdb_file}")
                        print (f"input X for sampling stored: {pdb_file}")

                        if IF_showfig==1:
                            view=show_pdb(
                                pdb_file=pdb_file, 
                                flag=flag, 
                                show_sidechains=show_sidechains,  
                                show_mainchains=show_mainchains, 
                                color=color
                            )
                            view.show()
                            
                        if calc_error:
                            print('On-fly check...')
                            if CKeys['Problem_ID']==7:
                                # only work for ModelA:SecStr
                                get_Model_A_error (pdb_file, X_data_reversed[samples], plotit=True)
                            else:
                                print("Error calculation on the predicted results is not applicable...")
                    
                            
                    else:
                        print("The predicted sequence is EMPTY...")
                        
