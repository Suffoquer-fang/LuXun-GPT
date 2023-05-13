# coding=utf-8
import sys
sys.path.append("./")
import logging
import os
from tqdm import tqdm
import json
import random
from random import choices
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np
import torch
import random
import pickle
import argparse
import numpy as np 
import torch
import logging
from datetime import datetime
import gzip
import os
import tarfile
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    set_seed, 
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from transformers.training_args import default_logdir
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter



@dataclass
class MyTrainingArguments(TrainingArguments):
    dataset_path: str = field(default="data/luxun")
    output_dir: str = field(default="output_luxun")
    lora_rank: int = field(default=8)
    
from transformers import TrainerCallback
from transformers.integrations import TensorBoardCallback
import torch

from peft import get_peft_model, LoraConfig, TaskType

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


