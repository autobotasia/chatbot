"""Back Translation to augment a dataset."""

from __future__ import print_function
from __future__ import division

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf
import os
import json
import gzip
import numpy as np
import re


@registry.register_problem
class Chatbot(text_problems.Text2TextProblem):
  """Problem spec for coresystem chatbot."""

  @property
  def approx_vocab_size(self):
    return 2**16  # 65536 

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def is_generate_per_split(self):
    return False  

  def gen_data(self, data_dir, tmp_dir):
    data = []
    with open(os.path.join("./utils", "data-nq-filtered.json"), 'r') as f:
      data = json.load(f)

    for example in data:
      yield example   

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    for example in self.gen_data(data_dir, tmp_dir):
      yield {
          "inputs": example["input"],
          "targets": example["target"]
      }
     



