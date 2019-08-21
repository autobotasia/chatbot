"""Back Translation to augment a dataset."""

from __future__ import print_function
from __future__ import division

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf
import os
import unidecode


@registry.register_problem
class Chatbot(text_problems.Text2TextProblem):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**18  # 32768

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 9,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def is_generate_per_split(self):
    return False  

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    #del data_dir
    del tmp_dir
    del dataset_split

    sPath = os.path.join(data_dir, 'source.txt')
    tPath = os.path.join(data_dir, 'target.txt')

    # Open the files and yield source-target lines.
    with tf.gfile.GFile(sPath, mode='r') as source_file:
      with tf.gfile.GFile(tPath, mode='r') as target_file:
        source, target = source_file.readline(), target_file.readline()
        while source and target:
          yield {'inputs': source.strip(), 'targets': target.strip()}
          source, target = source_file.readline(), target_file.readline()


