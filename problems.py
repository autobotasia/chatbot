"""Back Translation to augment a dataset."""

from __future__ import print_function
from __future__ import division

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf
import os


@registry.register_problem
class Chatbot(text_problems.QuestionAndContext2TextProblem):
  """Problem spec for coresystem chatbot."""

  @property
  def approx_vocab_size(self):
    return 2**18  # 32768 

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
    return True  

  def gen_data(self, tmp_dir, data_dir):
    number_of_lines = 0
    current_dialog_source = ''
    current_dialog_target = ''
    dialog_silenced = False
    # Iterate through the file and build list of dialogs separated by __eou__.
    for line in open(data_dir + '/train_none_original_no_cands.txt', errors='ignore'):
      if number_of_lines % 10000 == 0:
        print('Parsed ' + str(number_of_lines) + ' lines.')

      dialog_id = line.split()[0]
      # Check if this is a refurbished line.
      if ('__SILENCE__' not in line and
              ((dialog_silenced and dialog_id == '1') or not dialog_silenced)):
        dialog_silenced = False
        number_of_lines += 1

        # Get the utterances.
        current_dialog_source = ' '.join(line.split('\t')[0].split()[1:])
        current_dialog_target = line.split('\t')[1].strip('\n')
        
        # Whether this is a new dialog.
        if dialog_id == '1' and current_dialog_source != '':
          yield {
            'input': current_dialog_source,
            'target': current_dialog_target
          }                      
        else:
          current_dialog_source += current_dialog_source + '\n'
          current_dialog_target += current_dialog_target + '\n'

      else:
        dialog_silenced = True

      
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    for example in self.gen_data(tmp_dir, data_dir):
      print(example)
      yield {
          "inputs": example["input"],
          # TODO(ddohan, wgaj): Figure out a way of extracting all answers.
          "targets": example["target"],
          "context": example["input"]
      }



