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


@registry.register_problem
class Chatbot(text_problems.QuestionAndContext2TextProblem):
  """Problem spec for coresystem chatbot."""

  @property
  def approx_vocab_size(self):
    return 2**16  # 65536 

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 100,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def is_generate_per_split(self):
    return True  

  def gen_data(self, tmp_dir, data_dir):
    data = []
    with open(os.path.join(data_dir, "data.json"), 'r') as f:
      data = json.load(f)

    for example in data:
      yield example
      
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    #for example in self.gen_data(tmp_dir, data_dir):
    #  yield {
    #      "inputs": example["input"],
    #      "targets": example["target"],
    #      "context": example["context"]
    #  }
    filenames = [
      'nq-train-00.jsonl.gz',
      'nq-train-17.jsonl.gz',
      'nq-train-34.jsonl.gz',
      'nq-train-01.jsonl.gz',  
      'nq-train-18.jsonl.gz',  
      'nq-train-35.jsonl.gz',
      'nq-train-02.jsonl.gz',  
      'nq-train-19.jsonl.gz',  
      'nq-train-36.jsonl.gz',
      'nq-train-03.jsonl.gz',  
      'nq-train-20.jsonl.gz',  
      'nq-train-37.jsonl.gz',
      'nq-train-04.jsonl.gz',  
      'nq-train-21.jsonl.gz',  
      'nq-train-38.jsonl.gz',
      'nq-train-05.jsonl.gz',  
      'nq-train-22.jsonl.gz',  
      'nq-train-39.jsonl.gz',
      'nq-train-06.jsonl.gz',  
      'nq-train-23.jsonl.gz',  
      'nq-train-40.jsonl.gz',
      'nq-train-07.jsonl.gz',  
      'nq-train-24.jsonl.gz',  
      'nq-train-41.jsonl.gz',
      'nq-train-08.jsonl.gz',  
      'nq-train-25.jsonl.gz',  
      'nq-train-42.jsonl.gz',
      'nq-train-09.jsonl.gz',  
      'nq-train-26.jsonl.gz',  
      'nq-train-43.jsonl.gz',
      'nq-train-10.jsonl.gz',  
      'nq-train-27.jsonl.gz',  
      'nq-train-44.jsonl.gz',
      'nq-train-11.jsonl.gz',  
      'nq-train-28.jsonl.gz',  
      'nq-train-45.jsonl.gz',
      'nq-train-12.jsonl.gz',  
      'nq-train-29.jsonl.gz',  
      'nq-train-46.jsonl.gz',
      'nq-train-13.jsonl.gz',  
      'nq-train-30.jsonl.gz',  
      'nq-train-47.jsonl.gz',
      'nq-train-14.jsonl.gz',  
      'nq-train-31.jsonl.gz',  
      'nq-train-48.jsonl.gz',
      'nq-train-15.jsonl.gz',  
      'nq-train-32.jsonl.gz',  
      'nq-train-49.jsonl.gz',
      'nq-train-16.jsonl.gz',  
      'nq-train-33.jsonl.gz'
    ]

    for filename in filenames:
      g = gzip.open(os.path.join(data_dir, './natural_questions/v1.0/train/' + filename), 'r')
      for l in g:
        obj = json.loads(l)      
        document_html = obj['document_html'].encode('utf-8')      
        question_text = obj['question_text']
        context = ''
        answer_text = ''      
        anno = obj['annotations'][0]
        has_long_answer = anno['long_answer']['start_byte'] >= 0
        has_short_answer = anno['short_answers'] or anno['yes_no_answer'] != 'NONE'
        long_answers = [
          a['long_answer']
          for a in obj['annotations']
          if a['long_answer']['start_byte'] >=0
        ]
        short_answers = [
          a['short_answers']
          for a in obj['annotations']
          if a['short_answers'] and has_short_answer
        ]
        
        yes_no_answers = [
            a['yes_no_answer']
            for a in obj['annotations']
            if a['yes_no_answer'] != 'NONE' and has_short_answer
        ]

        if has_long_answer and has_short_answer:
          long_answer_bounds = [
            (la['start_byte'], la['end_byte']) for la in long_answers
          ]
          long_answer_counts = [
              long_answer_bounds.count(la) for la in long_answer_bounds
          ]
          long_answer = long_answers[np.argmax(long_answer_counts)]
          context = document_html[long_answer["start_byte"]:long_answer["end_byte"]]
        
          short_answers_ids = [[
              (s['start_byte'], s['end_byte']) for s in a
          ] for a in short_answers] + [a for a in yes_no_answers]
          short_answers_counts = [
              short_answers_ids.count(a) for a in short_answers_ids
          ]
          
          short_answers_texts = [
              ', '.join([
                  "%s"%document_html[s['start_byte']:s['end_byte']]
                  for s in short_answer
              ])
              for short_answer in short_answers
          ]

          short_answers_texts += yes_no_answers
          answer_text = short_answers_texts[np.argmax(short_answers_counts)]

          yield {
            "inputs"   : question_text,
            "targets"  : answer_text,
            "context"  : context
          } 



