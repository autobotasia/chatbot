"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.bin import t2t_decoder
from tensor2tensor.models import transformer

import decoding
import problems
import tensorflow as tf
import os


registry = problems.registry

tf.flags.DEFINE_string(
    'chatbot_problem', 
    'chatbot', 
    'Problem name for Vietnamese (notone) to Vietnamese translation.')
tf.flags.DEFINE_string(
    'chatbot_data_dir', 
    './data/chatbot', 
    'Data directory for Vietnamese to English translation.')
tf.flags.DEFINE_string(
    'chatbot_ckpt', 
    './checkpoints/chatbot', 
    'Pretrain checkpoint directory for Vietnamese to English translation.')
tf.flags.DEFINE_string(
    'paraphrase_from_file', 
    'test_input.vi', 
    'Input text file to paraphrase.')
tf.flags.DEFINE_string(
    'paraphrase_to_file', 
    'test_output.vi', 
    'Output text file to paraphrase.')
tf.flags.DEFINE_boolean(
    'chatbot_interactively',
    False,
    'Whether to translate interactively.')    

FLAGS = tf.flags.FLAGS
  


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  data_dir = FLAGS.chatbot_data_dir,
  problem_name = FLAGS.chatbot_problem,
  ckpt_path = FLAGS.chatbot_ckpt

  # Convert directory into checkpoints
  if tf.io.gfile.isdir(ckpt_path):
    ckpt_path = tf.train.latest_checkpoint(ckpt_path)

  # For back translation, we need a temporary file in the other language
  # before back-translating into the source language.
  tmp_file = os.path.join(
      '{}.tmp.chatbot.txt'.format(FLAGS.paraphrase_from_file)
  )

  if FLAGS.chatbot_interactively:
    print("%s %s %s" % (data_dir, problem_name, ckpt_path))
    #decoding.chatbot_interactively(problem_name, data_dir, ckpt_path)
    decoding.chatbot_interactively('chatbot', './data/chatbot', ckpt_path)
  else:
    # Step 1: Translating from source language to the other language.
    if not tf.io.gfile.exists(tmp_file):
      decoding.t2t_decoder(problem_name, data_dir,
                         FLAGS.paraphrase_from_file, tmp_file,
                         ckpt_path)
