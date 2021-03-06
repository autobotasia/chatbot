"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.bin import t2t_datagen
import problems
import tensorflow as tf


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.compat.v1.app.run(t2t_datagen.main)