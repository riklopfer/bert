# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import glob
import os

import numpy as np
import tensorflow as tf

import modeling
import optimization
import tokenization

from run_classifier_common import *

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "records_dir", None,
    "TF Records will be written and read from this directory. ")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("placeholders_file", None,
                    "The vocabulary file containing placeholder tokens which "
                    "appear in the training text.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "keep_all_checkpoints", False,
    "Eval option: if set to False, remove all checkpoints except for the best "
    "one.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)


  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in PROCESSORS:
    raise ValueError("Task not found: %s" % (task_name))

  processor = PROCESSORS[task_name]()

  label_list = processor.get_labels(FLAGS.data_dir)
  negative_label_idx = None
  if processor.get_negative_label() is not None:
    negative_label_idx = label_list.index(processor.get_negative_label())

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file,
      do_lower_case=FLAGS.do_lower_case,
      placeholders_file=FLAGS.placeholders_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  save_checkpoint_steps = None

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=save_checkpoint_steps,
      keep_checkpoint_max=100,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  # Default
  model_checkpoints = [FLAGS.init_checkpoint]

  if os.path.isdir(FLAGS.init_checkpoint):
    ckpt = tf.train.get_checkpoint_state(FLAGS.init_checkpoint)
    model_checkpoints = ckpt.all_model_checkpoint_paths

  # best checkpoint is determined by Overall F1
  best_checkpoint = None
  best_f1 = None
  for epoch_n, init_checkpoint in enumerate(model_checkpoints):
    tf.logging.info("\n\n"
                    "********************\n"
                    "Epoch N: %d\n"
                    "Initial Checkpoint: %s\n"
                    "********************\n",
                    epoch_n, init_checkpoint)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        negative_label_idx=negative_label_idx,
        init_checkpoint=init_checkpoint,
        learning_rate=None,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.records_dir, "eval.tf_record")
    if os.path.exists(eval_file):
      tf.logging.info("eval file (%s) already exists -- not overwriting",
                      eval_file)
    else:
      file_based_convert_examples_to_features(
          eval_examples, label_list, FLAGS.max_seq_length, tokenizer,
          eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    # Compute Precision, Recall, and F1
    exclude_ids = set()
    if processor.get_negative_label() is not None:
      exclude_ids.add(label_list.index(processor.get_negative_label()))

    total_tp, total_fp, total_fn = 0., 0., 0.
    total_f1 = 0.
    for label_id, label in enumerate(label_list):
      # use the label id here
      true_pos = result["{}_TP".format(label_id)]
      false_pos = result["{}_FP".format(label_id)]
      false_neg = result["{}_FN".format(label_id)]

      # Exclude negative label from overall metric
      if label_id in exclude_ids:
        tf.logging.info("Excluding '%s' from metrics", label)
        continue

      total_tp += true_pos
      total_fp += false_pos
      total_fn += false_neg

      if true_pos == 0:
        result["{} Precision".format(label)] = "{:.3%}".format(0)
        result["{} Recall".format(label)] = "{:.3%}".format(0)
        result["{} F1".format(label)] = "{:.3%}".format(0)
      else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 2 * precision * recall / (precision + recall)
        total_f1 += f1
        result["{} Precision".format(label)] = "{:.3%}".format(precision)
        result["{} Recall".format(label)] = "{:.3%}".format(recall)
        result["{} F1".format(label)] = "{:.3%}".format(f1)

    # Compute Overall F1
    if total_tp == 0:
      result["Overall Precision"] = "{:.3%}".format(0)
      result["Overall Recall"] = "{:.3%}".format(0)
      result["Overall F1"] = "{:.3%}".format(0)
    else:
      precision = total_tp / (total_tp + total_fn)
      recall = total_tp / (total_tp + total_fp)
      f1 = 2 * precision * recall / (precision + recall)
      result["Overall Precision"] = "{:.3%}".format(precision)
      result["Overall Recall"] = "{:.3%}".format(recall)
      result["Overall F1"] = "{:.3%}".format(f1)

    # Update best_checkpoint
    if best_checkpoint is None or result["Overall F1"] > best_f1:
      best_checkpoint = init_checkpoint
      best_f1 = result["Overall F1"]

    # Cannot use '_' or else 'Average' will be treated as int
    result["Average F1"] = '{:.3%}'.format(total_f1 / len(label_list))

    output_eval_file = os.path.join(FLAGS.output_dir,
                                    "eval_results_{}.txt".format(epoch_n))
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results for epoch %d *****", epoch_n)
      writer.write("Epochs = {}\n\n".format(epoch_n))
      for key in sorted(result.keys()):
        value = result[key]

        if key.endswith("_TP") or key.endswith("_FP") or key.endswith("_FN"):
          label_id, metric_name = key.rsplit("_", 1)
          key = "{} {}".format(label_list[int(label_id)], metric_name)
          # value = "{:.3%}".format(value)

        tf.logging.info("  %s = %s", key, str(value))
        writer.write("%s = %s\n" % (key, str(value)))

  tf.logging.info("Best checkpoint: %s", best_checkpoint)
  if not FLAGS.keep_all_checkpoints:
    for init_checkpoint in model_checkpoints:
      if init_checkpoint != best_checkpoint:
        for fname in glob.glob(init_checkpoint + "*"):
          tf.logging.info("Removing bad checkpoint file: %s", fname)
          os.remove(fname)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("records_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
