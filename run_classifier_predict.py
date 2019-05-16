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

import numpy as np

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

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

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
  init_checkpoint = FLAGS.init_checkpoint

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
      predict_batch_size=FLAGS.predict_batch_size)

  predict_examples = processor.get_test_examples(FLAGS.data_dir)
  num_actual_predict_examples = len(predict_examples)
  if FLAGS.use_tpu:
    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on.
    while len(predict_examples) % FLAGS.predict_batch_size != 0:
      predict_examples.append(PaddingInputExample())

  records_dir = FLAGS.records_dir
  if records_dir is None:
    records_dir = FLAGS.data_dir

  predict_file = os.path.join(records_dir, "predict.tf_record")
  if os.path.exists(predict_file):
    tf.logging.info("predict file (%s) already exists -- not overwriting",
                    predict_file)
  else:
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

  tf.logging.info("***** Running prediction *****")
  tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                  len(predict_examples), num_actual_predict_examples,
                  len(predict_examples) - num_actual_predict_examples)
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  predict_drop_remainder = True if FLAGS.use_tpu else False
  predict_input_fn = file_based_input_fn_builder(
      input_file=predict_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=predict_drop_remainder)

  result = estimator.predict(input_fn=predict_input_fn)

  # Write predictions and text
  predictions_file = os.path.join(FLAGS.output_dir, "predictions.tsv")
  with tf.gfile.GFile(predictions_file, "w") as writer:
    num_written_lines = 0
    tf.logging.info("***** Predict results *****")
    if predict_examples[0].text_b is None:
      tsv_header = ("TextA", "Predicted", "Actual", "Probability")
    else:
      tsv_header = ("TextA", "TextB", "Predicted", "Actual", "Probability")

    writer.write("\t".join(tsv_header) + "\n")
    for (i, prediction) in enumerate(result):
      if i % (num_actual_predict_examples // 10) == 0:
        tf.logging.info("Processing %d/%d", i, num_actual_predict_examples)

      if i >= num_actual_predict_examples:
        break

      probabilities = prediction["probabilities"]

      # Text A
      tsv_elements = [predict_examples[i].text_a]
      if predict_examples[i].text_b is not None:
        # (Optional) Text B
        tsv_elements.append(predict_examples[i].text_b)

      # Predicted label
      predicted_idx = np.argmax(probabilities)
      predicted_label = label_list[predicted_idx]
      tsv_elements.append(predicted_label)

      # Actual label
      tsv_elements.append(predict_examples[i].label)

      # Predicted probability
      tsv_elements.append(str(probabilities[predicted_idx]))

      output_line = "\t".join(tsv_elements) + "\n"
      writer.write(output_line)
      num_written_lines += 1
  assert num_written_lines == num_actual_predict_examples

  # # Write raw probabilities
  # output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
  # with tf.gfile.GFile(output_predict_file, "w") as writer:
  #   num_written_lines = 0
  #   tf.logging.info("***** Predict results *****")
  #   for (i, prediction) in enumerate(result):
  #     probabilities = prediction["probabilities"]
  #     if i >= num_actual_predict_examples:
  #       break
  #     output_line = "\t".join(
  #         str(class_probability)
  #         for class_probability in probabilities) + "\n"
  #     writer.write(output_line)
  #     num_written_lines += 1
  # assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("records_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
