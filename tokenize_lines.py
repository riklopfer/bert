#!/usr/bin/env python2.7
# coding=utf-8
"""script.py



Author(s): rklopfer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import mmpy
import tensorflow as tf

import tokenization


def main(argv):
  # parse args
  parser = argparse.ArgumentParser(prog=argv[0],
                                   description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)

  inputs_group = parser.add_argument_group('Input methods',
                                           'Must supply at least one of these')
  inputs_group.add_argument('inputs',
                            help='Directory or file to be processed',
                            nargs='*', type=unicode)
  inputs_group.add_argument('-l', '--list',
                            help='Newline delimited list of files to process',
                            type=unicode, default=None)
  parser.add_argument('--vocab_file',
                      help='Vocabulary file. ',
                      type=unicode, required=True)

  parser.add_argument('--placeholders_file',
                      help='Vocabulary of placeholder tokens only. ',
                      type=unicode, default=None)


  parser.add_argument('--do_lower_case',
                      help='Lowercase input',
                      action='store_true')

  parser.add_argument('-o', '--out',
                      help='Write output to this directory. ',
                      type=unicode, default=None)

  parser.add_argument("-v", "--verbosity", action="count",
                      help="Increase output verbosity", default=0)

  args = parser.parse_args(argv[1:])

  # Initialize logging
  global logger
  logger = mmpy.get_logger(verbosity=args.verbosity)

  # Extract ars from parser
  out_dir = args.out

  inputs = args.inputs
  list_input = args.list
  vocab_file = args.vocab_file
  do_lower_case = args.do_lower_case
  placeholders_file = args.placeholders_file

  if not (inputs or list_input):
    parser.print_help()
    raise AssertionError("You've gotta provide some input! "
                         "Either 'inputs' or '--list'")
  file_paths = []
  for search_path in inputs:
    file_paths.extend(mmpy.find_all_files(search_path))

  if list_input is not None:
    file_paths.extend(mmpy.read_lines(list_input))

  if not file_paths:
    raise Exception("Couldn't find any input files.")

  tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case, placeholders_file)

  for path_x, file_path in enumerate(file_paths):
    logger.info(
        "Processing %d/%d :: %s" % (path_x + 1, len(file_paths), file_path))

    with tf.gfile.GFile(file_path) as gfile:
      line = gfile.readline()
      while line is not None:
        line = line.strip()
        tokens = tokenizer.tokenize(line)

        logger.info("\nLINE:   %s"
                    "\nTOKENS: %s",
                    line, " ".join(tokens))
        line = gfile.readline()


if __name__ == "__main__":
  sys.exit(main(sys.argv))
