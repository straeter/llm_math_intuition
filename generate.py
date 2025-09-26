import argparse
import collections
import logging
import textwrap

import six
from mathematics_dataset.generate import _make_entropy_fn
from mathematics_dataset.modules import modules
from mathematics_dataset import generate_settings

from utils.io_utils import save_jsonl


def _filter_and_flatten(modules_, args):
  """Returns flattened dict, filtered according to FLAGS."""
  flat = collections.OrderedDict()

  def add(submodules, prefix=None):
    for key, module_or_function in six.iteritems(submodules):
      full_name = prefix + '__' + key if prefix is not None else key
      if isinstance(module_or_function, dict):
        add(module_or_function, full_name)
      else:
        if args.filter not in full_name:
          continue
        flat[full_name] = module_or_function

  add(modules_)

  # Make sure list of modules are in deterministic order. This is important when
  # generating across multiple machines.
  flat = collections.OrderedDict(
      [(key, flat[key]) for key in sorted(six.iterkeys(flat))])

  return flat


def init_modules(args):
    """Inits the dicts containing functions for generating modules."""
    counts = {}
    filtered_modules = collections.OrderedDict([])

    all_modules = collections.OrderedDict([])
    for j in range(args.n_difficulties):
        all_modules[f"train-{j + 1}_of_{args.n_difficulties}"] = modules.train(_make_entropy_fn(j, args.n_difficulties))
        counts[f"train-{j + 1}_of_{args.n_difficulties}"] = args.n_train // args.n_difficulties
    if args.n_test:
        all_modules['interpolate'] = modules.test()
        all_modules['extrapolate'] = modules.test_extra()
        counts['interpolate'] = args.n_test // 2
        counts['extrapolate'] = args.n_test // 2

    for regime_, modules_ in six.iteritems(all_modules):
        filtered_modules[regime_] = _filter_and_flatten(modules_, args)

    return filtered_modules, counts


def sample_from_module(module, args):
  """Samples a problem, ignoring samples with overly long questions / answers.

  Args:
    module: Callable returning a `Problem`.

  Returns:
    Pair `(problem, num_dropped)`, where `problem` is an instance of `Problem`
    and `num_dropped` is an integer >= 0 indicating the number of samples that
    were dropped.
  """
  num_dropped = 0
  while True:
    problem = module()
    question = str(problem.question)
    if len(question) > generate_settings.MAX_QUESTION_LENGTH:
      num_dropped += 1
      if args.show_dropped:
        logging.warning('Dropping question: %s', question)
      continue
    answer = str(problem.answer)
    if len(answer) > generate_settings.MAX_ANSWER_LENGTH:
      num_dropped += 1
      if args.show_dropped:
        logging.warning('Dropping question with answer: %s', answer)
      continue
    return problem, num_dropped

def main(args):
    """Prints Q&As from modules according to args"""

    text_wrapper = textwrap.TextWrapper(
        width=80, initial_indent=' ', subsequent_indent='  ')

    save_path = "data/questions.jsonl"

    filtered_modules, counts = init_modules(args)

    idx = 0

    for regime, flat_modules in six.iteritems(filtered_modules):
        per_module = counts[regime]
        for module_name, module in six.iteritems(flat_modules):
            # These magic print constants make the header bold.
            print('\033[1m{}/{}\033[0m'.format(regime, module_name))
            num_dropped = 0
            for _ in range(per_module):
                problem, extra_dropped = sample_from_module(module, args)
                num_dropped += extra_dropped
                text = text_wrapper.fill(
                    '{}  \033[92m{}\033[0m'.format(problem.question, problem.answer))
                print(text)
                result = {
                    "id": idx,
                    "question": str(problem.question),
                    "answer": str(problem.answer),
                    "module": module_name,
                    "regime": regime
                }
                save_jsonl(result, save_path)
                idx += 1
            if num_dropped > 0:
                logging.warning('Dropped %d examples', num_dropped)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_difficulties', type=int, default=3, help='Number of difficulty levels for training modules')
    parser.add_argument('--n_train', type=int, default=9, help='Number of training samples per category to generate')
    parser.add_argument('--n_test', type=int, default=0, help='Number of test samples per category to generate')
    parser.add_argument('--filter', type=str, default='', help='Filter for module names')
    parser.add_argument('--show_dropped', action='store_true', help='Whether to log dropped samples')
    args = parser.parse_args()

    main(args)
