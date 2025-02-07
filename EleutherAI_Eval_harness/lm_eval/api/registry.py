import logging
from typing import Callable, Dict

# import evaluate as hf_evaluate
import string
import numpy as np
import re

def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}


###
from lm_eval.api.model import LM


eval_logger = logging.getLogger("lm-eval")

MODEL_REGISTRY = {}


def register_model(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, LM
            ), f"Model '{name}' ({cls.__name__}) must extend LM class"

            assert (
                name not in MODEL_REGISTRY
            ), f"Model named '{name}' conflicts with existing model! Please register with a non-conflicting alias instead."

            MODEL_REGISTRY[name] = cls
        return cls

    return decorate


def get_model(model_name):
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load model '{model_name}', but no model for this name found! Supported model names: {', '.join(MODEL_REGISTRY.keys())}"
        )


TASK_REGISTRY = {}
GROUP_REGISTRY = {}
ALL_TASKS = set()
func2task_index = {}


def register_task(name):
    def decorate(fn):
        assert (
            name not in TASK_REGISTRY
        ), f"task named '{name}' conflicts with existing registered task!"

        TASK_REGISTRY[name] = fn
        ALL_TASKS.add(name)
        func2task_index[fn.__name__] = name
        return fn

    return decorate


def register_group(name):
    def decorate(fn):
        func_name = func2task_index[fn.__name__]
        if name in GROUP_REGISTRY:
            GROUP_REGISTRY[name].append(func_name)
        else:
            GROUP_REGISTRY[name] = [func_name]
            ALL_TASKS.add(name)
        return fn

    return decorate


OUTPUT_TYPE_REGISTRY = {}
METRIC_REGISTRY = {}
METRIC_AGGREGATION_REGISTRY = {}
AGGREGATION_REGISTRY: Dict[str, Callable[[], Dict[str, Callable]]] = {}
HIGHER_IS_BETTER_REGISTRY = {}

DEFAULT_METRIC_REGISTRY = {
    "loglikelihood": [
        "perplexity",
        "acc",
    ],
    "loglikelihood_rolling": ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "multiple_choice": ["acc", "acc_norm"],
    "generate_until": ["exact_match"],
}


def register_metric(**args):
    # TODO: do we want to enforce a certain interface to registered metrics?
    def decorate(fn):
        assert "metric" in args
        name = args["metric"]

        for key, registry in [
            ("metric", METRIC_REGISTRY),
            ("higher_is_better", HIGHER_IS_BETTER_REGISTRY),
            ("aggregation", METRIC_AGGREGATION_REGISTRY),
        ]:
            if key in args:
                value = args[key]
                assert (
                    value not in registry
                ), f"{key} named '{value}' conflicts with existing registered {key}!"

                if key == "metric":
                    registry[name] = fn
                elif key == "aggregation":
                    registry[name] = AGGREGATION_REGISTRY[value]
                else:
                    registry[name] = value

        return fn

    return decorate


def get_metric(name: str, hf_evaluate_metric=False) -> Callable:
    if not hf_evaluate_metric:
        if name in METRIC_REGISTRY:
            return METRIC_REGISTRY[name]
        else:
            eval_logger.warning(
                f"Could not find registered metric '{name}' in lm-eval, searching in HF Evaluate library..."
            )

    try:
        return exact_match_hf_evaluate
    except Exception:
        eval_logger.error(
            f"{name} not found in the evaluate library! Please check https://huggingface.co/evaluate-metric",
        )


def register_aggregation(name: str):
    def decorate(fn):
        assert (
            name not in AGGREGATION_REGISTRY
        ), f"aggregation named '{name}' conflicts with existing registered aggregation!"

        AGGREGATION_REGISTRY[name] = fn
        return fn

    return decorate


def get_aggregation(name: str) -> Callable[[], Dict[str, Callable]]:
    try:
        return AGGREGATION_REGISTRY[name]
    except KeyError:
        eval_logger.warning(f"{name} not a registered aggregation metric!")


def get_metric_aggregation(name: str) -> Callable[[], Dict[str, Callable]]:
    try:
        return METRIC_AGGREGATION_REGISTRY[name]
    except KeyError:
        eval_logger.warning(f"{name} metric is not assigned a default aggregation!")


def is_higher_better(metric_name) -> bool:
    try:
        return HIGHER_IS_BETTER_REGISTRY[metric_name]
    except KeyError:
        eval_logger.warning(
            f"higher_is_better not specified for metric '{metric_name}'!"
        )
