"""Test script for DeepEval framework integration."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from src.deepeval_model_connect import NvidiaLLM_Understanding

def test_deepeval_setup():
    print("Testing DeepEval with NvidiaLLM_Understanding...")
    evaluator = NvidiaLLM_Understanding(model_name="meta/llama-3.3-70b-instruct")
    
    congruency_metric = GEval(
        name="Cloze_Congruency",
        criteria="""
        Evaluate how congruent (aligned) the actual AI infill output is with the original expected text across:
        1. Thematic Alignment: Do both texts cover the same core themes?
        2. Structural Parallelism: Do the texts follow similar syntactic flow?
        3. Factual Consistency: Are statements compatible?
        4. Semantic Equivalence: Do corresponding points convey equivalent meaning?
        Provide a score from 0-1 and explain alignment vs divergence.
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=evaluator,
    )

    test_case = LLMTestCase(
        input="Artificial intelligence has revolutionized [MASK_1], playing a crucial role [MASK_2].",
        actual_output="Artificial intelligence has revolutionized modern technology, playing a crucial role in reshaping global industries.",
        expected_output="Artificial intelligence has revolutionized modern technological paradigms, playing a crucial role in reshaping industries worldwide.",
    )

    print("Test case configured successfully.")
    print("Metric name:", congruency_metric.name)
    print("Evaluation params:", congruency_metric.evaluation_params)

if __name__ == "__main__":
    test_deepeval_setup()
