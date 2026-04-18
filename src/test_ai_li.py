"""This class will create custom matrices to evalute the ai generated text"""

from deepeval_model_connect import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval , BERTScoreMetric
from deepeval_model_connect import NvidiaLLM


nvidia_evaluator = NvidiaLLM(model_name="sarvamai/sarvam-m")


def ai_detection_matrices(self):
    self.ai_gen_congurancy_detection_matric = GEval(
                name = "List_Congurancy",
                criteria="""
    Evaluate how congruent (aligned) these two lists are across four dimensions:
    1. Thematic Alignment: Do both lists cover the same core themes and subject matter?
    2. Structural Parallelism: Do the lists follow similar organizational logic and point sequencing?
    3. Factual Consistency: Are the facts, dates, and statistics compatible (not contradictory)?
    4. Semantic Similarity: Do corresponding points convey equivalent meaning even if phrased differently?
    
    Provide a score from 0-1 and explain which dimensions show alignment vs divergence.
    """,
     evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold = 0.7,
    model = nvidia_evaluator  
    )

    self.symantic_matric = BERTScoreMetric(
        model_type = "microsoft/deberta-xlarge-mnli",
        threshold = 0.7
    )
    