import json
from api_eval import BaseLLMEvaluator, extract_yes_no
from src.prompt_engineering import (
    load_coarse_grained_novelty_evaluation_prompt,
    load_correctness_evaluation_prompt,
    load_fine_grained_novelty_evaluation_prompt
) # these are prompt_engineering utils, will revise later to add task names to fitin the overall structure


class CorrectnessEvaluator(BaseLLMEvaluator):
    def preprocess_input(self, prompt, context):
        return context

    def generate_evaluation_prompt(self, prompt, context):
        return load_correctness_evaluation_prompt(
            problem=prompt,
            solutions=context.get("solutions", []),
            new_solution=context.get("new_solution", "")
        )

    def postprocess_results(self, raw_outputs):
        return extract_yes_no(raw_outputs[0])

    # not useful
    def aggregate_results(self, processed_results):
        return "YES" if all(decision == "YES" for decision in processed_results) else "NO"

    
class CoarseNoveltyEvaluator(BaseLLMEvaluator):
    def preprocess_input(self, prompt, context):
        return context

    def generate_evaluation_prompt(self, prompt, context):
        assert context.get("k") != 0 # should never be 0!
        return load_coarse_grained_novelty_evaluation_prompt(
            problem=prompt,
            solutions=context.get("solutions", []),
            k=context.get("k", 0),
            new_solution=context.get("new_solution", "")
        )

    def postprocess_results(self, raw_outputs):
        # return [extract_yes_no(output) for output in raw_outputs]
        return extract_yes_no(raw_outputs[0])
    # revise this function
    # not useful
    def aggregate_results(self, processed_results):
        yes_count = sum(1 for decision in processed_results if decision == "YES")
        return "YES" if yes_count > len(processed_results) / 2 else "NO"


class FineNoveltyEvaluator(BaseLLMEvaluator):
    def preprocess_input(self, prompt, context):
        return context

    def generate_evaluation_prompt(self, prompt, context):
        assert context.get("k") != 0 # should never be 0!
        return load_fine_grained_novelty_evaluation_prompt( # later revise back by remove the reasons
            problem=prompt,
            solutions=context.get("solutions", []),
            k=context.get("k", 0),
            new_solution=context.get("new_solution", "")
        )

    def postprocess_results(self, raw_outputs):
        return extract_yes_no(raw_outputs[0])
    
    # not useful
    def aggregate_results(self, processed_results):
        yes_count = sum(1 for decision in processed_results if decision == "YES")
        return "YES" if yes_count > len(processed_results) / 2 else "NO"

