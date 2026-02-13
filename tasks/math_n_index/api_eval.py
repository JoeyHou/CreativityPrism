import openai
from abc import ABC, abstractmethod
from api_warpper import ModelWrapper

def extract_yes_no(text):
    """
    A simple helper function to extract a YES/NO decision from a judge's output.
    This is an exmaple of the postprocessing function.
    """
    text = text.strip().upper()
    if "YES" in text:
        return "YES"
    elif "NO" in text:
        return "NO"
    return "UNCLEAR"

class BaseLLMEvaluator(ABC):
    """
    Abstract base class for an evaluation process using API-based models as judges.
    Supports multiple judges by iterating over a list of judge models.
    """

    def __init__(self, api_key, judge_models):
        """
        Parameters:
            api_key (str): Your API key.
            judge_models (list of str): List of API-based judge model names.
        """
        self.api_key = api_key
        self.judge_models = judge_models
        self.model_wrappers = {model: ModelWrapper(model, api_key) for model in judge_models} # note some eval tasks may need more than one judges

    @abstractmethod
    def preprocess_input(self, prompt, context):
        """
        Prepares the input data needed for evaluation.
        
        Parameters:
            prompt (str): The original problem or question.
            context (dict): Additional context (e.g. reference solutions, candidate response, etc.)
        
        Returns:
            dict: Processed input data.
        """
        pass

    @abstractmethod
    def generate_evaluation_prompt(self, prompt, context):
        """
        Constructs the evaluation prompt using the preprocessed data.
        
        Parameters:
            prompt (str): The original problem or question.
            context (dict): Processed context data.
        
        Returns:
            str: A fully formatted evaluation prompt to send to each judge.
        """
        pass

    @abstractmethod
    def postprocess_results(self, raw_outputs):
        """
        Processes the raw outputs from the judges into structured decisions.
        
        Parameters:
            raw_outputs (list of str): The raw text outputs from each judge.
        
        Returns:
            list: Processed decisions (e.g., a list of YES/NO decisions).
        """
        pass

    @abstractmethod
    def aggregate_results(self, processed_results):
        """
        Aggregates multiple judges' decisions into a final decision.
        
        For example, for correctness you might require that all judges say "YES",
        whereas for novelty you might use majority vote.
        
        Parameters:
            processed_results (list): List of individual judge decisions.
        
        Returns:
            str: The final aggregated decision.
        """
        pass

    def evaluate(self, prompt, context):
        """
        Executes the full evaluation pipeline:
        1. Preprocesses the input.
        2. Generates the evaluation prompt.
        3. Queries each judge model via its model wrapper.
        4. Postprocesses each judge's output.
        5. Aggregates the individual decisions into a final decision.
        
        Parameters:
            prompt (str): The original problem or question.
            context (dict): Additional context needed for evaluation.
            
        Returns:
            str: The final evaluation decision.
        """
        processed_context = self.preprocess_input(prompt, context)
        evaluation_prompt = self.generate_evaluation_prompt(prompt, processed_context)

        try:
            with open("eval_prompt_math_50_Claude.txt", "a") as f:
                f.write(f"--- Prompt ---\n{evaluation_prompt}\n\n")
        except Exception as e:
            print(f"Warning: Failed to write prompt to file: {e}")

        raw_outputs = []
        for model in self.judge_models:
            response = self.model_wrappers[model].generate_response(evaluation_prompt)
            raw_outputs.append(response)
        processed_results = self.postprocess_results(raw_outputs)
        print("The len of result: ", len(raw_outputs), raw_outputs)
        return processed_results, raw_outputs # (str, list)
