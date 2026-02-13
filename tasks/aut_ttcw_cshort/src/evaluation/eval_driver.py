from abc import abstractmethod
from src.inference.inference_driver import Driver

class EvalDriver(Driver):    

    def __init__(self, config = {}):
        super().__init__(config)

    @abstractmethod
    def create_batched_prompt(self):
        # depends on the subclass
        pass 
    
    @abstractmethod 
    # TODO: re-write this and parsing@parsing  
    def parse_llm_outputs(self, llm_outputs):
        pass 

    @abstractmethod
    def evaluation(self):
        # depends on the subclass
        pass 

    @abstractmethod
    def generate_eval_report(self, eval_outputs):
        # depends on the subclass
        pass