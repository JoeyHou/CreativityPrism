from abc import abstractmethod
from src.driver import Driver

class InferenceDriver(Driver):    
    
    def __init__(self, config = {}):
        super().__init__(config)

    @abstractmethod
    def create_batched_prompt(self):
        # depends on the subclass
        pass 
    
    @abstractmethod
    def parse_llm_outputs(self, llm_results):
        # depends on the subclass
        pass 

    @abstractmethod
    def inference(self):
        # depends on the subclass
        pass 

