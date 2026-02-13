import sys
import argparse
import os 

import logging 
logging.basicConfig(
    # level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.utils.helpers import load_json, write_json
from src.inference.creative_writing import CreativeWritingInference
from src.inference.aut_push import AUTInference
from src.inference.creative_short_story import CreativeShortStoryInference


def run_inference(config):
    logger = logging.getLogger(__name__)
    if config.get('debug', False):
        logger_level = logging.DEBUG
    else:
        logger_level = logging.INFO
    config['logger'] = logger

    for exp_config in config['experiments_list']:
        print('=> current config:', exp_config)
        
        ## 1.1 setting up logger
        file_handler = logging.FileHandler(os.path.join('log', f"{exp_config['run_id']}_eval.log"))
        file_handler.setLevel(logger_level)
        logger.setLevel(logger_level)
        logger.addHandler(file_handler)
        exp_config['logger'] = logger 
         
        ## 1.2 initialize driver
        task = exp_config.get('task', 'creative_writing')
        if task == 'creative_writing':
            inference_driver = CreativeWritingInference(exp_config)
        elif task == 'aut_push':
            inference_driver = AUTInference(exp_config)
        elif task == 'creative_short':
            inference_driver = CreativeShortStoryInference(exp_config)
        else:
            raise NotImplementedError
        
        ## 2. running inference 
        inference_results = inference_driver.inference()
        os.makedirs('data/output/{}'.format(exp_config['run_id']), exist_ok = True)
        write_json(inference_results, 'data/output/{}/{}'.format(exp_config['run_id'], 'inference_output.json'))
        # print(inference_results)
        
        ## 3. cleaning up
        logger.removeHandler(file_handler)
        file_handler.close()
        del inference_driver.llm
        del inference_driver

if __name__ == '__main__':
    if len(sys.argv) == 1:
        config = load_json('configs/default.json')
    else:
        config = load_json(sys.argv[1])
    # print(config)
    run_inference(config)