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
from src.inference.ttct import TTCTInference
from src.inference.creativity_index import CreativityIndexInference
from src.inference.creative_math import CreativeMathInference
from src.inference.dat import DATInference
from src.inference.neocoder import NeoCoderInference

# Adding a task means adding one line here and one class under src/inference/.
# The name is what a config's "task" field has to say.
INFERENCE_DRIVERS = {
    'creative_writing': CreativeWritingInference,
    'aut_push': AUTInference,
    'creative_short': CreativeShortStoryInference,
    'ttct': TTCTInference,
    'creativity_index': CreativityIndexInference,
    'creative_math': CreativeMathInference,
    'dat': DATInference,
    'neocoder': NeoCoderInference,
}


def run_inference(config):
    logger = logging.getLogger(__name__)
    if config.get('debug', False):
        logger_level = logging.DEBUG
    else:
        logger_level = logging.INFO
    config['logger'] = logger

    failures = []
    for exp_config in config['experiments_list']:
        print('=> current config:', exp_config)
        
        # One bad experiment must not take the rest of the list with it.
        # An experiments_list is often hours of work; losing the tail because
        # entry three hit a missing dependency is how an overnight run comes
        # back empty. Failures are collected and re-raised at the end so the
        # exit code still says something went wrong.
        try:
            ## 1.1 setting up logger
            script_dir = os.path.dirname(os.path.abspath(__file__))   # path of this script
            log_dir = os.path.join(script_dir, "log")
            os.makedirs(log_dir, exist_ok=True)
            print("We are at log dir: ", log_dir)
            log_path = os.path.join(log_dir, f"{exp_config['run_id']}_eval.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            print(log_path)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logger_level)
            logger.setLevel(logger_level)
            logger.addHandler(file_handler)
            exp_config['logger'] = logger 
         
            # os.makedirs('data/output/{}'.format(exp_config['run_id']), exist_ok = True)
            data_dir = os.path.join(script_dir, "data", "output", exp_config['run_id'])
            os.makedirs(data_dir, exist_ok=True)

            ## 1.2 initialize driver
            task = exp_config.get('task', 'creative_writing')
            if task not in INFERENCE_DRIVERS:
                raise NotImplementedError(
                    'unknown task {!r}; known: {}'.format(task, sorted(INFERENCE_DRIVERS))
                )
            inference_driver = INFERENCE_DRIVERS[task](exp_config)
        
            ## 2. running inference 
            inference_results = inference_driver.inference()
        
            # write_json(inference_results, 'data/output/{}/{}'.format(exp_config['run_id'], 'inference_output.json'))
            write_json(inference_results, os.path.join(data_dir, "inference_output.json"))
            # print(inference_results)
        
            ## 3. cleaning up
            logger.removeHandler(file_handler)
            file_handler.close()
            # Drop the engine before the next experiment: vLLM does not free the
            # GPU when the driver alone goes out of scope, which is why chaining
            # experiments used to fail on the second one.
            del inference_driver.llm
            del inference_driver
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append((exp_config.get('run_id', '?'), exc))
            logging.getLogger(__name__).exception(
                '%s failed: %s', exp_config.get('run_id', '?'), exc)
            print('=> FAILED {}: {}: {}'.format(
                exp_config.get('run_id', '?'), type(exc).__name__, exc))

    if failures:
        print('\n=== {} of {} experiment(s) failed ==='.format(
            len(failures), len(config['experiments_list'])))
        for run_id, exc in failures:
            print('  {}: {}: {}'.format(run_id, type(exc).__name__, exc))
        raise SystemExit(1)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        config = load_json('configs/default.json')
    else:
        config = load_json(sys.argv[1])
    print("Using Config: ", config)
    run_inference(config)
