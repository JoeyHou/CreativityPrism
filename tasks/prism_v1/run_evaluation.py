import sys
import argparse
import os 
import json 

import logging 
logging.basicConfig(
    # level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.utils.helpers import load_json, write_json
from src.evaluation.creative_writing import CreativeWritingEval
from src.evaluation.aut_push import AUTEval
from src.evaluation.creative_short_story import CreativeShortStoryEval
from src.evaluation.ttct import TTCTEval
from src.evaluation.creativity_index import CreativityIndexEval
from src.evaluation.creative_math import CreativeMathEval
from src.evaluation.dat import DATEval
from src.evaluation.neocoder import NeoCoderEval

# Adding a task means adding one line here and one class under src/evaluation/.
EVAL_DRIVERS = {
    'creative_writing': CreativeWritingEval,
    'aut_push': AUTEval,
    'creative_short': CreativeShortStoryEval,
    'ttct': TTCTEval,
    'creativity_index': CreativityIndexEval,
    'creative_math': CreativeMathEval,
    'dat': DATEval,
    'neocoder': NeoCoderEval,
}

import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

def evaluator_check(run_id, likert):
    ground_truth = load_json('./data/output/{}/inference_output.json'.format(run_id))
    eval_predict = load_json('./data/output/{}/eval_output_cleaned.json'.format(run_id))

    all_comp_data = []
    for story_key in ground_truth:
        model_output = {k: -1 for k in range(1, 15)}
        found_match = False
        for pred_dp in eval_predict:
            if pred_dp['prompt_id'][0] == story_key:
                model_output[pred_dp['prompt_id'][1]] = pred_dp['cleaned_output']
                found_match = True
        if found_match:
            all_comp_data.append({
                'story_id': story_key,
                'human_output': ground_truth[story_key]['human_output'],
                'model_output': model_output
            })
    

    acc_by_q = {k: 0 for k in range(1, 15)}
    pos_true = 0
    pos_pred = 0
    invalid_pred = 0
    all_true = []
    all_pred = []
    
    for k in range(1, 15):
        true = [comp['human_output'][str(k)] for comp in all_comp_data]
        pred = [comp['model_output'][k] for comp in all_comp_data]
        invalid_pred += sum([p == -1 for p in pred])
        all_true.extend(true)
        all_pred.extend(pred)
        if likert:
            acc_by_q[k] = round(pearsonr(true, pred).statistic, 2)
        else:
            acc_by_q[k] = round(accuracy_score(true, pred), 2)
        pos_true += sum(true)
        pos_pred += sum(pred)
    acc_by_q['avg_acc'] = round(np.mean(list(acc_by_q.values())), 2)
    if likert:
        acc_by_q['all_acc'] = round(pearsonr(all_true, all_pred).statistic, 2)
        pos_pred = round(pos_pred / (5 * len(all_comp_data) * 14), 2)
    else:
        acc_by_q['all_acc'] = round(accuracy_score(all_true, all_pred), 2)
    return acc_by_q, pos_true, pos_pred, invalid_pred

def run_evaluation(config):
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
            if '/' in exp_config['run_id']:
                os.makedirs('log/' + '/'.join(exp_config['run_id'].split('/')[:-1]), exist_ok = True)
            file_handler = logging.FileHandler(os.path.join('log', f"{exp_config['run_id']}_eval.log"))
            file_handler.setLevel(logger_level)
            logger.setLevel(logger_level)
            logger.addHandler(file_handler)
            logger.info("=> current config:")
            logger.info(exp_config)
            exp_config['logger'] = logger

            ## 1.2 initialize eval driver
            task = exp_config.get('task', 'creative_writing')
            report_format = 'csv'
            if task not in EVAL_DRIVERS:
                raise NotImplementedError(
                    'unknown task {!r}; known: {}'.format(task, sorted(EVAL_DRIVERS))
                )
            eval_driver = EVAL_DRIVERS[task](exp_config)

            ## 2. running evaluation
            os.makedirs('data/output/{}'.format(exp_config['run_id']), exist_ok = True)
            eval_report, eval_output_cleaned = eval_driver.evaluation()
            write_json(eval_output_cleaned, 'data/output/{}/{}'.format(exp_config['run_id'], 'eval_output_cleaned.json'))
        
            if report_format == 'csv':
                eval_report.to_csv('data/output/{}/{}'.format(exp_config['run_id'], 'eval_report.csv'), index = False)
            elif report_format == 'json':
                write_json(eval_report, 'data/output/{}/{}'.format(exp_config['run_id'], 'eval_report.json'))
            else:
                print('ERROR: unknown report format:', report_format)
        
            ## 3. callback functions
            # if 'evaluator_check' in exp_config['run_id']:
            #     likert = exp_config.get('5_scale', False)
            #     acc_by_q, pos_true, pos_pred, invalid_pred = evaluator_check(exp_config['run_id'], likert)
            #     logger.info(
            #         'acc_by_q: {}'.format(
            #             json.dumps(acc_by_q, indent = 4)
            #             ) + \
            #         'pos_true: {}, pos_pred: {}, invalid_pred: {}'.format(
            #             pos_true, pos_pred, invalid_pred
            #         )
            #     )

            ## 4. cleaning up
            logger.removeHandler(file_handler)
            file_handler.close()
            del eval_driver
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
    run_evaluation(config)
