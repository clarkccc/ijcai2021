# pylint: disable=logging-fstring-interpolation, broad-except
"""ingestion program for autoWSL"""
import os
from os.path import join
import sys
from sys import path
import argparse
import time
import datetime
import yaml
import subprocess
import threading
import mmap
import contextlib
from filelock import FileLock
import numpy as np

from common import get_logger, init_usermodel

import timing
from timing import Timer
from dataset import Dataset

import pickle
from env import Env


# Verbosity level of logging:
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)
TRAIN_TIME_BUDGET = 60
TEST_TIME_BUDGET = 60*2
N_TEST = 1
TMAX = 1500

class RewardsError(Exception):
    """Agent pickle error"""

def _here(*args):
    """Helper function for getting the current directory of this script."""
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(here, *args))


def write_start_file(output_dir):
    """Create start file 'start.txt' in `output_dir` with updated timestamp
    start time.

    """
    LOGGER.info('===== alive_thd started')
    start_filepath = os.path.join(output_dir, 'start.txt')

    # initialize start.txt for mmap
    # max timestamp chars length is 17, set file size to 20 chars for enough mmap space.
    with open(start_filepath, "w") as f:
        f.truncate(20)

    with open(start_filepath, 'r+b') as ftmp:
        LOGGER.debug(f"***** open start.txt for mmap ...")
        with contextlib.closing(mmap.mmap(ftmp.fileno(), 0, access=mmap.ACCESS_WRITE)) as mm:
            LOGGER.debug(f"***** mmap start.txt for reading ...")
            while True:
                current_time = datetime.datetime.now().timestamp()
                current_time = str(current_time) + '\n'
                current_time = bytes(current_time, encoding='utf-8')
                # LOGGER.info(f"***** Start to write current time to start.txt...")
                mm.seek(0)
                mm.write(current_time)
                # LOGGER.info(f"***** Write current time to start.txt done. current_time={current_time}")
                time.sleep(10)


class IngestionError(RuntimeError):
    """Model api error"""


def _parse_args():
    root_dir = _here(os.pardir)
    default_dataset_dir = join(root_dir, "sample_data")
    default_output_dir = join(root_dir, "sample_result_submission")
    default_ingestion_program_dir = join(root_dir, "ingestion_program")
    default_code_dir = join(root_dir, "code_submission")
    default_score_dir = join(root_dir, "scoring_output")
    default_temp_dir = join(root_dir, 'temp_output')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        # default=default_dataset_dir,
                        default='./data/public/01',
                        help="Directory storing the dataset (containing "
                             "e.g. adult.data/)")
    parser.add_argument('--output_dir', type=str,
                        default=default_output_dir,
                        help="Directory storing the predictions. It will "
                             "contain e.g. [start.txt, predictions, end.yaml]"
                             "when ingestion terminates.")
    parser.add_argument('--ingestion_program_dir', type=str,
                        default=default_ingestion_program_dir,
                        help="Directory storing the ingestion program "
                             "`ingestion.py` and other necessary packages.")
    parser.add_argument('--code_dir', type=str,
                        # default=default_code_dir,
                        default='/home/quicktron_ye/Documents/competition/autorl_starting_kit/code_submission',
                        help="Directory storing the submission code "
                             "`model.py` and other necessary packages.")
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help="Directory storing the scoring output "
                             "e.g. `scores.txt` and `detailed_results.html`.")
    parser.add_argument('--temp_dir', type=str,
                        default=default_temp_dir,
                        help="Directory storing the temporary output."
                             "e.g. save the participants` model after "
                             "trainning.")

    args = parser.parse_args()
    LOGGER.debug(f'Parsed args are: {args}')
    LOGGER.debug("-" * 50)
    if (args.dataset_dir.endswith('run/input') and
            args.code_dir.endswith('run/program')):
        LOGGER.debug("Since dataset_dir ends with 'run/input' and code_dir "
                     "ends with 'run/program', suppose running on "
                     "CodaLab platform. Modify dataset_dir to 'run/input_data'"
                     " and code_dir to 'run/submission'. "
                     "Directory parsing should be more flexible in the code of"
                     " compute worker: we need explicit directories for "
                     "dataset_dir and code_dir.")

        args.dataset_dir = args.dataset_dir.replace(
            'run/input', 'run/input_data')
        args.code_dir = args.code_dir.replace(
            'run/program', 'run/submission')

        # Show directories for debugging
        LOGGER.debug(f"sys.argv = {sys.argv}")
        LOGGER.debug(f"Using dataset_dir: {args.dataset_dir}")
        LOGGER.debug(f"Using output_dir: {args.output_dir}")
        LOGGER.debug(
            f"Using ingestion_program_dir: {args.ingestion_program_dir}")
        LOGGER.debug(f"Using code_dir: {args.code_dir}")
    return args


def _init_python_path(args):
    path.append(args.ingestion_program_dir)
    path.append(args.code_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    os.environ['PYTHONPATH'] = args.code_dir

import matplotlib.pyplot as plt
def _train_predict(umodel, dataset, train_timer, test_timer):
    # Train the model

    with train_timer.time_limit('train'):
        agent = umodel.train(TRAIN_TIME_BUDGET)
    def draw(iter,job_statuses,rewards):
        import time
        machines={}
        jobs_op=set()
        timeStart=0
        fig=plt.figure()
        # ax=fig.gca(projection='3d')
        process_time_remain={}
        colors=["g","b","y",'pink','b','b','b']
        track=[]
        job_machine={}
        breack_down=False
        
        for dataIn in job_statuses:
            timeStart+=1
            availabe={jobname:info for jobname,info in dataIn.items() if info['machine']}
            for job in track :
                if(dataIn[job]['status']!='done'):
                    print(dataIn[job])
            for jobname,jobinfo in availabe.items():
                if(jobname+jobinfo["op"] not in jobs_op):
                    machine_name=jobinfo["machine"]
                    machine_no=machines[machine_name] = machines.get(machine_name, int(machine_name[1:]))
                    jobs_op.add(jobname+jobinfo["op"])
                    job_machine[machine_name]=jobname
                    if jobname in process_time_remain and jobinfo["priority"] >0:
                        print(process_time_remain[jobname],jobinfo,jobname)
                    if  jobname  in  process_time_remain or jobname in track:
                        color='r'
                    else:
                        color=colors[jobinfo["priority"]]
                    plt.barh(y=[machine_no],width=[jobinfo['remain_process_time']],height=[1],left=[timeStart],color=color )
                else: 
                    if(jobinfo['status']!='work'):
                        print(jobinfo)
            process_time_remain={jobname:info["remain_pending_time"] for jobname,info in dataIn.items() if info["remain_pending_time"]<0 and info['priority']> 0}
            if(process_time_remain):print(process_time_remain)
        # ax.set_zlabel(str(rewards))
        keys=[it for it,index in sorted(machines.items(),key=lambda x:x[1])]
        plt.yticks(np.arange(len(keys))+0.5, keys)
        plt.xlabel(str(rewards))
        plt.savefig("result/"+time.strftime(str(-rewards)+"_%m_%d_%H_%M_%S", time.localtime())+'.png',dpi=1000, format='png',bbox_inches="tight") 
    def test(conf, agent,iter):
        rewards = 0
        done = False
        env = Env(conf)
        machine_status, job_status, t, job_list = env.reset()
        job_statuses=[]
        ptv=0
        while not done:
            action = agent.act(machine_status, job_status, t, job_list)
            machine_status, job_status, t, reward, job_list, done = env.step(action)
            rewards += sum(reward.values())
            if(sum(reward.values())!=-1):
                ptv+= sum(reward.values())+1
            job_statuses.append(job_status)
        print(conf,rewards,ptv)
        if('04' in conf):
            draw(iter,job_statuses,rewards)
        return rewards

    with test_timer.time_limit('test'):
        test_dataset_dir = dataset.test_dataset_dir
        conf_dir = os.listdir(test_dataset_dir)
        test_conf_list = [join(test_dataset_dir, file) for file in conf_dir]
        rewards = []
        for file in test_conf_list:
            reward = []
            for iter in range(N_TEST):
                reward.append(test(file, agent,iter))
            rewards.append(sum(reward)/len(reward))
        return rewards


def _finalize(args, timer):
    # Finishing ingestion program
    end_time = time.time()

    time_stats = timer.get_all_stats()
    for pname, stats in time_stats.items():
        for stat_name, val in stats.items():
            LOGGER.info(f'the {stat_name} of duration in {pname}: {val} sec')

    overall_time_spent = timer.get_overall_duration()

    # Write overall_time_spent to a end.yaml file
    end_filename = 'end.yaml'
    content = {
        'ingestion_duration': overall_time_spent,
        'time_stats': time_stats,
        'end_time': end_time}

    with open(join(args.output_dir, end_filename), 'w') as ftmp:
        yaml.dump(content, ftmp)
        LOGGER.info(
            f'Wrote the file {end_filename} marking the end of ingestion.')

        LOGGER.info("[+] Done. Ingestion program successfully terminated.")
        LOGGER.info(f"[+] Overall time spent {overall_time_spent:5.2} sec")

    # Copy all files in output_dir to score_dir
    os.system(
        f"cp -R {os.path.join(args.output_dir, '*')} {args.score_dir}")
    LOGGER.debug(
        "Copied all ingestion output to scoring output directory.")

    LOGGER.info("[Ingestion terminated]")


def _write_predict(output_dir, rewards):
    """prediction should be list"""
    os.makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'rewards.pkl'), 'wb') as f:
        try:
            pickle.dump(rewards, f)
        except:
            raise RewardsError(
                f"Your rewards information can't be pickled")



def _init_timer(time_budgets, t):
    timer = Timer()
    timer.add_process(t, time_budgets, timing.RESET)
    LOGGER.debug(
        f"init time budget of train_predict: {time_budgets} "
        f"mode: {timing.RESET}")
    return timer

def _install_requirements(code_dir):
    if os.path.exists(join(code_dir, 'requirements.txt')):
        requirements = join(code_dir, 'requirements.txt')
        os.system(f'pip install -r {requirements}')
    else:
        LOGGER.info("===== No requirements")


def main():
    """main entry"""
    LOGGER.info('===== Start ingestion program.')
    # Parse directories from input arguments
    LOGGER.info('===== Initialize args.')
    args = _parse_args()
    # class Test:
    #     def __init__(self) -> None:
    #         pass
    # args = Test()
    # args.dataset_dir = './data/public/01'
    # args.code_dir = '/home/quicktron_ye/Documents/competition/autorl_starting_kit/code_submission'
    _init_python_path(args)

    LOGGER.info('===== Set alive_thd')
    alive_thd = threading.Thread(target=write_start_file, name="alive",
                                 args=(args.output_dir,))
    alive_thd.daemon = True
    alive_thd.start()

    LOGGER.info('===== Install requirements.')
    _install_requirements(args.code_dir)

    LOGGER.info('===== Load data.')
    dataset = Dataset(args.dataset_dir)

    LOGGER.info(f"Training Time budget: {TRAIN_TIME_BUDGET}")
    LOGGER.info(f"Testing Time budget: {TEST_TIME_BUDGET}")

    LOGGER.info("===== import user model")
    train_data_dir = dataset.train_dataset_dir
    umodel = init_usermodel(train_data_dir)

    LOGGER.info("===== Begin training user model")
    train_timer = _init_timer(TRAIN_TIME_BUDGET, 'train')
    test_timer = _init_timer(TEST_TIME_BUDGET, 'test')
    rewards = _train_predict(umodel, dataset, train_timer, test_timer)
    _write_predict(args.output_dir, rewards)

    _finalize(args, train_timer)


if __name__ == "__main__":
    main()
 