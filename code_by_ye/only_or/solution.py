import math
import collections
import copy
import time
# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box, Tuple, Dict

from copy import deepcopy
from visualdl import LogWriter

import datetime
import numpy as np
import random

from collections import defaultdict
#%%

class Trainer:
    def __init__(self, Env, conf_list):
        self.conf_list = conf_list
        self.Env = Env
        self.checkpoint = None
        self.iter = 0

    def train(self, run_time):
        env = self.Env(self.conf_list[0])
        obs = env.reset()
        return Agent(env.job_types, env)

class Agent:
    def __init__(self, job_types, env):
        self.job_types = job_types
        self.env = env
    
    def act(self, machine_status, job_status, time, job_list):
        self.machine_status = machine_status
        self.job_status = job_status
        self.time = time
        self.job_list = job_list

        self.valid_action = {}
        self.valid_machine_type = defaultdict(list)
        for machine in self.machine_status:
            if len(self.machine_status[machine]['job_list']) != 0:
                self.valid_action[machine] = self.machine_status[machine]['job_list']
                
        job_appear_first_dict_by_machine_type = self.or_act_rules()
        
        action = dict()
        if len(self.valid_action) > 0:
            for machine_id in self.valid_action:
                machine_type = self.machine_status[machine_id]['type']
                job_id = job_appear_first_dict_by_machine_type[machine_type]
                action[machine_id] = job_id
        return action
    
    def or_act_rules(self):
        if len(self.valid_action) == 0:
            return {}
        # extended_selected_num = selected_num * 2
        lot_selected_dict_by_machine_type = defaultdict(defaultdict)
        idle_machine_id = set()
        job_infos_dict = defaultdict()
        feature_map_count = defaultdict(lambda:int(0))
        idle_num_by_machine_type = defaultdict(lambda:int(0))
        for machine_id in self.machine_status:
            if self.machine_status[machine_id]['status'] == 'idle':
                idle_num_by_machine_type[self.machine_status[machine_id]['type']] += 1
        
        max_num = 4
        or_choice_set = set()
        selected_num_by_machine_type = defaultdict(lambda:int(0))
        for agent_id in self.valid_action:
            machine_type = self.machine_status[agent_id]['type']
            selected_num = min(idle_num_by_machine_type[machine_type]*2, max_num)
            # selected_num = max_num
            lot_collect = []
            qt_select_lot = self.qtfirst(agent_id, 100)
            pt_select_lot = self.ptfirst(agent_id, 100)
            
            if selected_num_by_machine_type[machine_type] == selected_num:
                continue
            n = 0
            while len(qt_select_lot) + len(pt_select_lot) > 0:
                tag = n % 2
                if tag == 0:
                    select_list = qt_select_lot
                else:
                    select_list = pt_select_lot
                if len(select_list) > 0:
                    lot = select_list.pop(0)
                    or_choice_figure = (machine_type, self.job_status[lot]['type'], self.job_status[lot]['priority'], 
                                        self.job_status[lot]['remain_process_time'], self.job_status[lot]['remain_pending_time'])

                    while or_choice_figure in or_choice_set:
                        if len(select_list) == 0:
                            break
                        else:
                            lot = select_list.pop(0)
                            or_choice_figure = (machine_type, self.job_status[lot]['type'], self.job_status[lot]['priority'], 
                                                self.job_status[lot]['remain_process_time'], self.job_status[lot]['remain_pending_time'])
                    else:
                        lot_collect.append(lot)
                        or_choice_set.add(or_choice_figure)
                        selected_num_by_machine_type[machine_type] += 1
                if selected_num_by_machine_type[machine_type] >= selected_num:
                    break
                n += 1
            lot_selected_dict_by_machine_type[machine_type] = lot_collect
            idle_machine_id.add(agent_id)
            for job_id in lot_collect:
                rest_procedures = self.get_job_procedure(job_id)
                rest_info = rest_procedures[:2]
                job_infos_dict[job_id] = [self.job_status[job_id]['arrival'], rest_info]
                
        # 指定type空闲的机器有多少coff就初始值为多少，然后加上work的机器除2向下取整。
        machine_status_trance_coff = defaultdict(lambda:int(0))
        for machine_id in self.machine_status:
            machine_type = self.machine_status[machine_id]['type']
            if self.machine_status[machine_id]['status'] == 'work':
                job_id = self.machine_status[machine_id]['job']
                arrive_time = self.machine_status[machine_id]['remain_time']
                rest_procedures = self.get_job_procedure(job_id)
                rest_info = rest_procedures[:2]
                machine_status_trance_coff[machine_type] += 0.5
                if len(rest_info) <= 1:
                    continue
                else:
                    job_infos_dict[job_id] = [arrive_time, rest_info]
                
            elif self.machine_status[machine_id]['status'] == 'idle':
                machine_status_trance_coff[machine_type] += 1
        
        for machine_type, coff in machine_status_trance_coff.items():
            machine_status_trance_coff[machine_type] = math.floor(coff)
        
        # jobs_data = [  # task = (machine_id, processing_time).
        #     [(0, 3), (1, 2), (2, 2)],  # Job0
        #     [(0, 2), (2, 1), (1, 4)],  # Job1
        #     [(1, 4), (2, 3)]  # Job2
        # ]
        
        # suspend_infos = [[0,5,10],[4,4,5],[2,5,5]] # job_2 1 has a conflict supend with optimal
        
        # job_arrive_time = [0]*len(jobs_data)
        
        jobs_status_infos = defaultdict(list)
        n = 0
        n_machine_type_id = 0
        job_name_dict_by_n = dict()
        machine_type_dict_by_n = dict()
        for job_id, infos in job_infos_dict.items():
            job_arrive_time = infos[0]
            job_info = []
            suspend_info = []
            k = 0
            for details in infos[1]:
                machine_type = details['machine_type']
                if machine_type not in machine_type_dict_by_n:
                    machine_type_dict_by_n[machine_type] = n_machine_type_id
                    single_machine_id = n_machine_type_id
                    n_machine_type_id += 1
                else:
                    single_machine_id = machine_type_dict_by_n[machine_type]
                process_coff = machine_status_trance_coff[machine_type]
                if k == 0:
                    process_coff_init = process_coff
                k += 1
                suspend_time = details['max_pend_time']
                suspend_info.append(math.floor(suspend_time/process_coff))
                process_time = math.floor(details['process_time']/process_coff)
                job_info.append((single_machine_id, process_time))
            feature_map_count[(tuple(suspend_info),tuple(job_info))] += 1
            if feature_map_count.get((tuple(suspend_info),tuple(job_info))) > 2:
                continue
            elif job_info == []:
                continue
            job_name_dict_by_n[n] = job_id
            n += 1
            jobs_status_infos['job_arrive_time'].append(math.floor(job_arrive_time/process_coff_init))
            jobs_status_infos['suspend_infos'].append(suspend_info)
            jobs_status_infos['jobs_data'].append(job_info)
            jobs_status_infos['priority_punish'].append(int(self.job_status[job_id]['priority']*self.env.ptv_weight + 1))

        # {'job_arrive_time': [0, 0, 0, 0, 0, 0, 0, 6, 5, 15, 12, 19], 
        #  'suspend_infos': [[4, 96], [4, 96], [4, 96], [56, 74, 10], [74, 10], [74, 10], [56, 74, 10], [56, 74, 10], [4, 96], [74, 10], [74, 10], [10]], 
        #  'jobs_data': [[(0, 9), (1, 26)], [(0, 9), (1, 26)], [(0, 9), (1, 26)], [(0, 19), (2, 33), (1, 16)], [(2, 33), (1, 16)], [(2, 33), (1, 16)], [(0, 19), (2, 33), (1, 16)], [(0, 19), (2, 33), (1, 16)], [(0, 9), (1, 26)], [(2, 33), (1, 16)], [(2, 33), (1, 16)], [(1, 16)]], 
        #  'priority_punish': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
        machine_type_dict_by_n = {v: k for k, v in machine_type_dict_by_n.items()}

        if len(jobs_status_infos['jobs_data']) == 1:
            job_id = list(job_infos_dict.keys())
            job_id = job_id[0]
            rest_procedures = self.get_job_procedure(job_id)
            machine_type = rest_procedures[0]['machine_type']
            job_appear_first_dict_by_machine_type = {machine_type:job_id}
            return job_appear_first_dict_by_machine_type
        
        output_data_dict = MinimalJobshopSat(jobs_status_infos)
            
        job_appear_first_dict_by_machine_type = dict()
        for machine_type_index, job_infos in output_data_dict.items():
            machine_type = machine_type_dict_by_n[machine_type_index]
            for (job_index, process_index) in job_infos:
                if process_index == 0:
                    job_id = job_name_dict_by_n[job_index]
                    job_appear_first_dict_by_machine_type[machine_type] = job_id
                    break
                    
        return job_appear_first_dict_by_machine_type

    def get_job_procedure(self, job_id):
        index = int(self.job_status[job_id]['op'][1:]) - 1
        procedures = self.job_types[self.job_status[job_id]['type']][index:]
        return procedures

    def qtfirst(self, agent_id, choice_num = 3):
        candi_list = []
        for job in self.valid_action[agent_id]:
            candi_list.append(job)
        if len(candi_list) == 0:
            return None
        else:
            sorted_list = [a for a in candi_list if self.job_status[a]['priority']>0]
            if len(sorted_list) > 0:
                a = sorted(sorted_list, key=lambda x: (self.job_status[x]['remain_pending_time']/self.job_status[x]['priority'],self.job_status[x]['remain_process_time']))[0:choice_num]
            else:
                a = sorted(candi_list, key=lambda x: self.job_status[x]['remain_pending_time'])[0:choice_num]
            return a

    def ptfirst(self, agent_id, choice_num = 3):
        candi_list = []
        for job in self.valid_action[agent_id]:
            candi_list.append(job)
        if len(candi_list) == 0:
            return None
        else:
            a = sorted(candi_list, key=lambda x: (self.job_status[x]['remain_process_time'], self.job_status[x]['remain_pending_time']))[0:choice_num]
            return a






#%%


def MinimalJobshopSat(jobs_status_infos):
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    # jobs_data = [  # task = (machine_id, processing_time).
    #     [(0, 3), (1, 2), (2, 2)],  # Job0
    #     [(0, 2), (2, 1), (1, 4)],  # Job1
    #     [(1, 4), (2, 3)]  # Job2
    # ]
    jobs_data = jobs_status_infos['jobs_data']
    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    
    # suspend_infos = [[0,5,10],[4,4,5],[2,5,5]] # job_2 1 has a conflict supend with optimal
    suspend_infos = jobs_status_infos['suspend_infos']
    # TODO 之后再改，暂时可选的都是已经到了的
    job_arrive_time = jobs_status_infos['job_arrive_time']
    coff_list = jobs_status_infos['priority_punish']
    
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)
    horizon += sum([sum(_suspend_infos) for _suspend_infos in suspend_infos]) + sum(job_arrive_time)
    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)
    job_starts_dict = defaultdict(dict)
    job_ends_dict = defaultdict(dict)
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(job_arrive_time[job_id], horizon, 'start' + suffix)
            end_var = model.NewIntVar(job_arrive_time[job_id], horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)
            job_starts_dict[job_id][task_id] = start_var
            job_ends_dict[job_id][task_id] = end_var
            
    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)
    
    suspend_dict = defaultdict(list)
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            task_bool = model.NewBoolVar('presence_' + str(job_id) + '_' + str(task_id))
            suspend = model.NewIntVar(0, horizon, 'suspend_' + str(job_id) + '_' + str(task_id))
            if task_id == 0:
                model.Add(job_starts_dict[job_id][task_id] - job_arrive_time[job_id] - suspend_infos[job_id][0] < 0).OnlyEnforceIf(task_bool.Not())
                model.Add(job_starts_dict[job_id][task_id] - job_arrive_time[job_id] - suspend_infos[job_id][0] == suspend).OnlyEnforceIf(task_bool)
            else:
                model.Add(job_starts_dict[job_id][task_id] - job_ends_dict[job_id][task_id-1] - suspend_infos[job_id][task_id] < 0).OnlyEnforceIf(task_bool.Not())
                model.Add(job_starts_dict[job_id][task_id] - job_ends_dict[job_id][task_id-1] - suspend_infos[job_id][task_id] == suspend).OnlyEnforceIf(task_bool)
            suspend_dict[job_id].append(suspend)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    suspend_punishment = []
    for job_id, _suspend_time_list in suspend_dict.items():
        for _suspend_time in _suspend_time_list:
            suspend_punishment.append(coff_list[job_id]*_suspend_time)
    model.Minimize(obj_var+sum(suspend_punishment))

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(start=solver.Value(
                        all_tasks[job_id, task_id].start),
                                       job=job_id,
                                       index=task_id,
                                       duration=task[1]))

        # Create per machine output lines.
        output = ''
        output_data_dict = defaultdict(list)
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-10s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-10s' % sol_tmp
                output_data_dict[machine].append((assigned_task.job, assigned_task.index))
            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        # print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
        # print(output)
    e = time.time()
    # if e-s > 4:
    #     import pdb
    #     for key, value in jobs_status_infos.items():
            
    #         print(key)
    #         print(value)
    #     pdb.set_trace()
    return output_data_dict
# MinimalJobshopSat()