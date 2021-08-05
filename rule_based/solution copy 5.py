import math
import numpy as np
# import collections
import copy
# import time
# Import Python wrapper for or-tools CP-SAT solver.
# from ortools.sat.python import cp_model
import pdb
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box, Tuple, Dict
from itertools import permutations
# import matplotlib.pyplot as plt
import time
# from copy import deepcopy
# from visualdl import LogWriter
# import pdb
# import datetime
# import numpy as np
# import random
# from parl.utils import logger
from collections import defaultdict
# over priority not in 
# over arrival replace processtime  
# over remove already use
# over permu priority
# todo breakdown
# todo hybirid
class Trainer:
    def __init__(self, Env, conf_list):
        self.conf_list = conf_list
        self.Env = Env
        self.checkpoint = None
        self.iter = 0

    def train(self, run_time):
        env = self.Env(self.conf_list[0])
        obs = env.reset()
        
        return Agent(env.job_types)

class Agent:
    def __init__(self, job_types, test=False):
        self.job_types = job_types
        self.test = test
        # ('A01', {'machine_type': 'A', 'max_pend_time': 32, 'op_name': 'A01', 'process_time': 19})
        self.opInfos={op['op_name']:op for ops in job_types.values() for op in ops }
        self.opNextOp={}
        self.mtNextOp={}
        self.opNextProcess={}
        for ops in job_types.values():
            for i in range(len(ops)):
                thisName=ops[i]["op_name"]
                mt=ops[i]["machine_type"]
                self.opNextProcess[thisName]=len(ops)-i
                if i<len(ops)-1:
                    self.opNextOp[thisName]=ops[i+1]
                    self.mtNextOp[mt]=ops[i+1]                
    
    def priMin(self,job_status):
        
        def get_next_op_info(self,job,job_status):
            if job_status[job]['status'] == 'to_arrive':
                job_type = job_status[job]['type']
                next_op = self.job_types[job_type][0]
                return {'machine':'A', 'next_max_pending_time':next_op['max_pend_time']}
            else:
                job_type = job_status[job]['type']
                now_op = job_status[job]['op']
                next_op= self.opNextOp[now_op] if now_op in self.opNextOp else None
                next_op_info = {'machine':next_op['machine_type'], 'next_max_pending_time':next_op['max_pend_time']} if next_op is not None \
                    else {'machine':None, 'next_max_pending_time':0} if job_type!='d' else {
                        'machine':'D', 'next_max_pending_time':self.opInfos[now_op]['max_pend_time']
                    }
                return next_op_info  
        
        timeMin={}
        for job in [jobName  for jobName,jobItem in job_status.items() if jobItem['priority']> 0]:
            if (job_status[job]['status'] == 'work' or job_status[job]['status'] == 'to_arrive'):
                next_op_info = get_next_op_info(self,job,job_status)
                mt=next_op_info['machine']
                base=job_status[job]["remain_process_time"]
                if(job_status[job]['status']=='to_arrive'):
                    base=job_status[job]['arrival']
                timeMin[mt]=min(timeMin.get(mt,99999), base + next_op_info['next_max_pending_time'])        
        return timeMin
    
    def mcLoad(self,machine_status):
        return

    def simulation_by_data(self, action, machine_status_copy, job_status_copy, t):
        
        ## 输入上一个状态，获得下一个状态
        to_arrive_job = set()
        just_arrive_job = set()
        
        working_jobs = set()
        just_working_finished_jobs = set()
        
        done_jobs = set()
        
        pending_jobs = set()
        jobs_list_rebuild_by_machine_type = defaultdict(list)
        for job, job_state in job_status_copy.items():
            job_type = job_state['type']
            op_index = int(job_state['op'][-1])
            need_machine_type = self.job_types[job_type][op_index-1]['machine_type']
            
            if job_state['status'] == 'work':
                if job_state['remain_process_time'] == 1:
                    just_working_finished_jobs.add(job)
                    if op_index < len(self.job_types[job_type]):
                        need_machine_type = self.job_types[job_type][op_index]['machine_type']
                        jobs_list_rebuild_by_machine_type[need_machine_type].append(job)
                else:
                    working_jobs.add(job)
                    
            elif job_state['status'] == 'to_arrive':
                if job_state['arrival'] == 1:
                    just_arrive_job.add(job)
                else:
                    to_arrive_job.add(job)
                    
            elif job_state['status'] == 'done':
                done_jobs.add(job)
                
            else:
                pending_jobs.add(job)
                jobs_list_rebuild_by_machine_type[need_machine_type].append(job)        
        
        working_machines = set()
        
        just_working_finished_machine = set()
        
        idle_machine = set()
        down_machine_set = set()
        for machine, machine_status in machine_status_copy.items():
            
            if machine_status['status'] == 'idle':
                idle_machine.add(machine)
                
            elif machine_status['status'] == 'work':
                if machine_status['remain_time'] == 1:
                    just_working_finished_machine.add(machine)
                else:
                    working_machines.add(machine)
            elif machine_status['status'] == 'down':
                if not self.have_re_test:
                    self.re_test =True
                else:
                    self.re_test =False
                down_machine_set.add(machine)

        selected_job_set = set()
        selected_machine_set = set()
        selected_job_by_machine_type = defaultdict(set)
        for machine, job_id in action.items():
            if machine not in idle_machine or machine in down_machine_set:
                continue
            elif job_id in working_jobs or job_id in just_working_finished_machine or job_id in done_jobs:
                continue
            elif job_id in selected_job_set:
                continue
            elif job_id in to_arrive_job or job_id in just_arrive_job:
                continue
            else:
                ## change job state
                # machine# 7 'machine'
                machine_type = machine_status_copy[machine]['type']
                job_type = job_status_copy[job_id]['type']
                
                now_op = job_status_copy[job_id]['op'] # 4 'op'
                now_state_index = int(job_status_copy[job_id]['op'][-1]) - 1
                status = 'work' # 2 'status'
                    
                remain_process_time = self.job_types[job_type][now_state_index]['process_time'] # 5 'remain_process_time'
                remain_pending_time = math.inf # 6 'remain_pending_time'
                selected_job_set.add(job_id)
                selected_machine_set.add(machine)
                pending_jobs.remove(job_id)
                jobs_list_rebuild_by_machine_type[machine_type].remove(job_id)
                
                job_status_copy[job_id]['status'] = status
                job_status_copy[job_id]['op'] = now_op
                job_status_copy[job_id]['remain_process_time'] = remain_process_time - 1
                job_status_copy[job_id]['remain_pending_time'] = remain_pending_time
                job_status_copy[job_id]['machine'] = machine
                
                ## change machine state
                # type 01; status work 02; remain_time remain_process_time 03; job job_id 04; job_list set() 05;
                machine_status_copy[machine]['status'] = status
                machine_status_copy[machine]['remain_time'] = remain_process_time - 1
                machine_status_copy[machine]['job'] = job_id
                machine_status_copy[machine]['job_list'] = set()
                
                selected_job_by_machine_type[machine_type].add(job_id)
        
        arrive_job_by_machine_type = defaultdict(set)
        now_finished_set = set()
        for job_id in just_working_finished_jobs:
            job_type = job_status_copy[job_id]['type']
            now_state_index = int(job_status_copy[job_id]['op'][-1]) - 1
            
            # now_option = self.job_types[job_type][now_state_index]['op_name'] # 4 'op'
            ## 考虑任务为done的情况
            if len(self.job_types[job_type]) > now_state_index+1:
                next_option = self.job_types[job_type][now_state_index+1]['op_name'] # 4 'op'
                status = 'pending'
                remain_process_time = self.job_types[job_type][now_state_index+1]['process_time'] # 5 'remain_process_time'
                remain_pending_time = self.job_types[job_type][now_state_index+1]['max_pend_time'] # 6 'remain_pending_time'
                machine_type = self.job_types[job_type][now_state_index+1]['machine_type']
                arrive_job_by_machine_type[machine_type].add(job_id)
            else:
                next_option = job_status_copy[job_id]['op'] # 4 'op'
                status = 'done'
                remain_process_time = 0
                remain_pending_time = math.inf # 6 'remain_pending_time'
                now_finished_set.add(job_id)
                
            job_status_copy[job_id]['status'] = status
            job_status_copy[job_id]['op'] = next_option
            job_status_copy[job_id]['remain_process_time'] = remain_process_time
            job_status_copy[job_id]['remain_pending_time'] = remain_pending_time
            job_status_copy[job_id]['machine'] = None
            
        if len(now_finished_set) + len(done_jobs) >= len(job_status_copy.keys()):
            done = True
        else:
            done = False
        
        for job_id in just_arrive_job:
            job_type = job_status_copy[job_id]['type']
            now_state_index = int(job_status_copy[job_id]['op'][-1])-1
            machine_type = self.job_types[job_type][now_state_index]['machine_type']
            arrive_job_by_machine_type[machine_type].add(job_id)
            next_option = self.job_types[job_type][now_state_index]['op_name'] # 4 'op'
            status = 'pending'
            remain_process_time = self.job_types[job_type][now_state_index]['process_time'] # 5 'remain_process_time'
            remain_pending_time = self.job_types[job_type][now_state_index]['max_pend_time'] # 6 'remain_pending_time'

            job_status_copy[job_id]['status'] = status
            job_status_copy[job_id]['op'] = next_option
            job_status_copy[job_id]['remain_process_time'] = remain_process_time
            job_status_copy[job_id]['remain_pending_time'] = remain_pending_time
            job_status_copy[job_id]['machine'] = None
            job_status_copy[job_id]['arrival'] = 0
        
        for machine_type, job_list in jobs_list_rebuild_by_machine_type.items():
            selected_job_set = selected_job_by_machine_type[machine_type]
            add_job_set = arrive_job_by_machine_type[machine_type]
            job_set = set(job_list)
            job_set -= selected_job_set 
            job_set |= add_job_set
            jobs_list_rebuild_by_machine_type[machine_type] = list(job_set)
        
        for machine_type, jobset in arrive_job_by_machine_type.items():
            add_job_set = arrive_job_by_machine_type[machine_type]
            job_set = set(jobs_list_rebuild_by_machine_type[machine_type])
            job_set |= add_job_set
            jobs_list_rebuild_by_machine_type[machine_type] = list(job_set)
        
        for machine in just_working_finished_machine:
            if machine in down_machine_set:
                continue
            machine_type = machine_status_copy[machine]['type']
            job_list = jobs_list_rebuild_by_machine_type[machine_type]
            machine_status_copy[machine]['status'] = 'idle'
            machine_status_copy[machine]['remain_time'] = 0
            machine_status_copy[machine]['job'] = None
            machine_status_copy[machine]['job_list'] = job_list
        
        job_list_out = defaultdict(list)
        for machine, machine_info in machine_status_copy.items():
            machine_type = machine_status_copy[machine]['type']
            if machine_info['status'] == 'idle':
                job_list = jobs_list_rebuild_by_machine_type[machine_type]
                machine_status_copy[machine]['job_list'] = job_list
                job_list_out[machine] = job_list
            else:
                job_list_out[machine] = []

            if machine_info['status'] == 'work' and machine not in selected_machine_set:
                machine_status_copy[machine]['remain_time'] -= 1
        
        pri_reward = 0
        for job, info in job_status_copy.items():
            if job in pending_jobs:
                status = info['status']
                if status == 'pending':
                    info['remain_pending_time'] -= 1
                    if info['priority'] > 0:
                        pri_reward += 10 * info['priority']
            elif job in to_arrive_job:
                # elif status == 'to_arrive':
                #     job_status_copy[]
                info['arrival'] -= 1
            elif job in working_jobs:
                info['remain_process_time'] -= 1
        t += 1
        reward = -1 - pri_reward
        return machine_status_copy, job_status_copy, t, reward, job_list_out, done
    
    def act(self, machine_status, job_status, time, job_list):

        if time ==0 and self.test == False:
            job_types_list = list(self.job_types.keys())
            job_type_minimaze_dict = defaultdict(dict)
            for job_type, infos in self.job_types.items():
                time_curr = 0
                for info in infos:
                    machine_type = info['machine_type']
                    process_time = info['process_time']
                    job_type_minimaze_dict[machine_type][job_type] = time_curr
                    time_curr += process_time
            
            
            total_time_by_machine_type = defaultdict(lambda:int(0))
            
            for job_id, infos in job_status.items():
                job_type = infos['type']
                for info in self.job_types[job_type]:
                    machine_type = info['machine_type']
                    process_time = info['process_time']
                    total_time_by_machine_type[machine_type] += process_time
            
            
            c_num, d_num = 0, 0
            for machine, info in machine_status.items():
                if info['type'] == 'C':
                    c_num += 1
                elif info['type'] == 'D':
                    d_num += 1
            
            mean_time_c = total_time_by_machine_type['C'] / c_num
            mean_time_d = total_time_by_machine_type['D'] / d_num
            
            emergency_machine_type = 'C' if mean_time_c > mean_time_d else 'D'
            print(c_num, mean_time_c)
            print(d_num, mean_time_d)
            if mean_time_c - mean_time_d > 10:
                diff = mean_time_c - mean_time_d - 30
            elif mean_time_d - mean_time_c > 0:
                diff = mean_time_d - mean_time_c + 30
            else:
                diff = 0
                
            if abs(diff) <= max(mean_time_c, mean_time_d)/10:
                print('beyond')
                priority_dict_base = dict()
                for job in job_status:
                    priority_dict_base[job] = job_status[job]['priority']
                self.dynamic_priority_change = priority_dict_base
            else:
                print('change')
                job_types_list = list(self.job_types.keys())
                job_type_minimaze_dict = defaultdict(dict)
                for job_type, infos in self.job_types.items():
                    time_curr = 0
                    for info in infos:
                        machine_type = info['machine_type']
                        process_time = info['process_time']
                        job_type_minimaze_dict[machine_type][job_type] = time_curr
                        time_curr += process_time
                        
                                
                check_time_total = 5
                max_iter = 5
                priority_record_dict = dict()
                while check_time_total > 0:
                    machine_status_copy = copy.deepcopy(machine_status)
                    job_status_copy = copy.deepcopy(job_status)
                    t = copy.deepcopy(time)
                    job_list_copy = copy.deepcopy(job_list)
                    close_list = defaultdict(set)
                    draw_infos = defaultdict(dict)
                    job_stage = defaultdict(lambda:defaultdict(lambda:False))
                    breakdown_info = defaultdict(list)
                    if check_time_total == max_iter:
                        priority_dict_base = dict()
                        for job in job_status_copy:
                            priority_dict_base[job] = job_status_copy[job]['priority']
                            self.dynamic_priority_change = priority_dict_base
                    else:
                        for job in change_job_set:
                            job_status_copy[job]['priority'] = priority_dict_base[job]
                            self.dynamic_priority_change = priority_dict_base
                    rewards = 0
                    agent_test = Agent(self.job_types, True)
                    done = False
                    change_job_set = set()
                    while not done and t < 1000:
                        action = agent_test.act(machine_status_copy, job_status_copy, t, job_list_copy)
                        machine_status_copy, job_status_copy, t, reward, job_list_copy, done = self.simulation_by_data(action, machine_status_copy, job_status_copy, t)
                        rewards += reward
                    else:
                        idle_time_mean_max, delay_job_info_dict = self.draw_ye(draw_infos, job_stage, breakdown_info, machine_status, check_time_total)
                        emergency_type = idle_time_mean_max[1]
                        if emergency_type == emergency_machine_type:
                            print('--------check--------')
                            print(emergency_type)
                            print(emergency_machine_type)
                            check_time_total -= 1
                        else:
                            if (rewards, -idle_time_mean_max[0]) not in priority_record_dict:
                                priority_record_dict[(rewards, -idle_time_mean_max[0])] = copy.deepcopy(priority_dict_base)
                            check_time_total -= 1
                            # priority_record_dict.keys().tolist()
                            job_types_list_copy = job_types_list[:]
                            job_type_minimaze_dict_copy = copy.deepcopy(job_type_minimaze_dict)

                            # emergency_time = idle_time_mean_max[0]
                            for job_type in job_types_list_copy:
                                if job_type not in job_type_minimaze_dict_copy[emergency_type]:
                                    job_types_list_copy.remove(job_type)
                            
                            job_types_list_copy = sorted(job_types_list_copy, key=lambda x:job_type_minimaze_dict_copy[emergency_type][x], reverse=True)
                            job_choice_num = 0
                            for machine_name, info in machine_status.items():
                                if info['type'] == emergency_type:
                                    job_choice_num += 1
                                    
                            job_choice_num = int(2*job_choice_num)
                            while job_choice_num > 0 and len(job_types_list_copy)>0:
                                choice_job_type = job_types_list_copy.pop(0)
                                job_candidate = [a for a in job_status if job_status[a]['status'] == 'pending' and job_status[a]['type']==choice_job_type]
                                job_candidate = sorted(job_candidate, key=lambda x:(self.dynamic_priority_change[x], job_status[x]['remain_pending_time']), reverse=True)
                                while job_choice_num > 0:
                                    if len(job_candidate) > 0:
                                        job_choice = job_candidate.pop(0)
                                        if job_choice not in change_job_set:
                                            priority_dict_base[job_choice] += 1
                                        else:
                                            priority_dict_base[job_choice] += 0.25
                                        # print(job_choice, ':', priority_dict_base[job_choice])
                                        change_job_set.add(job_choice)
                                        job_choice_num -= 1
                                    else:
                                        break
                            for delay_job in delay_job_info_dict:
                                priority_dict_base[delay_job] += 5
                                change_job_set.add(delay_job)
                    
                
                job_status.keys()
                sort_key_list = list(priority_record_dict.keys())
                
                sort_key_list.sort()
                if len(sort_key_list) >0:
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!change!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print(sort_key_list)
                    self.dynamic_priority_change = priority_record_dict[sort_key_list[-1]]
                else:
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!reserve!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    priority_dict_base = dict()
                    for job in job_status:
                        priority_dict_base[job] = job_status[job]['priority']
                    self.dynamic_priority_change = priority_dict_base

        if self.test:
            priority_dict_base = dict()
            for job in job_status:
                priority_dict_base[job] = job_status[job]['priority']
            self.dynamic_priority_change = priority_dict_base
            
        action = {}
        priMin=self.priMin(job_status)
        alreadIn=set()
        for machine in sorted(job_list):
            job = self.adv_wwsqt(machine, machine_status, job_status, time,[a for a in job_list[machine] if a not in alreadIn ],priMin)
            if job is not None:
                action[machine] = job
                alreadIn.add(job)
        return action


    def adv_wwsqt(self, machine, machine_status, job_status, time, job_list,priMin):
        if len(job_list) == 0:
            return None
        else:
            sorted_list = [a for a in job_list if self.dynamic_priority_change[a]>0]

            if len(sorted_list) == 0:
                machine_type = machine_status[machine]['type']
                max_time=priMin.get(machine_type,999999)
                job_list=[a for a in job_list if job_status[a]['remain_process_time']<max_time]
                if len(job_list)>0:
                    return sorted(job_list, key=lambda x: (job_status[x]['remain_process_time']-math.sqrt(self.opNextProcess[job_status[x]['op']]),job_status[x]['remain_pending_time']))[0]
                else:
                    return None
            else:
                return self.getJob(sorted_list,job_status)

    def getJob(self,sorted_list,job_status):
        if(len(sorted_list)==1):return sorted_list[0]
        if(sum([job_status[x]['remain_process_time'] for x in sorted_list])< min([job_status[x]['remain_pending_time'] for x in sorted_list])):
            return sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/self.dynamic_priority_change[x],job_status[x]['remain_process_time']))[0]
        best_set=None
        number=9999999
        permutations_list = sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/self.dynamic_priority_change[x], job_status[x]['remain_process_time']))[:6]
        for sor in list(permutations(permutations_list)):
            process_time=0
            score=0
            for job in sor:
                thisJob=job_status[job]
                if(thisJob["remain_pending_time"]<process_time):
                    score+=(process_time-thisJob["remain_pending_time"])*self.dynamic_priority_change[job]
                process_time+=thisJob["remain_process_time"]
            if(score<number):
                best_set=sor
                number=score
                if(score==0): 
                    return sor[0]
        return best_set[0]



    # def adv_wwsqt(self, machine, machine_status, job_status, time, job_list,priMin):
    #     if len(job_list) == 0:
    #         return None
    #     else:
    #         sorted_list = [a for a in job_list if job_status[a]['priority']>0]
    #         if len(sorted_list) == 0:
    #             sorted_list = [a for a in job_list if self.dynamic_priority_change[a]>0]
    #             if len(sorted_list) > 0:
    #                 return self.getJob_dg(sorted_list, job_status)
    #             else:
    #                 sorted_list = []
            
    #         if len(sorted_list) == 0:
    #             machine_type = machine_status[machine]['type']
    #             max_time=priMin.get(machine_type,999999)
    #             job_list=[a for a in job_list if job_status[a]['remain_process_time']<max_time]
    #             if len(job_list)>0:
    #                 return sorted(job_list, key=lambda x: (job_status[x]['remain_process_time']-math.sqrt(self.opNextProcess[job_status[x]['op']]),job_status[x]['remain_pending_time']))[0]
    #             else:
    #                 return None
    #         else:
    #             return self.getJob(sorted_list,job_status)

    # def getJob(self,sorted_list,job_status):
    #     if(len(sorted_list)==1):return sorted_list[0]
    #     if(sum([job_status[x]['remain_process_time'] for x in sorted_list])< min([job_status[x]['remain_pending_time'] for x in sorted_list])):
    #         return sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/self.dynamic_priority_change[x],job_status[x]['remain_process_time']))[0]
    #     best_set=None
    #     number=9999999
    #     permutations_list = sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/self.dynamic_priority_change[x], job_status[x]['remain_process_time']))[:6]
    #     for sor in list(permutations(permutations_list)):
    #         process_time=0
    #         score=0
    #         for job in sor:
    #             thisJob=job_status[job]
    #             if(thisJob["remain_pending_time"]<process_time):
    #                 score+=(process_time-thisJob["remain_pending_time"])*thisJob["priority"]
    #             process_time+=thisJob["remain_process_time"]
    #         if(score<number):
    #             best_set=sor
    #             number=score
    #             if(score==0): 
    #                 return sor[0]
    #     return best_set[0]


    # def getJob_dg(self,sorted_list,job_status):
    #     if(len(sorted_list)==1):return sorted_list[0]
    #     if(sum([job_status[x]['remain_process_time'] for x in sorted_list])< min([job_status[x]['remain_pending_time'] for x in sorted_list])):
    #         return sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/self.dynamic_priority_change[x], job_status[x]['remain_process_time']))[0]
    #     best_set=None
    #     number=9999999
    #     permutations_list = sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/self.dynamic_priority_change[x], job_status[x]['remain_process_time']))[:6]
    #     for sor in list(permutations(permutations_list)):
    #         process_time=0
    #         score=0
    #         for job in sor:
    #             thisJob=job_status[job]
    #             if(thisJob["remain_pending_time"]<process_time):
    #                 score+=(process_time-thisJob["remain_pending_time"])*self.dynamic_priority_change[job]
    #             process_time+=thisJob["remain_process_time"]
    #         if(score<number):
    #             best_set=sor
    #             number=score
    #             if(score==0): 
    #                 return sor[0]
    #     return best_set[0]
    
    

    def draw_ye(self, draw_infos, job_stage, breakdown_info, machine_status, check_time_total):
        colors=["greenyellow","g","yellow",'cyan','dodgerblue','darkblue','violet','purple']
        alarm = 'r'
        pending = 'lightgray'
        # print('breakdown_info:')
        # print(breakdown_info)
        # fig=plt.figure()
        # draw_infos[machine][job] = (t, process_info['remain_time']-pending_time, job_status['priority'], pending_time)
        machines_idle_time={}
        delay_job_info_dict = {}
        machines_name_list = []
        for machine, jobs_info_dict in draw_infos.items():
            machine_no = len(machines_name_list) 
            machines_name_list.append(machine + '_' + machine_status[machine]['type'])
            
            for job, job_info in jobs_info_dict.items():
                if job_info[3] >= 0:
                    # plt.barh(y=[machine_no], width=[job_info[3]], height=0.5, left=[job_info[0]],color=pending)
                    pass
                else:
                    # plt.barh(y=[machine_no], width=[-job_info[3]], height=0.5, left=[job_info[0]],color=alarm)
                    delay_job_info_dict[job] = (job_info[2], -job_info[3]) ## priority, delay_time
                
                if machines_name_list[-1] in machines_idle_time:
                    machines_idle_time[machines_name_list[-1]] = min(machines_idle_time[machines_name_list[-1]], job_info[0])
                else:
                    machines_idle_time[machines_name_list[-1]] = job_info[0]
                # plt.barh(y=[machine_no], width=[job_info[1]], height=0.5, left=[job_info[0]+abs(job_info[3])],color=colors[int(job_info[2])])
                # plt.barh(y=[machine_no], width=[1], height=0.5, left=[job_info[0]+abs(job_info[3])+job_info[1]],color='k')

        cal_dict = defaultdict(list)
        for key, value in machines_idle_time.items():
            machine = key.split('_')[0]
            machine_type = key.split('_')[1]
            idle_time = value
            cal_dict[machine_type].append(idle_time)
        
        machine_list = list(cal_dict.keys())
        machine_list = sorted(machine_list, key=lambda x: np.mean(cal_dict[x]), reverse=True)
        idle_time_mean_max = (np.mean(cal_dict[machine_list[0]]), machine_list[0])
        
        # plt.yticks(np.arange(len(machines_name_list))+0.25, machines_name_list)
        # plt.savefig("/home/quicktron_ye/Documents/competition/yu_test_pro/rule_based/result_pic/"+time.strftime("_%m_%d_%H_%M_%S", time.localtime())
        #             + '_' + str(check_time_total) + '.png', dpi=1000, format='png', bbox_inches="tight") 
        # plt.close()
        return idle_time_mean_max, delay_job_info_dict