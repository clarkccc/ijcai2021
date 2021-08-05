import numpy as np
from collections import defaultdict
from copy import deepcopy

import math
import collections
import copy
from ortools.sat.python import cp_model

class Agent:
    def __init__(self, trainer, env):
        self.trainer = trainer
        self.env = env()
        self.env.reset()
        self.last_valid_action = None
        self.action_size = 4

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

        self.job_loc = {}

        machine_dict = {
            'A':0,
            'B':1,
            'C':2,
            'D':3
        }
        for job in self.job_status:
            job_op = self.job_status[job]['op']
            job_type = self.job_status[job]['type']

            for op in self.env.env.job_types[job_type]:
                if op['op_name'] == job_op:
                    op_machine = op['machine_type']
                    self.job_loc[job] = machine_dict[op_machine]
                    break

        for machine in self.valid_action:
            self.valid_machine_type[self.machine_status[machine]['type']].append(machine)

        if self._valid_action_step():
            self.last_valid_action = deepcopy(self.valid_action)

            return {}

        else:
            self.last_valid_action = deepcopy(self.valid_action)
            self.get_candidates()
            obs = {}
            for machine_type in self.valid_machine_type:
                if len(self.valid_machine_type[machine_type]) > 0:
                    agent_id = self.valid_machine_type[machine_type][0]
                    obs[agent_id] = {
                        'obs':self.gen_observation(agent_id),
                        'action_mask':self.get_action_mask(agent_id)
                    }

            rl_actions = {}
            for agent_id in obs:
                rl_actions[agent_id] = self.trainer.compute_action(obs[agent_id], explore=False)

            step_actions = {}
            eval_actions = {}
            for key in rl_actions:
                if rl_actions[key] == 0:
                    continue
                else:
                    act_choice = rl_actions[key]-1
                    try:
                        # 这里暂时没有排除的方案，所以不会有none
                        a_ = self.candidates[key][act_choice]
                    except:
                        a_ = None
                        
                    if a_ is not None:
                        eval_actions[key] = a_
                        self.job_list[key] = []
                        del self.valid_action[key]
                        for tool in self.valid_action:
                            if a_ in self.valid_action[tool]:
                                self.valid_action[tool].remove(a_)
                                
            for k in eval_actions:
                self.job_loc[eval_actions[k]] += 1
            step_actions.update(eval_actions)

            while self.check_real_step():
                for key in eval_actions:
                    job = eval_actions[key]
                    for machine in self.machine_status:
                        if key == machine:
                            self.machine_status[machine].update({
                                'job_list':[],
                                'status':'work',
                                'job': job,
                            })
                        else:
                            if job in self.machine_status[machine]['job_list']:
                                self.machine_status[machine]['job_list'].remove(job)
                    self.job_status[job].update({
                        'status':'work',
                        'op': self.get_job_op(job, key),
                        'machine':key
                    })
                    self.job_list[key] = []

                for key in self.job_list:
                    if job in self.job_list[key]:
                        self.job_list[key].remove(job)

                self.valid_action = {}
                for machine in self.machine_status:
                    if machine in self.job_list and len(self.job_list[machine]) != 0:
                        self.valid_action[machine] = list(self.job_list[machine])

                self.get_candidates()

                obs = {}
                for machine_type in self.valid_machine_type:
                    if len(self.valid_machine_type[machine_type]) > 0:
                        agent_id = self.valid_machine_type[machine_type][0]
                        obs[agent_id] = {
                            'obs':self.gen_observation(agent_id),
                            'action_mask':self.get_action_mask(agent_id)
                        }

                rl_actions = {}
                for agent_id in obs:
                    rl_actions[agent_id] = self.trainer.compute_action(obs[agent_id], explore=False)

                eval_actions = {}
                for key in rl_actions:
                    if rl_actions[key] == 0:
                        continue
                    else:
                        act_choice = rl_actions[key]-1
                        try:
                            # 这里暂时没有排除的方案，所以不会有none
                            a_ = self.candidates[key][act_choice]
                        except:
                            a_ = None
                            
                        if a_ is not None:
                            eval_actions[key] = a_
                            self.job_list[key] = []
                            del self.valid_action[key]
                            for tool in self.valid_action:
                                if a_ in self.valid_action[tool]:
                                    self.valid_action[tool].remove(a_)
                                
                for k in eval_actions:
                    self.job_loc[eval_actions[k]] += 1
                step_actions.update(eval_actions)

        return step_actions


    def get_candidates(self):
        # TODO
        self.candidates = dict()
        candidates_by_type = dict()
        for machine_id in self.valid_action:
            machine_type = self.machine_status[machine_id]['type']
            if machine_type in candidates_by_type:
                self.candidates[machine_id] = candidates_by_type[machine_id]
            else:
                _temp_candidate = []
                check_set = set()
                nop_list = self.next_op_first(machine_id)
                qtf_list = self.qtfirst(machine_id)
                ptf_list = self.ptfirst(machine_id)
                for target_list in [nop_list, qtf_list, ptf_list]:
                    choice_job_id = self.get_first_choice_without_duplicate(target_list, machine_type, check_set)
                    _temp_candidate.append(choice_job_id)
                self.candidates[machine_id] = _temp_candidate

        self.lots_all = {}

        for i in range(len(self.env.rl_agent_id)):
            tool_id = self.env.rl_agent_id[i]
            if self.machine_status[tool_id]['type'] not in self.lots_all:
                self.lots_all[self.machine_status[tool_id]['type']] = []
            if tool_id in self.candidates:
                self.lots_all[self.machine_status[tool_id]['type']] += [list(self.candidates[tool_id])]
            else:
                self.lots_all[self.machine_status[tool_id]['type']] += [[]]*(self.env.action_size-1)


            self.candidates = dict()
            candidates_by_type = dict()
            for machine_id in self.valid_action:
                machine_type = self.machine_status[machine_id]['type']
                if machine_type in candidates_by_type:
                    self.candidates[machine_id] = candidates_by_type[machine_id]
                else:
                    _temp_candidate = []
                    check_set = set()
                    nop_list = self.next_op_first(machine_id)
                    qtf_list = self.qtfirst(machine_id)
                    ptf_list = self.ptfirst(machine_id)
                    for target_list in [nop_list, qtf_list, ptf_list]:
                        choice_job_id = self.get_first_choice_without_duplicate(target_list, machine_type, check_set)
                        _temp_candidate.append(choice_job_id)
                    self.candidates[machine_id] = _temp_candidate

            self.lots_all = {}

            for i in range(len(self.env.rl_agent_id)):
                tool_id = self.env.rl_agent_id[i]
                if self.machine_status[tool_id]['type'] not in self.lots_all:
                    self.lots_all[self.machine_status[tool_id]['type']] = []
                if tool_id in self.candidates:
                    self.lots_all[self.machine_status[tool_id]['type']] += [list(self.candidates[tool_id])]
                else:
                    self.lots_all[self.machine_status[tool_id]['type']] += [[]]*(self.action_size-1)
    
    def get_job_procedure(self, job_id):
        index = int(self.job_status[job_id]['op'][1:]) - 1
        procedures = self.env.env.job_types[self.job_status[job_id]['type']][index:]
        return procedures
        
    def get_job_feature(self, machine_type, job_id):
        feature = (machine_type, self.job_status[job_id]['type'], self.job_status[job_id]['priority'], 
                                    self.job_status[job_id]['remain_process_time'], self.job_status[job_id]['remain_pending_time'])
        return feature
    
    def get_first_choice_without_duplicate(self, choice_list, machine_type, check_set):
        temp_choice_job_id = choice_list.pop(0)
        first_try_job_id = temp_choice_job_id
        temp_choice_feature = self.get_job_feature(machine_type, temp_choice_job_id)
        while temp_choice_feature in check_set:
            if len(choice_list) > 0:
                temp_choice_job_id = choice_list.pop(0)
                temp_choice_feature = self.get_job_feature(machine_type, temp_choice_job_id)
            else:
                return first_try_job_id
        else:
            check_set.add(temp_choice_feature)
            return temp_choice_job_id
        

    def next_op_first(self, agent_id):
        candi_list = []
        candi_list_sub = []
        candi_list_info = dict()
        for job_id in self.valid_action[agent_id]:
            rest_procedures = self.get_job_procedure(job_id)
            if len(rest_procedures) >= 2:
                candi_list.append(job_id)
                candi_list_info[job_id] = rest_procedures
            else:
                candi_list_sub.append(job_id)
        
        if len(candi_list) == 0:
            if len(candi_list_sub) == 0:
                return None
            else:
                a = sorted(candi_list_sub, key=lambda x: (self.job_status[x]['remain_process_time'], self.job_status[x]['remain_pending_time']))[:]
                return a
        else:
            a = sorted(candi_list, key=lambda x: (candi_list_info[x][0]['process_time']/candi_list_info[x][1]['process_time']))[:]
            return a

    def qtfirst(self, agent_id):
        candi_list = []
        for job in self.valid_action[agent_id]:
            candi_list.append(job)
        if len(candi_list) == 0:
            return None
        else:
            sorted_list = [a for a in candi_list if self.job_status[a]['priority']>0]
            if len(sorted_list) > 0:
                a = sorted(sorted_list, key=lambda x: (self.job_status[x]['remain_pending_time']/self.job_status[x]['priority'],self.job_status[x]['remain_process_time']))[:]
            else:
                a = sorted(candi_list, key=lambda x: self.job_status[x]['remain_pending_time'])[:]
            return a

    def ptfirst(self, agent_id):
        candi_list = []
        for job in self.valid_action[agent_id]:
            candi_list.append(job)
        if len(candi_list) == 0:
            return None
        else:
            a = sorted(candi_list, key=lambda x: (self.job_status[x]['remain_process_time'], self.job_status[x]['remain_pending_time']))[:]
            return a


    def gen_observation(self, agent_id):
        '''
        :param agent_id:
        :return:
            features
            (length 95)
        '''
        agent_obs = []

        valid_actions = self.valid_action[agent_id] if agent_id in self.valid_action else []
        ft = self.gen_lots_features(valid_actions)
        agent_obs += ft

        ft = self.get_act_rule_feature(agent_id)
        agent_obs += ft

        ft = self.get_working_jobs_feature(agent_id)
        agent_obs += ft

        ft = self.get_pending_jobs_feature(agent_id)
        agent_obs += ft

        ft = self.get_machine_feature(agent_id)
        agent_obs += ft

        machine_type = self.machine_status[agent_id]['type']
        ft_dict = {
            'A':0,
            'B':1,
            'C':2,
            'D':3
        }
        ft = [0,0,0,0]
        ft[ft_dict[machine_type]] = 1
        agent_obs += ft
        return np.array(agent_obs)

    def get_act_rule_feature(self, agent_id):
        # length: 20
        ft = []
        ft += self.get_act_rule_job_feature(agent_id)
        ft += self.get_act_rule_machine_feature(agent_id)
        ft += self.get_act_rule_prev_feature(agent_id)

        return ft

    def gen_lots_features(self, lots):
        # length: 10
        ft = []
        pt = np.array(self.get_attr(lots, 'remain_process_time'))
        rqt = np.array(self.get_attr(lots, 'remain_pending_time'))
        # length of neg candidate jobs
        neg_num = sum(rqt<0)
        ft.append(neg_num)

        ## 超时的时间长度
        if neg_num > 0 :
            ft.append(np.std(rqt<0)/np.mean(rqt<0))
            ft.append(max(np.min(rqt<0),-10))
        else:
            ft += [0,0]

        pos_num = len(rqt) - neg_num
        ## 可挂件的总体特征
        if pos_num > 0:
            ft.append(pos_num)
            ft.append(np.std(rqt>0))
            ft.append(np.max(rqt>0))
            ft.append(np.min(rqt>0))
        else:
            ft += [0, 0, 0, 0]
        
        balance_value = ft[-1] + 1
        ## 处理时间综合优先级的特征
        if len(pt) > 0:
            ft.append(np.std(pt)/np.mean(pt))
            ft.append(np.max(pt)-balance_value)
            ft.append(np.min(pt)-balance_value)
        else:
            ft += [0,0,0]
        return ft
    
    def get_act_rule_job_feature(self, agent_id):
        # job features when action rule done
        # length: 4
        # TODO 加选择这里要+2个0
        ft = []
        if agent_id not in self.candidates:
            return [0,0,0,0]
        
        # TODO 这里做个比较
        ## 候选任务这里允许重复，选取两类候选任务，把两类候选任务按照两个组特征方式，展开为四维
        candi_jobs = self.candidates[agent_id]
        pc = []
        pd = []
        for job in candi_jobs:
            pc.append(self.job_status[job]['remain_process_time'])
            if self.job_status[job]['priority'] == 0:
                pending_time = 99
            else:
                pending_time = self.job_status[job]['remain_pending_time'] / self.job_status[job]['priority']
            pd.append(pending_time)
            
        pc = [x - pc[0] for x in pc[1:]]
        pd = [x - pd[0] for x in pd[1:]]
        ft += pc
        ft += pd
        return ft

    def get_act_rule_machine_feature(self, agent_id):
        # qtime features of jobs when action rule done
        # 这里也改为与第一类选择的差值
        # length: 8
        temp_ft = []
        # TODO 加选择这里要+4个0
        if agent_id not in self.candidates:
            return [0,0,0,0,0,0,0,0]
        candi_jobs = self.candidates[agent_id]
        for job in candi_jobs:
            machine_info = []
            for other_job in self.valid_action[agent_id]:
                if job != other_job and self.job_status[other_job]['remain_pending_time']-self.job_status[job]['remain_process_time']<0:
                    machine_info.append(self.job_status[other_job]['remain_pending_time']-self.job_status[job]['remain_process_time'])
            temp_ft += [len(machine_info), max(machine_info), min(machine_info), sum(machine_info)/len(machine_info)] if len(machine_info) > 0 \
                else[0,0,0,0]
        ft = []
        tempt_ft = temp_ft[4:]
        for choice_index in range(len(tempt_ft)):
            indeed_index = choice_index%4
            ft.append(tempt_ft[choice_index]-temp_ft[indeed_index])
        return ft

    def get_act_rule_job_feature(self, agent_id):
        # job features when action rule done
        # length: 4
        # TODO 加选择这里要+2个0
        ft = []
        if agent_id not in self.candidates:
            return [0,0,0,0]
        
        # TODO 这里做个比较
        ## 候选任务这里允许重复，选取两类候选任务，把两类候选任务按照两个组特征方式，展开为四维
        candi_jobs = self.candidates[agent_id]
        pc = []
        pd = []
        for job in candi_jobs:
            pc.append(self.job_status[job]['remain_process_time'])
            if self.job_status[job]['priority'] == 0:
                pending_time = 99
            else:
                pending_time = self.job_status[job]['remain_pending_time'] / self.job_status[job]['priority']
            pd.append(pending_time)
            
        pc = [x - pc[0] for x in pc[1:]]
        pd = [x - pd[0] for x in pd[1:]]
        ft += pc
        ft += pd
        return ft

    def get_act_rule_prev_feature(self, agent_id):
        # 动作规则下即将到此类机器的jobs（还在上一类机器中work或者还未arrive）中可能出现的qtime超时信息
        # length: 8
        def get_next_op_info(job):
            if self.job_status[job]['status'] == 'to_arrive':
                job_type = self.job_status[job]['type']
                next_op = self.env.env.job_types[job_type][0]
                return {'machine':'A', 'next_max_pending_time':next_op['max_pend_time']}
            else:
                job_type = self.job_status[job]['type']
                now_op = self.job_status[job]['op']
                for op_idx, op in enumerate(self.env.env.job_types[job_type]):
                    if op['op_name'] == now_op:
                        break
                next_op = self.env.env.job_types[job_type][op_idx+1] if op_idx < len(self.env.env.job_types[job_type]) - 1 else None
                next_op_info = {'machine':next_op['machine_type'], 'next_max_pending_time':next_op['max_pend_time']} if next_op is not None \
                    else {'machine':None, 'next_max_pending_time':None}
                return next_op_info


        working_job_dict = defaultdict(dict)
        for job in self.job_status:
            status = self.job_status[job]['status']
            if status == 'work' or status == 'to_arrive':
                next_op_info = get_next_op_info(job)
                if next_op_info['machine'] is not None:
                    working_job_dict[next_op_info['machine']][job] = {
                        'remain_process_time': self.job_status[job]['arrival'] if status == 'to_arrive' else self.job_status[job]['remain_process_time'],
                        'max_pending_time': next_op_info['next_max_pending_time']
                    }

        machine_type = self.machine_status[agent_id]['type']

        if agent_id not in self.candidates:
            return [0,0,0,0,0,0,0,0]
        
        ## working_job_dict
        ## 1. 还需要处理的时间，或者将要到达的时间
        ## 2. 之后的最大挂起时间
        working_job_dict = defaultdict(dict)
        for job in self.job_status:
            status = self.job_status[job]['status']
            if status == 'work' or status == 'to_arrive':
                next_op_info = get_next_op_info(job)
                if next_op_info['machine'] is not None:
                    working_job_dict[next_op_info['machine']][job] = {
                        'remain_process_time': self.job_status[job]['arrival'] if status == 'to_arrive' else self.job_status[job]['remain_process_time'],
                        'max_pending_time': next_op_info['next_max_pending_time']
                    }

        machine_type = self.machine_status[agent_id]['type']

        next_op_first_info = []
        next_op_first_job = self.candidates[agent_id][0]
        next_op_process_time = self.job_status[next_op_first_job]['remain_process_time']

        for job in working_job_dict[machine_type]:
            next_op_job_info = working_job_dict[machine_type][job]
            if  next_op_job_info['remain_process_time'] + next_op_job_info['max_pending_time'] < next_op_process_time:
                next_op_first_info.append(next_op_process_time-(next_op_job_info['remain_process_time'] + next_op_job_info['max_pending_time']))
                
        qtime_first_info = []
        qtime_first_job = self.candidates[agent_id][1]
        qtime_process_time = self.job_status[qtime_first_job]['remain_process_time']

        for job in working_job_dict[machine_type]:
            next_op_job_info = working_job_dict[machine_type][job]
            if  next_op_job_info['remain_process_time'] + next_op_job_info['max_pending_time'] < qtime_process_time:
                qtime_first_info.append(qtime_process_time-(next_op_job_info['remain_process_time'] + next_op_job_info['max_pending_time']))

        ptime_first_info = []
        ptime_first_job = self.candidates[agent_id][2]
        ptime_process_time = self.job_status[ptime_first_job]['remain_process_time']

        for job in working_job_dict[machine_type]:
            next_op_job_info = working_job_dict[machine_type][job]
            if  next_op_job_info['remain_process_time'] + next_op_job_info['max_pending_time'] < ptime_process_time:
                ptime_first_info.append(ptime_process_time-(next_op_job_info['remain_process_time'] + next_op_job_info['max_pending_time']))
                
        next_op_info = [len(next_op_first_info), max(next_op_first_info), min(next_op_first_info), sum(next_op_first_info)/len(next_op_first_info)] if len(next_op_first_info) > 0 \
            else [0,0,0,0]
        ptime_info = [len(ptime_first_info), max(ptime_first_info), min(ptime_first_info), sum(ptime_first_info)/len(ptime_first_info)] if len(ptime_first_info) > 0 \
            else [0,0,0,0]
        qtime_info = [len(qtime_first_info), max(qtime_first_info), min(qtime_first_info), sum(qtime_first_info)/len(qtime_first_info)] if len(qtime_first_info) > 0 \
            else [0,0,0,0]
        ft = []
        n = 0
        for pv, qv in zip(ptime_info, qtime_info):
            ft.append(pv - next_op_info[n])
            ft.append(qv - next_op_info[n])
            n += 1
        return ft




















    def get_working_jobs_feature(self, agent_id):
        # length: 20
        ft = []
        pt_dict = {'A':[],'B':[], 'C':[], 'D':[]}
        for machine in self.machine_status:
            status = self.machine_status[machine]['status']
            if status == 'work':
                job = self.machine_status[machine]['job']
                pt_dict[self.machine_status[machine]['type']].append(self.job_status[job]['remain_process_time'])

        machine_type = self.machine_status[agent_id]['type']

        ft += [len(pt_dict[machine_type]), max(pt_dict[machine_type]), min(pt_dict[machine_type]), sum(pt_dict[machine_type])/len(pt_dict[machine_type])] \
            if len(pt_dict[machine_type]) > 0 else [0,0,0,0]

        for k in pt_dict:
            type_info = pt_dict[k]
            if len(type_info) > 0:
                ft += [len(type_info), max(type_info), min(type_info), sum(type_info)/len(type_info)]
            else:
                ft += [0,0,0,0]
        return ft

    def get_pending_jobs_feature(self, agent_id):
        # length: 40
        ft = []
        pending_time_dict = {'A':[],'B':[], 'C':[], 'D':[]}
        process_time_dict = {'A':[],'B':[], 'C':[], 'D':[]}
        for job in self.job_status:
            if self.job_status[job]['status'] == 'pending':
                op_name = self.job_status[job]['op']
                if self.job_status[job]['priority'] == 0:
                    pending_time = 99
                else:
                    pending_time = self.job_status[job]['remain_pending_time'] / self.job_status[job]['priority']
                process_time = self.job_status[job]['remain_process_time']
                machine_type = self.get_op_machine_type(op_name)
                pending_time_dict[machine_type].append(pending_time)
                process_time_dict[machine_type].append(process_time)
        machine_type = self.machine_status[agent_id]['type']
        ft += [len(pending_time_dict[machine_type]), max(pending_time_dict[machine_type]), min(pending_time_dict[machine_type]), sum(pending_time_dict[machine_type])/len(pending_time_dict[machine_type])] \
            if len(pending_time_dict[machine_type]) > 0 else [0,0,0,0]
        ft += [len(process_time_dict[machine_type]), max(process_time_dict[machine_type]), min(process_time_dict[machine_type]), sum(process_time_dict[machine_type])/len(process_time_dict[machine_type])] \
            if len(process_time_dict[machine_type]) > 0 else [0,0,0,0]

        for k in pending_time_dict:
            pending_info = pending_time_dict[k]
            process_info = process_time_dict[k]
            if len(pending_info) > 0:
                ft += [len(pending_info), max(pending_info), min(pending_info), sum(pending_info)/len(pending_info)]
                ft += [len(process_info), max(process_info), min(process_info), sum(process_info)/len(process_info)]
            else:
                ft += [0,0,0,0]
                ft += [0,0,0,0]
        return ft

    def get_op_machine_type(self, op_name):
        for type in self.env.env.job_types:
            job_type = self.env.env.job_types[type]
            for op in job_type:
                if op['op_name'] == op_name:
                    return op['machine_type']

    def get_machine_feature(self, agent_id):
        num_pending_machine = 0
        machine_type = self.machine_status[agent_id]
        for machine in self.machine_status:
            if agent_id == machine:
                continue
            else:
                if self.machine_status[machine]['type'] == machine_type:
                    if self.machine_status[machine]['status'] == 'idle':
                        num_pending_machine += 1
        return [num_pending_machine]



    def check_real_step(self):
        for machine_type in self.valid_machine_type:
            if len(self.valid_machine_type[machine_type]) > 0:
                return False
        return True

    def get_job_op(self, job, machine):
        return self.job_status[job]['op']

    def get_action_mask(self, agent_id):
        res = np.ones(self.env.action_size)
        machine_type = self.machine_status[agent_id]['type']
        machine_dict = {
            'A':0,
            'B':1,
            'C':2,
            'D':3
        }
        can_wait = False

        for job in self.job_loc:
            if not job in self.valid_action[agent_id]:
                if self.job_loc[job] <= machine_dict[machine_type]:
                    can_wait = True

        res[0] = 1 if can_wait else 0

        return res


    def _valid_action_step(self):
        if self.last_valid_action is None:
            return False
        if len(self.valid_action) == 0:
            return True
        if len(self.last_valid_action) != len(self.valid_action):
            return False
        for key in self.last_valid_action:
            if key in self.valid_action:
                if set(self.valid_action[key]) != set(self.last_valid_action[key]):
                    return False
            else:
                return False
        return True


    def get_attr(self, lots, attr):
        res = []
        if isinstance(lots, list) or isinstance(lots, set):
            for lot in lots:
                if attr == 'remain_pending_time':
                    if self.job_status[lot]['priority'] == 0:
                        res.append(99)
                    else:
                        res.append(self.job_status[lot][attr] / self.job_status[lot]['priority'])
                else:
                    res.append(self.job_status[lot][attr])
        else:
            lot = lots
            if attr == 'remain_pending_time':
                if self.job_status[lot]['priority'] == 0:
                    res.append(99)
                else:
                    res.append(self.job_status[lot][attr] / self.job_status[lot]['priority'])
            else:
                res.append(self.job_status[lot][attr])
        return res
