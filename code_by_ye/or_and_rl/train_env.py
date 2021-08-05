"""
baseline: Training Environment

"""
#%%
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
def get_env(Env, conf_list, conf=None):
    class RLEnv(MultiAgentEnv):

        def __init__(self, config={}):
            self.Env = Env
            self.recode_num = 0
            self.conf_list = conf_list
            self.env = self.Env(conf_list[0])
            self.action_size = 4
            self.machine_length = 0
            for key in self.env.machines:
                self.machine_length += len(self.env.machines[key])
            self.rl_agent_id = ['M00' + str(i) for i in range(1,10)] if self.machine_length <10 else \
                ['M00' + str(i) for i in range(1,10)] + ['M0' + str(i) for i in range(10,self.machine_length+1)]

            # TODO 这里的观测值发生了变化
            self.obs_size = 95
            self.action_space = Discrete(self.action_size)
            self.observation_space = Dict({
                'obs':Box(float("-inf"), float("inf"), shape=(self.obs_size,)),
                'action_mask':Box(0,1,shape=(self.action_size,))
            })
            tag = datetime.datetime.timestamp(datetime.datetime.now())
            self.logw = LogWriter("./ft_appearance_log_{}".format(tag))
            self.p_coeff = 10   
            self.q_coeff = 1


        def reset(self):
            if conf is None:
                self.config = random.choice(self.conf_list)
            else:
                self.config = conf
            
            # import pdb
            # print('reset!!!')
            # pdb.set_trace()
            self.env = self.Env(self.config)
            self.valid_action = {}
            self.last_valid_action = {}
            self.lot_dict = {}
            self.step_actions = defaultdict()
            self.reward = None
            self.env_reward = {
                'makespan': 0,
                'PTV': 0
            }
            self.step_reward = {}
            self.last_step_reward = {}
            for agent_id in self.rl_agent_id:
                self.step_reward[agent_id] = 0
                self.last_step_reward[agent_id] = 0
            self.done = None
            self.action_memory = defaultdict(list)
            # reset simulator
            self.machine_status, self.job_status, self.time, self.job_list = self.env.reset()
            self.job_loc = {}
            for job in self.job_status:
                self.job_loc[job] = 0

            # get valid action
            for machine in self.machine_status:
                if machine in self.job_list:
                    if len(self.job_list[machine]) != 0:
                        self.valid_action[machine] = list(self.job_list[machine])

            # get valid machines
            self.valid_machine_type = defaultdict(list)
            for machine in self.valid_action:
                self.valid_machine_type[self.machine_status[machine]['type']].append(machine)

            self.get_candidates()
            self.time_start = self.env.clock.time

            obs = {}
            for machine_type in self.valid_machine_type:
                # 选第一个可选择的机器作为当前观测obs中的obs
                agent_id = self.valid_machine_type[machine_type][0]
                obs[agent_id] = {
                    'obs':self.gen_observation(agent_id),
                    'action_mask':self.get_action_mask(agent_id),
                }
            return obs

        def step(self, actions):
            # import pdb
            # print('---step---')
            # print(actions)
            # pdb.set_trace()
            eval_actions = {}
            ## actions {} -> key = agent_id ; value = 0 skip 1 qtfirst 2 ptfirst
            ## eval_actions -> key = agent_id ; value = job_id
            self.get_candidates() # 初始化 self.candidates
            for key in actions:
                if actions[key] == 0:
                    continue
                else:
                    act_choice = actions[key]-1
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
                                
            for key in eval_actions:
                recipe_time = self.get_attr(eval_actions[key], 'remain_process_time')
                self.lot_dict[eval_actions[key]] = {'tool_id':key,'end_time':self.env.clock.time+recipe_time[0]}
                self.job_loc[eval_actions[key]] += 1
                self.action_memory[key].append(eval_actions[key])

            self.last_valid_action = deepcopy(self.valid_action)
            self.actions_dict = copy.deepcopy(eval_actions)
            ## 删除已经选择好机器的可选项
            self.step_actions.update(eval_actions)
            for machine_type in self.valid_machine_type:
                if len(self.valid_machine_type[machine_type]) > 0:
                    del self.valid_machine_type[machine_type][0]

            ## 检测剩余可选机器是否还有任务可以选，没有就把其中剩余的机型删除
            for machine_type in self.valid_machine_type:
                del_machines = []
                for machine in self.valid_machine_type[machine_type]:
                    if len(self.valid_action[machine]) == 0:
                        del_machines.append(machine)
                for del_machine in del_machines:
                    self.valid_machine_type[machine_type].remove(del_machine)


            # check simulator step
            if self.check_real_step(): # -> 是否不剩下可选的机器
                ## 无机器可选直接快进
                self.machine_status, self.job_status, self.time, self.reward, self.job_list, self.done = self.env.step(self.step_actions)
                for key in self.reward:
                    self.env_reward[key] += self.reward[key]
                for key in self.rl_agent_id:
                    self.step_reward[key] += self.get_rewards(self.reward, self.step_actions, 'action')
                self.step_actions = {}
                self.valid_action = {}

                # check if validation changed
                while self._valid_action_step() and not self.done:

                    self.machine_status, self.job_status, self.time, self.reward, self.job_list, self.done = self.env.step({})
                    for key in self.reward:
                        self.env_reward[key] += self.reward[key]
                    for key in self.rl_agent_id:
                        self.step_reward[key] += self.get_rewards(self.reward, {}, 'action')
                    self.valid_machine_type = defaultdict(list)
                    self.valid_action = {}
                    for machine in self.machine_status:
                        if machine in self.job_list and len(self.job_list[machine]) > 0:
                            self.valid_action[machine] = list(self.job_list[machine])
                    for machine in self.valid_action:
                        self.valid_machine_type[self.machine_status[machine]['type']].append(machine)

                self.get_candidates()
                rewards = {}
                dones = {}
                obs = {}

                if self.done:
                    for agent_id in self.machine_status:
                        obs[agent_id] = {
                            'obs':self.gen_observation(agent_id),
                            'action_mask':np.ones(self.action_size)
                        }
                        rewards[agent_id] = (self.step_reward[agent_id] - self.last_step_reward[agent_id]) / len(self.rl_agent_id) / 1000
                        self.last_step_reward[agent_id] = self.step_reward[agent_id]
                        dones[agent_id] = self.done
                        dones['__all__'] = True

                else:
                    for machine_type in self.valid_machine_type:
                        agent_id = self.valid_machine_type[machine_type][0]
                        obs[agent_id] = {
                            'obs':self.gen_observation(agent_id),
                            'action_mask':self.get_action_mask(agent_id)
                        }
                        rewards[agent_id] = (self.step_reward[agent_id] - self.last_step_reward[agent_id]) / len(self.rl_agent_id) / 1000
                        self.last_step_reward[agent_id] = self.step_reward[agent_id]
                        dones[agent_id] = self.done
                    dones['__all__'] = False

                return obs, rewards, dones, {}

            # simu-step
            else:
                for key in self.step_actions:
                    job = self.step_actions[key]
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

                if self.check_real_step(): # 如果无可选的机器进行安排

                    self.machine_status, self.job_status, self.time, self.reward, self.job_list, self.done = self.env.step(self.step_actions)
                    for key in self.reward:
                        self.env_reward[key] += self.reward[key]
                    for key in self.step_reward:
                        self.step_reward[key] += self.get_rewards(self.reward, self.step_actions, 'action')
                    self.step_actions = {}
                    while self._valid_action_step() and not self.done:

                        self.machine_status, self.job_status, self.time, self.reward, self.job_list, self.done = self.env.step({})
                        for key in self.reward:
                            self.env_reward[key] += self.reward[key]
                        for key in self.step_reward:
                            self.step_reward[key] += self.get_rewards(self.reward, self.step_actions, 'action')
                        self.valid_action = {}
                        self.valid_machine_type = defaultdict(list)
                        for machine in self.machine_status:
                            if len(self.machine_status[machine]['job_list']) != 0:
                                self.valid_action[machine] = self.machine_status[machine]['job_list']
                        for machine in self.valid_action:
                            self.valid_machine_type[self.machine_status[machine]['type']].append(machine)

                    self.get_candidates()
                    rewards = {}
                    dones = {}
                    obs = {}

                    if self.done:
                        for agent_id in self.machine_status:
                            obs[agent_id] = {
                                'obs':self.gen_observation(agent_id),
                                'action_mask':np.ones(self.action_size)
                            }
                            rewards[agent_id] = (self.step_reward[agent_id] - self.last_step_reward[agent_id]) / len(self.rl_agent_id) / 1000
                            self.last_step_reward[agent_id] = self.step_reward[agent_id]
                            dones[agent_id] = self.done
                            dones['__all__'] = True

                    else:
                        for machine_type in self.valid_machine_type:
                            if len(self.valid_machine_type[machine_type]) > 0:
                                agent_id = self.valid_machine_type[machine_type][0]
                                obs[agent_id] = {
                                    'obs':self.gen_observation(agent_id),
                                    'action_mask':self.get_action_mask(agent_id)
                                }
                                rewards[agent_id] = (self.step_reward[agent_id] - self.last_step_reward[agent_id]) / len(self.rl_agent_id) / 1000
                                self.last_step_reward[agent_id] = self.step_reward[agent_id]
                                dones[agent_id] = False
                        dones['__all__'] = False

                    return obs, rewards, dones, {}

                else:
                    self.get_candidates() # 还有可选的机器可以安排
                    rewards = {}
                    dones = {}
                    obs = {}

                    if self.done:
                        for agent_id in self.machine_status:
                            obs[agent_id] = {
                                'obs':self.gen_observation(agent_id),
                                'action_mask':np.ones(self.action_size)
                            }
                            rewards[agent_id] = (self.step_reward[agent_id] - self.last_step_reward[agent_id]) / len(self.rl_agent_id) / 1000
                            self.last_step_reward[agent_id] = self.step_reward[agent_id]
                            dones[agent_id] = self.done
                            dones['__all__'] = True

                    else:
                        for machine_type in self.valid_machine_type:
                            if len(self.valid_machine_type[machine_type]) > 0:
                                agent_id = self.valid_machine_type[machine_type][0]
                                obs[agent_id] = {
                                    'obs':self.gen_observation(agent_id),
                                    'action_mask':self.get_action_mask(agent_id)
                                }
                                rewards[agent_id] = (self.step_reward[agent_id] - self.last_step_reward[agent_id]) / len(self.rl_agent_id) / 1000
                                self.last_step_reward[agent_id] = self.step_reward[agent_id]
                                dones[agent_id] = False
                        dones['__all__'] = False
                    for key, value_ in obs.items():
                        

                        if value_['obs'].shape[0] != 95:
                            import pdb
                            pdb.set_trace()
                    return obs, rewards, dones, {}


        def _valid_action_step(self):
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


        def gen_observation(self, agent_id):
            '''
            :param agent_id:
            :return:
                features
                (length 92)
            '''
            agent_obs = []
            self.recode_num += 1
            # candidate jobs features
            valid_actions = self.valid_action[agent_id] if agent_id in self.valid_action else []
            ## 任务时间信息的特征10个
            ft = self.gen_lots_features(valid_actions)
            agent_obs += ft

            # rule action (ptf，qtf) features
            ## 任务行为选取特征20个
            ft = self.get_act_rule_feature(agent_id)
            agent_obs += ft

            # working job features
            ## 机器当前加工时间特征 20
            ft = self.get_working_jobs_feature(agent_id)
            self.logw.add_scalar('choice_agent_type_work_len', ft[0], self.recode_num)
            self.logw.add_scalar('choice_agent_type_work_max', ft[1], self.recode_num)
            self.logw.add_scalar('choice_agent_type_work_min', ft[2], self.recode_num)
            self.logw.add_scalar('choice_agent_type_work_sul', ft[3], self.recode_num)
            self.logw.add_scalar('a_type_work_len', ft[4], self.recode_num)
            self.logw.add_scalar('a_type_work_max', ft[5], self.recode_num)
            self.logw.add_scalar('a_type_work_min', ft[6], self.recode_num)
            self.logw.add_scalar('a_type_work_sul', ft[7], self.recode_num)
            self.logw.add_scalar('b_type_work_len', ft[8], self.recode_num)
            self.logw.add_scalar('b_type_work_max', ft[9], self.recode_num)
            self.logw.add_scalar('b_type_work_min', ft[10], self.recode_num)
            self.logw.add_scalar('b_type_work_sul', ft[11], self.recode_num)
            self.logw.add_scalar('c_type_work_len', ft[12], self.recode_num)
            self.logw.add_scalar('c_type_work_max', ft[13], self.recode_num)
            self.logw.add_scalar('c_type_work_min', ft[14], self.recode_num)
            self.logw.add_scalar('c_type_work_sul', ft[15], self.recode_num)
            self.logw.add_scalar('d_type_work_len', ft[16], self.recode_num)
            self.logw.add_scalar('d_type_work_max', ft[17], self.recode_num)
            self.logw.add_scalar('d_type_work_min', ft[18], self.recode_num)
            self.logw.add_scalar('d_type_work_sul', ft[19], self.recode_num)
            agent_obs += ft

            # pending job features
            ## 所有挂起件的挂起特征和加工时间特征 40
            ft = self.get_pending_jobs_feature(agent_id)
            agent_obs += ft

            # machine features 1
            ft = self.get_machine_feature(agent_id)
            agent_obs += ft

            # machine type one-hot
            machine_type = self.machine_status[agent_id]['type']
            ft_dict = {
                'A':0,
                'B':1,
                'C':2,
                'D':3
            }
            # 4
            ft = [0,0,0,0]
            ft[ft_dict[machine_type]] = 1
            agent_obs += ft
            if np.array(agent_obs).shape[0] != 95:
                import pdb
                pdb.set_trace()
            return np.array(agent_obs)

        def get_act_rule_feature(self, agent_id):
            # length: 20
            ft = []
            ft += self.get_act_rule_job_feature(agent_id)
            self.logw.add_scalar('candidate_2-1_pctime', ft[-1], self.recode_num)
            self.logw.add_scalar('candidate_3-1_pctime', ft[-2], self.recode_num)
            self.logw.add_scalar('candidate_2-1_pd_pri', ft[-3], self.recode_num)
            self.logw.add_scalar('candidate_3-1_pd_pri', ft[-4], self.recode_num)

            ft += self.get_act_rule_machine_feature(agent_id)
            self.logw.add_scalar('candidate_2-1_other_pd_less_len', ft[-8], self.recode_num)
            self.logw.add_scalar('candidate_2-1_other_pd_less_max', ft[-7], self.recode_num)
            self.logw.add_scalar('candidate_2-1_other_pd_less_min', ft[-6], self.recode_num)
            self.logw.add_scalar('candidate_2-1_other_pd_less_sul', ft[-5], self.recode_num)
            self.logw.add_scalar('candidate_3-1_other_pd_less_len', ft[-4], self.recode_num)
            self.logw.add_scalar('candidate_3-1_other_pd_less_max', ft[-3], self.recode_num)
            self.logw.add_scalar('candidate_3-1_other_pd_less_min', ft[-2], self.recode_num)
            self.logw.add_scalar('candidate_3-1_other_pd_less_sul', ft[-1], self.recode_num)
            ft += self.get_act_rule_prev_feature(agent_id)
            self.logw.add_scalar('candidate_2-1_act_rule_prev_feature_len', ft[-8], self.recode_num)
            self.logw.add_scalar('candidate_2-1_act_rule_prev_feature_max', ft[-7], self.recode_num)
            self.logw.add_scalar('candidate_2-1_act_rule_prev_feature_min', ft[-6], self.recode_num)
            self.logw.add_scalar('candidate_2-1_act_rule_prev_feature_sul', ft[-5], self.recode_num)
            self.logw.add_scalar('candidate_3-1_act_rule_prev_feature_len', ft[-4], self.recode_num)
            self.logw.add_scalar('candidate_3-1_act_rule_prev_feature_max', ft[-3], self.recode_num)
            self.logw.add_scalar('candidate_3-1_act_rule_prev_feature_min', ft[-2], self.recode_num)
            self.logw.add_scalar('candidate_3-1_act_rule_prev_feature_sul', ft[-1], self.recode_num)
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

        def get_act_rule_prev_feature(self, agent_id):
            # qtime features of jobs which is still working in last op when action rule done
            # length: 8
            def get_next_op_info(job):
                if self.job_status[job]['status'] == 'to_arrive':
                    ## 工件还没到，则返回该机器将选A，next_max_pending_time，下一个工序的最大挂起时间
                    for type in self.env.jobs:
                        if job in self.env.jobs[type]:
                            job_type = type
                            break
                    next_op = self.env.job_types[job_type][0]
                    return {'machine':'A', 'next_max_pending_time':next_op['max_pend_time']}
                else:
                    for type in self.env.jobs:
                        if job in self.env.jobs[type]:
                            job_type = type
                            break
                    ## 获取正在操作工件的当前操作状态
                    now_op = self.job_status[job]['op']
                    for op_idx, op in enumerate(self.env.job_types[job_type]):
                        if op['op_name'] == now_op:
                            break
                    ## 根据环境中的操作索引，获得下一个操作的操作状态
                    next_op = self.env.job_types[job_type][op_idx+1] if op_idx < len(self.env.job_types[job_type]) - 1 else None
                    next_op_info = {'machine':next_op['machine_type'], 'next_max_pending_time':next_op['max_pend_time']} if next_op is not None \
                        else {'machine':None, 'next_max_pending_time':None}
                    return next_op_info

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
            # working job features
            # length: 20  4个当前机器类型，16个所有机器当前加工情况
            ft = []
            pt_dict = {'A':[],'B':[], 'C':[], 'D':[]}
            ## 找到正在加工的零件，获取其剩余加工时间
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
            # pending job features
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
        # def get_candidates(self):
        #     # candidate job with different action rule
        #     self.candidates = {}
        #     for i in self.valid_action:
        #         self.candidates[i] = self._get_candidates(agent_id=i)

        #     self.lots_all = {}

        #     for i in range(len(self.rl_agent_id)):
        #         tool_id = self.rl_agent_id[i]
        #         if self.machine_status[tool_id]['type'] not in self.lots_all:
        #             self.lots_all[self.machine_status[tool_id]['type']] = []
        #         if tool_id in self.candidates:
        #             self.lots_all[self.machine_status[tool_id]['type']] += [list(self.candidates[tool_id])]
        #         else:
        #             self.lots_all[self.machine_status[tool_id]['type']] += [[]]*(self.action_size-1)
        
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
        
        def get_candidates(self):
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

            for i in range(len(self.rl_agent_id)):
                tool_id = self.rl_agent_id[i]
                if self.machine_status[tool_id]['type'] not in self.lots_all:
                    self.lots_all[self.machine_status[tool_id]['type']] = []
                if tool_id in self.candidates:
                    self.lots_all[self.machine_status[tool_id]['type']] += [list(self.candidates[tool_id])]
                else:
                    self.lots_all[self.machine_status[tool_id]['type']] += [[]]*(self.action_size-1)

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
                        # import pdb
                        # pdb.set_trace()
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
            procedures = self.env.job_types[self.job_status[job_id]['type']][index:]
            return procedures

        def act_rules(self, agent_id, action, filter_lots=set()):
            if agent_id not in self.valid_action:
                return []

            lots = [job_id for job_id in self.valid_action[agent_id] if (self.machine_status[agent_id]['type'], self.job_status[job_id]['type'], self.job_status[job_id]['priority'], 
                                                    self.job_status[job_id]['remain_process_time'], self.job_status[job_id]['remain_pending_time']) not in filter_lots]
            if len(lots) == 0:
                return []

            if action == 2:
                lot = self.qtfirst(agent_id)
            elif action == 3:
                lot = self.ptfirst(agent_id)

            return lot

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

        def gen_lots_features(self, lots):
            # length: 10
            ft = []
            pt = np.array(self.get_attr(lots, 'remain_process_time'))
            rqt = np.array(self.get_attr(lots, 'remain_pending_time'))
            # length of neg candidate jobs
            neg_num = sum(rqt<0)
            ft.append(neg_num)
            self.logw.add_scalar('neg_job_len', neg_num, self.recode_num)
            ## 超时的时间长度
            if neg_num > 0 :
                ft.append(np.std(rqt<0)/np.mean(rqt<0))
                ft.append(max(np.min(rqt<0),-10))
            else:
                ft += [0,0]
            self.logw.add_scalar('neg_pdtime_std/mean', ft[-1], self.recode_num)
            
            
            pos_num = len(rqt) - neg_num
            ## 可挂件的总体特征
            if pos_num > 0:
                ft.append(pos_num)
                ft.append(np.std(rqt>0))
                ft.append(np.max(rqt>0))
                ft.append(np.min(rqt>0))
            else:
                ft += [0, 0, 0, 0]
            self.logw.add_scalar('pos_pdtime_len', pos_num, self.recode_num)
            self.logw.add_scalar('pos_pdtime_std', ft[-3], self.recode_num)
            self.logw.add_scalar('pos_pdtime_max', ft[-2], self.recode_num)
            self.logw.add_scalar('pos_pdtime_min', ft[-1], self.recode_num)
            
            balance_value = ft[-1] + 1
            ## 处理时间综合优先级的特征
            if len(pt) > 0:
                ft.append(np.std(pt)/np.mean(pt))
                ft.append(np.max(pt)-balance_value)
                ft.append(np.min(pt)-balance_value)
            else:
                ft += [0,0,0]
            self.logw.add_scalar('pctime_std/min', ft[-3], self.recode_num)
            self.logw.add_scalar('pctime_max_mpd', ft[-2], self.recode_num)
            self.logw.add_scalar('pctime_min_mpd', ft[-1], self.recode_num)
            return ft

        def check_real_step(self):
            for machine_type in self.valid_machine_type:
                if len(self.valid_machine_type[machine_type]) > 0:
                    return False
            return True

        def get_job_op(self, job, machine):
            for job_type in self.env.jobs:
                if job in self.env.jobs[job_type]:
                    type = job_type
                    for op in self.env.job_types[type]:
                        if op['machine_type'] == self.machine_status[machine]['type']:
                            return op['op_name']

        def get_op_machine_type(self, op_name):
            for type in self.env.job_types:
                job_type = self.env.job_types[type]
                for op in job_type:
                    if op['op_name'] == op_name:
                        return op['machine_type']

        # def get_machine_type(self, machine_id):
        #     for job_type in self.env.jobs:
        #         if job in self.env.jobs[job_type]:
        #             type = job_type
        #             for op in self.env.job_types[type]:
        #                 if op['machine_type'] == self.machine_status[machine]['type']:
        #                     return op['op_name']
                        
        def get_action_mask(self, agent_id):
            res = np.ones(self.action_size)
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

        def get_rewards(self, reward, action, type):
            def get_jobs_process_time(a):
                process_time = 0
                for key in a:
                    process_time += self.job_status[a[key]]['remain_process_time']
                return process_time

            reward_dict = {
                'env': sum(reward.values()),
                'action': self.p_coeff * get_jobs_process_time(action) + reward['makespan'] + self.q_coeff * reward['PTV']
            }

            return reward_dict[type]

    return RLEnv

#%%

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer

import ray
from ray import tune
from gym.spaces import Discrete, Box, Dict

class MaskedActionsModel(TFModelV2):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(4, ),
                 action_embed_size=8,
                 **kw):
        super(MaskedActionsModel, self).__init__(obs_space, action_space,
                                                 num_outputs, model_config, name)

        action_size = action_space.n
        n_dims = obs_space.shape[0]-action_size

        self.action_embed_model = FullyConnectedNetwork(
            Box(float("-inf"), float("inf"), shape=(n_dims,)), action_space, action_size,
            model_config, name + "_action_embed")
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_embed, _ = self.action_embed_model({
            "obs": input_dict["obs"]["obs"]
        })

        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


#%%

def MinimalJobshopSat(jobs_status_infos):
    """Minimal jobshop problem."""
    s = time.time()
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
    
    if status == cp_model.OPTIMAL:
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