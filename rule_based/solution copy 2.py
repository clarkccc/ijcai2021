import time
import random
from collections import defaultdict
from itertools import permutations
import math
# over priority not in 
# over arrival replace processtime  
# over remove already use
# over permu priority
# todo breakdown
# todo hybirid
# todo large process time
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
    def __init__(self, job_types):
        self.job_types = job_types
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
                return {'machine':'A', 'next_max_pending_time':next_op['max_pend_time'],'process_time':next_op['process_time']}
            else:
                job_type = job_status[job]['type']
                now_op = job_status[job]['op']
                next_op= self.opNextOp[now_op] if now_op in self.opNextOp else None
                next_op_info = {'machine':next_op['machine_type'], 'next_max_pending_time':next_op['max_pend_time'],'process_time':next_op['process_time']} if next_op is not None \
                    else {'machine':None, 'next_max_pending_time':0,'process_time':0} 
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
        
    def act(self, machine_status, job_status, time, job_list):
        action = {}
        priMin=self.priMin(job_status)
        mtMin=self.mcLoad(machine_status)
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
            sorted_list = [a for a in job_list if job_status[a]['priority']>0]
            if len(sorted_list) == 0:
                machine_type = machine_status[machine]['type']
                max_time=priMin.get(machine_type,999999)
                job_list=[a for a in job_list if job_status[a]['remain_process_time']<max_time]
                if len(job_list)>0:
                    return sorted(job_list, key=lambda x: (job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                else:
                    return None
            else:
                return self.getJob(sorted_list,job_status)
                # return sorted(sorted_list, key=lambda x: (((job_status[x]['remain_pending_time']-job_status[x]['remain_process_time'])/job_status[x]['priority'],job_status[x]['remain_process_time'])))[0]
    def getJob(self,sorted_list,job_status):
        if(len(sorted_list)==1):return sorted_list[0]
        if(sum([job_status[x]['remain_process_time'] for x in sorted_list])< min([job_status[x]['remain_pending_time'] for x in sorted_list])):
            # return sorted(sorted_list, key=lambda x: ((-job_status[x]['priority'])))[0]
            return sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/job_status[x]['priority'],job_status[x]['remain_process_time']))[0]
            # return sorted(sorted_list, key=lambda x: (((job_status[x]['remain_pending_time']-job_status[x]['remain_process_time'])/job_status[x]['priority'],job_status[x]['remain_process_time'])))[0]
        best_set=None
        number=9999999
        for sor in list(permutations(sorted_list)):
            process_time=0
            score=0
            for job in sor:
                thisJob=job_status[job]
                if(thisJob["remain_pending_time"]<process_time):
                    score+=(process_time-thisJob["remain_pending_time"])*thisJob["priority"]
                process_time+=thisJob["remain_process_time"]
            if(score<number):
                best_set=sor
                number=score
                if(score==0): 
                    return sor[0]
        return best_set[0]
