import time
import random
from collections import defaultdict,Counter
from itertools import permutations
import math
import numpy
# over priority not in 
# over arrival replace processtime  
# over remove already use
# over permu priority
# todo breakdown
# todo hybirid
# todo large process time
import heapq
class PriMin:
    def __init__(self,job_status,job_types,opNextOp):
        self.job_status=job_status
        self.job_types=job_types
        self.opNextOp=opNextOp
        self.timeMin=defaultdict(list)
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
        for job in [jobName  for jobName,jobItem in job_status.items() if jobItem['priority']> 0]:
            if (job_status[job]['status'] == 'work' or job_status[job]['status'] == 'to_arrive'):
                next_op_info_s = get_next_op_info(self,job,job_status)
                base=job_status[job]['arrival'] if job_status[job]['status'] == 'to_arrive' else job_status[job]["remain_process_time"]
                self.timeMin[next_op_info_s['machine']].append( next_op_info_s['next_max_pending_time'] + base)
                
        for mt in self.timeMin.values():
            heapq.heapify(mt)            
    def getNext(self,mt):
        return min(self.timeMin[mt]) if len(self.timeMin[mt])!=0 else 999999
    
    def popOut(self,mt):
        bbc=heapq.heappop(self.timeMin[mt]) if mt in self.timeMin and len(self.timeMin[mt])!=0 else 999999
        return bbc 

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
        opMachineCounter={}
        self.op_machine_load={}
        self.op_allmt={}
        for ops in job_types.values():
            for i in range(len(ops)):
                thisName=ops[i]["op_name"]
                mt=ops[i]["machine_type"]
                self.opNextProcess[thisName]=len(ops)-i
                opMachineCounter[thisName]=Counter({mt:ops[i]['process_time']})
                self.op_allmt[thisName]=set()
                if i<len(ops)-1:
                    self.opNextOp[thisName]=ops[i+1]
                    self.mtNextOp[mt]=ops[i+1]
                for j in range(i+1,len(ops)):
                    self.op_allmt[thisName].add(ops[j]["machine_type"])
        for op in opMachineCounter.keys():
            opThis=op
            counterAll=opMachineCounter[op]
            while opThis in self.opNextOp:
                opThis=self.opNextOp[opThis]['op_name']
                counterAll=counterAll+opMachineCounter[opThis]
            self.op_machine_load[op]=counterAll
    def preferOP(self,machine_status, job_status):
        machine_count=dict(Counter([m_info['type'] for m_info in  machine_status.values() if m_info['status']!= 'down' ]))
        idle_count=dict(Counter([m_info['type'] for m_info in  machine_status.values() if m_info['status']=='idle' ]))
        op_count_all=Counter([job_info['op'] for job_info in  job_status.values()])
        mt_all={"A":0,"B":0,"C":0,"D":0}
        for op,op_count in op_count_all.items():
            for mt,mt_pt in self.op_machine_load[op].items():
                mt_all[mt]+=mt_pt*op_count
        for mt in mt_all.keys(): mt_all[mt]/=machine_count[mt]
        mtIndex={mt:-(index+1) for index,mt in enumerate([mt  for mt,pt in  Counter(mt_all).most_common(4)])}
        return {
            "pri":{op:numpy.mean([mtIndex[mt] for mt in self.op_allmt[op]]) for op in op_count_all.keys()},
            "long":{op:sum([mtIndex[mt] for mt in self.op_allmt[op]]) for op in op_count_all.keys()},
            'idle':{op:sum([-idle_count[mt] if mt in idle_count else 0 for mt in self.op_allmt[op]]) for op in op_count_all.keys()}
        }
        # return {"long":{job_info['op']:-len(self.op_allmt[job_info['op']]) for job_info in  job_status.values()}}
    def act(self, machine_status, job_status, time, job_list):
        l=dict(Counter([m_info['type'] for m_info in  machine_status.values() if m_info['status']!= 'down' ])).values()
        machine_count=(max(l),max(l)-min(l))
        action = {}
        op=self.preferOP( machine_status, job_status)
        priMin=PriMin(job_status,self.job_types,self.opNextOp)
        alreadIn=set()
        
        iter=0
        for mt in ['A','B','C','D']:
            machines=[m for m,m_info in  machine_status.items() 
                            if m_info['status']=='idle' and
                            m_info['type']==mt and 
                            len(job_list[m])!=0
                            ]
            for machine in machines:
                jobs=[a for a in job_list[machine] if a not in alreadIn ]
                job = self.adv_wwsqt(machine, machine_status, job_status, time+iter,jobs,priMin,op,machine_count)
                if job is not None:
                    action[machine] = job
                    alreadIn.add(job)
                iter+=1
                # if(job_status[job]['priority']>0):
                #     thisJob=job_status[job]
                #     thisJob['status']="work"
                #     thisJob['machine']=machine
                #     priMin=PriMin(job_status,self.job_types,self.opNextOp)
                    # priMin.popOut(machine_status[machine]['type'])
        return action
    def adv_wwsqt(self, machine, machine_status, job_status, time, job_list,priMin,op,count):
        if len(job_list) == 0:
            return None
        else:
            sorted_list = [a for a in job_list if job_status[a]['priority']>0]
            if len(sorted_list) == 0:
                machine_type = machine_status[machine]['type']
                max_time=priMin.getNext(machine_type)
                job_list=[a for a in job_list if job_status[a]['remain_process_time']<max_time]
                if len(job_list)>0:
                    if(count[1]<3 and time%count[0] > count[0]/4):
                        return sorted(job_list, key=lambda x: (op["idle"][job_status[x]['op']],op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    if   (time %5 ==1):
                        return sorted(job_list, key=lambda x: (op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    # elif (time %5 == 3):
                    #     return sorted(job_list, key=lambda x: (op["idle"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    else :
                        return sorted(job_list, key=lambda x: (job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                else:
                    return None
            else:
                return self.getJob(sorted_list,job_status,op)
    
    
    
    def getJob(self,sorted_list,job_status,op):
        if(len(sorted_list)==1):return sorted_list[0]
        def abc(job_status,x):
            op=job_status[x]['op']
            if op in self.opNextOp:
                return (self.opNextOp[op]['process_time'])/job_status[x]['priority']
            else:
                return 999999
        if(sum([job_status[x]['remain_process_time'] for x in sorted_list])< min([job_status[x]['remain_pending_time'] for x in sorted_list])):
            # return sorted(sorted_list, key=lambda x: ((-job_status[x]['priority'])))[0]
            return sorted(sorted_list, key=lambda x: (op["idle"][job_status[x]['op']],op["long"][job_status[x]['op']],abc(job_status,x),job_status[x]['remain_process_time']/job_status[x]['priority']))[0]
            # return sorted(sorted_list, key=lambda x: (((job_status[x]['remain_pending_time']-job_status[x]['remain_process_time'])/job_status[x]['priority'],job_status[x]['remain_process_time'])))[0]
        best_set=None
        best_sets=set()
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
                    best_sets.add(sor[0])
        if(len(best_sets)!=0):
            return sorted(best_sets, key=lambda x: (abc(job_status,x),job_status[x]['remain_process_time']/job_status[x]['priority']))[0]
        return best_set[0]
