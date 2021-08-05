from copy import deepcopy
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

# todo stage 3 
# todo priority view
# todo overview
import heapq
class PriMin:
    def __init__(self,job_status,job_types,opNextOp,machine_status):
        self.job_status=job_status
        self.job_types=job_types
        self.opNextOp=opNextOp
        self.timeMin=defaultdict(list)
        self.machine_status=machine_status
        def get_next_op_info(self,job,job_status):
            if job_status[job]['status'] == 'to_arrive':
                job_type = job_status[job]['type']
                next_op = self.job_types[job_type][0]
                return [
                    {'machine':'A', 'next_max_pending_time':next_op['max_pend_time']}
                ]
            else:
                job_type = job_status[job]['type']
                now_op = job_status[job]['op']
                next_op= self.opNextOp[now_op] if now_op in self.opNextOp else None
                next_op_info = [{'machine':next_op['machine_type'], 'next_max_pending_time':next_op['max_pend_time'],} if next_op is not None \
                    else {'machine':None, 'next_max_pending_time':0}] 
                return next_op_info  
        
        for job in [jobName  for jobName,jobItem in job_status.items() if jobItem['priority']> 0]:
            if (job_status[job]['status'] == 'work' or job_status[job]['status'] == 'to_arrive'):
                next_op_info_ss = get_next_op_info(self,job,job_status)
                base=0
                if job_status[job]['status'] == 'to_arrive':
                    base=job_status[job]['arrival'] 
                elif job_status[job]['status'] == 'work':
                    base=job_status[job]["remain_process_time"]
                for next_op_info_s in next_op_info_ss:
                    self.timeMin[next_op_info_s['machine']].append(
                        {
                            'time':base
                            ,'pend':next_op_info_s['next_max_pending_time'] + base
                        }
                    )
        for mt,values in self.timeMin.items():
            self.timeMin[mt]=sorted(values,key=lambda x:x['pend'])
    
    def getNext(self,mt):
        return self.timeMin[mt][0]['pend'] if len(self.timeMin[mt])!=0 else 999999
    
    def popOut(self,mt):
        if mt not in self.timeMin or not self.timeMin[mt]:return
        bbc=self.timeMin[mt].pop()
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
        self.mt_op_before={}
        op_jp=[]
        for jp,ops in job_types.items():
            for i in range(len(ops)):
                thisName=ops[i]["op_name"]
                mt=ops[i]["machine_type"]
                self.opNextProcess[thisName]=len(ops)-i
                opMachineCounter[thisName]=Counter({mt:ops[i]['process_time']})
                self.op_allmt[thisName]=set()
                op_jp.append({'op':thisName,'jp':jp})
                if i<len(ops)-1:
                    self.opNextOp[thisName]=ops[i+1]
                    self.mtNextOp[mt]=ops[i+1]
                for j in range(i+1,len(ops)):
                    self.op_allmt[thisName].add(ops[j]["machine_type"])
                timeBefore=sum([ops[k]['process_time'] for k in range(0,i)])
                if mt not in self.mt_op_before:
                    self.mt_op_before[mt]=[{'op':thisName,'before':timeBefore,'cur':ops[i]['process_time']}]
                else:
                    self.mt_op_before[mt].append({'op':thisName,'before':timeBefore,'cur':ops[i]['process_time']})
        def getPrefer(mt):
            cf=[]
            for op in sorted(self.mt_op_before[mt],key=lambda x:x['before']-x['cur']):
                jp=[it['jp'] for it in op_jp if it['op']==op['op']][0]
                cf.extend([it['op'] for it in op_jp if it['jp']==jp])
            return cf
        self.f={mt:getPrefer(mt)  for mt in ["A",'B',"C","D"]}
        for op in opMachineCounter.keys():
            opThis=op
            counterAll=opMachineCounter[op]
            while opThis in self.opNextOp:
                opThis=self.opNextOp[opThis]['op_name']
                counterAll=counterAll+opMachineCounter[opThis]
            self.op_machine_load[op]=counterAll
            
    def preferOP(self,machine_status, job_status):
        has_down=len([m_info['type'] for m_info in  machine_status.values() if m_info['status']== 'down' ])!=0
        machine_count=dict(Counter([m_info['type'] for m_info in  machine_status.values() if m_info['status']!= 'down' ]))
        idle_count=dict(Counter([m_info['type'] for m_info in  machine_status.values() if m_info['status']=='idle' ]))
        op_count_all=Counter([job_info['op'] for job_info in  job_status.values() if job_info['status']!='done'])
        op_count_pending=Counter([self.opInfos[job_info['op']]['machine_type'] for job_info in  job_status.values() if job_info['status']=='pending'])
        mt_pri={'A':[],'B':[],'C':[],'D':[]}
        mt_remain={}
        mt_all={"A":0,"B":0,"C":0,"D":0}
        jobPending=[job_info for job_info in  job_status.values() if job_info['status']=='pending' and job_info['priority']>0]
        for jobInfo in jobPending:
            opInfo=self.opInfos[jobInfo['op']]
            mt=opInfo['machine_type']
            opBefore=[op for op,info in self.opNextOp.items() if info['op_name']==jobInfo['op']]
            if opBefore:
                mt_pri[self.opInfos[opBefore[0]]['machine_type']].append({'arrive':0,'process':opInfo['process_time'],'pend':opInfo['process_time'],'mtNext':mt})

        for machine in [mc for mc in machine_status.values() if mc['status']!='down']:
            mt=machine['type']
            r_t=machine['remain_time']
            job_name=machine['job']
            if(job_name):
                this_job=job_status[job_name]
                this_op=this_job['op']
                if(this_job['priority']>0 and this_op in self.opNextOp):
                    this_job_remain=this_job['remain_process_time']
                    next_op=self.opNextOp[this_op]
                    next_process=next_op['process_time']
                    next_pend=next_op['max_pend_time']
                    next_type=next_op['machine_type']
                    mt_pri[mt].append({'arrive':this_job_remain,'process':next_process,'pend':next_pend,'mtNext':next_type})
            if mt in mt_remain.keys():
                mt_remain[mt].append(r_t)
            else :
                mt_remain[mt]=[r_t]
        for mt in mt_remain.values():
            heapq.heapify(mt)
        for op,op_count in op_count_all.items():
            for mt,mt_pt in self.op_machine_load[op].items():
                mt_all[mt]+=mt_pt*op_count
        for mt in mt_all.keys(): mt_all[mt]/=machine_count[mt]
        mtIndex={mt:-(index+1) for index,mt in enumerate([mt  for mt,pt in  Counter(mt_all).most_common(4)])}
        # after breakdown
        # all idle ,all pending ,idle pending
        #=============================================
        mts={machine_type[0]:machine_type[1]  for machine_type in Counter(mt_all).most_common(4) 
             if machine_type[0] not in op_count_pending or  op_count_pending[machine_type[0]] < machine_count[machine_type[0]]*1.5}
        nf={}
        for mt,mc in mts.items():
            for m in self.f[mt]:
                if m not in nf:
                    nf[m]=[-mc]
                else:
                    nf[m].append(-mc)
        for op in self.opInfos.keys():
            if op not in nf:
                nf[op]=1
            else:
                nf[op]=numpy.min(nf[op])
        #==========================================
        #before breakdonw
        #all idle
        class Before:
            def __init__(self,weights,prefer) -> None:
                self.mtlog={"A":-1,"B":-1,"C":-1,"D":-1}
                self.weights=weights
                opall=set([ item for items in prefer.values() for item in items])
                self.prefer={key:{index:it for it,index in enumerate(item)} for key,item in prefer.items()}
                self.prefer={key:{op:item[op] if op in item else 99 for op in opall} for key,item in self.prefer.items()}
                self.ops=[]
                ppt=[]
                while(weights):
                    del_list=[]
                    for w_k,value in sorted(weights.items(),key=lambda x:x[1]):
                        if(value==0):
                            del_list.append(w_k)
                            break
                        weights[w_k]-=1
                        ppt.append(w_k)
                        self.ops.append(self.prefer[w_k])
                    for del_k in del_list: del weights[del_k]
            def getPrefer(self,mt):
                iter=self.mtlog[mt]=self.mtlog[mt]+1
                return self.ops[iter % (len(self.ops))]
                
            
        return {
            "long":{op:sum([mtIndex[mt] for mt in self.op_allmt[op]]) for op in op_count_all.keys()},
            'idle':{op:sum([-idle_count[mt] if mt in idle_count else 0 for mt in self.op_allmt[op]]) for op in op_count_all.keys()}
            ,"mp":mt_remain
            ,"mt":mt_pri
            ,'nf':nf
            ,'before':Before(weights={"C":1,"D":3},prefer=self.f)
            ,"max":max(mt_all.values())
            ,'bk':has_down
        }
    
    def act(self, machine_status, job_status, time, job_list):
        l=dict(Counter([m_info['type'] for m_info in  machine_status.values() if m_info['status']!= 'down' ])).values()
        machine_count=(max(l),max(l)-min(l))
        action = {}
        op=self.preferOP( machine_status, job_status)
        priMin=PriMin(job_status,self.job_types,self.opNextOp,machine_status)
        alreadIn=set()
        iter=0
        # for mt in ['A','B','C','D']:
        # self.track_time=[]
        # self.track_job=[]
        # self.track_machine=[]
        self.track_time=[]
        self.track_job=['J0094']
        self.track_machine=['M006']
        for mt in ['D','C','B','A']:
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
                    if(job_status[job]['priority']>0):
                        thisJob=job_status[job]
                        thisJob['status']="work"
                        thisJob['machine']=machine
                        machine_status[machine]['remain_time']=job_status[job]["remain_process_time"]
                        machine_status[machine]['job']=job
                        priMin=PriMin(job_status,self.job_types,self.opNextOp,machine_status)
                        op=self.preferOP( machine_status, job_status)
                    if(job in self.track_job):
                        print("JOB===",time,machine,job,job_status[job])
                    if(machine in self.track_machine):
                        print("MAC=J===",time,machine,job,job_status[job])
                        # print(time,op)
                else:
                    if(machine in self.track_machine):
                         print("MAC=N=="time,op)
                iter+=1
        # if(action):
        #     print(time,{job_status[job]['op']:job_status[job]['priority'] for machine_n,job in action.items()})
        return action
    def adv_wwsqt(self, machine, machine_status, job_status, time, job_list,priMin,op,count):
        machine_type = machine_status[machine]['type']
        if len(job_list) == 0:
            return None
        else:
            sorted_list = [a for a in job_list if job_status[a]['priority']>0]
            max_time=priMin.getNext(machine_type)
            if len(sorted_list) == 0:
                origin_job_size=len(job_list)
                job_list=[a for a in job_list if job_status[a]['remain_process_time']<max_time]
                if(origin_job_size>len(job_list)):
                    priMin.popOut(machine_type)
                if len(job_list)>0:
                    if(op['bk']):
                        return sorted(job_list, key=lambda x: (op['idle'][job_status[x]['op']],op['nf'][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    else:
                        prefer=op['before'].getPrefer(machine_type)
                        return sorted(job_list, key=lambda x: (prefer[job_status[x]['op']],job_status[x]['remain_process_time']))[0]
                    # if  time<op['max']*0:
                    #         return sorted(job_list, key=lambda x: (op['df'][job_status[x]['op']],op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    #     elif (time%2==0):
                    #         return sorted(job_list, key=lambda x: (op['cf'][job_status[x]['op']],op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    #     # elif (time%4==2):
                    #     #     return sorted(job_list, key=lambda x: (-op['df'][job_status[x]['op']],op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    #     # elif (time%4==3):       
                    #     #     return sorted(job_list, key=lambda x: (-op['cf'][job_status[x]['op']],op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    # else :
                    #     if  (time%5==1 ):
                    #         return sorted(job_list, key=lambda x: (op["long"][job_status[x]['op']],op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    #     else :
                    #         return sorted(job_list, key=lambda x: (job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    # else:
                    #     if(time%2==1):
                    #         return sorted(job_list, key=lambda x: (op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    #     else:
                        # elif (time%4==2):
                        #     return sorted(job_list, key=lambda x: (op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                        # else:
                        #     return sorted(job_list, key=lambda x: (job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]

                    # if(count[1]<3 and time%count[0] > 1):
                    #     return sorted(job_list, key=lambda x: (op["idle"][job_status[x]['op']],op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    # if   (time %5 ==1):
                    #     return sorted(job_list, key=lambda x: (op["long"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    # elif (time %5 == 3):
                    #     return sorted(job_list, key=lambda x: (op["idle"][job_status[x]['op']],job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                    return sorted(job_list, key=lambda x: (job_status[x]['remain_process_time'],job_status[x]['remain_pending_time']))[0]
                else:
                    return None
            else:
                origin_job_size=len(sorted_list)
                rst=[a for a in sorted_list if job_status[a]['remain_process_time']<=max_time]
                if(origin_job_size>len(rst)):
                    priMin.popOut(machine_type)
                return self.getJob(sorted_list,job_status,op) 
    
    
    
    def getJob(self,sorted_list,job_status,op):
        def bbc(job_name,job_status,op):
            op_name=job_status[job_name]['op']
            mc=op['mp']
            mtype=job_status[job_name]['type'].upper()
            otherPri=op['mt'][mtype]
            remain_pending=job_status[job_name]['remain_pending_time']
            remain_process_time=job_status[job_name]['remain_process_time']
            base=0
            if(job_name in self.track_job):
                print("JOB===",base,job_status[job_name],remain_pending,remain_process_time)
            if op_name in self.opNextOp:
                next_info=self.opNextOp[op_name]
                next_maxPending=next_info['max_pend_time']
                next_machine_type=next_info['machine_type']
                base=mc[next_machine_type]
                priNext=sorted([pri  for pri in otherPri if pri['arrive']<=remain_process_time and  pri['mtNext']==next_machine_type],key=lambda x:x['arrive'])
                if(priNext):
                    base=[max(base[i],priNext[i]['arrive'])+priNext[i]['process'] for i in range(min(len(base),len(priNext)))]
                base=min(base)
                if(job_name in self.track_job):
                    print(base,remain_pending,remain_process_time,next_maxPending,priNext)
                if( base >remain_process_time+next_maxPending and remain_pending>0):
                    return False
                else :
                    return True
            else:
                return True
        if(len(sorted_list)==1):
            return sorted_list[0] if bbc(sorted_list[0],job_status,op) else None
        def abc(job_status,x):
            op=job_status[x]['op']
            if op in self.opNextOp:
                return (job_status[x]['remain_process_time'])/job_status[x]['priority']
            else:
                return 999999
        if(sum([job_status[x]['remain_process_time'] for x in sorted_list])< min([job_status[x]['remain_pending_time'] for x in sorted_list])):
            # return sorted(sorted_list, key=lambda x: ((-job_status[x]['priority'])))[0]
            better=[s for s in sorted_list if bbc(s,job_status,op) ]
            if(better):sorted_list=better 
            else:return None
            return sorted(sorted_list, key=lambda x: (abc(job_status,x),job_status[x]['remain_process_time']/job_status[x]['priority'],job_status[x]['remain_process_time'],-job_status[x]['priority']))[0]
        best_set=None
        best_sets=set()
        number=9999999
        if(len(sorted_list)>6):
            sorted_list=sorted_list[0:6]
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
            newRst=list(filter(lambda x:bbc(x,job_status,op),best_sets))
            if(len(newRst)>0):
                best_sets=newRst
            else:
                return None
            return sorted(best_sets, key=lambda x: (job_status[x]['remain_process_time']/job_status[x]['priority'],job_status[x]['remain_process_time'],-job_status[x]['priority']))[0]
        return best_set[0] if (bbc(best_set[0],job_status,op)) else None