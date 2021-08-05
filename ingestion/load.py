import pickle
from time import time
import os
import matplotlib.pyplot as plt
import math
if __name__ == "__main__":
    os.system('export DISPLAY=:0.0')
    color=[""]
    with open('0.pkl', 'rb') as f:
        datas=pickle.load(f)
        machines={}
        jobs=set()
        timeStart=0
        fig=plt.figure()
        ax=fig.gca(projection='3d')
        process_time_remain={}
        colors=["g","b","y",'pink','r']
        for dataIn in datas:
            timeStart+=1
            availabe={jobname:info for jobname,info in dataIn.items() if info['machine']}
            for jobname,jobinfo in availabe.items():
                if(jobname not in jobs):
                    machine_name=jobinfo["machine"]
                    machine_no=machines[machine_name] = machines.get(machine_name, len(machines))
                    jobs.add(jobname)
                    ax.bar3d(
                        x=timeStart,
                        y=machine_no*2,
                        z=0,
                        dx=jobinfo['remain_process_time'],
                        dy=0.1,
                        dz=-process_time_remain.get(jobname,-0.1),
                        color=colors[jobinfo["priority"]],
                        alpha=0.5
                    )
                    ax.text(timeStart,machine_no*2,0,"\n".join(jobname),fontsize=jobinfo['remain_process_time']/10)
            process_time_remain={jobname:info["remain_pending_time"] for jobname,info in dataIn.items()}
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig('甘特图.png',dpi=1000, format='png',bbox_inches="tight") #bbox_inches="tight"解决X轴时间两个字不被保存的问题
