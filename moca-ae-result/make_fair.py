import os
import sys
import re
import csv
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import statistics

def Average(lst):
    return sum(lst) / len(lst)


num_workload = []
num_core = 8
app_name = ['resnet', 'alexnet', 'googlenet', 'squeezenet', 'kwsnet', 'yolonet', 'yololitenet']
total_workload_type = list(range(len(app_name)))
total_priority_type = list(range(12))
target_scale = 1.0

#test_name = ['prema', 'priority_baseline', 'planaria', 'priority_bubble']
test_name = ['static baseline', 'MoCA']

name_list = []
workload_type_list = []
priority_list = []
target_list = []
result_list = []
qos_list = []
success_list = []
# single program time
#app_single = [0, 11047170, 6330629, 7483605, 3495019, 4417141, 10807095, 3256638]
app_single = [8740139,5549700,5161288,2094997,3144113,6496378,1738151]
app_single = [14186582,7451485,6838849,2514085,4940160,8730125,1414779]
#success = []
#ToDo: mp/sp

#thread_cycle_storage = [[0 for i in range(len(core_count))] for j in range(num_app)]
#print(num_app, thread_cycle_storage)
#total_cycle_storage = [[0 for i in range(len(core_count))] for j in range(num_app)]

name_regex1 = r"Test priority_mp8_(\S*)ic(\S)_(\S)\b"
end_regex = r"Script done \S*\b"

type_regex = r"queue id (\d*) workload type: (\d*)\b"
priority_regex = r"queue id (\d*) priority: (\d*)\b"
qos_regex = r"queue id (\d*) qos: (\d*)\b"
target_regex = r"queue id (\d*) target: (\d*)\b"
result_regex = r"queue id (\d*) dispatch to finish time: (\d*)\b"
busy_regex = r"gemmini core id (\d*) runtime: (\d*)\b"

#cycle_regex = r"^turn (\d*) total thread cycles: (\d*)\b"
#file_regex = r"\W*(\w*)_(\w*)_orow\S*\b"
#cycle_regex = r"^Cycle\s*\b(\d*)\b"
#data_regex = r"^PerfCounter (\S*)_FireSim\S*:\s*(\d*)\b"
test = 0
test_name = ["static", "dynamic"]

workload_type = []
qos = []
priority = []
target = []
result = []
workload_set = str(sys.argv[2])

qos_type = ['l', 'm', 'h']
with open(sys.argv[1], "r") as f:
    for line in f.readlines():
        name_search1 = re.search(name_regex1, line)
        type_search = re.search(type_regex, line)
        priority_search = re.search(priority_regex, line)
        qos_search = re.search(qos_regex, line)
        target_search = re.search(target_regex, line)
        result_search = re.search(result_regex, line)
        busy_search = re.search(busy_regex, line)

        if name_search1:
            name = None
            if name_search1:
                name = str(name_search1.group(1)) + "ic"
                name_print = name_search1.group(1)+'_workload'+name_search1.group(2)+'_qos'+name_search1.group(3)
                print(name_print)
            test += 1
            name_index = test_name.index(name)
            name_list.append(name_index)
            qos_index = qos_type.index(str(name_search1.group(3)))
            qos_list.append(qos_index)
            # initialize
            workload_type = []
            priority = []
            target = []
            result = []

        elif busy_search:
            core = int(busy_search.group(1))
            if core == 0:
                # store to list
                num_workload.append(len(workload_type))
                workload_type_list.append(workload_type)
                priority_list.append(priority)
                target_list.append(target)
                result_list.append(result)
                success = [1 if x*target_scale > y else 0 for x, y in zip(target, result)]
                success_list.append(success)

        elif type_search:
            workload_type.append(int(type_search.group(2))-1)


        elif priority_search:
            priority.append(int((int(priority_search.group(2))+1)))


        elif target_search:
            this_target = int(target_search.group(2))
            target.append(this_target)

        elif result_search:
            result.append(int(result_search.group(2)))


success_ratio = []
for i in range(test):
    ratio = sum(success_list[i]) / num_workload[i]
    success_ratio.append(ratio)


print(name_list)
print(success_ratio)

# ToDo: have to sort baseline names on final plot
#result_list = [x for _,x in sorted(zip(name_list, result_list))]
#workload_type_list = [x for _,x in sorted(zip(name_list, workload_type_list))]
#priority_list = [x for _,x in sorted(zip(name_list, priority_list))]
#name_list.sort()

all_list = [[[] for i in range(len(app_name))] for j in range(len(name_list))]
priority_all_list = [[[] for i in range(len(app_name))] for j in range(len(name_list))]
for i in range(len(name_list)):
    for j in range(len(workload_type_list[i])):
        workload_type = workload_type_list[i][j]
        all_list[i][workload_type].append(result_list[i][j])
        priority_all_list[i][workload_type].append(priority_list[i][j])



#print(all_list)
#print(priority_all_list)
average_all = [[] for i in range(len(name_list))]
median_all = [[] for i in range(len(name_list))]

priority_average_all = [[] for i in range(len(name_list))]

for i in range(len(all_list)):
    average_all[i].append(0)
    median_all[i].append(0)
    priority_average_all[i].append(0)
    for j in range(len(all_list[i])):
        if len(all_list[i][j]) != 0:
            average_all[i].append(Average(all_list[i][j]))
            median_all[i].append(statistics.median(all_list[i][j]))
            priority_average_all[i].append(Average(priority_all_list[i][j]))
        else:
            average_all[i].append(0)
            median_all[i].append(0)
            priority_average_all[i].append(0)
print(average_all)
#print(median_all)
#print(priority_average_all)
#fairness: min((Csp/Cmp) / 1) -> (same priority)
#STP: sum(Csp/Cmp)

fair_list = []
stp_list = []

pp = []
# baseline (equal priority)
for i in range(len(name_list)):
    pp = []
    stp = 0
    for j in range(len(app_name)):
        if average_all[i][j] != 0:
            ppi = app_single[j] / average_all[i][j]
            stp += ppi
            p_factor = priority_average_all[i][j] / sum(priority_average_all[i])
            pp.append(ppi/p_factor)
    min_fair = 1
    for j in range(len(pp)):
        for k in range(len(pp)):
            min_fair = min(min_fair, pp[j] / pp[k])
    fair_list.append(min_fair)
    stp_list.append(stp)
print(fair_list)
print(stp_list)
print(name_list)
name_list = [x for _,x in sorted(zip(qos_list, name_list))]
fair_list = [x for _,x in sorted(zip(qos_list, fair_list))]
stp_list = [x for _,x in sorted(zip(qos_list, stp_list))]


print(fair_list)
print(stp_list)

static_fair = []
dynamic_fair = []
for i in range(len(fair_list)):
    if name_list[i] == 0:
        static_fair.append(fair_list[i])
    elif name_list[i] == 1:
        dynamic_fair.append(fair_list[i])
    else:
        print('invalid')

dynamic_fair = [dynamic_fair[i]/static_fair[i] for i in range(len(dynamic_fair))]
static_fair = [1 for i in static_fair]

static_stp = []
dynamic_stp = []
for i in range(len(stp_list)):
    if name_list[i] == 0:
        static_stp.append(stp_list[i])
    elif name_list[i] == 1:
        dynamic_stp.append(stp_list[i])
    else:
        print('invalid')

dynamic_stp = [dynamic_stp[i]/static_stp[i] for i in range(len(dynamic_stp))]
static_stp = [1 for i in static_stp]

xlabels = ['MoCA','Static Partition']
c = ['blue', 'orange']
xticks = ['Low', 'Medium', 'High', 'Total']
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.figure(figsize=(6, 4))
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
x_axis = ["QoS-L", "QoS-M", "QoS-H"]
x = np.arange(len(x_axis))
axis_adjust = 0.3
x_adjust = axis_adjust * int(len(x)/2)
plt.bar(x, static_fair, width=axis_adjust, label=xlabels[1], color=c[1])
plt.bar(x-axis_adjust, dynamic_fair, width=axis_adjust, label=xlabels[0], color=c[0])
plt.xlabel('QoS levels')
plt.xticks(x-axis_adjust/2, x_axis)
plt.ylabel('Normalized Fairness')
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('fairness_workload'+workload_set+'.png')


plt.figure(figsize=(6, 4))
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
x_axis = ["QoS-L", "QoS-M", "QoS-H"]
x = np.arange(len(x_axis))
axis_adjust = 0.3
x_adjust = axis_adjust * int(len(x)/2)
plt.bar(x, static_stp, width=axis_adjust, label=xlabels[1], color=c[1])
plt.bar(x-axis_adjust, dynamic_stp, width=axis_adjust, label=xlabels[0], color=c[0])
plt.xlabel('QoS levels')
plt.xticks(x-axis_adjust/2, x_axis)
plt.ylabel('Normalized STP')
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('STP_workload'+workload_set+'.png')


