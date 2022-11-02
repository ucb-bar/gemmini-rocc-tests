import sys
import re
import csv
import numpy as np
from matplotlib import pyplot as plt

num_workload = []
num_core = 8
app_name = ['resnet', 'alexnet', 'googlenet', 'squeezenet', 'kwsnet', 'yolonet', 'yololitenet']
total_workload_type = list(range(len(app_name)))
total_priority_type = list(range(12))
target_scale = 1.0

priority_low_start = 0
priority_low_end = 2 # or till 1?
priority_mid_start = 3
priority_mid_end = 8
priority_high_start = 9
priority_high_end = 11


name_list = []
workload_type_list = []
priority_list = []
target_list = []
result_list = []
success_list = []

qos_list = []

#success = []
#ToDo: mp/sp

#thread_cycle_storage = [[0 for i in range(len(core_count))] for j in range(num_app)]
#print(num_app, thread_cycle_storage)
#total_cycle_storage = [[0 for i in range(len(core_count))] for j in range(num_app)]

#stat, dynam
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
            raw = int(priority_search.group(2))
            if raw <= priority_low_end:
                priority.append(0)
            elif raw <= priority_mid_end:
                priority.append(1)
            elif raw <= priority_high_end:
                priority.append(2)
#priority.append(int(priority_search.group(2)))

        elif target_search:
            this_target = int(target_search.group(2))
            target.append(this_target)

        elif result_search:
            result.append(int(result_search.group(2)))


success_ratio = []
for i in range(test):
    ratio = sum(success_list[i]) / num_workload[i]
    success_ratio.append(ratio)


# mod2
priority_dict0 = []
priority_dict1 = []
priority_dict2 = []
for i in range(len(name_list)):
    priority_dict = {}
    priority_dist = {}
    for key, value in zip(priority_list[i], success_list[i]):
        if key in priority_dict:
            priority_dict[key] += value
            priority_dist[key] += 1
        else:
            priority_dict[key] = value
            priority_dist[key] = 1

    for j in priority_dict:
        priority_dict[j] = priority_dict[j] / priority_dist[j]
    priority_dict = dict(sorted(priority_dict.items()))
    priority_dict['total'] = success_ratio[i]
    priority_dict0.append(priority_dict[0])
    priority_dict1.append(priority_dict[1])
    priority_dict2.append(priority_dict[2])
    #priority_dict_list.append(priority_dict)


print(name_list)
print(success_ratio)
print(priority_dict0, priority_dict1, priority_dict2)
name_list = [x for _,x in sorted(zip(qos_list, name_list))]
result_list = [x for _,x in sorted(zip(qos_list, result_list))]
priority_dict0 = [x for _,x in sorted(zip(qos_list, priority_dict0))]
priority_dict1 = [x for _,x in sorted(zip(qos_list, priority_dict1))]
priority_dict2 = [x for _,x in sorted(zip(qos_list, priority_dict2))]
success_list = [x for _,x in sorted(zip(qos_list, success_list))]
success_ratio = [x for _,x in sorted(zip(qos_list, success_ratio))]
qos_list.sort()
print("after sorted")
print(qos_list)
print(name_list)
print(success_ratio)
print(priority_dict0, priority_dict1, priority_dict2)

static = []
dynamic = []
for i in range(len(success_ratio)):
    if name_list[i] == 0:
        static.append(success_ratio[i])
    elif name_list[i] == 1:
        dynamic.append(success_ratio[i])
    else:
        print('invalid')


xlabels = ['MoCA','Static Partition']
c = ['blue', 'orange']
xticks = ['Low', 'Medium', 'High', 'Total']
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14


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
plt.bar(x, static, width=axis_adjust, label=xlabels[1], color=c[1])
plt.bar(x-axis_adjust, dynamic, width=axis_adjust, label=xlabels[0], color=c[0])
plt.xlabel('QoS levels')
plt.xticks(x-axis_adjust/2, x_axis)
plt.ylabel('SLA Satisfaction Rate')
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('total_workload'+workload_set+'.png')

x_axis = ['p-Low', 'p-Mid', 'p-High']
x_label = ['Static Partition', 'MoCA']
color = ['orange', 'blue']

fig, axs = plt.subplots(1, 3, figsize=(12,4))
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#axis_adjust = 0.7 / (len(x_axis))
for i in range(len(qos_type)):
    this_priority_dict_list = []
    this_priority_dict_list.append([priority_dict0[2*i], priority_dict1[2*i], priority_dict2[2*i]])
    this_priority_dict_list.append([priority_dict0[2*i+1], priority_dict1[2*i+1], priority_dict2[2*i+1]])
    print(this_priority_dict_list)
    axis_adjust = 0.5 / len(this_priority_dict_list)
    #plt.figure(figsize=(12,8))
    #fig = plt.figure(figsize=(6,4))
#plt.figure(figsize=(6, 4))
    #ax = fig.add_subplot(111)
    x = np.arange(len(this_priority_dict_list[0]))
    x_adjust = axis_adjust * int(len(this_priority_dict_list)/2)
    for j in range(len(this_priority_dict_list)):
        axs[i].bar(x-axis_adjust*j, this_priority_dict_list[j], width=axis_adjust, label=x_label[j], color=color[j])
    axs[i].set_xticks(x-axis_adjust/2, x_axis)#, fontsize=BIGGER_SIZE)
    axs[i].set_xlabel("Priority Level", fontweight='bold', fontsize=BIGGER_SIZE)
    axs[i].set_ylabel("SLA Satisfaction Rate", fontweight='bold', fontsize=BIGGER_SIZE)
    axs[i].set(ylim=(0, 1))
    axs[i].set_title('QoS-'+qos_type[i].upper(), fontsize = 16)
    #axs[i].set_yticks(fontsize=16)
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('priority_workload'+workload_set+'.png')



'''
for i in range(num_app):
    total_cycle_storage[i] = [total_cycle_storage[i][0] / x for x in total_cycle_storage[i]]
    thread_cycle_storage[i] = [thread_cycle_storage[i][0] / x for x in thread_cycle_storage[i]]

print(thread_cycle_storage)
print(total_cycle_storage)

plt.figure(figsize=(12,8))
x_axis = core_count
for i in range(num_app):
    plt.plot(x_axis, total_cycle_storage[i], marker='x', label=app_name[i])
plt.xlabel('Core count')
plt.ylabel('Normalized improvement')
plt.ylim([1, core_count[-1]])
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('total.png')

plt.figure(figsize=(12,8))
x_axis = core_count
for i in range(num_app):
    plt.plot(x_axis, thread_cycle_storage[i], marker='x', label=app_name[i])
plt.xlabel('Core count')
plt.ylabel('Normalized improvement')
plt.ylim([1, core_count[-1]])
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('thread.png')
'''
