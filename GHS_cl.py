import math


def lc50_cl(values):
    labels_cl = []
    for i in range(len(values)):
        if values[i] <= math.log(1, 10): # 0
            labels_cl.append(1)
        elif values[i] > math.log(1, 10) and values[i] <= math.log(10, 10): # 1
            labels_cl.append(2)
        elif values[i] > math.log(10, 10) and values[i] <= math.log(100, 10): # 2
            labels_cl.append(3)
        else:
            labels_cl.append(0)
    return labels_cl


def oral_ld50_cl(values):
    labels_cl = []
    for i in range(len(values)):
        if values[i] <= math.log(5, 10): # 0.7
            labels_cl.append(1)
        elif values[i]>math.log(5, 10) and values[i]<=math.log(50, 10): # 1.7
            labels_cl.append(2)
        elif values[i]>math.log(50, 10) and values[i]<=math.log(300, 10): # 2.5
            labels_cl.append(3)
        elif values[i]>math.log(300, 10) and values[i]<=math.log(2000, 10): # 3.3
            labels_cl.append(4)
        elif values[i] >math.log(2000, 10) and values[i]<=math.log(5000, 10): # 3.7
            labels_cl.append(5)
        else:
            labels_cl.append(0)
    return labels_cl


def skin_ld50_cl(values):
    labels_cl = []
    for i in range(len(values)):
        if values[i] <= math.log(50, 10): # 1.7
            labels_cl.append(1)
        elif values[i] > math.log(50, 10) and values[i] <= math.log(200, 10): # 2.3
            labels_cl.append(2)
        elif values[i] > math.log(200, 10) and values[i] <= math.log(1000, 10): # 3
            labels_cl.append(3)
        elif values[i] > math.log(1000, 10) and values[i] <= math.log(2000, 10): # 3.3
            labels_cl.append(4)
        elif values[i] > math.log(2000, 10) and values[i] <= math.log(5000, 10): # 3.7
            labels_cl.append(5)
        else:
            labels_cl.append(0)
    return labels_cl


def simple_cl(values, label_name):
    rank = {'Gambusia affinis_4.0d_LC50': [10, 1000], 'Rat_oral_LD50': [50, 2000], 'Rabbit_skin_LD50': [200, 2000]}

    if label_name == 'Gambusia affinis_4.0d_LC50':
        labels_cl = []
        for i in range(len(values)):
            if values[i] <= math.log(rank[label_name][0], 10):
                labels_cl.append('High')
            elif values[i] > math.log(rank[label_name][0], 10) and values[i] <= math.log(rank[label_name][1], 10):
                labels_cl.append('Low')
            else:
                labels_cl.append('Non')
    else:
        labels_cl = []
        for i in range(len(values)):
            if values[i] <= math.log(rank[label_name][0], 10):
                labels_cl.append('High')
            elif values[i] > math.log(rank[label_name][0], 10) and values[i] <= math.log(rank[label_name][1], 10):
                labels_cl.append('Low')
            else:
                labels_cl.append('Non')

    return labels_cl









