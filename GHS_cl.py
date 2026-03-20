import math


def lc50_cl(values):
    """
    Classifies LC50 values into categories based on toxicity thresholds.
    Thresholds are in log10 scale (log10(mg/L)).
    
    Categories:
    1: High Toxicity (<= 1 mg/L)
    2: Moderate Toxicity (1-10 mg/L)
    3: Low Toxicity (10-100 mg/L)
    0: Non-toxic / Out of range (> 100 mg/L)
    """
    labels_cl = []
    for i in range(len(values)):
        if values[i] <= math.log(1, 10): # 1 mg/L -> log10(1) = 0
            labels_cl.append(1)
        elif values[i] > math.log(1, 10) and values[i] <= math.log(10, 10): # 10 mg/L
            labels_cl.append(2)
        elif values[i] > math.log(10, 10) and values[i] <= math.log(100, 10): # 100 mg/L
            labels_cl.append(3)
        else:
            labels_cl.append(0)
    return labels_cl


def oral_ld50_cl(values):
    """
    GHS Classification for Oral LD50 (log10(mg/kg)).
    Levels 1-5 correspond to increasing severity defined by the GHS system.
    """
    labels_cl = []
    for i in range(len(values)):
        if values[i] <= math.log(5, 10): # Level 1: <= 5 mg/kg
            labels_cl.append(1)
        elif values[i]>math.log(5, 10) and values[i]<=math.log(50, 10): # Level 2
            labels_cl.append(2)
        elif values[i]>math.log(50, 10) and values[i]<=math.log(300, 10): # Level 3
            labels_cl.append(3)
        elif values[i]>math.log(300, 10) and values[i]<=math.log(2000, 10): # Level 4
            labels_cl.append(4)
        elif values[i] >math.log(2000, 10) and values[i]<=math.log(5000, 10): # Level 5
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

