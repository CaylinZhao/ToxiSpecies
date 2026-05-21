import pandas as pd
import numpy as np

dose_tasks = pd.read_csv('Data/Sample_num_dose.csv')  # 59
concentration_tasks = pd.read_csv('Data/Sample_num_concentration.csv')  # 48
tasks_all = pd.concat([dose_tasks, concentration_tasks])
tasks_all = tasks_all.sort_values(by='number', ascending=False).reset_index(drop=True)  # 107


# Setting_1_1: Mix all tasks, from head tasks to tail tasks, split rate 8:2
print('Setting_1_1')
tasks_head = tasks_all.iloc[0:int(0.8*len(tasks_all)), :]
tasks_tail = tasks_all.iloc[int(0.8*len(tasks_all))+1:, :]
tasks_head.to_csv('Data/3.Experiment setting/Setting_1_1/tasks_train.csv', index=False)
tasks_tail.to_csv('Data/3.Experiment setting/Setting_1_1/tasks_test.csv', index=False)

print('task num', len(tasks_head), len(tasks_tail))

data_train = pd.DataFrame()
label_col = []
for t in tasks_head.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Experiment setting/Setting_1_1/data_train.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_tail.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Experiment setting/Setting_1_1/data_test.csv', index=False)

print('Sample num:', len(data_train), len(data_test))


# Setting_1_2: Mix all tasks, from head tasks to tail tasks, split rate 7:3
print('Setting_1_2')
tasks_head = tasks_all.iloc[0:int(0.7*len(tasks_all)), :]
tasks_tail = tasks_all.iloc[int(0.7*len(tasks_all))+1:, :]
tasks_head.to_csv('Data/3.Experiment setting/Setting_1_2/tasks_train.csv', index=False)
tasks_tail.to_csv('Data/3.Experiment setting/Setting_1_2/tasks_test.csv', index=False)
print(len(tasks_head), len(tasks_tail))

data_train = pd.DataFrame()
label_col = []
for t in tasks_head.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Experiment setting/Setting_1_2/data_train.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_tail.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Experiment setting/Setting_1_2/data_test.csv', index=False)

print('Sample num:', len(data_train), len(data_test))


# Setting_1_3: Mix all tasks, from head tasks to tail tasks, split rate 6:4
print('Setting_1_3')
tasks_head = tasks_all.iloc[0:int(0.6*len(tasks_all)), :]
tasks_tail = tasks_all.iloc[int(0.6*len(tasks_all))+1:, :]
tasks_head.to_csv('Data/3.Experiment setting/Setting_1_3/tasks_train.csv', index=False)
tasks_tail.to_csv('Data/3.Experiment setting/Setting_1_3/tasks_test.csv', index=False)
print(len(tasks_head), len(tasks_tail))

data_train = pd.DataFrame()
label_col = []
for t in tasks_head.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Experiment setting/Setting_1_3/data_train.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_tail.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Experiment setting/Setting_1_3/data_test.csv', index=False)

print('Sample num:', len(data_train), len(data_test))


# Setting_2_1: From animal to human, i.e. from LDLo to TDLo
print('Setting_2_1')
tasks_train = tasks_all[tasks_all['label'].str.contains('LDLo')]
tasks_test = tasks_all[tasks_all['label'].str.contains('TDLo')]
tasks_train.to_csv('Data/3.Experiment setting/Setting_2_1/tasks_train.csv', index=False)
tasks_test.to_csv('Data/3.Experiment setting/Setting_2_1/tasks_test.csv', index=False)
print(len(tasks_train), len(tasks_test))

data_train = pd.DataFrame()
label_col = []
for t in tasks_train.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Experiment setting/Setting_2_1/data_train.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_test.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Experiment setting/Setting_2_1/data_test.csv', index=False)

print('Sample num:', len(data_train), len(data_test))


# Setting_2_2: From dose to concentration, i.e. from LD50 to LC50
print('Setting_2_2')
tasks_train = tasks_all[tasks_all['label'].str.contains('LD50')]
tasks_test = tasks_all[tasks_all['label'].str.contains('LC50')]
tasks_train.to_csv('Data/3.Experiment setting/Setting_2_2/tasks_train.csv', index=False)
tasks_test.to_csv('Data/3.Experiment setting/Setting_2_2/tasks_test.csv', index=False)
print(len(tasks_train), len(tasks_test))

data_train = pd.DataFrame()
label_col = []
for t in tasks_train.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Experiment setting/Setting_2_2/data_train.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_test.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Experiment setting/Setting_2_2/data_test.csv', index=False)

print('Sample num:', len(data_train), len(data_test))


# Setting_3_1: Same species and endpoints, different conditions, i.e. LD50 in different routes of rodent animals
print('Setting_3_1')
tasks_all = tasks_all[tasks_all['label'].str.contains('LD50')
                      & (tasks_all['label'].str.contains('mouse')
                         | tasks_all['label'].str.contains('rat'))]
tasks_train = tasks_all.iloc[0:int(0.7*len(tasks_all)), :]
tasks_test = tasks_all.iloc[int(0.7*len(tasks_all))+1:, :]
tasks_train.to_csv('Data/3.Experiment setting/Setting_3_1/tasks_train.csv', index=False)
tasks_test.to_csv('Data/3.Experiment setting/Setting_3_1/tasks_test.csv', index=False)
print(len(tasks_train), len(tasks_test))

data_train = pd.DataFrame()
label_col = []
for t in tasks_train.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Experiment setting/Setting_3_1/data_train.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_test.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Experiment setting/Setting_3_1/data_test.csv', index=False)

print('Sample num:', len(data_train), len(data_test))


# Setting_3_2: Same species and endpoints, different conditions, i.e. LC50 in different fish
print('Setting_3_2')
tasks_all = tasks_all[tasks_all['label'].str.contains('LC50')
                      & (tasks_all['label'].str.contains('Danio rerio')
                         | tasks_all['label'].str.contains('Oryzias latipes')
                         | tasks_all['label'].str.contains('Pimephales promelas')
                         | tasks_all['label'].str.contains('Oncorhynchus mykiss'))]

tasks_train = tasks_all.iloc[0:int(0.7*len(tasks_all)), :]
tasks_test = tasks_all.iloc[int(0.7*len(tasks_all))+1:, :]
tasks_train.to_csv('Data/3.Experiment setting/Setting_3_2/tasks_train.csv', index=False)
tasks_test.to_csv('Data/3.Experiment setting/Setting_3_2/tasks_test.csv', index=False)
print(len(tasks_train), len(tasks_test))

data_train = pd.DataFrame()
label_col = []
for t in tasks_train.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Experiment setting/Setting_3_2/data_train.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_test.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Experiment setting/Setting_3_2/data_test.csv', index=False)

print('Sample num:', len(data_train), len(data_test))












