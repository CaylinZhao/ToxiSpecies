import pandas as pd
import numpy as np

# Load task lists for dose-based and concentration-based toxicity endpoints
dose_tasks = pd.read_csv('Data/Sample_num_dose.csv')  # 59 tasks
concentration_tasks = pd.read_csv('Data/Sample_num_concentration.csv')  # 48 tasks
tasks_all = pd.concat([dose_tasks, concentration_tasks])
# Sort tasks by sample count to ensure consistency across splits
tasks_all = tasks_all.sort_values(by='number', ascending=False).reset_index(drop=True)  # 107 total


# Scenario 1_1: Mixed all species/endpoints, 8:1:1 split for generalization testing
print('Setting_1_1')
tasks_train = tasks_all.iloc[0:int(0.8*len(tasks_all)), :]
tasks_tail = tasks_all.iloc[int(0.8*len(tasks_all)):, :]
tasks_valid = tasks_tail.sample(frac=0.5, random_state=0)
tasks_test = tasks_tail.drop(tasks_valid.index, axis=0)
# Save split metadata for reproducibility
tasks_train.to_csv('Data/3.Task split/Setting_1_1/tasks_train.csv', index=False)
tasks_valid.to_csv('Data/3.Task split/Setting_1_1/tasks_valid.csv', index=False)
tasks_test.to_csv('Data/3.Task split/Setting_1_1/tasks_test.csv', index=False)
print('task num 1_1', len(tasks_train), ':', len(tasks_valid), ':', len(tasks_test))

data_train = pd.DataFrame()
label_col = []
for t in tasks_train.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Task split/Setting_1_1/data_train.csv', index=False)

data_valid = pd.DataFrame()
label_col = []
for t in tasks_valid.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_valid = data_valid._append(data_add)
    label_col += [t]*len(data_add)
data_valid.insert(loc=1, column='Label_name', value=label_col)
data_valid.to_csv('Data/3.Task split/Setting_1_1/data_valid.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_test.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Task split/Setting_1_1/data_test.csv', index=False)

print('Sample num 1_1:', len(data_train), ':', len(data_valid), ':', len(data_test))
print('Chemical 1_1:', len(data_train['Canonical SMILES'].unique()), ':', len(data_valid['Canonical SMILES'].unique()), ':', len(data_test['Canonical SMILES'].unique()))


# Setting_1_2: Mix all tasks, from head tasks to tail tasks, split rate 6:2:2
print('Setting_1_2')
tasks_train = tasks_all.iloc[0:int(0.6*len(tasks_all)), :]
tasks_tail = tasks_all.iloc[int(0.6*len(tasks_all)):, :]
tasks_valid = tasks_tail.sample(frac=0.5, random_state=0)
tasks_test = tasks_tail.drop(tasks_valid.index, axis=0)
tasks_train.to_csv('Data/3.Task split/Setting_1_2/tasks_train.csv', index=False)
tasks_valid.to_csv('Data/3.Task split/Setting_1_2/tasks_valid.csv', index=False)
tasks_test.to_csv('Data/3.Task split/Setting_1_2/tasks_test.csv', index=False)
print('task num 1_2', len(tasks_train), ':', len(tasks_valid), ':', len(tasks_test))

data_train = pd.DataFrame()
label_col = []
for t in tasks_train.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Task split/Setting_1_2/data_train.csv', index=False)

data_valid = pd.DataFrame()
label_col = []
for t in tasks_valid.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_valid = data_valid._append(data_add)
    label_col += [t]*len(data_add)
data_valid.insert(loc=1, column='Label_name', value=label_col)
data_valid.to_csv('Data/3.Task split/Setting_1_2/data_valid.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_test.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Task split/Setting_1_2/data_test.csv', index=False)

print('Sample num 1_2:', len(data_train), ':', len(data_valid), ':', len(data_test))
print('Chemical 1_2:', len(data_train['Canonical SMILES'].unique()), ':', len(data_valid['Canonical SMILES'].unique()), ':', len(data_test['Canonical SMILES'].unique()))


# Setting_2_1: From animal to human, i.e. from LDLo,LD50,LC50 to TDLo
print('Setting_2_1')
tasks_trva = tasks_all[tasks_all['label'].str.contains('LDLo') | tasks_all['label'].str.contains('LD50') | tasks_all['label'].str.contains('LC50')]

tasks_train = tasks_trva.iloc[0:int(0.8*len(tasks_trva)), :]
tasks_valid = tasks_trva.iloc[int(0.8*len(tasks_trva)):, :]
tasks_test = tasks_all[tasks_all['label'].str.contains('TDLo')]
tasks_train.to_csv('Data/3.Task split/Setting_2_1/tasks_train.csv', index=False)
tasks_valid.to_csv('Data/3.Task split/Setting_2_1/tasks_valid.csv', index=False)
tasks_test.to_csv('Data/3.Task split/Setting_2_1/tasks_test.csv', index=False)
print('task num 2_1', len(tasks_train), ':', len(tasks_valid), ':', len(tasks_test))

data_train = pd.DataFrame()
label_col = []
for t in tasks_train.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Task split/Setting_2_1/data_train.csv', index=False)

data_valid = pd.DataFrame()
label_col = []
for t in tasks_valid.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_valid = data_valid._append(data_add)
    label_col += [t]*len(data_add)
data_valid.insert(loc=1, column='Label_name', value=label_col)
data_valid.to_csv('Data/3.Task split/Setting_2_1/data_valid.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_test.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Task split/Setting_2_1/data_test.csv', index=False)

print('Sample num 2_1:', len(data_train), ':', len(data_valid), ':', len(data_test))
print('Chemical 2_1:', len(data_train['Canonical SMILES'].unique()), ':', len(data_valid['Canonical SMILES'].unique()), ':', len(data_test['Canonical SMILES'].unique()))


# Setting_2_2: ：From dose to concentration, i.e. from LDLo,LD50,TDLo to LC50
print('Setting_2_2')
tasks_trva = tasks_all[tasks_all['label'].str.contains('LDLo') | tasks_all['label'].str.contains('LD50') | tasks_all['label'].str.contains('TDLo')]
tasks_train = tasks_trva.iloc[0:int(0.9*len(tasks_trva)), :]
tasks_valid = tasks_trva.iloc[int(0.9*len(tasks_trva)):, :]
tasks_test = tasks_all[tasks_all['label'].str.contains('LC50')]
tasks_train.to_csv('Data/3.Task split/Setting_2_2/tasks_train.csv', index=False)
tasks_valid.to_csv('Data/3.Task split/Setting_2_2/tasks_valid.csv', index=False)
tasks_test.to_csv('Data/3.Task split/Setting_2_2/tasks_test.csv', index=False)
print('task num 2_2', len(tasks_train), ':', len(tasks_valid), ':', len(tasks_test))

data_train = pd.DataFrame()
label_col = []
for t in tasks_train.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Task split/Setting_2_2/data_train.csv', index=False)

data_valid = pd.DataFrame()
label_col = []
for t in tasks_valid.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_valid = data_valid._append(data_add)
    label_col += [t]*len(data_add)
data_valid.insert(loc=1, column='Label_name', value=label_col)
data_valid.to_csv('Data/3.Task split/Setting_2_2/data_valid.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_test.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Task split/Setting_2_2/data_test.csv', index=False)

print('Sample num 2_2:', len(data_train), ':', len(data_valid), ':', len(data_test))
print('Chemical 2_2:', len(data_train['Canonical SMILES'].unique()), ':', len(data_valid['Canonical SMILES'].unique()), ':', len(data_test['Canonical SMILES'].unique()))


# Setting_2_3: ：Same species and endpoints, different conditions, i.e. LC50 in different fish
print('Setting_2_3')
tasks_trvate = tasks_all[tasks_all['label'].str.contains('LC50')]
tasks_train = tasks_trvate.iloc[0:int(0.8*len(tasks_trvate)), :]
tasks_tail = tasks_trvate.iloc[int(0.8*len(tasks_trvate)):, :]
tasks_valid = tasks_tail.sample(frac=0.5, random_state=0)
tasks_test = tasks_tail.drop(tasks_valid.index, axis=0)
tasks_train.to_csv('Data/3.Task split/Setting_2_3/tasks_train.csv', index=False)
tasks_valid.to_csv('Data/3.Task split/Setting_2_3/tasks_valid.csv', index=False)
tasks_test.to_csv('Data/3.Task split/Setting_2_3/tasks_test.csv', index=False)
print('task num 2_3', len(tasks_train), ':', len(tasks_valid), ':', len(tasks_test))

data_train = pd.DataFrame()
label_col = []
for t in tasks_train.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_train = data_train._append(data_add)
    label_col += [t]*len(data_add)
data_train.insert(loc=1, column='Label_name', value=label_col)
data_train.to_csv('Data/3.Task split/Setting_2_3/data_train.csv', index=False)

data_valid = pd.DataFrame()
label_col = []
for t in tasks_valid.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_valid = data_valid._append(data_add)
    label_col += [t]*len(data_add)
data_valid.insert(loc=1, column='Label_name', value=label_col)
data_valid.to_csv('Data/3.Task split/Setting_2_3/data_valid.csv', index=False)

data_test = pd.DataFrame()
label_col = []
for t in tasks_test.iloc[:, 0]:
    data_add = pd.read_csv('Data/4.All data/' + str(t) + '.csv')
    data_add = data_add.rename(columns={data_add.columns.tolist()[0]: 'ChemID', data_add.columns.tolist()[1]: 'Label_value'})
    data_test = data_test._append(data_add)
    label_col += [t]*len(data_add)
data_test.insert(loc=1, column='Label_name', value=label_col)
data_test.to_csv('Data/3.Task split/Setting_2_3/data_test.csv', index=False)

print('Sample num 2_3:', len(data_train), ':', len(data_valid), ':', len(data_test))
print('Chemical 2_3:', len(data_train['Canonical SMILES'].unique()), ':', len(data_valid['Canonical SMILES'].unique()), ':', len(data_test['Canonical SMILES'].unique()))



