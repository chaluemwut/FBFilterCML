from db.mysql_conn import MysqlDb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from scipy import stats
import numpy as np
import pickle
from row_index import RowIndex


def read_line(file_name):
    lines = [line.rstrip('\n') for line in open(file_name)]
    return lines
        
def process(data_size):
    full_training_size = 150
    training_size = full_training_size - data_size
    half_training_size = training_size / 2
    
    db = MysqlDb()
    rowIdx = RowIndex() 
    measure_data = read_line('data/measurement.txt')[:data_size]
    un_measure = read_line('data/unmeasurement.txt')[:data_size]
    
#     print 'measure {} unmeasurement {}'.format(len(measure_data), len(un_measure))
    
    yes_training = [str(data) for data in db.get_n_row('yes', 75)]
    no_training = [str(data) for data in db.get_n_row('no', 75)]
    training_id = []
    training_id.extend(yes_training)
    training_id.extend(no_training)
    str_trainind_id = '(' + ','.join(training_id) + ')'
    x_train = db.get_feature_by_row(str_trainind_id)
    y_train = ([1] * 75) + ([0] * 75)
    
    rowIdx.training150_data_x = training_id
    rowIdx.training150_data_y = y_train
    
    x_train_other_id = []
    x_train_other_id.extend(yes_training[:half_training_size])
    x_train_other_id.extend(no_training[:half_training_size])
    str_trainind_other_id = '(' + ','.join(x_train_other_id) + ')'    
    x_train_other = db.get_feature_by_row(str_trainind_other_id)
    
    y_train_other = ([1] * half_training_size) + ([0] * half_training_size)
    
    rowIdx.training100_data_x = x_train_other_id
    rowIdx.training100_data_y = y_train_other
    
    yes_test = [str(data) for data in db.get_25_row('yes')]
    no_test = [str(data) for data in db.get_25_row('no')]
    test_id = []
    test_id.extend(yes_test)
    test_id.extend(no_test)
    str_test_id = '(' + ','.join(test_id) + ')'    
    x_test = db.get_feature_by_row(str_test_id)
    y_test = [1] * 25 + [0] * 25
    
    rowIdx.test50_data_x = test_id
    rowIdx.test50_data_y = y_test
    
#     print 'normal x {} y {}'.format(len(x_train), len(y_train))
# Case 1
    m1 = RandomForestClassifier()
    m1.fit(x_train, y_train)
    y_pred = m1.predict(x_test)
    fsc1 = f1_score(y_test, y_pred)
    

# Case 2
    m2_x_train = []
    m2_y_train = []
    str_measure_id = '(' + ','.join(measure_data) + ')'
    y_training_measure = db.get_label_data(str_measure_id)
    x_training_measure = db.get_feature_by_row(str_measure_id)
    for x in x_train_other:
        m2_x_train.append(x)
    for x in x_training_measure:
        m2_x_train.append(x) 
    
    for y in y_train_other:
        m2_y_train.append(y)
    for y in y_training_measure:
        m2_y_train.append(y)
        
#     print 'good x {} y {}'.format(len(m2_x_train), len(m2_y_train))   
 
    m2 = RandomForestClassifier()
    m2.fit(m2_x_train, m2_y_train)
    m2_y_pred = m2.predict(x_test)
    fsc2 = f1_score(y_test, m2_y_pred)
    

# Case 3
    m3_x_train = []
    m3_y_train = []
    str_unmeasure_id = '(' + ','.join(un_measure) + ')'
    y_training_measure3 = db.get_label_data(str_unmeasure_id)
    x_training_measure3 = db.get_feature_by_row(str_unmeasure_id)
    for x in x_train_other:
        m3_x_train.append(x)
    for x in x_training_measure3:
        m3_x_train.append(x) 
    
    for y in y_train_other:
        m3_y_train.append(y)
    for y in y_training_measure3:
        m3_y_train.append(y)
        
#     print 'good x {} y {}'.format(len(m3_y_train), len(m3_y_train)) 
       
    m3 = RandomForestClassifier()
    m3.fit(m3_x_train, m3_y_train)
    m3_y_pred = m3.predict(x_test)
    fsc3 = f1_score(y_test, m3_y_pred)
    
# case 4
    m4 = RandomForestClassifier()
    m4.fit(x_train_other, y_train_other)
    m4_y_pred = m4.predict(x_test)
    fsc4 = f1_score(y_test, m4_y_pred)
    
    return fsc1, fsc2, fsc3, fsc4, rowIdx

def run_test(data_size):
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    rowIdxLst = []
    for _ in range(0, 2000):
        fsc1, fsc2, fsc3, fsc4, rowId = process(data_size)
        res1.append(fsc1)
        res2.append(fsc2)
        res3.append(fsc3)
        res4.append(fsc4)
        rowIdxLst.append(rowId)
    return np.average(res1), np.average(res2), np.average(res3), np.average(res4)

def run_range():
    map_ret = {}
    for i in [10, 20, 30, 40, 50]:
        res1, res2, res3, res4 = run_test(i)
        map_ret[i] = [res1, res2, res3, res4]
        print '{},{},{}'.format(res1, res2, res3)
    
if __name__ == '__main__':
    run_range()
