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
        
def process():
    db = MysqlDb()
    rowIdx = RowIndex() 
    measure_data = read_line('data/measurement.txt')
    un_measure = read_line('data/unmeasurement.txt')
    
    yes_training = [str(data) for data in db.get_75_row('yes')]
    no_training = [str(data) for data in db.get_75_row('no')]
    training_id = []
    training_id.extend(yes_training)
    training_id.extend(no_training)
    str_trainind_id = '(' + ','.join(training_id) + ')'
    x_train = db.get_feature_by_row(str_trainind_id)
    y_train = [1] * 75 + [0] * 75
    
    rowIdx.training150_data_x = training_id
    rowIdx.training150_data_y = y_train
    
    x_train_other_id = []
    x_train_other_id.extend(yes_training[:50])
    x_train_other_id.extend(no_training[:50])
    str_trainind_other_id = '(' + ','.join(x_train_other_id) + ')'    
    x_train_other = db.get_feature_by_row(str_trainind_other_id)
    
    y_train_other = [1] * 50 + [0] * 50
    
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

def run_test():
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    rowIdxLst = []
    for _ in range(0, 200):
        fsc1, fsc2, fsc3, fsc4, rowId = process()
        res1.append(fsc1)
        res2.append(fsc2)
        res3.append(fsc3)
        res4.append(fsc4)
        rowIdxLst.append(rowId)
    pickle.dump(res1, open('result/res1.obj', 'wb'))
    pickle.dump(res2, open('result/res2.obj', 'wb'))
    pickle.dump(res3 , open('result/res3.obj', 'wb'))
    pickle.dump(res4, open('result/res4.obj', 'wb'))
    pickle.dump(rowIdxLst, open('result/row_id.obj', 'wb'))
    print 'fsc1 : {}, fsc2 : {}, fsc3 : {}, fsc4 : {}'.format(np.average(res1), np.average(res2), np.average(res3), np.average(res4))

def dump_test():
#     import matplotlib.pyplot as plt
    res1 = pickle.load(open('result/2000_1/res1.obj', 'rb'))
    res2 = pickle.load(open('result/2000_1/res2.obj', 'rb'))
    res3 = pickle.load(open('result/2000_1/res3.obj', 'rb'))
    res4 = pickle.load(open('result/2000_1/res4.obj', 'rb'))
#     plt.plot(res2)
#     plt.show()
#     print stats.ttest_ind(res1, res3)    
    print 'fsc1 : {}, fsc2 : {}, fsc3 : {}, fsc4 : {}'.format(np.average(res1), np.average(res2), np.average(res3), np.average(res4))
    
          
if __name__ == '__main__':
    dump_test()
    
