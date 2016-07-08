from db.mysql_conn import MysqlDb
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy import stats
import numpy as np
import pickle
from row_index import RowIndex
import filter01
import copy, random

db = MysqlDb()

def read_line(file_name):
    lines = [line.rstrip('\n') for line in open(file_name)]
    return lines
    
def get_training_data():
    yes_test = [str(data) for data in db.get_25_row('yes')]
    no_test = [str(data) for data in db.get_25_row('no')]
    test_id = []
    test_id.extend(yes_test)
    test_id.extend(no_test)
    str_test_id = '(' + ','.join(test_id) + ')'    
    x_test = db.get_feature_by_row(str_test_id)
    y_test = [1] * 25 + [0] * 25
    return x_test, y_test

def get_notin_training_data():
    num_row = 50
    measure = [str(x) for x in pickle.load(open('data/data100/measure.obj', 'rb'))]
    un_measure = [str(x) for x in pickle.load(open('data/data100/un_measure.obj', 'rb'))]
    str_lst = '({},{})'.format(','.join(measure), ','.join(un_measure))   
    
    yes_test = [str(data) for data in db.get_notin_row('yes', str_lst, num_row)]
    no_test = [str(data) for data in db.get_notin_row('no', str_lst, num_row)]
    test_id = []
    test_id.extend(yes_test)
    test_id.extend(no_test)
    str_test_id = '(' + ','.join(test_id) + ')'    
    x_test = db.get_feature_by_row(str_test_id)
    y_test = [1] * num_row + [0] * num_row
    return x_test, y_test

def get_random_training_data():
    num_row = 1300
    measure = [str(x) for x in pickle.load(open('data/data_new_100/measure.obj', 'rb'))]
    un_measure = [str(x) for x in pickle.load(open('data/data_new_100/un_measure.obj', 'rb'))]
    str_lst = '({},{})'.format(','.join(measure), ','.join(un_measure))
     
    random_id_list = [str(x) for x in db.get_random_training_data(str_lst, num_row)];
    str_id = '(' + ','.join(random_id_list) + ')'
    x_test = db.get_feature_by_row(str_id)
    y_test = db.get_label_data(str_id) 
    return x_test, y_test    
    
def str_join(lst):
    return '(' + ','.join(lst) + ')'
    
def process_100():
    fsc1 = []
    fsc2 = []
    fsc3 = [] 
    fsc4 = []
    fsc5 = []
    for _ in range(200):
        x_test, y_test = get_random_training_data()                
        fsc1.append(process_un_measure(0, x_test, y_test, 0))
        fsc2.append(process_un_measure(25, x_test, y_test, 25))
        fsc3.append(process_un_measure(50, x_test, y_test, 50))
        fsc4.append(process_un_measure(75, x_test, y_test, 75))
        fsc5.append(process_un_measure(100, x_test, y_test, 100))
    
    pickle.dump(fsc1, open('data/result/fsc1.obj', 'wb'))
    pickle.dump(fsc2, open('data/result/fsc2.obj', 'wb'))
    pickle.dump(fsc3, open('data/result/fsc3.obj', 'wb'))
    pickle.dump(fsc4, open('data/result/fsc4.obj', 'wb'))
    pickle.dump(fsc5, open('data/result/fsc5.obj', 'wb'))
               
    print np.average(fsc1)
    print np.average(fsc2)
    print np.average(fsc3)
    print np.average(fsc4)
    print np.average(fsc5)
            
def process_un_measure(n, x_test, y_test, data_size):    
    measure = [str(x) for x in pickle.load(open('data/data_new_100/measure.obj', 'rb'))]
    un_measure = [str(x) for x in pickle.load(open('data/data_new_100/un_measure.obj', 'rb'))]
    
    random.shuffle(measure)
    random.shuffle(measure)
    
    random.shuffle(un_measure)
    random.shuffle(un_measure) 
    
    measure = measure[0:100]
    un_measure = un_measure[0:100]
    
    n_measure = len(measure) - n
    adding_measure = measure[0:n_measure]
    adding_unmeasure = un_measure[0:n]
        
    x_train = []
    y_train = []
    
    if n != 100:
        x_measure = db.get_feature_by_row(str_join(adding_measure))
        y_measure = db.get_label_data(str_join(adding_measure))
        x_train.extend(x_measure)
        y_train.extend(y_measure)
        
    if n != 0:
        x_un_measure = db.get_feature_by_row(str_join(adding_unmeasure))
        y_un_measure = db.get_label_data(str_join(adding_unmeasure))
        x_train.extend(x_un_measure)
        y_train.extend(y_un_measure)
    
    m = tree.DecisionTreeClassifier()
    ml = copy.deepcopy(m)
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    return recall_score(y_test, y_pred)

def create_tree_img(clf, loop, data_size):
    from sklearn.externals.six import StringIO
    with open(str(loop)+'_'+str(data_size)+".dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)


def prediction_process(train_index, x_test, y_test):
    x_train = db.get_feature_by_row(str_join(train_index))
    y_train = db.get_label_data(str_join(train_index))
        
    m = tree.DecisionTreeClassifier()
    ml = copy.deepcopy(m)
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    return f1_score(y_test, y_pred)   

def lab2():
    import copy, random
    measure = [str(x) for x in pickle.load(open('data/data_new_100/measure.obj', 'rb'))]
    un_measure = [str(x) for x in pickle.load(open('data/data_new_100/un_measure.obj', 'rb'))]
        
    measure = measure[0:100]
    un_measure = un_measure[0:100]
    
    result1 = []
    result2 = []
    result3 = []
    for _ in range(200):
#     case1   
        x_test, y_test = get_random_training_data()
        random.shuffle(measure)
        random.shuffle(measure)
        random.shuffle(un_measure)
        random.shuffle(un_measure)
               
        c1_measure = measure[0:50]
        f1_c1 = prediction_process(c1_measure, x_test, y_test)
        result1.append(f1_c1)
        
#         case 2
        c2_measure = measure[0:100]
        f1_c2 = prediction_process(c2_measure, x_test, y_test)
        result2.append(f1_c2)
        
#         case 3
        c3_measure = measure[0:50]
        c3_measure.extend(un_measure[0:50])
        f1_c3 = prediction_process(c3_measure, x_test, y_test)
        result3.append(f1_c3)
    
    pickle.dump(result1, open('data/result/lab2/case1.obj', 'wb'))
    pickle.dump(result2, open('data/result/lab2/case2.obj', 'wb'))
    pickle.dump(result3, open('data/result/lab2/case3.obj', 'wb'))    
    
    print np.average(result1)
    print np.average(result2)
    print np.average(result3)         
    

         
if __name__ == '__main__':
    lab2()
   
# 0.976068230431
# 0.928097299897
# 0.915756956285
# 0.895935715245
# 0.654371587486    
