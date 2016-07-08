from filter01 import read_line
from sets import Set
import pickle

def merage_process_measure():
    data_new = Set([int(x) for x in read_line('data/new/new_measurement.txt')])
    data_old = Set([int(x) for x in read_line('data/measurement.txt')])
    data_new.update(data_old)
    lst = sorted(list(data_new))
    pickle.dump(lst, open('data/data100/measure.obj', 'wb'))

def merage_process_un_measure():
    un_20 = [1817, 1844, 1854, 1881, 1883, 1886, 1892, 1904, 1906, 1909, 1913, 1947, 1948, 1968, 1976, 1984, 1987, 1994, 1996, 2020]
    data_new = Set([int(x) for x in read_line('data/new/new_unmeasurement.txt')])
    data_new2 = Set([int(x) for x in read_line('data/new/new_unmeasurement2.txt')])    
    data_old = Set([int(x) for x in read_line('data/unmeasurement.txt')])
    data_new.update(data_old)
    data_new.update(data_new2)
    data_new.update(un_20)
    lst = sorted(list(data_new))
    pickle.dump(lst, open('data/data_new_100/un_measure.obj', 'wb'))

def load_data():
    d = pickle.load(open('data/data_new_100/un_measure.obj', 'rb'))
    print len(d)   
    
if __name__ == '__main__':
    load_data()