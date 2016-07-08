import pickle
import numpy as np
from scipy import stats

def read_result():
    fsc0 = pickle.load(open('data/result/fsc1.obj', 'rb'))
    fsc25 = pickle.load(open('data/result/fsc2.obj', 'rb'))
    fsc50 = pickle.load(open('data/result/fsc3.obj', 'rb'))
    fsc75 = pickle.load(open('data/result/fsc4.obj', 'rb'))
    fsc100 = pickle.load(open('data/result/fsc5.obj', 'rb'))
    
    print np.mean(fsc0), np.var(fsc0)
    print np.mean(fsc25), np.var(fsc25), stats.ttest_ind(fsc0, fsc25)
    print np.mean(fsc50), np.var(fsc50), stats.ttest_ind(fsc25, fsc50)
    print np.mean(fsc75), np.var(fsc75), stats.ttest_ind(fsc50, fsc75)
    print np.mean(fsc100), np.var(fsc75), stats.ttest_ind(fsc100, fsc75)

def read_result2():
    result1 = pickle.load(open('data/result/lab2/case1.obj', 'rb'))
    result2 = pickle.load(open('data/result/lab2/case2.obj', 'rb'))
    result3 = pickle.load(open('data/result/lab2/case3.obj', 'rb'))
    
    print np.mean(result1), np.var(result1)
    print np.mean(result2), np.var(result2), stats.ttest_ind(result1, result2)
    print np.mean(result3), np.var(result3), stats.ttest_ind(result2, result3)        
       
if __name__ == '__main__':
    read_result2()    
