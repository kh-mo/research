def get_far(tp, fn, fp, tn):
    '''
    실제 abnormal 중 normal로 예측한 비율
    지나친 generalization
    정상으로 간주하는 범위가 커질수록 값이 커진다
    '''

    return fp / (fp+tn)

def get_frr(tp, fn, fp, tn):
    '''
    실제 normal중 abnormal로 예측한 비율
    지나친 specialization
    정상으로 간주하는 범위가 작아질수록 값이 커진다
    '''
    return fn / (tp+fn)




#########################################
'''
0 : normal
1 : abnormal
  
  | pred
------------------
r |   |  0 |  1
e | 0 | tp | fn  
a |--------------
l | 1 | fp | tn

관심있는 값을 작은 숫자로 놓는것이 디폴트 설정을 그대로 쓰는 방법이다.
'''

import numpy as np
from sklearn.metrics import confusion_matrix

real = np.random.randint(low=0, high=2, size=10)
pred = np.random.randint(low=0, high=2, size=10)
print("real : ", real, "\npred : ", pred)

confusion_matrix(real, pred, labels=[0,1])
tp, fn, fp, tn = confusion_matrix(real, pred, labels=[0,1]).ravel()

get_far(tp, fn, fp, tn)
get_frr(tp, fn, fp, tn)