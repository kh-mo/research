'''
Batchnormalization

Batchnorm의 평균, 분산 계산은 batch 단위로 이루어진다
그리고 dimension별로 계산을 한다
'''

import torch
import torch.nn as nn

input = torch.randn(2,3)
m = nn.BatchNorm1d(3, affine=False)

input_mean = input.mean(dim=0)
input_var = ((input-input_mean)**2).mean(dim=0)

our_batchnorm_cal_result = (input-input_mean) / torch.sqrt(input_var+m.eps)
batchnorm_function_result = m(input)

print(our_batchnorm_cal_result)
print(batchnorm_function_result)

'''
torch.var(unbiased=True or False)
unbiased 옵션은 Bessel's correction을 이용할지 말지를 결정하는 옵션
차이는 편차 제곱의 평균을 n을 쓸지 n-1을 쓸지 나타내는 것이다

unbiased = True : ((x-mean(x))**2) / n
unbiased = False : ((x-mean(x))**2) / (n-1)

'''
input_var = ((input-input_mean)**2).mean(dim=0)
var_function_result = torch.var(input, dim=0, unbiased=False)
print(input_var)
print(var_function_result)

input_var = ((input-input_mean)**2).sum(dim=0) / (input.shape[0]-1)
var_function_result = torch.var(input, dim=0, unbiased=True)
print(input_var)
print(var_function_result)
