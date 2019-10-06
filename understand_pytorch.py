'''
pytorch 함수를 쓰다보면 왜 이렇게 동작하는지 의문이 들 때가 있다.
혹은 정확한 동작, 내가 원하는 방식으로 동작하는건지 궁금해 질 때가 있고 또 논문을 정확히 이해하고 구현한건지 고민스러울 때도 있다.
이 모든 상황에 올바르게 계산, 구현을 하고 있는지 참고 자료 용도로써 본 파일을 작성하고 유지한다.
'''

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

'''
learning rate scheduling
example : cosine annealing, iclr 2017
'''

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
    def forward(self, x):
        return x

learning_rate_max = 0.01
learning_rate_min = 0.00001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_max)
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0.01, eta_min=learning_rate_min)