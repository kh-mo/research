import scipy.stats as stats
import numpy as np


dist = stats.beta
n_trials = [0,1,2,3,4,5,8,15,50,500,1000,2000]
data = stats.bernoulli.rvs(0.5, size=n_trials[-1]) # 베르누이 확률로 n_trials[-1]개 만큼 0,1 생성
x = np.linspace(0,1,100) # [0~1] 100개 포인트 생성

for k, N in enumerate(n_trials):
    print(k)
    heads = data[:N].sum() ## heads는 N의 절반값에 가깝다
    y = dist.pdf(x,1+heads,1+N-heads) ## 본 수식(베타분포)의 기대값은 0.5에 수렴하게 된다



from IPython.core.pylabtools import figsize

figsize(11, 9)

from matplotlib import pyplot as plt

plt.subplot(len(n_trials)/2, 2, k+1) ## (len(n_trials)/2, 2)의 서브플랏에서 k+1번째 위치를 차지하고 있음을 의미



