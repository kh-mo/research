'''
과제 : plt 클래스의 객체 단위, 인스턴스 단위를 이해할 필요가 있음
해당 예시는 동전던지기가 시행횟수가 늘어남에 따라 어떤 확률 분포를 띄게 되는지 나타내는 코드
데이터 생성 : 베르누이 분포
prior, posterior : 베타분포(인자에 대한 이해가 더 필요)
베이지안 추론은 람다를 찾아가는 과정으로 이해할 수도 있음
'''
import scipy.stats as stats
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc("font", family="Malgun Gothic")

figsize(11, 9) # 이미지 전체 크기 결정(세로, 가로)

dist = stats.beta
n_trials = [0,1,2,3,4,5,8,15,50,500,1000,2000]
data = stats.bernoulli.rvs(0.5, size=n_trials[-1]) # 베르누이 확률로 n_trials[-1]개 만큼 0,1 생성
x = np.linspace(0,1,100) # [0~1] 100개 포인트 생성

for k, N in enumerate(n_trials):
    sx = plt.subplot(len(n_trials)/2, 2, k+1) ## (len(n_trials)/2, 2)의 서브플랏에서 k+1번째 위치를 차지하고 있음을 의미
    plt.xlabel("$p$, 앞면의 확률", fontsize=13) if k in [0, len(n_trials)-1] else None # 처음과 마지막에만 xlabel 지정
    plt.setp(sx.get_yticklabels(), visible=False) # set property(y label 안보이게 하자)

    heads = data[:N].sum() ## heads는 N의 절반값에 가깝다
    y = dist.pdf(x,1+heads,1+N-heads) ## 본 수식(베타분포)의 기대값은 0.5에 수렴하게 된다

    plt.plot(x, y, label="%d 번의 동전 던지기, \n앞면 %d 번 관측"%(N, heads)) # label이 잘 안나오는데 이것은 아래 leg 설정을 이용해서 해결한다
    plt.fill_between(x, 0, y, color="#348ABD", alpha=0.4) # 두 라인 y1~y2사이를 칠해주는 기능
    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1) # 0.5 position에 0~4 길이의 -- 선 그리기

    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)

plt.suptitle("사후확률의 베이지안 업데이트", y=1.02, fontsize=14)
plt.tight_layout() # 레이아웃 간격을 자동 조절
