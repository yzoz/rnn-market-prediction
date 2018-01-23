import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from inc import prices, toTS
from sklearn import preprocessing
import numpy as np

def placeNum(nums):

    arr = []

    for x in nums:
        if x <= -0.275: arr.append(0)
        elif x <= -0.075: arr.append(1)
        elif x >= 0.275: arr.append(4)
        elif x >= 0.075: arr.append(3)
        else: arr.append(2)

    return arr

start = toTS('2015-01-05T00:00:00')
finish = toTS('2017-12-15T00:00:00')
back = 8
win = '60m'

pr, times = prices(start * 1000000000, finish * 1000000000, win, 'rts')
mn = (len(pr) // back) * back
pr = pr[0:mn]
times = times[0:mn]
#print(pr)
pr = np.reshape(pr, [-1, back])
pr = preprocessing.normalize(pr)
pr = np.round(pr, decimals=3)
pr = np.reshape(pr, -1)
pr = placeNum(pr)

unique, counts = np.unique(pr, return_counts=True)
print(dict(zip(unique, counts)), len(unique))


fig = plt.figure(figsize=(70,35))
a=fig.gca()
a.set_frame_on(False)
plt.plot(times, pr)
#plt.show()
plt.grid(True)
plt.savefig('norm.png', dpi=72, bbox_inches='tight', pad_inches=0)
plt.close()
