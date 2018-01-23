import pandas as pd
from inc import indb, toTS

def ugly(_d, _t):
    _d = str(_d)
    _t = str(_t)
    y = _d[0:4]
    m = _d[4:6]
    d = _d[6:8]

    hh = _t[0:2]
    mm = _t[2:4]
    ss = _t[4:6]

    f = y + '-' + m + '-' + d + 'T' + hh + ':' + mm + ':' + ss
    #print(f)
    return toTS(f)

A = pd.read_csv('data/RTS_150101_171201_5.csv', sep=',', usecols=[2, 3, 4, 5, 6, 7, 8]).to_records()
#123492
for i, _d, _t, o, h, l, c, v in A:
    times = ugly(_d, _t)
    p = (o + h + l + c) / 4
    data = [
        {
            'measurement': 'rts',
            'time': times,
            'fields': {
                'p': round(p, 2),
                'v': v
            }
        }
    ]
    indb('micex').write_points(data, time_precision='s')
    #print(times, p, v)
    print("\r%s"%(i), end="")