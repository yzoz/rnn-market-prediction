from influxdb import InfluxDBClient
from datetime import datetime as dt
import pandas as pd
import numpy as np

inUser = 'ad'
inPwd = 'pass'
InHost = 'localhost'
inPort = 8086

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def fromTS(ts):
    date = dt.fromtimestamp(ts)
    return date

def toTS(date):
    ts = int(dt.timestamp(pd.Timestamp(date)))
    return ts

def indb(inDB):
    db = InfluxDBClient(InHost, inPort, inUser, inPwd, inDB)
    return db

def min_max(start, end, ticker, db='micex'):

    buy_max = 0
    sell_max = 0

    min_max = indb(db).query('SELECT max("size"), min("size") FROM %s WHERE time >= %s AND time <= %s' % (ticker, start, end), epoch='s').get_points()

    for row in min_max:
        buy_max = row['max']
        sell_max = row['min']

    return buy_max, sell_max

def getLast(ticker, db):
    cursor = indb(db).query('SELECT LAST("c") FROM "%s" ' % (ticker), epoch='s').get_points()
    for row in cursor:
        times = row['time']
    return times

#now()
def prices(start, end, group, ticker, db='micex'):
    cursor = indb(db).query('SELECT FIRST("p"), LAST("p") FROM %s WHERE time >= %s AND time <= %s GROUP BY time(%s)' % (ticker, start, end, group), epoch='s').get_points()
    price = []
    times = []
    for pr in cursor:
        if pr['first'] and pr['last'] and pr['time']:
            _pr = pr['first'] - pr['last']
            _times = fromTS(pr['time'])
            price.append(_pr)
            times.append(_times)
        else:
            continue
    return price, times
