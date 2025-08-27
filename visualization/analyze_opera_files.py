import h5py
import os
import sys


#f = h5py.File('../dataset/w4c24/2020/OPERA-CONTEXT/boxi_0015.train20.rates.crop.h5', "r")
f = h5py.File('../dataset/w4c24/2020/OPERA-CONTEXT/roxi_0004.train20.rates.crop.h5', "r")
data = f['rates.crop']
print('Data', data.shape)
print('data[0]:',data[100])


print('HRIT:')


f = h5py.File('../dataset/w4c24/2020/HRIT/roxi_0008.cum1test20.reflbt0.ns.h5', "r")
data = f['REFL-BT']
print('Data', data.shape)
print('data[0]:',data[0])
