##delft 2 oil
import numpy as np
from netCDF4 import Dataset as dat
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import matplotlib.dates as dates

limn=1

file=dat('transfer_Fernando/ext-PSY4V3R1_1dAV_20190301_20190302_gridS_R20190313.nc')

input = 'transfer_Fernando/ext-PSY4V3R1_1dAV_'

dateini='2019-03-01'
dateend='2019-03-02'
a=[dateini, dateend]

date=dates.datestr2num(a)

ntimes=len(np.arange(date[0], date[1]+1))

date = np.arange(date[0], date[1]+1)

time_oil=np.arange(ntimes)

time_oil=time_oil*24*60 

lat= np.ma.filled(file['nav_lat'][::],fill_value=0)

lon= np.ma.filled(file['nav_lon'][::],fill_value=0)

#[lon,lat]=np.meshgrid(lon,lat)

fileu=dat('transfer_Fernando/ext-PSY4V3R1_1dAV_20190301_20190302_gridU_R20190313.nc')
u = np.ma.filled(fileu['vozocrtx'][::],fill_value=0)
filev=dat('transfer_Fernando/ext-PSY4V3R1_1dAV_20190301_20190302_gridV_R20190313.nc')
v = np.ma.filled(filev['vomecrty'][::],fill_value=0)

utim=np.zeros([ntimes,limn, u.shape[2], u.shape[3]])
vtim=np.zeros([ntimes,limn, v.shape[2], v.shape[3]])

depth = np.ma.filled(fileu['vozocrtx'][::],fill_value=19999999)
depth = depth[0,0,:,:]
depth[depth<19999999]=1000;depth[depth>=19999999]=-1000

layer =  -np.ma.filled(fileu['deptht'][::],fill_value=0)
layer=layer[0:limn]



for i in range(ntimes):
  fileu=input + dates.num2date(date[i]).strftime("%Y%m%d")+'_'+dates.num2date(date[i]+1).strftime("%Y%m%d")+'_gridU_R20190313.nc'
  filev=input + dates.num2date(date[i]).strftime("%Y%m%d")+'_'+dates.num2date(date[i]+1).strftime("%Y%m%d")+'_gridV_R20190313.nc'
  fileu = dat(fileu)
  filev = dat(filev)
  u = np.ma.filled(fileu['vozocrtx'][:,limn-1,::],fill_value=0)
  v = np.ma.filled(filev['vomecrty'][:,limn-1,::],fill_value=0)
  utim[i,::] = u
  vtim[i,::] = v

u1=np.squeeze(utim).copy()
v1=np.squeeze(vtim).copy()

time=pd.read_csv('time_hours.txt', delim_whitespace=True).values   ###output data
time=time[:,0]*60 


u_interp=np.zeros([u1.shape[1], u1.shape[2], len(time)])
v_interp=np.zeros([v1.shape[1], v1.shape[2], len(time)])

for j in np.arange(u_interp.shape[0]):
   for k in np.arange(u_interp.shape[1]):
      u_interp[j,k,:] = np.interp(time, time_oil, np.squeeze(u1[:,j,k]))



for j in range(v_interp.shape[0]):
  for k in range(v_interp.shape[1]):
     v_interp[j,k,:] = np.interp(time, time_oil, v1[:,j,k])


np.save('u1', u_interp)

np.save('v1', v_interp)
