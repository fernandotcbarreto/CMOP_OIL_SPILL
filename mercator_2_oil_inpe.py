##delft 2 oil
import numpy as np
from netCDF4 import Dataset as dat
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import matplotlib.dates as dates

limn=20

path = '/home/fernando/mercator_regrid/'

file=dat(path+'regrid_recorte_ext-PSY4V3R1_1dAV_20190301_20190302_gridS_R20190313.nc')

input = path+'regrid_recorte_ext-PSY4V3R1_1dAV_'
endfnameu='_gridU_R20190313.nc'
endfnamev='_gridV_R20190313.nc'
endfnamet='_gridT_R20190313.nc'
endfnames='_gridS_R20190313.nc'


dateini='2019-03-01'
dateend='2019-03-03'
a=[dateini, dateend]

date=dates.datestr2num(a)

ntimes=len(np.arange(date[0], date[1]+1))

date = np.arange(date[0], date[1]+1)

time_oil=np.arange(ntimes)

time_oil=time_oil*24*60 

lat= np.ma.filled(file['lat'][::],fill_value=0)

lon= np.ma.filled(file['lon'][::],fill_value=0)

[lon,lat]=np.meshgrid(lon,lat)

fileu=dat(path+'regrid_recorte_ext-PSY4V3R1_1dAV_20190301_20190302_gridU_R20190313.nc')
u = np.ma.filled(fileu['u'][::],fill_value=0)
filev=dat(path+'regrid_recorte_ext-PSY4V3R1_1dAV_20190301_20190302_gridV_R20190313.nc')
v = np.ma.filled(filev['v'][::],fill_value=0)

utim=np.zeros([ntimes,limn, u.shape[2], u.shape[3]])
vtim=np.zeros([ntimes,limn, v.shape[2], v.shape[3]])
ttim=np.zeros([ntimes,limn, v.shape[2], v.shape[3]])
stim=np.zeros([ntimes,limn, v.shape[2], v.shape[3]])


depth = np.ma.filled(fileu['u'][::],fill_value=19999999)
depth = depth[0,0,:,:]
depth[depth<19999999]=1000;depth[depth>=19999999]=-1000

layer =  -np.ma.filled(fileu['deptht'][::],fill_value=0)
layer=layer[0:limn]



for i in range(ntimes):
  fileu=input + dates.num2date(date[i]).strftime("%Y%m%d")+'_'+dates.num2date(date[i]+1).strftime("%Y%m%d")+endfnameu
  filev=input + dates.num2date(date[i]).strftime("%Y%m%d")+'_'+dates.num2date(date[i]+1).strftime("%Y%m%d")+endfnamev
  filet=input + dates.num2date(date[i]).strftime("%Y%m%d")+'_'+dates.num2date(date[i]+1).strftime("%Y%m%d")+endfnamet
  files=input + dates.num2date(date[i]).strftime("%Y%m%d")+'_'+dates.num2date(date[i]+1).strftime("%Y%m%d")+endfnames
  fileu = dat(fileu)
  filev = dat(filev)
  filet = dat(filet)
  files = dat(files)
  u = np.ma.filled(fileu['u'][:,0:limn,::],fill_value=0)
  v = np.ma.filled(filev['v'][:,0:limn,::],fill_value=0)
  t = np.ma.filled(filet['t'][:,0:limn,::],fill_value=0)
  s = np.ma.filled(files['s'][:,0:limn,::],fill_value=0)  
  utim[i,::] = u
  vtim[i,::] = v
  ttim[i,::] = t
  stim[i,::] = s

counttimeh=np.zeros([1])
#counttimeh=np.array([12*60])


u,v=utim.copy(), vtim.copy()

with open('time_delft.txt', 'w') as f:
   np.savetxt(f, np.array(len(time_oil)).reshape(1,), fmt = '%i')
   np.savetxt(f, np.array(int(lon.shape[0])).reshape(1,), fmt = '%i')
   np.savetxt(f, np.array(int(lon.shape[1])).reshape(1,), fmt = '%i')   
   np.savetxt(f, np.array(int(u.shape[1])).reshape(1,), fmt = '%i')      
   np.savetxt(f, time_oil,fmt='%10.4f')
   np.savetxt(f, layer,fmt='%10.4f')
   np.savetxt(f, counttimeh,fmt='%10.4f')
#   np.savetxt(f, time_oil,fmt="%s")


for i in range(len(time_oil)):
  with open(str(i+1) + 'v.txt', 'w') as f:
    for j in range(v.shape[1]):
      np.savetxt(f, np.squeeze(vtim[i,j, :,:]),fmt='%10.8f')

for i in range(len(time_oil)):
  with open(str(i+1) + 'u.txt', 'w') as f:
    for j in range(u.shape[1]):
      np.savetxt(f, np.squeeze(utim[i,j, :,:]),fmt='%10.8f')

for i in range(len(time_oil)):
  with open(str(i+1) + 'temp.txt', 'w') as f:
    for j in range(u.shape[1]):
      np.savetxt(f, np.squeeze(ttim[i,j, :,:]),fmt='%10.8f')

for i in range(len(time_oil)):
  with open(str(i+1) + 'salt.txt', 'w') as f:
    for j in range(u.shape[1]):
      np.savetxt(f, np.squeeze(stim[i,j, :,:]),fmt='%10.8f')

with open('lat_delft.txt', 'w') as f:
     np.savetxt(f, lat,fmt='%10.8f')


with open('lon_delft.txt', 'w') as f:
     np.savetxt(f, lon,fmt='%10.8f')


with open('depth.txt', 'w') as f:
  np.savetxt(f, depth, fmt='%10.1f')



