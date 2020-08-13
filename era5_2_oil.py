##delft 2 oil
import numpy as np
from netCDF4 import Dataset as dat
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import matplotlib.dates as dates

fileu='download_u_10_20190301_20190303.nc'

filev='download_v_10_20190301_20190303.nc'

uwind=dat(fileu)

vwind=dat(filev)

time=uwind['time'][::]

lon=uwind['longitude'][::]

lat=uwind['latitude'][::]

long, latg = np.meshgrid(lon,lat)


u10a=uwind['u10'][::]

v10a=vwind['v10'][::]

timeinioil = '2019-03-25 00:00:00'

a=['1900-01-01 00:00:00', timeinioil]

numhours= np.diff(dates.datestr2num(a))*24    # it is in hours since Era 5 is given in 'hours since 1900-01-01 00:00:00.0'

ind= np.argmin(abs(time-numhours))



if ind==0 and numhours < time[ind]:
    print('EXECUTION TIME STARTS BEFORE DATA')
#    sys.exit()
elif numhours == time[ind]:
  counttimeh = numhours - time[ind]
elif numhours < time[ind]:
  ind=ind-1 
  counttimeh = numhours - time[ind]
elif numhours > time[ind]:
  counttimeh = numhours - time[ind]


u10=u10a[ind::]

v10=v10a[ind::]

timemod = time - time[ind]

timemod = timemod[ind::]

time_wind = timemod*60

counttimeh = counttimeh*60


with open('time_era_5.txt', 'w') as f:
   np.savetxt(f, np.array(len(time_wind)).reshape(1,), fmt = '%i')                 #number of time series
   np.savetxt(f, np.array(int(long.shape[0])).reshape(1,), fmt = '%i')             #number of latitudes
   np.savetxt(f, np.array(int(long.shape[1])).reshape(1,), fmt = '%i')             #number of longitudes
   np.savetxt(f, time_wind,fmt='%10.4f')                                           #time gauging points
   np.savetxt(f, counttimeh,fmt='%10.4f')                                           #time gauging points


for i in range(len(time_wind)):
  with open(str(i+1) + 'u_wind.txt', 'w') as f:
      np.savetxt(f, np.squeeze(u10[i,:,:]),fmt='%12.8f')

for i in range(len(time_wind)):
  with open(str(i+1) + 'v_wind.txt', 'w') as f:
      np.savetxt(f, np.squeeze(v10[i,:,:]),fmt='%12.8f')



with open('lat_era_5.txt', 'w') as f:
     np.savetxt(f, latg,fmt='%12.8f')



with open('lon_era_5.txt', 'w') as f:
     np.savetxt(f, long,fmt='%12.8f')

