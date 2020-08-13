import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.font_manager import FontProperties
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
from matplotlib.ticker import EngFormatter, StrMethodFormatter



lat_env=pd.read_csv('lat_delft.txt', delim_whitespace=True, header=None).values
lat_env = ma.masked_where(lat_env==0, lat_env)


lon_env=pd.read_csv('lon_delft.txt', delim_whitespace=True, header=None).values
lon_env = ma.masked_where(lon_env==0, lon_env)


lat_part=pd.read_csv('lat_part_inpe.txt', delim_whitespace=True).values
lat_part = ma.masked_where(lat_part==0, lat_part)


lon_part=pd.read_csv('lon_part_inpe.txt', delim_whitespace=True).values
lon_part = ma.masked_where(lon_part==0, lon_part)


#depth=pd.read_csv('depth.txt', delim_whitespace=True,header=None).values
#depth= ma.masked_where(depth==0, depth)


diam=pd.read_csv('diam.txt', delim_whitespace=True)

font0 = FontProperties()
font = font0.copy()
font2=font0.copy()
font.set_style('normal')
font.set_size('x-large')
font.set_weight('bold')
font2.set_size('medium')
font2.set_weight('bold')
#prob=pd.read_csv('prob_func_cont.txt', delim_whitespace=True, header=None).values

#num=pd.read_csv('num_part_cont_vitoria_summer.txt', delim_whitespace=True, header=None).values

#probabi=pd.read_csv('probabilistic_vitoria_summer.txt', delim_whitespace=True, header=None).values


#conc=pd.read_csv('concentration_vitoria_summer.txt', delim_whitespace=True, header=None).values

#time_prob=pd.read_csv('time_prob_vitoria_summer.txt', delim_whitespace=True, header=None).values


lat1=lat_env.min()
lat2=lat_env.max()
lon1=lon_env.min()
lon2=lon_env.max()


u1=np.load('u1.npy')
v1=np.load('v1.npy')
u1= ma.masked_where(u1==0, u1)
v1= ma.masked_where(v1==0, v1)


m = Basemap(projection='merc',llcrnrlat=lat1,urcrnrlat=lat2,\
            llcrnrlon=lon1,urcrnrlon=lon2,lat_ts=20,resolution='l')

lon,lat=m(lon_env, lat_env)

lonp,latp=m(lon_part, lat_part)


m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='coral',lake_color='aqua')
m.drawcoastlines(color = '0.15')
m.drawparallels(np.arange(lat1, lat2, 1), labels=[1,0,0,0])
m.drawmeridians(np.arange(lon1, lon2, 1),labels=[0,0,0,1])

m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)

m.scatter(lonp,latp, c='b')


fig= plt.figure(figsize=(7, 7))
path='/home/fernando/codegfortran/figure/'
#for i in np.arange(lonp.shape[0]):
for i in np.arange(400):
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='coral',lake_color='aqua', alpha=0.1)
    m.drawcoastlines(color = '0.15')   
    m.drawparallels(np.arange(lat1, lat2, 0.1), labels=[1,0,0,0])
    m.drawmeridians(np.arange(lon1, lon2, 0.1),labels=[0,0,0,1])
    m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)
    m.scatter(lonp[i,:],latp[i,:], c='b')
    filename= path+str(i)+'south'+'.png'  
#    plt.savefig(filename, dpi=96)
    plt.savefig(filename)
    plt.gca()
    fig= plt.figure(figsize=(7, 7))

plt.show()


map = Basemap(llcrnrlon=-40.5,llcrnrlat=-20.5,urcrnrlon=-40.1,urcrnrlat=-20.2, epsg=4326)
#http://server.arcgisonline.com/arcgis/rest/services

map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 3000, verbose= True)
plt.show()


#https://www.bdmweather.com/2018/04/python-m-arcgisimage-basemap-options/


fig= plt.figure(figsize=(7, 7))
path='/home/valdir/Documentos/oil_model/input_data/figure_es/'
#for i in np.arange(lonp.shape[0]):
for i in np.arange(400):
    m.scatter(lonp[i,:],latp[i,:], c='b')
    filename= path+str(i)+'south'+'.png'  
#    plt.savefig(filename, dpi=96)
    plt.savefig(filename)
    plt.gca()
    fig= plt.figure(figsize=(7, 7))





depth=pd.read_csv('depth.txt', delim_whitespace=True).values
depth= ma.masked_where(depth==0, depth)
depth=depth*0 -1000

fig= plt.figure(figsize=(7, 7))
path='/home/valdir/Documentos/oil_model/input_data/figure_es/'
for i in np.arange(lon_part.shape[0]):
#for i in np.arange(600):
  plt.pcolor(lon_env, lat_env, depth, color='y')
  plt.scatter(lon_part[i,:], lat_part[i,:], color='b')
  filename= path+str(i)+'south_parti'+'.png'  
  plt.savefig(filename)
  plt.gca()
  fig= plt.figure(figsize=(7, 7))



depth=pd.read_csv('depth.txt', delim_whitespace=True,header=None).values
depth[np.where(depth<=3)] = -1000
depth= ma.masked_where(depth>3, depth) 
ad=3
lon_envq=lon_env.copy()
lat_envq=lat_env.copy()
a=np.ma.filled(lon_envq, fill_value=0)
a=np.where(a==0)
lon_envq=np.ma.filled(lon_envq, fill_value=lon_envq.min())
lat_envq=np.ma.filled(lat_envq, fill_value=lat_envq.min())
u1=np.ma.filled(u1, fill_value=0)
v1=np.ma.filled(v1, fill_value=0)
u1[a]=0
v1[a]=0
fig, ax = plt.subplots()
i=lon_part.shape[0]-1
for i in range(2):
  ax.scatter(lon_part[i,:], lat_part[i,:], color='b')
  ax.pcolor(lon_env, lat_env, depth, color='g')
  q=ax.quiver(lon_envq[0::ad,0::ad], lat_envq[0::ad,0::ad], u1[0::ad,0::ad, i],v1[0::ad,0::ad, i]) #quiver doesnot accept mask
  ax.quiverkey(q, X=0.3, Y=1.1, U=1,
             label='Quiver key', labelpos='E')
  axes = plt.gca()
  axes.set_xlim([-40.325,-40.250])
  axes.set_ylim([-20.36, -20.30])
  fig, ax = plt.subplots()



  
plt.show()




#################





##################################  current

ad=1
fig, ax = plt.subplots()
path='/home/valdir/Documentos/oil_model/input_data/figure_es/'
depth=pd.read_csv('depth.txt', delim_whitespace=True,header=None).values
depth= ma.masked_where(depth==0, depth)
depth= ma.masked_where(depth==0, depth)
#for i in np.arange(lon_part.shape[0]):
for i in [10]:
#  plt.pcolor(lon_env, lat_env, depth, color='y')
  lon_envq=lon_env.copy()
  lat_envq=lat_env.copy()
  lon_envq=np.ma.filled(lon_envq, fill_value=lon_envq.min())
  lat_envq=np.ma.filled(lat_envq, fill_value=lat_envq.min())
 # ax.pcolor(lon_env, lat_env, depth)
  un = u1[0::ad,0::ad, i] 
  vn = v1[0::ad,0::ad, i] 
  uv= np.sqrt(u1[0::ad,0::ad, i]**2 + v1[0::ad,0::ad, i]**2)
  y=ax.pcolor(lon_env[0::ad,0::ad], lat_env[0::ad,0::ad],uv, vmin=-0.5, vmax=0.5)
  q=ax.quiver(lon_envq[0::ad,0::ad], lat_envq[0::ad,0::ad], u1[0::ad,0::ad, i],v1[0::ad,0::ad, i], units='width')
  filename= path+str(i)+'current'+'.png'  
  ax.quiverkey(q, X=0.3, Y=1.1, U=0.5,
             label='1 m/s', labelpos='E')
  plt.gca()
#  axes = plt.gca()
#  axes.set_xlim([-40.330,-40.250])
#  axes.set_ylim([-20.34, -20.30])
  plt.colorbar(y)
  plt.savefig(filename)
  fig, ax = plt.subplots()


##################################


###particles

ad=1
ad2=3
fig, ax = plt.subplots()
path='/home/fernando/codegfortran/figure/'
#depth=pd.read_csv('depth.txt', delim_whitespace=True,header=None).values
#depth= ma.masked_where(depth==0, depth)
#depth= ma.masked_where(depth==0, depth)
for i in np.arange(lon_part.shape[0]):
#for i in np.arange(5):
#for i in [100]:
#  plt.pcolor(lon_env, lat_env, depth, color='y')
  lon_envq=lon_env.copy()
  lat_envq=lat_env.copy()
  lon_envq=np.ma.filled(lon_envq, fill_value=lon_envq.min())
  lat_envq=np.ma.filled(lat_envq, fill_value=lat_envq.min())
#  for k in np.arange(1)+1:
#   print(i)
#   land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
#   ax.fill(land[:,0], land[:,1], color='grey', alpha=1)
 # ax.pcolor(lon_env, lat_env, depth)
  un = u1[0::ad,0::ad, i] 
  vn = v1[0::ad,0::ad, i] 
  uv= np.sqrt(u1[0::ad,0::ad, i]**2 + v1[0::ad,0::ad, i]**2)
#  y=ax.pcolor(lon_env[0::ad,0::ad], lat_env[0::ad,0::ad],uv, vmin=-0.5, vmax=0.5)
  ax.scatter(lon_part[i,:], lat_part[i,:], color='r', s=1)
  q=ax.quiver(lon_envq[0::ad2,0::ad2], lat_envq[0::ad2,0::ad2], u1[0::ad2,0::ad2, i],v1[0::ad2,0::ad2, i], units='width')
 # q=ax.quiver(lon_envq[0::ad,0::ad], lat_envq[0::ad,0::ad], un,vn, units='width')
  filename= path+str(i)+'particle_tuba_era'+'.png'  
#  ax.quiverkey(q, X=0.3, Y=1.1, U=0.5,
#             label='1 m/s', labelpos='E')
  plt.gca()
  axes = plt.gca()
#  ax.set_xlim([-40.36,-40.20])    #model vel  vitoria bay
#  ax.set_ylim([-20.34, -20.26])    #model vel vitoria bay
#  ax.set_xlim([-50,-30])    #model vel
#  ax.set_ylim([-7,3])    #model vel
#  ax.set_xlim([-35,-28])    #model vel
#  ax.set_ylim([-12.5,-5])    #model vel
#  plt.colorbar(y)
  plt.savefig(filename)
  fig, ax = plt.subplots()    # 

plt.show()

#subplots

ad=1
ad2=4
fig = plt.figure(figsize=(10, 7))
lon_envq=lon_env.copy()
lat_envq=lat_env.copy()
lon_envq=np.ma.filled(lon_envq, fill_value=lon_envq.min())
lat_envq=np.ma.filled(lat_envq, fill_value=lat_envq.min())
for j in range(12):
  j=j+1
  i=j+13
  ax = fig.add_subplot(4,3,j)
  for k in np.arange(46)+1:
    land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
    ax.fill(land[:,0], land[:,1], color='grey', alpha=1)
  un = u1[0::ad,0::ad, i] 
  vn = v1[0::ad,0::ad, i] 
  uv= np.sqrt(u1[0::ad,0::ad, i]**2 + v1[0::ad,0::ad, i]**2)
  ax.scatter(lon_part[i,:], lat_part[i,:], color='r', s=5)
  q=ax.quiver(lon_envq[0::ad2,0::ad2], lat_envq[0::ad2,0::ad2], u1[0::ad2,0::ad2, i],v1[0::ad2,0::ad2, i], units='width')
  plt.gca()
  axes = plt.gca()
  ax.set_xlim([-40.35,-40.25])    #model vel  vitoria bay
  ax.set_ylim([-20.34, -20.29])    #model vel vitoria bay
  ax.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
  plt.rcParams['xtick.labelsize']=6
  plt.rcParams['ytick.labelsize']=6
  ax.yaxis.set_major_formatter(EngFormatter(unit=u"°"))
  plt.xticks(np.arange(-40.35,-40.25, step=0.03))
  textstr=str(j)+' h'
  props = dict(boxstyle='round', facecolor='wheat', alpha=1)
  ax.text(-40.34, -20.3, textstr,bbox=props, fontsize=8) 
#  ax.set_xlim([-41.025,-40.92])    #model vel
#  ax.set_ylim([-22, -21.6])    #model vel

plt.show()







#logo = plt.imread('logo.jpg')

ad=1
ad2=3
fig, ax = plt.subplots()
path='/home/valdir/Documentos/oil_model/run_summer/input/gif_clima/'
depth=pd.read_csv('depth.txt', delim_whitespace=True,header=None).values
depth= ma.masked_where(depth==0, depth)
depth= ma.masked_where(depth==0, depth)
#for i in np.arange(lon_part.shape[0]):
for i in np.arange(16):
#for i in [lon_part.shape[0]-1]:
#  plt.pcolor(lon_env, lat_env, depth, color='y')
  lon_envq=lon_env.copy()
  lat_envq=lat_env.copy()
  lon_envq=np.ma.filled(lon_envq, fill_value=lon_envq.min())
  lat_envq=np.ma.filled(lat_envq, fill_value=lat_envq.min())
  for k in np.arange(46)+1:
   print(i)
   land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
   ax.fill(land[:,0], land[:,1], color='grey', alpha=1)
 # ax.pcolor(lon_env, lat_env, depth)
  un = u1[0::ad,0::ad, i] 
  vn = v1[0::ad,0::ad, i] 
  uv= np.sqrt(u1[0::ad,0::ad, i]**2 + v1[0::ad,0::ad, i]**2)
 # y=ax.pcolor(lon_env[0::ad,0::ad], lat_env[0::ad,0::ad],uv, vmin=-0.5, vmax=0.5)  #model vel
  ax.scatter(lon_part[i,:], lat_part[i,:], color='r', s=10)
#  ax.scatter(lon_part[i,:], lat_part[i,:], color='r', s=diam.iloc[i].values)
  q=ax.quiver(lon_envq[0::ad2,0::ad2], lat_envq[0::ad2,0::ad2], u1[0::ad2,0::ad2, i],v1[0::ad2,0::ad2, i], units='width')
 # q=ax.quiver(lon_envq[0::ad,0::ad], lat_envq[0::ad,0::ad], un,vn, units='width')
  filename= path+str(i)+'tese'+'.png'  
  ax.quiverkey(q, X=0.3, Y=1.1, U=0.5,
             label='1 m/s', labelpos='E')
#  ax.figure.figimage(logo, xo=100, yo=150, alpha=1, zorder=10, origin = 'upper', resize=True)
  plt.gca()
  axes = plt.gca()
  axes.set_xlim([-40.330,-40.250])
  axes.set_ylim([-20.34, -20.30])
 # plt.colorbar(y)                      #model vel
  plt.savefig(filename)
  fig, ax = plt.subplots()


plt.show()



######################################33  Probability maps


fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=1)

s=ax.pcolor(lon_env, lat_env, prob, cmap='Reds')
plt.colorbar(s)
plt.show()





fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=1)

s=ax.pcolor(lon_env, lat_env,num, cmap='Reds')
plt.colorbar(s)
plt.show()


fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=1)

s=ax.pcolor(lon_env, lat_env,num/(sum(sum(num)))*100, cmap='Reds')
plt.colorbar(s)
plt.show()


fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=1)

s=ax.plot(lon_env, lat_env)
plt.show()

#NORMALIZE


ori=num/(sum(sum(num)))*100
t=np.ma.masked_where(ori==0, ori)
norm = (ori-t.min())/(t.max()-t.min())

fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=1)

s=ax.pcolor(lon_env, lat_env,norm, cmap='Reds')
circle1=plt.Circle((-40.30, -20.34), 0.02, edgecolor='black', facecolor='None')
ax.add_artist(circle1)
plt.colorbar(s)
plt.show()


  ###BEST PROBABILITY MAP SO FAR
ori=num/(sum(sum(num)))*100
t=np.ma.masked_where(ori==0 , ori)

fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=0.7)

s=ax.pcolor(lon_env, lat_env,t, cmap='coolwarm', vmin=0, vmax=0.4, linewidth=0, rasterized=True)
#plt.plot(-40.278, -20.3168, 'bo', label='Vitória bay spill', color = 'black')
plt.plot(-40.2422, -20.2986, 'bo', label='Tubarão port spill', color = 'black')
#circle1=plt.Circle((-40.2911, -20.3211), 0.004, edgecolor='red', facecolor='None')
#circle2=plt.Circle((-40.2562, -20.3114), 0.003, edgecolor='red', facecolor='None')
#circle3=plt.Circle((-40.2482, -20.3135), 0.003, edgecolor='red', facecolor='None')
#circle4=plt.Circle((-40.2668, -20.3097), 0.003, edgecolor='red', facecolor='None')
circle2=plt.Circle((-40.2487, -20.2893), 0.003, edgecolor='red', facecolor='None')
circle3=plt.Circle((-40.2439, -20.2979), 0.003, edgecolor='red', facecolor='None')
circle4=plt.Circle((-40.2306, -20.2991), 0.003, edgecolor='red', facecolor='None')
ax.set_xlim([-40.36,-40.20])    #model vel
ax.set_ylim([-20.34, -20.26])    #model vel
ax.legend(loc=4)
plt.xticks(np.arange(-40.36,-40.20, step=0.03))
#ax.add_artist(circle1)
#ax.add_artist(circle2)
#ax.add_artist(circle3)
#ax.add_artist(circle4)
cbar=plt.colorbar(s)
cbar.ax.get_yaxis().labelpad = 15
plt.xlabel('longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
ax.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
ax.yaxis.set_major_formatter(EngFormatter(unit=u"°"))
textstr='Camburi beach'
props = dict(boxstyle='round', facecolor='wheat', alpha=1)
ax.text(-40.299, -20.272, textstr,bbox=props, fontsize=6) 
cbar.ax.set_ylabel('Distribution of the oil slick (%)', rotation=270,fontsize=12)
plt.tight_layout()
#plt.savefig('figures_mooring/distribution_summer_tuba_mooring_cont.eps', format='eps', dpi=600)
plt.show()


#########new probability

t=np.ma.masked_where(probabi==0, probabi)


plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2
fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=0.7)

s=ax.pcolor(lon_env, lat_env,t, cmap='coolwarm', vmin=0, linewidth=0, rasterized=True)
#plt.plot(-40.278, -20.3168, 'bo', label='Vitória bay spill', color = 'black')
plt.plot(-40.2422, -20.2986, 'bo', label='Tubarão port spill', color = 'black')
ax.legend(loc=4)
cbar=plt.colorbar(s)
cbar.ax.get_yaxis().labelpad = 15
ax.set_xlim([-40.36,-40.20])    #model vel
#ax.set_xlim([-40.36,-40.22])    #model vel vitoria bay spill case
ax.set_ylim([-20.34, -20.26])    #model vel
plt.xticks(np.arange(-40.36,-40.20, step=0.03))
ax.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
ax.yaxis.set_major_formatter(EngFormatter(unit=u"°"))
plt.xlabel('longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
cbar.ax.set_ylabel('Probability of ocurrence (%)', rotation=270,fontsize=12)
plt.tight_layout()
textstr='Camburi beach'
props = dict(boxstyle='round', facecolor='wheat', alpha=1)
ax.text(-40.299, -20.272, textstr,bbox=props, fontsize=6) 
#plt.savefig('figures_mooring/probability_summer_tuba_mooring_cont.eps', format='eps', dpi=600)
ax.grid
plt.show()




#########areal density

conc1 = conc*1000
t=np.ma.masked_where((conc1==0)|(conc1<10), conc1)

fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=0.7)

s=ax.pcolor(lon_env, lat_env,t, cmap='coolwarm', vmin=10, vmax=250, linewidth=0, rasterized=True)
#plt.plot(-40.278, -20.3168, 'bo', label='Vitória bay spill', color = 'black')
plt.plot(-40.2422, -20.2986, 'bo', label='Tubarão port spill', color = 'black')
ax.legend(loc=4)
cbar=plt.colorbar(s)
#circle1=plt.Circle((-40.2911, -20.3211), 0.004, edgecolor='red', facecolor='None')
#circle2=plt.Circle((-40.2562, -20.3114), 0.003, edgecolor='red', facecolor='None')
#circle3=plt.Circle((-40.2482, -20.3135), 0.003, edgecolor='red', facecolor='None')
#circle4=plt.Circle((-40.2668, -20.3097), 0.003, edgecolor='red', facecolor='None')
circle2=plt.Circle((-40.2487, -20.2893), 0.003, edgecolor='red', facecolor='None')
circle3=plt.Circle((-40.2439, -20.2979), 0.003, edgecolor='red', facecolor='None')
circle4=plt.Circle((-40.2306, -20.2991), 0.003, edgecolor='red', facecolor='None')
cbar.ax.get_yaxis().labelpad = 15
ax.set_xlim([-40.36,-40.20])    #model vel
ax.set_ylim([-20.34, -20.26])    #model vel
plt.xticks(np.arange(-40.36,-40.20, step=0.03))
ax.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
ax.yaxis.set_major_formatter(EngFormatter(unit=u"°"))
#ax.add_artist(circle1)
#ax.add_artist(circle2)
#ax.add_artist(circle3)
#ax.add_artist(circle4)
plt.xlabel('longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
cbar.ax.set_ylabel('Area density (g/m2)', rotation=270,fontsize=12)
plt.tight_layout()
#plt.savefig('figures_mooring/density_summer_tuba_mooring.eps', format='eps', dpi=600)
plt.show()



#########time_prob

t=np.ma.masked_where(time_prob==0, time_prob)

fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=0.7)

s=ax.pcolor(lon_env, lat_env,t*100, cmap='coolwarm', vmin=0, vmax=30, linewidth=0, rasterized=True)
cbar=plt.colorbar(s)
cbar.ax.get_yaxis().labelpad = 15
#plt.plot(-40.278, -20.3168, 'bo', label='Vitória bay spill', color = 'black')
plt.plot(-40.2422, -20.2986, 'bo', label='Tubarão port spill', color = 'black')
ax.legend(loc=4)
ax.set_xlim([-40.36,-40.20])    #model vel
ax.set_ylim([-20.34, -20.26])    #model vel
plt.xticks(np.arange(-40.36,-40.20, step=0.03))
ax.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
ax.yaxis.set_major_formatter(EngFormatter(unit=u"°"))
plt.xlabel('longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
cbar.ax.set_ylabel('Time percentage (%)', rotation=270,fontsize=12)
plt.tight_layout()
#plt.savefig('figures_mooring/time_summer_tuba_mooring.eps', format='eps', dpi=600)
plt.show()



####################################points

depth[depth<0]=0
t=np.ma.masked_where(depth<-100, depth)
fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=1)

ax.set_xlim([-40.36,-40.22])    #model vel
ax.set_ylim([-20.35, -20.27])    #model vel
plt.show()



#################################################points
Study region

depth=pd.read_csv('depth.txt', delim_whitespace=True,header=None).values
depth= ma.masked_where(depth>-1, depth)


fig, ax = plt.subplots()
for k in np.arange(46)+1:
 land=pd.read_csv('land_b/saida'+str(k)+'.ldb', delim_whitespace=True, header=None).values
 ax.fill(land[:,0], land[:,1], color='grey', alpha=0.7)

ax.set_xlim([-40.395,-40.22])    #model vel
ax.set_ylim([-20.36, -20.23])    #model vel
plt.xticks(np.arange(-40.395,-40.18, step=0.05))
ax.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
ax.yaxis.set_major_formatter(EngFormatter(unit=u"°"))
plt.plot(-40.278, -20.3168,'^', label='Vitória bay spill', color = 'black')
plt.plot(-40.2422, -20.2986 ,'bo', label='Tubarão port spill', color = 'black')
plt.arrow(-40.2347, -20.3139, 0.0000,0.010)
plt.arrow(-40.2429, -20.2754, 0.0000,-0.010)
ax.legend(loc=4)
plt.xlabel('longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
textstpier1='Outer pier'
textstpier2='Inner pier'
props = dict(boxstyle='round', facecolor='wheat', alpha=1)
ax.text(-40.2416, -20.3178, textstpier1,bbox=props, fontsize=8) 
ax.text(-40.2547, -20.2754, textstpier2,bbox=props, fontsize=8) 
plt.tight_layout()
plt.savefig('figures_mooring/spill_location.eps', format='eps', dpi=300)
plt.show()




m = Basemap(projection='merc',llcrnrlat=lat1,urcrnrlat=lat2,\
            llcrnrlon=lon1,urcrnrlon=lon2,lat_ts=20,resolution='f')

lon,lat=m(lon_env, lat_env)

lonp,latp=m(lon_part, lat_part)


m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='coral',lake_color='aqua')
m.drawcoastlines(color = '0.15')
m.drawparallels(np.arange(lat1, lat2, 0.1), labels=[1,0,0,0])
m.drawmeridians(np.arange(lon1, lon2, 0.1),labels=[0,0,0,1])

m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)

m.scatter(lonp[-1,:],latp[-1,:], c='b')

plt.show()



