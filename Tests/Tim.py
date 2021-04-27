import numpy as np
import pylab as pl
import ephem as eph
import math

q = np.ones(4)

urloc = eph.Observer();
urloc.lon = '60:0.0'
#urloc.lon = '33'
urloc.lat = '35:05.8'
urloc.lat = '33.8'
urloc.elevation = 2198
urloc.date = '1986/3/13'
v = eph.Venus(urloc);

z = math.cos(v.alt)
hyp = math.cos(v.alt)
y = hyp*math.cos(v.az)
x = hyp*math.sin(v.az)



print x,y,z

print('%s %s' % (urloc.lon, urloc.lat))
print('%f %f' % (urloc.lon, urloc.lat))
print('%f %f' % (v.alt, v.az))
print('%s %s' % (v.alt, v.az))
print q[0], q[1], q[2],q[3]





#q = np.ones(4)
#
#urloc = eph.Observer();
#urloc.lon = '-111:32.1'
#urloc.lon = '111.1'
#urloc.lat = '35:05.8'
#urloc.lat = '33.8'
#urloc.elevation = 2198
#urloc.date = '1986/3/13'
#
#print('%s %s' % (urloc.lon, urloc.lat))
#print q[0], q[1], q[2],q[3]