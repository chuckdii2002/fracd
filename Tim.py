import numpy as np
import pylab as pl
import ephem as eph
import math

#variable definition
#1.  q: is the rotation quaternion. It is obtained from the axis of orientation and the orientation angle between initial camera view and the direction of the celestial body
#2.  cameraview: is the initial view of the camera, before orientation
#3.  venusvec: The vector pointing in the direction of the celestial body. The celestial body in this example is venus. The vector in the direction of venus is computed using vnenus' relative position with respect to an observer. The trig identities for computting the three components of this vector assumes a 3D triangle. Also, I don't know your reference cordinate axes but I have assumed the z-direction is upwards.
#4.  rotationaxis: is the axis about which the camera is rotated. This direction is obtained as the cross product between venusvec and venusvec. Remember that the cross product between two vectors gives you a vector which is normal to both. If we assume venusvec and venusvec acts on a plane, then their cross product is the normal to the plain and about which orientation takes place.

#Other comments: So far, I have assumed a random cameraview of [1,0,0]. You'll need to set the correct value based on yuor refrence direction.

q = np.zeros(4)
cameraview = np.zeros(3)
venusvec = np.zeros(3)
newview = np.zeros(3)
rotationaxis = np.zeros(3)

cameraview[0] = 1.
cameraview[1] = 0.
cameraview[2] = 0.


# Observer. You'll need to set this properly as well
urloc = eph.Observer();
urloc.lon = '60:0.0'
urloc.lat = '33.8'
urloc.elevation = 2198
urloc.date = '1986/3/13'
v = eph.Venus(urloc)

hyp = math.cos(v.alt)
venusvec[0] = hyp*math.sin(v.az)
venusvec[1] = hyp*math.cos(v.az)
venusvec[2] = math.cos(v.alt)
magcameraview = np.power(cameraview[0]*cameraview[0]+cameraview[1]*cameraview[1]+cameraview[2]*cameraview[2],0.5)
magvenusvec = np.power(venusvec[0]*venusvec[0]+venusvec[1]*venusvec[1]+venusvec[2]*venusvec[2],0.5)
dotproduct = 0
for i in np.arange(0,3):
    venusvec[i] = venusvec[i]/magvenusvec
    cameraview[i] = cameraview[i]/magcameraview
    dotproduct = dotproduct + cameraview[i]*venusvec[i]

theta = dotproduct/(magcameraview*magvenusvec)
theta = math.acos(theta)
rotationaxis[0] = cameraview[1]*venusvec[2]-cameraview[2]*venusvec[1]
rotationaxis[1] = cameraview[2]*venusvec[0]-cameraview[0]*venusvec[2]
rotationaxis[2] = cameraview[0]*venusvec[1]-cameraview[1]*venusvec[0]
magrotationvec = np.power(rotationaxis[0]*rotationaxis[0]+rotationaxis[1]*rotationaxis[1]+rotationaxis[2]*rotationaxis[2],0.5)
for i in np.arange(0,3):
    rotationaxis[i] = rotationaxis[i]/magrotationvec

q[0] = math.cos(theta/2)
for i in np.arange(0,3):
    q[i+1] = rotationaxis[i]*math.sin(theta/2)
qmag = np.power(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3],0.5)
for i in np.arange(0,4):
    q[i] =q[i]/qmag

print q
