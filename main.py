from math import *
import numpy as np
from matplotlib import pyplot as plt
import copy
#rotate


#possbile bug sources:
#numbers must be rounded becasue of pi and radians givin imperfet results
#mainly in simulator, so not too big of a deal for production code

p1 = [0,0]
p2 = [90,0]
p3 = [130, -50]
p4 = [150, -50]
vp1 = np.array(p1)
vp2 = np.array(p2)
vp3 = np.array(p3)
vp4 = np.array([150, -50])

##vp1 = np.array([0,0])
##vp2 = np.array([10,0])
##vp3 = np.array([20,0])
##vp4 = np.array([30,0])
##vp5 = np.array([40,0])
##vp6 = np.array([50,0])
##vp7 = np.array([60,0])
##vp8 = np.array([70,0])
##vp9 = np.array([80,0])
##vp10 = np.array([96.6666,-5.5555])
##vp11 = np.array([103.333,-11.11111])
##vp12 = np.array([109.99999,-16.6667])
##vp13 = np.array([116.6666,-22.2222])
##vp14 = np.array([123.3333,-27.7777])
##vp15 = np.array([130.0000,-33.333313])
##vp16 = np.array([136.6666,-38.8888])
##vp17 = np.array([143.3333,-44.4444])
##vp18 = np.array([150,-50])
##
##pointlist = [vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10,vp11,vp12,vp13,vp14,vp15,vp16,vp17,vp18]


npoints = []
pointlist = [vp1, vp2, vp3,vp4]




def graph(l):
    plt.rcParams["figure.figsize"] = [7.00,3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.xlim(0,170)
    plt.ylim(-60,0)
    for x in l:
        plt.plot(x[0],x[1],marker="o",markersize=20,markeredgecolor="red",markerfacecolor="green")
    plt.show()

def  smoother( path,  weight_data,  weight_smooth,  tolerance) :
        # copy array
        newPath = copy.deepcopy(path)
        change = tolerance
        while (change >= tolerance) :
            change = 0.0
            i = 1
            while (i < len(path) - 1) :
                j = 0
                while (j < len(path[i])) :
                    aux = newPath[i][j]
                    newPath[i][j] += weight_data * (path[i][j] - newPath[i][j]) + weight_smooth * (newPath[i - 1][j] + newPath[i + 1][j] - (2.0 * newPath[i][j]))
                    change += abs(aux - newPath[i][j])
                    j += 1
                i += 1
        return newPath


def smooth(path, a, b, tol):
    npath = path.copy()
    change = tol
    while change >= tol:
        change = 0.0
##        for i in range(0, len(path)-1):
        i = 1
        #print(len(path)-1)
        while i<len(path)-1:
            #for j in range(path[i].length):
            for j in range(2):    
                aux = npath[i][j]
                npath[i][j] += a* (path[i][j] - npath[i][j]) + b * (npath[i-1][j] + npath[i+1][j]-(2.0*npath[i][j]))
                #print("t"+str(npath[i][j]))
                change += abs(float(aux)-npath[i][j])
                #print(change)
            #print(npath[i])
            i = i + 1
    return npath

#def convert_to_path

def magnitude(vec):
    return np.sqrt(vec.dot(vec))

def normalize(vec):
    normalized_v = vec / np.sqrt(np.sum(vec**2))
    return normalized_v

for x in range(len(pointlist)-1):
    #print(x)
    vector = pointlist[x+1] - pointlist[x]
    points_that_fit = ceil(magnitude(vector) / 10)
    #print(points_that_fit)
    #print(vector)

    vector = normalize(vector) * 10
    #print(vector)
    for i in range(points_that_fit):
        npoints.append(pointlist[x]+vector*i)

npoints.append(pointlist[-1])

#print(npoints)
#graph(npoints)
graph(pointlist)
print(pointlist)
graph(npoints)
b = smoother(npoints,0.25,0.75,0.001)
print("")
print("npath")
print("")
print(b)
graph(b)


def rotate(px,py,rx,ry,deg):
    rot = deg
    temp1 = px - rx
    first = temp1 * cos(rot)
    temp2 = py - ry
    sec = temp2 * sin(rot)
    x1 = first - sec + rx
    #print(x1)
    third = temp1 * sin(rot)
    fourth = temp2*cos(rot)
    y1 = third + fourth + ry
    #print(y1)
    l = [round(x1,9),round(y1,9),round(rot,9)]
    return l


def get_rot_point(offset,objectunitrot):
    roaxispointx = 0 + offset
    z = rotate(roaxispointx,0,0,0,objectunitrot)
    return z

print(get_rot_point(1,radians(-90)))




def radiget(vo,vi,wid):
    top = vo + vi
    bot = vo - vi
    side = wid/2
    t1 = top/bot
    ret = t1 * side
    return ret

##def anguget(vo,vi,rad):
##    t1 = vo + vi
##    b1 = 2*rad
##    return t1/b1

def anguget(vo,vi,l,wid):
    mid = vo/vi
    par = mid - l
    top = vi * par
    return top/wid

def rot_with_offset(x1,y1,cd,offset,rad):
    rotate(0,0,0+offset,0)

def get_point(x1,y1,vo,vi,leng,wid):
    rad = radiget(vo,vi,wid)
    angvelo = anguget(vo,vi,leng,wid)
    print(degrees(angvelo))
    x2 = x1 + rad
    y2 = y1
    print(rotate(x1,y1,x2,y2,angvelo))


##class Bot():
##    def __init__(self,posx,posy,radrot):
##        self.x = posx
##        self.y = posy
##        self.radrot = radrot
##    def move(t1speed,t2speed,time):
##        if t1speed == t2speed:
            

#we should be able to just add
