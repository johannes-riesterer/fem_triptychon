#----------------------------------------------------------------
#load modules
#----------------------------------------------------------------
import numpy as np

from scipy.spatial import Delaunay
from scipy import sparse
import scipy.sparse.linalg as spla
from numpy import linalg

import matplotlib.pyplot as plt 
import math



#----------------------------------------------------------------
#mesh generation
#-----------------------------------------------------------------

points = []

#number of points per side
bound = 60
for i in range(bound):
    for j in range(bound):
        points.append([2*float(i)/(bound-1),2*float(j)/(bound-1)]) 
mesh = Delaunay(points)


points_np = np.array(points)


#---------------------------------------------------------------
#assembling 
#---------------------------------------------------------------

#find boundary nodes and store their indices in list boundary
boundary = []
def addNode(node):
    found = False
    for n in boundary:
        if n==node:
            found = True
    if found == False:
        boundary.append(node)

for i in range(len(mesh.simplices)):
    for j in range(3):
        if mesh.neighbors[i][j] == -1:
            addNode(mesh.simplices[i][(j+1)%3])
            addNode(mesh.simplices[i][(j+2)%3])


#store values for boundary elements


boundaryValues = {}
for i in boundary:
    boundaryValues[i] =  0.0
#set values for f at nodes
f = []
for i in range(len(mesh.points)):
    f.append(1.0)
               
    
#assemble local quantities

#lists for local quantities stored in same order as mesh.simplices and simplex.points (counterclockwise)
M_local = []
b_local = []

def computeVolume(simplex):
    p0 = mesh.points[simplex[0]]
    p1 = mesh.points[simplex[1]]
    p2 = mesh.points[simplex[2]]
    
    vm = np.mat([ [p0[0], p0[1], 1], [p1[0], p1[1], 1], [p2[0], p2[1], 1]])
    vol = 0.5*abs(linalg.det(vm))
    return vol

def computeJacobian(simplex):
    p0 = mesh.points[simplex[0]]
    p1 = mesh.points[simplex[1]]
    p2 = mesh.points[simplex[2]]
    jacobian = np.mat([ [p1[0] - p0[0], p2[0] - p0[0]], [p1[1] - p0[1], p2[1] - p0[1]]]); 
    return jacobian


def assemble_local(simplex):
    
    vol = computeVolume(simplex)
    J = computeJacobian(simplex)
    #linear shape functions!
    C = (J.I) * (J.I.T)
    d = abs(linalg.det(J))
    phi_1 = np.mat([[-1], [-1]])
    phi_2 = np.mat([[1], [0]])
    phi_3 = np.mat([[0], [1]])
    
    
    a_11 = float((C*phi_1).T * phi_1)
    a_12 = float((C*phi_1).T * phi_2)
    a_13 = float((C*phi_1).T * phi_3)
    
    a_21 = float((C*phi_2).T * phi_1)
    a_22 = float((C*phi_2).T * phi_2)
    a_23 = float((C*phi_2).T * phi_3)
    
    a_31 = float((C*phi_3).T * phi_1)
    a_32 = float((C*phi_3).T * phi_2)
    a_33 = float((C*phi_3).T * phi_3)
    
    
    m = ( d/2.0)* np.mat([[a_11, a_12, a_13], [a_21, a_22, a_23], [a_31, a_32, a_33] ])
    M_local.append(m)
    
    #assemble right hand side via linear interpolation of f and very simple quadrature rule
    s_0 = f[simplex[0]]
    s_1 = f[simplex[1]]
    s_2 = f[simplex[2]]
    

    f_l = d*0.166
    b_l = np.mat([[f_l], [f_l], [f_l]])
    b_local.append(b_l)
    

    
    
    
#assemble global quantities

gdof = len(mesh.points) 
M = sparse.lil_matrix((gdof, gdof))
b = [0]*(gdof)

def assemble():
    for simplex in mesh.simplices:
        assemble_local(simplex)
        
    for i in range(len(mesh.simplices)):
        simplex = mesh.simplices[i]
        i0 = simplex[0]
        i1 = simplex[1]
        i2 = simplex[2]
        
        M[i0,i0] += M_local[i][0,0]
        M[i0,i1] += M_local[i][0,1]
        M[i0,i2] += M_local[i][0,2]
        
        M[i1,i0] += M_local[i][1,0]
        M[i1,i1] += M_local[i][1,1]
        M[i1,i2] += M_local[i][1,2]
        
        M[i2,i0] += M_local[i][2,0]
        M[i2,i1] += M_local[i][2,1]
        M[i2,i2] += M_local[i][2,2]
        
        #b[i0]
        b[i0] += float(b_local[i][0])
        b[i1] += float(b_local[i][1])
        b[i2] += float(b_local[i][2])
        
assemble()   

#boundary conditions
for i in boundary:
    #set column and row of boundary element to zero
    for j in range(gdof):
        M[j,i] = 0
        M[i,j] = 0
    M[i,i] = 1
    b[i] = boundaryValues[i]


#-------------------------------------------------------------------------------
#SOLVE
#-------------------------------------------------------------------------------
xx = spla.gmres(M, b, tol=1e-7)
x = xx[0]
err = xx[1]

print x
print err


#visualise the mesh
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(1)
plt.subplot(2,2,1)
plt.title(r'Mesh $\Omega$')
plt.triplot(points_np[:,0], points_np[:,1], mesh.simplices.copy())
plt.plot(points_np[:,0], points_np[:,1], 'o')
plt.subplot(2,2,3)
#plt.figure(2)
plt.title(r'Sparse matrix pattern')
plt.spy(M)
plt.subplot(2,2,2)
#plt.figure(2)
plt.title(r"Solution $u$ on $\Omega$ of $- \Delta u  = 1, \; u = 0$ on $\partial \Omega$")
plt.tripcolor(points_np[:,0], points_np[:,1], mesh.simplices.copy(), x, edgecolors='k')
plt.colorbar()
plt.show()



