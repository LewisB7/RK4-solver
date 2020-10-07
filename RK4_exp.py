from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
   
"""Calculate an analytical exponential profile from z=0 to z=top_h. Given the
value of this profile at z=0 and z=top_h as BCs and using the same parameter 
values as in the analytical solution, calculate a numerical solution using RK4."""
 
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]    
    
plt.close('all')    

#maximum iterations for shooting method
maxIter = 10000
#tolerance for shooting method
tol = 0.00001
#von karman const
kappa = 0.325
#packing density
lam = 0.25
#Volume fraction
beta=0.25
#canopy top
H = 1.0 
#solver top height
top_h = 1*H
#surface roughness
z0 = 0.082*H # 0.005*H
#zero-plane displacement
#d = (1+(lam-1)*4**(-lam))*H
d = 0.6985*H
#constant l_c
l_c_const = kappa/(1/(H-d)-1/H)
#Drag coefficient
CD=1.2 # Chosen to replicate Branford 0deg cubes
#constant L_c
L_c_const = 2*H*(1-beta)/(CD*lam)
#Lc/h
Lch = L_c_const/H
#number of spatial steps in vertical direction
nmax = 75
#step size
dz = top_h/nmax
#analytical and numerical grid
z = np.linspace(0,nmax*dz,nmax+1)

### Analytical

# exponential velocity profile components
ust = 1 # DNS 0deg u* 
Uh = ust/kappa*np.log((H-d)/z0)
kv = (2*(kappa*(H-d))**2*L_c_const)**(-1/3)
#computation
anal = np.zeros(len(z))
for i in np.arange(0,len(z)):
    anal[i] = Uh*np.exp(kv*(z[i]-H))

### Numerical functions

def dif(x,b):
    val = x-b
    return val

def f(z, u1, u2):
    val = u2
    return val
    
def g(z, u1, u2, kappa, l_c_const, d, L_c_const, h):
    l_ = l(z, h, kappa, l_c_const, d)
    dl_ = dl(z, h, kappa, l_c_const)
    L_c_ = L_c(z, h, L_c_const)
    val = (-l_ * dl_ * u2**2 + 0.5 * u1**2/L_c_) / (l_**2 * u2)
    return val
 
def dl(z, h, kappa, l_c_const):
    val = 0
    return val

def l(z, h, kappa, l_c_const, d):
    val = kappa * (h - d)
    return val

def L_c(z, h, L_c_const):
    val = L_c_const
    return val

### Numerical initial conditions
    
#arrays for u and u'
u1 = np.zeros(len(z))
u2 = np.zeros(len(z))

#value which we are iterating for
targetval = anal[-1]

#index of velocity shooting for
idx = nmax # nmax if top of domain but could be different if e.g. canopy top

#Numerical solution value at surface
u1[0] = Uh*np.exp(kv*(z[0]-H))

# initial guesses for u2(0) interval that must bracket the real value
a = 0.01 #0.000001 # any tiny but non-zero number (Lewis: the solution seems to be sensitive to this)
b = 100.0 # any sufficiently large number

### Numerical solution

i = 0
while i <= maxIter:
    i = i+1 #shooting method so iterate   

    #RK method
    u2[0] = np.copy(a)
    for j in np.arange(idx):
        k1 = dz * f(z[j], u1[j], u2[j])
        l1 = dz * g(z[j], u1[j], u2[j], kappa, l_c_const, d, L_c_const, H)
        k2 = dz * f(z[j] + 0.5*dz, u1[j] + 0.5*k1, u2[j] + 0.5*l1)
        l2 = dz * g(z[j] + 0.5*dz, u1[j] + 0.5*k1, u2[j] + 0.5*l1, kappa, l_c_const, d, L_c_const, H)
        k3 = dz * f(z[j] + 0.5*dz, u1[j] + 0.5*k2, u2[j] + 0.5*l2)
        l3 = dz * g(z[j] + 0.5*dz, u1[j] + 0.5*k2, u2[j] + 0.5*l2, kappa, l_c_const, d, L_c_const, H)
        k4 = dz * f(z[j] + dz, u1[j] + k3, u2[j] + l3)
        l4 = dz * g(z[j] + dz, u1[j] + k3, u2[j] + l3, kappa, l_c_const, d, L_c_const, H)
        
        u1[j+1] = u1[j] + (k1 + 2*k2 + 2*k3 + k4)/6
        u2[j+1] = u2[j] + (l1 + 2*l2 + 2*l3 + l4)/6
    ua = np.copy(u1[idx])
       
    c = (a+b)/2
    u2[0] = np.copy(c)
    for j in np.arange(idx):
        k1 = dz * f(z[j], u1[j], u2[j])
        l1 = dz * g(z[j], u1[j], u2[j], kappa, l_c_const, d, L_c_const, H)
        k2 = dz * f(z[j] + 0.5*dz, u1[j] + 0.5*k1, u2[j] + 0.5*l1)
        l2 = dz * g(z[j] + 0.5*dz, u1[j] + 0.5*k1, u2[j] + 0.5*l1, kappa, l_c_const, d, L_c_const, H)
        k3 = dz * f(z[j] + 0.5*dz, u1[j] + 0.5*k2, u2[j] + 0.5*l2)
        l3 = dz * g(z[j] + 0.5*dz, u1[j] + 0.5*k2, u2[j] + 0.5*l2, kappa, l_c_const, d, L_c_const, H)
        k4 = dz * f(z[j] + dz, u1[j] + k3, u2[j] + l3)
        l4 = dz * g(z[j] + dz, u1[j] + k3, u2[j] + l3, kappa, l_c_const, d, L_c_const, H)
        
        u1[j+1] = u1[j] + (k1 + 2*k2 + 2*k3 + k4)/6
        u2[j+1] = u2[j] + (l1 + 2*l2 + 2*l3 + l4)/6
    uc = np.copy(u1[idx])
    
    #check if value at top of profile is within tolerance
    #if it is, exit else continue    
    if (dif(uc,targetval) == 0 or (b - a)/2 < tol): # solution found
        #print('converged after')
        #print(i)
        grad = np.copy(u2[0])
        break
    if (np.sign(dif(uc,targetval)) == np.sign(dif(ua,targetval))):
        a = np.copy(c) 
    else:
        b = np.copy(c) # new interval
    if i == maxIter:
            print('maxIter reached in shooting method')

### Plot
plt.figure()
plt.plot(u1,z/H,marker="",linestyle="-",label="numerical")
plt.plot(anal,z/H,marker="x",linestyle="none",label="analytical")
#plt.xlim(0,15)
plt.xlabel(r"$u$")
plt.ylabel(r"$z/H$")
plt.legend(loc=2)
