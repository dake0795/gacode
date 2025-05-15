from pygacode import expro
import sys
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text',usetex=True)
rc('font',size=18)

# Number of theta-points for plotting
narc = 300

surf  = sys.argv[1]
n     = int(sys.argv[2])
ftype = sys.argv[3]

# Read profiles
expro.expro_read('input.gacode',0)
nexp = int(expro.expro_n_exp)

print('nexp = {:d}'.format(nexp))

fig = plt.figure(figsize=(8,12))
ax = fig.add_subplot(111,aspect='equal')
ax.set_xlabel(r'$R$')
ax.set_ylabel(r'$Z$')

t = 2*np.pi*np.linspace(0,1,narc)

if n > 0:
   rlist = np.arange(0,nexp,n)
else:
   rlist = np.arange(abs(n),abs(n)+1)

# HAM geometry flux-surfaces
for i in rlist:
   rmaj = expro.expro_rmaj[i]
   zmaj = expro.expro_zmag[i]
   r    = expro.expro_rmin[i]
   k    = expro.expro_kappa[i]
   s1   = np.arcsin(expro.expro_delta[i])
   s2   = -expro.expro_zeta[i]
   s3   = expro.expro_shape_sin3[i]
   c0   = expro.expro_shape_cos0[i]
   c1   = expro.expro_shape_cos1[i]
   c2   = expro.expro_shape_cos2[i]
   c3   = expro.expro_shape_cos3[i]

   x = rmaj+r*np.cos(t
                     +c0
                     +s1*np.sin(t)  +c1*np.cos(t)
                     +s2*np.sin(2*t)+c2*np.cos(2*t)
                     +s3*np.sin(3*t)+c3*np.cos(3*t))
   y = zmaj+k*r*np.sin(t)

   if i == rlist[0]:
      ax.plot(x,y,'-k',linewidth=1,label=r'$\mathrm{HAM~3}$')
   else:
      ax.plot(x,y,'-k',linewidth=1)


ax.legend()
plt.tight_layout()

if ftype == 'screen':
   plt.show()
else:
   outfile = ftype
   plt.savefig(outfile)
