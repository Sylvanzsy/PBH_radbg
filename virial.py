import matplotlib.pyplot as plt
from cosmology import *
from txt import *
from radcool import cool
import matplotlib
#import mpl_toolkits.mplot3d
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import sys
import os
#plt.style.use('test2')
plt.style.use('tableau-colorblind10')
from injection import heat, heat_subgrid, mdot_bondi

Mdown = lambda z: 1.54e5*((1+z)/31)**-2.074

"""
xh20 = 0.0005722534911268776
lfpbh = [1e-4, 1e-3, 1e-2, 1e-1]
lxh2 = np.array([0.0005540624323109008, 
0.000902877589285586, 
0.0009975958404269216, 
0.0017898655625888066])

lx = np.geomspace(1e-4, 1e-1, 100)
alp = 0.15
plt.loglog(lfpbh, lxh2/xh20, label='Simulation')
plt.loglog(lx, 3*(lx/lfpbh[-1])**alp, '--', label='Fit')
plt.xlabel(r'$f_{\rm PBH}$')
plt.ylabel(r'$x_{\rm H_{2}, PBH}/x_{\rm H_{2}, CDM}$')
plt.tight_layout()
plt.savefig('xh2rat_fpbh.pdf')
plt.close()
#plt.show()
#exit()
"""

def fh2_pbh(fpbh, alp=0.15, fref=0.1, rref=3):
	return max(rref*(fpbh/fref)**alp, 1.0)

def fh2_max(T):
	return 3.5e-4*(T/1e3)**1.52
	
def trat_cdm(m, z, delt=6.0, Ob=0.04930, Om=0.3153, h=0.6736, X=0.76, gamma=5.0/3, T0=2.73):
	T = Tvir(m, z, 200, Om=Om, h=h)
	n = rhom(1/(1+z), Om, h)*delt*200*Ob/Om*X/PROTON
	ny = np.zeros(17)
	ny[0] = n
	ny[3] = n*fh2_max(T)
	lamb = cool(T, n/0.93, ny, 0, z, gamma, X, T0)
	tdyn = tff(z, delt*200, Om, h)
	tcool = -T/lamb
	return tcool/tdyn
	
def lum_pbh(m, mh, z, delt=6.0, Ob=0.04930, Om=0.3153, h=0.6736, X=0.76, er=0.057):
	v = Vcir(mh, z, Om=Om, h=h)/1e5
	n = rhom(1/(1+z), Om, h)*delt*200*Ob/Om*X/PROTON
	lum = heat_subgrid(m, n, v, er=er) * mdot_bondi(m, n, v, er) *6.7e-16*0.1/er*m*Msun*SPEEDOFLIGHT**2
	return lum

def trat_pbh(m, fpbh, mh, z, delt=6.0, Ob=0.04930, Om=0.3153, h=0.6736, X=0.76, gamma=5.0/3, T0=2.73, er=0.057):
	T = Tvir(mh, z, 200, Om=Om, h=h)
	n = rhom(1/(1+z), Om, h)*delt*200*Ob/Om*X/PROTON
	ny = np.zeros(17)
	ny[0] = n
	ny[3] = n*fh2_max(T)*fh2_pbh(fpbh)
	lamb = cool(T, n/0.93, ny, 0, z, gamma, X, T0)
	tdyn = tff(z, delt*200, Om, h)
	npbh = mh*(Om-Ob)/Om*fpbh/m
	Nb = mh*Ob*Msun/PROTON/mmw(X=X)
	gamm = lum_pbh(m, mh, z, delt, Ob, Om, h, X, er) * npbh * (gamma-1.0)/BOL/Nb
	tcool = -T/(gamm+lamb)
	if tcool<0:
		return -tcool/tdyn
	else:
		return tcool/tdyn

#"""
mpbh, fpbh = 33, 1e-3
lab = r'PBH, $m_{\rm PBH}='+str(mpbh)+r'\ \rm M_{\odot}$, $f_{\rm PBH}='+str(fpbh)+'$'

delt = 6.5
z = 20

m1, m2, nm = 1e5, 1e6, 1000
lm = np.geomspace(m1, m2, nm)
lrat = np.array([trat_cdm(x, z, delt) for x in lm])
lrat1 = np.array([trat_pbh(mpbh, fpbh, x, z, delt) for x in lm])
#print(lrat)
y1, y2 = 1e-2, 1e2
mref = Mdown(z)
plt.figure()
plt.loglog(lm, lrat, label=r'CDM, $z={}$, $\Delta={:.0f}$'.format(z, delt*200))
plt.loglog(lm, lrat1, '-.', label=lab)
plt.plot([m1, m2], [1]*2, 'r:')
plt.plot([mref]*2, [y1, y2], 'k--', label='TS09')
plt.xlabel(r'$M_{\rm h}\ [\rm M_{\odot}]$')
plt.ylabel(r'$t_{\rm cool}/t_{\rm dyn}$')
plt.legend(loc=2)
plt.xlim(m1, m2)
plt.ylim(y1, y2)
plt.tight_layout()
plt.savefig('rat_m.pdf')
plt.close()
#"""

m1, m2 = 3e4, 3e6

def mth_cdm(z, m1=m1, m2=m2, nm=100, delt=5, Ob=0.04930, Om=0.3153, h=0.6736, X=0.76, gamma=5.0/3, T0=2.73):
	m1, m2 = Mdown(z)*0.1, Mdown(z)*10
	lm = np.geomspace(m1, m2, nm+1)
	lrat = np.array([trat_cdm(x, z, delt, Ob, Om, h, X, gamma, T0) for x in lm])
	func = interp1d(np.log10(lrat[lrat>0]), np.log10(lm[lrat>0]))
	return 10**func(0)
	
def mth_pbh(mpbh, fpbh, z, m1=m1, m2=m2, nm=100, delt=5, Ob=0.04930, Om=0.3153, h=0.6736, X=0.76, gamma=5.0/3, T0=2.73, er=0.057):
	m1, m2 = Mdown(z)*0.1, Mdown(z)*10
	lm = np.geomspace(m1, m2, nm+1)
	lrat = np.array([trat_pbh(mpbh, fpbh, x, z, delt, Ob, Om, h, X, gamma, T0, er) for x in lm])
	ind = np.argmax(lrat)
	if lrat[ind]<1:
		return lm[ind]
	else:
		func = interp1d(np.log10(lrat[ind:]), np.log10(lm[ind:]))
	return 10**func(0)
	
lls = ['-', '--', '-.', ':']
llc = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
	
if __name__=="__main__":
	delt = 6.5
	nm = 1000
	mpbh, fpbh = 33, 1e-3
	lf = [1e-4, 1e-3, 1e-2, 1e-1]
	
	out = []
	
	z1, z2, nz = 5, 40, 70
	lz = np.linspace(z1, z2, nz+1)
	out.append(lz)
	lmth = np.array([mth_cdm(z, delt=delt, nm=nm) for z in lz])
	out.append(lmth)
	lmref = Mdown(lz)
	y1, y2 = 1e5, 1e7
	plt.figure()
	plt.plot(lz, lmref, 'k', ls=(0,(10,5)), label=r'TS09', zorder=3)
	plt.plot(lz, lmth, color='gray', label=r'$f_{\rm PBH}=0$, '+'$\Delta={:.0f}$'.format(delt*200), lw=4.5, alpha=0.5)
	for i in range(len(lf)):
		fpbh = lf[i]
		lab = r'$f_{\rm PBH}='+str(fpbh)+'$'
		lmth1 = np.array([mth_pbh(mpbh, fpbh, z, delt=delt, nm=nm) for z in lz])
		sel = lmth1 * 0.04864/0.3089*fpbh/mpbh > 0
		plt.plot(lz[sel], lmth1[sel], color=llc[i], ls=lls[i], label=lab)
		out.append(lmth1)
	plt.yscale('log')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$M_{\rm mol}\ [\rm M_{\odot}]$')
	plt.legend(ncol=2, loc=1)
	plt.xlim(z1, z2)
	plt.ylim(y1, y2)
	plt.tight_layout()
	plt.savefig('mth_z.pdf')
	plt.close()
	totxt('mth_z.txt', out)
	

