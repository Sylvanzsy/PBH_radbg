import numpy as np
#from scipy.integrate import quad
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.integrate import quad, solve_ivp
from scipy.optimize import root
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from colossus.cosmology import cosmology
#plt.style.use('test2')
#plt.style.use('tableau-colorblind10')

# constants
GRA = 6.672e-8
#GRV = 1.0
BOL = 1.3806e-16
#BOL = 1.0
PROTON = 1.6726e-24
ELECTRON = 9.10938356e-28
HBAR = 1.05457266e-27
PLANCK = HBAR*2*np.pi
CHARGE = 4.80320451e-10
SPEEDOFLIGHT = 2.99792458e+10
eV = 1.60218e-12
YR = 3600*24*365
PC = 3.085678e18
KPC = PC*1e3
MPC = PC*1e6
SIGMATH = 8*np.pi*(CHARGE**2/ELECTRON/SPEEDOFLIGHT**2)**2/3
STEFAN = 2*np.pi**5*BOL**4/(15*PLANCK**3*SPEEDOFLIGHT**2)
RC = 4*STEFAN/SPEEDOFLIGHT

AU = 1.495978707e13
Lsun = 3.828e33
Rsun = 6.96342e10

H00 = 1e7/MPC

# choose a cosmology model
cosname = 'planck18'
cosmo = cosmology.setCosmology(cosname)
print(cosmo)
h = cosmo.H0/100
#print('log10(h) = {:.3f}'.format(np.log10(h)))
H0 = H00
rho0 = cosmo.Om0*H0**2*3/8/np.pi/GRA
# Units
UL = 3.085678e24/h
Msun = 1.989e33
UM = 1.989e33/h
UV = 1.0e5
# The internal units have h built-in because 
# this is what colossus uses for the power spectrum. 

Om = cosmo.Om0
Ob = cosmo.Ob0
Or = cosmo.Or0
aeq = Or/Om # scale factor at matter radiation equity
lo = np.array([Om, 1-Om, 1./24000./h**2])
lw = np.array([0., -1., 1./3.])

# cosmology
def Hubble_a(a, lo=lo, lw=lw):
	assert len(lw)==len(lo)
	H0 = H00
	out = 0
	for o, w in zip(lo, lw):
		out += o * a**(-3*(1+w))
	return H0*out**(0.5)

def dt_da(a, lo, lw, h=h):
	return 1/a/Hubble_a(a, lo, lw)/h

def age_a(a, lo=lo, lw=lw, h=h):
	def dt_da(a):
		return 1./(a*h*Hubble_a(a, lo, lw))
	I = quad(dt_da, 0., a, epsrel = 1e-4)
	return I[0]

def horizon(a, lo=lo, lw=lw, a0=1e-10):
	def integrand(loga):
		return SPEEDOFLIGHT/np.exp(loga)/Hubble_a(np.exp(loga),lo,lw)
	I = quad(integrand, np.log(a0), np.log(a), epsrel=1e-8)
	return I[0]

# growth factor in linear perturbation theory (non-normalized)
def delta_plus(z, Om = Om, Ol = 1-Om):
	lo = np.array([Om, Ol, 1./24000./h**2])
	lw = np.array([0., -1., 1./3.])
	a = 1/(1+z)
	def integrand(a):
		return 1./(a*Hubble_a(a, lo, lw))**3
	I = quad(integrand, 0., a, epsrel = 1e-4)
	return I[0]*Hubble_a(a, lo, lw)

lz0 = np.hstack([[0],10**np.linspace(-2, 4, 1000)])
lt0 = [np.log10(age_a(1/(x+1))/1e9/YR) for x in lz0]
ZT = interp1d(lt0, lz0)

# matter density
def rhom(a, Om = Om, h = h):
	H0 = h*H00
	rho0 = Om*H0**2*3/8/np.pi/GRA
	return rho0/a**3
	
# halo virial radius
def RV(m = 1e10, z = 10.0, delta = 200):
	M = m*UM
	return (M/(rhom(1/(1+z))*delta)*3/4/np.pi)**(1/3)

# halo virial temperature
def Tvir(m = 1e10, z = 10.0, delta = 200):
	M = m*UM
	Rvir = (M/(rhom(1/(1+z))*delta)*3/4/np.pi)**(1/3)
	return GRA*M*mmw(xeH)*PROTON/Rvir/(3*BOL)

# halo mass corresponding to a given scale k
def Mk(k):
	R = 1/k*UL
	M = R**3*4*np.pi/3*rho0/UM
	return M

# normalized growth linear factor from numerical integration
def Dgrow_(z, Om = Om, Ol = 1-Om):
	return delta_plus(z, Om, Ol)/delta_plus(0, Om, Ol)

# normalized growth linear factor from the fitting formula in the book 
# Galaxy Formation and Evolution by Mo et al, 2010 (equ 4.76) 
def Dgrow(z, Om = Om, Ol = 1-Om):
	Omz = Om*(1+z)**3/(Om*(1+z)**3 + Ol)
	Olz = 1-Omz
	gz = 2.5*Omz*(Omz**(4/7)-Olz+(1+Omz/2)*(1+Olz/70))**-1
	gz0 = 2.5*Om*(Om**(4/7)-Ol+(1+Om/2)*(1+Ol/70))**-1
	return gz/(1+z)/gz0

# critical overdensity for halo formation (~1.69)
from colossus.lss import peaks
deltac0 = peaks.collapseOverdensity(corrections = True, z = 0)
def deltac_z(z, Om = Om, Ol = 1-Om, mode=1):
	if mode==0:
		deltac = deltac0
	else:
		deltac = peaks.collapseOverdensity(corrections = True, z = z)
	Dz = Dgrow(z, Om, Ol)
	return deltac/Dz

# PS function
# mode=1: PS formalism that includes corrections for ellipsoidal dynamics
# mode=0: original PS formalism
def fPS(nu0, mode = 0, A = 0.322, q = 0.3, fac = 0.84, norm=1):
	nu = nu0*fac
	if mode>0:
		return (2/np.pi)**0.5*nu*np.exp(-nu**2/2) * A * (1+1/nu**(2*q)) /norm
	else:
		return (2/np.pi)**0.5*nu0*np.exp(-nu0**2/2) /norm
	
# kernel function and normalization
# top-hat sphere in real space
gf1 = 4*np.pi/3
def wf1(k, R):
	return 3*(np.sin(k*R)-k*R*np.cos(k*R))/(k*R)**3

# cutoff in k-space
gf2 = 6*np.pi**2
def wf2(k, R):
	return 1*(k*R<=1) + 0*(k*R>1)

# Gaussian
gf3 = (2*np.pi)**1.5
def wf3(k, R):
	return np.exp(-(k*R)**2/2)
	
# variance of density field for a given nass scale M
def sigma2_M(M, ps, wf = wf1, gf = gf1, Om = Om, norm=1.0):
	rho0 = Om*H0**2*3/8/np.pi/GRA
	R = (M*UM/rho0/gf)**(1/3)/UL
	x, y = ps[0], ps[1]
	#sel = lk<kc
	#x, y = lk[sel], lPk[sel]
	return np.trapz(y*x**2*wf(x, R)**2, x)/2/np.pi**2/norm

# correction function
corr = lambda z: 1

from scipy.misc import derivative

# halo mass function dn/dM, # density in units of (Mpc/h)^-3, where M is in units of solar mass in comoving units
# given the power spectrum, kernel and cosmological parameters 
# return the halo mass function at redshift z
# ps = [lk, lPk]: power spectrum
# lm: mass/scale grid over which the integration is performed
def halomassfunc(lm, z, ps, wf=wf1, gf=gf1, Om = Om, Ol = 1-Om, dx = 5e-2, corr = corr, mode=0):
	dc = deltac_z(z, Om, Ol)
	lognu = np.log([dc/sigma2_M(m, ps, wf, gf)**0.5 for m in lm])
	logm = np.log(lm)
	func = interpolate.interp1d(logm, lognu, fill_value='extrapolate')
	rho0 = Om*H0**2*3/8/np.pi/GRA
	def hmf(M):
		nu = dc/sigma2_M(M, ps, wf, gf)**0.5
		logM = np.log(M)
		return rho0/M**2*fPS(nu, mode)*np.abs(derivative(func, logM, dx))*UL**3/UM / corr(z)
	return hmf

# power spectrum regulated by PBHs
# ps: input CDM power spectrum
# mpbh: PBH mass, fpbh: fraction of PBHs in dark matter
# aeq: scale factor at matter radiation equity
# mode=0: truncate the isocurvature mode at large scales, =1: no truncation
# iso=0: consider mode mixing, >0: only have pure isocurvature mode from PBHs
# seed=0: only consider the Poisson effect, >0: include the seed effect (tentative)
# dmax: upper limit of overdensity (if != 0), =0: no upper limit
# cut=0: add the isocurvature term at all scales, >0: truncate the PBH perturbation at scales smaller than mpbh
# mfac: normalization factor
def PBH(ps, mpbh, fpbh, aeq, h, mode=1, mfac=1.0, alpha=0, Ob=Ob, Om=Om, iso=1, seed=0, dmax=178, out=0, cut=1):
	npbh = fpbh*rhom(1, Om, h)*(Om-Ob)/Om/mpbh/Msun
	if out>0:
		print('PBH number density: {:.2e} Mpc^-3'.format(npbh*(MPC)**3))
	ks = (2*np.pi**2*npbh)**(1.0/3.0)*MPC/h*mfac #4*1e3*(fpbh*(20/h/mpbh))**(1.0/3.0)
	kcut = 3*ks
	#print('PBH scale: {:.2e} h/Mpc'.format(ks))
	lk, lPk = ps[0], ps[1]
	kH = MPC/horizon(aeq)
	#print('Horizon size at aeq: {:.2e} kpc/h'.format(1/kH))
	selH = lk<kH
	aeq_ = np.ones(len(lk))*aeq
	aeq_[selH] = aeq * (kH/lk[selH])**2
	kts = max((2*np.pi**2*npbh/fpbh*aeq)**(1.0/3.0)*MPC/h, ks)
	ksub = (2*np.pi**2*npbh/fpbh)**(1.0/3.0)*MPC/h*mfac
	#gam = Om*h*np.exp(-Ob*(1+np.sqrt(2*h)/Om))
	#q = lk*1e3/gam
	#T_k = (1+(15*q+(0.9*q)**1.5+(5.6*q)**2)**1.24)**(-1.24)
	grow0 = Dgrow(0)/Dgrow(1.0/aeq_-1)
	fc = (Om-Ob)/Om
	a_ = 0.25*((1+24*fc)**0.5-1)
	grow = ( (1.0+1.5*fc/a_/aeq_)**a_ - 1.0 ) #* T_k *(lk/kH)**2
	#print('Ratio of growth factor at z=0:', np.max(grow)/np.max(grow0), np.max(grow), np.max(grow0))
	#print(grow, grow0)
	if mode==0:
		sel = lk>ks
	else:
		sel = lk>0 #lPk < pkpbh
	delta2 = lk[sel]**3/ks**3 * (grow[sel])**2 * fpbh**2
	if seed>0:
		delta2 = delta2*(lk[sel]<kts) + delta2*(lk[sel]>=kts)*(lk[sel]/kts)**3
	if dmax>0:
		delta2[delta2>dmax**2] = dmax**2
	pkpbh =  2*np.pi**2 * delta2/lk[sel]**3
	lPk_ = np.copy(lPk)
	kcrit = ks #kcut #(ks*kcut)**0.5
	delt = (lk[sel]/kcrit)**3
	sel0 = lk[sel]>kcut
	delt[sel0] *= 0 #(lk[sel][sel0]/(kcut))**-3 #fpbh
	#delt[delt<1] = 1
	#delt[delt>mfac] = mfac
	#if seed>0:
	#	pkpbh = pkpbh*mfac*(lk[sel]<ks) + pkpbh*mfac*(lk[sel]>=ks)*(lk[sel]/ks)**3
	if iso>0:
		piso = pkpbh*mfac #+ lPk[sel] * delt * grow**2 * fpbh / grow0
	else:
		piso = pkpbh*mfac + lPk[sel] * delt * grow**2 * fpbh / grow0
	#piso[lk>ks*fpbh*1e4] = 0
	if cut>0:
		piso[lk[sel]>ksub] = 0
	lPk_[sel] += piso #* 1e3
	return [lk, lPk_]
