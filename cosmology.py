import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Constants
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

# Primordial abundances
Hfrac = 0.75
XeH = 0.0
XeHe = 0.0
# mean molecular weight, xeH/xeHe: hydrogen/helium ionization fraction
def mmw(xeH = XeH, xeHe = XeHe, X = Hfrac):
	xh = 4*X/(1+3*X)
	return 4.0/(1.0+3*X)/(xh*(1+xeH)+(1-xh)*(1+xeHe))

# Units
UL = 3.085678e21
#UL = 1.0
UM = 1.989e43
Msun = 1.989e33
#UM = 1.0
UV = 1.0e5
#UV = 1.0

UT = UL/UV
UD = UM/UL**3
UP = UM/UL/UT**2
UE = UM*UL**2/UT**2

G = GRA*UM*UT**2/UL**3

def midbin(l):
	return (l[1:]+l[:-1])*0.5

def sumy(l):
	return np.array([np.min(l), np.max(l), np.average(l), np.median(l)])

# black-body spectrum
def BB_spectrum(nu0, T):
  numax = 50.*BOL*T/PLANCK
  nu = nu0 * (nu0<numax) + numax * (nu0>=numax)
  return 2*PLANCK/SPEEDOFLIGHT**2 * nu**3/(np.exp(PLANCK*nu/(BOL*T))-1) * (nu<=numax)

# Cosmology
def H(a, Om = 0.3153, h = 0.6736, OR = 9.54e-5):
	H0 = h*100*UV/UL/1e3
	H = H0*(Om/a**3+(1-Om-OR)+OR/a**4)**0.5
	return H

# co-moving distance
def DZ(z, Om = 0.3153, h = 0.6736, OR = 9.54e-5):
	def integrand(a):
		return SPEEDOFLIGHT/(a**2)/H(a, Om, h, OR)
	I = quad(integrand, 1/(1+z), 1, epsrel = 1e-8)
	return I[0]

def dt_da(a, Om = 0.3153, h = 0.6736, OR = 9.54e-5):
	return 1/a/H(a, Om, h, OR)

# cosmic age
def TZ0(z, Om = 0.3153, h = 0.6736, OR = 9.54e-5):
	I = quad(dt_da, 0, 1/(1+z), args = (Om, h, OR), epsrel = 1e-8)
	return I[0]

#lz0 = np.linspace(0,3300,3301)
lz0 = np.hstack([[0],10**np.linspace(-2, 4, 1000)])
lt0 = np.array([np.log10(TZ0(x)/1e9/YR) for x in lz0])
ld0 = [DZ(x)/UL/1e3 for x in lz0]

# interpolation functions for the relation between t, d and z
TZint = interp1d(np.log10(1/(1+lz0)), lt0)
TZ = lambda z: 10**TZint(np.log10(1/(1+z)))*1e9*YR
ZTint = interp1d(lt0, lz0)
ZT = lambda x: ZTint(np.log10(x/1e9/YR))
ZD = interp1d(ld0, lz0)

# matter density
def rhom(a, Om = 0.3153, h = 0.6736):
	H0 = h*100*UV/UL/1e3
	rho0 = Om*H0**2*3/8/np.pi/GRA
	return rho0/a**3

# gas temperature [K] in the standard CDM cosmology
# fit from https://journals.aps.org/prd/abstract/10.1103/PhysRevD.82.083520
def T_b(z, a1=1./119, a2=1./115, T0=2.726):
	a = 1./(1+z)
	return T0/(a*(1+a/(a1*(1+(a2/a)**1.5))))

# I think this is another version of T_b...
def T_cosmic(z, alpha = -4, beta = 1.27, z0 = 189.6, zi = 1020, T0 = 2.726*1021):
	def integrand(logt):
		return alpha/3.0-(2+alpha)/3.0*(1-np.exp(-(ZT(logt)/z0)**beta))
	I = quad(integrand, np.log10(TZ(zi)/1e9/YR), np.log10(TZ(z)/1e9/YR), epsrel = 1e-6)[0]
	temp = T0*10**I
	return temp

# DM temperature, m: DM particle mass in eV
def T_dm(z, m = 1., T0=2.726):
	zc = m*1e9*eV / ((3./2)*BOL*T0) - 1
	Tc = T0*(1+zc)
	return T0*(1+z) * (z>zc) + Tc*((1+z)/(1+zc))**2 * (z<=zc)

# DM haloes, halo mass is always in Msun

# NFW halo concentration
# fit from Dutton & Macci�� 2014, for z = 0-5, logMvir = 10-15
def concentration_z(z, m, h=0.6774):
	a = 0.52 + (0.905-0.52)*np.exp(-0.617*z**1.21) # M200-c200
	# a = 0.537 + (1.025-0.537)*np.exp(-0.718*z**1.08) # Mvir-cvir
	b = -0.101 + 0.026*z
	# b = -0.097 + 0.025*z
	logc = a + b*np.log10(m*h/1e12)
	return 10**logc

# free-fall timescale for a halo of overdensity delta, at redshift z
def tff(z = 10.0, delta = 200, Om=0.3153, h=0.6736):
	return (3*np.pi/(32*GRA*delta*rhom(1/(1+z),Om,h)))**0.5

# virial radius
def RV1(m = 1e10, z = 10.0, delta = 200, Om=0.3153, h=0.6736):
	M = m*Msun
	return (M/(rhom(1/(1+z),Om,h)*delta)*3/4/np.pi)**(1/3)

# circular velovity
def Vcir(m = 1e10, z = 10.0, delta = 200, Om=0.3153, h=0.6736):
	M = m*Msun
	Rvir = (M/(rhom(1/(1+z),Om,h)*delta)*3/4/np.pi)**(1/3)
	return (GRA*M/Rvir)**0.5

# halo mass for a given circular velocity
def M_vcir(z, v, delta = 200, Om=0.3153, h=0.6736):
	M = v**3 / GRA**1.5 /(4*np.pi*rhom(1/(1+z),Om,h)*delta/3)**0.5
	return M/Msun

# energy loss rate in virialization
def Lvir(m = 1e10, z = 10.0, delta = 200, Om=0.3153, h=0.6736):
	M = m*Msun
	Rvir = (M/(rhom(1/(1+z),Om,h)*delta)*3/4/np.pi)**(1/3)
	return 3*GRA*M**2/Rvir/tff(z, delta,Om,h)/5

# virial temperature
def Tvir(m = 1e10, z = 10.0, delta = 200, xeH=0, Om=0.3153, h=0.6736):
	M = m*Msun
	Rvir = (M/(rhom(1/(1+z),Om,h)*delta)*3/4/np.pi)**(1/3)
	return GRA*M*mmw(xeH)*PROTON/Rvir/(2*BOL) # *3/5
	
# halo mass for a given virial temperature
def M_Tvir(T, z = 10.0, delta = 200, xe=0, Om=0.3153, h=0.6736):
	y = 2*T*BOL*(3/(4*np.pi*delta*rhom(1/(1+z),Om,h)))**(1/3)/GRA/(mmw(xe)*PROTON)
	return y**1.5/Msun

# Jeans mass
def Jeansm(T, rho, mu = 0.63, gamma=5./3):
	cs = (gamma*T*BOL/(mu*PROTON))**0.5
	MJ = np.pi/6 * cs**3/(GRA**3*rho)**0.5
	return MJ/Msun
	
# halo mass threshold of star formation in the post-reionization epoch
def Mreion(z, delta=200, T=2e4, Om=0.3153, h=0.6736):#, Ob=0.048, Om=0.3089):
	rho = rhom(1/(1+z),Om,h) * delta
	MJ = Jeansm(T, rho)
	return MJ


if __name__=="__main__":
	print([M_Tvir(1e4, z) for z in np.linspace(100,6,95)])
