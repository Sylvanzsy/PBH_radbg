# module for baryon-darkmatter scattering (BDSM)
from cosmology import *
from scipy.optimize import fsolve, curve_fit
from scipy.special import erf
from scipy.integrate import odeint, ode
from scipy.interpolate import interp1d
from injection import *
#from numba import njit

# dT/dt in the standard CDM cosmology based on the fit from 
# https://journals.aps.org/prd/abstract/10.1103/PhysRevD.82.083520
def Tdot(z, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76, a1=1./119, a2=1./115, T0=2.726):
	mu = 4/(1+3*X)
	a = 1/(1+z)
	Denorm = a**2 * (a+a1+a1*(a2/a)**1.5)**2
	Norm = a1*(1.5*a2*(a2/a)**0.5 + 2*a*(1+(a2/a)**1.5) \
		 + a1*(1+2*(a2/a)**1.5+(a2/a)**3))
	return - T0 * Norm/Denorm * a*H(a, Om, h, OR)  #* 1.5*BOL*rhom(a, Om, h)*Ob/(PROTON*mu)

GeV_to_mass = eV*1e9/SPEEDOFLIGHT**2	

# cooling/heating in the IGM for the standard CDM cosmology
def GammaC(z, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76, a1=1./119, a2=1./115, T0=2.726):
	a = 1/(1+z)
	dT0 = Tdot(z, Om, Ob, OR, h, X, a1, a2, T0)/(a*H(a, Om, h, OR))
	dT1 = -2*T_b(z, a1, a2, T0)/a
	gamma = (dT0-dT1)*a*H(a, Om, h, OR)/(T0/a-T_b(z, a1, a2, T0))
	return gamma

# relative velocity between gas and DM at redshift z
def vbdm_z(z, v0 = 30., z0 = 1100.):
	return v0*1e5*(1+z)/(1+z0)

# critical velocity for heating-cooling transition
def uthf(mb, mdm, Tb, Tdm):
	return (Tb*BOL/mb+Tdm*BOL/mdm)**0.5

# dv/dt due to friction between gas and DM
#@jit(nopython=True)
def drag(rho, v, Tb, Tdm, mb = PROTON, mdm = 0.3*GeV_to_mass, sigma = 8e-20):
	uth = (Tb*BOL/mb+Tdm*BOL/mdm)**0.5
	if v**2<=0:
		return 0
	else:
		r = v/uth
		A = (erf(r/2**0.5)-(2/np.pi)**0.5*np.exp(-r**2/2)*r)/v**2 * (r>1e-5) + (2/np.pi)**0.5*r/(3*uth**2) * (r<=1e-5)#eq(6)
		return rho*sigma*1e20/(mb+mdm) * A  #eq(5)

#@jit(nopython=True)
def Q_IDMB(rho, v, Tb, Tdm, mb = PROTON, mdm = 0.3*GeV_to_mass, sigma = 8e-20, gamma = 5/3):
	uth = (Tb*BOL/mb+Tdm*BOL/mdm)**0.5
	r = max(v/uth, 0)
	c = (Tdm-Tb)/uth**3 * ((2/np.pi)**0.5*np.exp(-r**2/2)) #first term in eq(4)
	if v<=0:
		d = 0
	else:
		d = mdm/v * (erf(r/2**0.5)-(2/np.pi)**0.5*np.exp(-r**2/2)*r)/BOL * (r>1e-5) + mdm/BOL * (2/np.pi)**0.5*r**2/(3*uth) * (r<=1e-5) #second term in eq(4)
	out = mb*rho*sigma*1e20/(mdm+mb)**2 * (c + d) #sum'em up
	return out*(gamma-1)
	
# evolution of gas temperature, DM temperature, and their relative velocity 
# by BDMS, similar to the cool function in radcool
def bdmscool(Tdm, Tb, v, rhob, rhodm, Mdm, sigma, gamma, X):
	xh = 4*X/(1+3*X)
	#a = 1/(1+z)
	rho = rhob + rhodm
	#rhob = Ob/Om * rhom(a, Om, h)
	QH = Q_IDMB(rhob, v, Tdm, Tb, Mdm*GeV_to_mass, PROTON, sigma, gamma)*xh
	QHe = Q_IDMB(rhob, v, Tdm, Tb, Mdm*GeV_to_mass, 4*PROTON, sigma, gamma)*(1-xh)
	dTdm = (QH+QHe)
	#rhodm = (Om-Ob)/Om * rhom(a, Om, h)
	QH = Q_IDMB(rhodm, v, Tb, Tdm, PROTON, Mdm*GeV_to_mass, sigma, gamma)*xh
	QHe = Q_IDMB(rhodm, v, Tb, Tdm, 4*PROTON, Mdm*GeV_to_mass, sigma, gamma)*(1-xh)
	dTb = (QH+QHe) #+ GammaC(1/a-1, Om, Ob, OR, h, X, T0 = T0)*(T0/a-Tb)
	DH = drag(rho, v, Tb, Tdm, PROTON, Mdm*GeV_to_mass, sigma)
	DHe = drag(rho, v, Tb, Tdm, 4*PROTON, Mdm*GeV_to_mass, sigma)	
	dv = - (xh*DH + (1-xh)*DHe)
	return [dTdm, dTb, dv]




# thermal history of the IGM modified by BDMS, where the cooling/heating
# and chemistry in gas are assumed to be identical to the CDM case
def thermalH(z0 = 1000., z1 = 9.0, v0 = 30., Mdm = 0.3, sigma = 8e-20, Om = 0.315, Ob = 0.048, OR = 9.54e-5, h = 0.6774, X = 0.76, a1=1./119, a2=1./115, T0=2.726, nb = 100000, vmin = 1e-20):
	Mb = PROTON*4/(1+3*X)/GeV_to_mass*(Om-Ob)/Ob
	Tmin = T_b(z1)/(1+Mb/Mdm)
	#print(Tmin)
	xh = 4*X/(1+3*X)
	def func(y, a):
		uth = 0 #(y[1]*BOL/PROTON+y[0]*BOL/(Mdm*GeV_to_mass))**0.5
		if y[2]<=vmin*uth:
			v = vmin*uth
			dv = 0.0#-y[2]/a
		else:
			DH = drag(rhom(a, Om, h), y[2], y[1], y[0], PROTON, Mdm*GeV_to_mass, sigma)
			DHe = drag(rhom(a, Om, h), y[2], y[1], y[0], 4*PROTON, Mdm*GeV_to_mass, sigma)
			v = y[2]	
			dv = -v/a - (xh*DH + (1-xh)*DHe)/(a*H(a, Om, h, OR))
		if y[1]<=y[0] and y[2]<=vmin*uth:
			dTdm = -2*y[0]/a
			dTb = -2*y[1]/a + GammaC(1/a-1, Om, Ob, OR, h, X, a1, a2, T0)*(T0/a-y[1])/ (a*H(a, Om, h, OR))
		elif y[1]<=Tmin:
			dTb = 0
			dTdm = 0
		else:
			rhob = Ob/Om * rhom(a, Om, h)
			QH = Q_IDMB(rhob, v, y[0], y[1], Mdm*GeV_to_mass, PROTON, sigma)*xh
			QHe = Q_IDMB(rhob, v, y[0], y[1], Mdm*GeV_to_mass, 4*PROTON, sigma)*(1-xh)
			dTdm = -2*y[0]/a + (QH+QHe)/ (a*H(a, Om, h, OR))
			rhodm = (Om-Ob)/Om * rhom(a, Om, h)
			QH = Q_IDMB(rhodm, v, y[1], y[0], PROTON, Mdm*GeV_to_mass, sigma)*xh
			QHe = Q_IDMB(rhodm, v, y[1], y[0], 4*PROTON, Mdm*GeV_to_mass, sigma)*(1-xh)
			dTb = -2*y[1]/a + (GammaC(1/a-1, Om, Ob, OR, h, X, a1, a2, T0)*(T0/a-y[1]) + (QH+QHe))/ (a*H(a, Om, h, OR))
		return [dTdm, dTb, dv]
	ai = 1/(1+z0)
	af = 1/(1+z1)
	#lz = np.linspace(z0, z1, nb)
	la = np.logspace(np.log10(ai), np.log10(af), nb)
	#la = 1/(lz+1)
	#print(la)
	lz = 1/la - 1
	y0 = [T_dm(z0, Mdm), T_b(z0), vbdm_z(z0, v0)]
	sol = odeint(func, y0, la)#, mxstep = 1000, hmin = 1e-30)
	d = {}
	sol = sol.T
	d['lz'] = lz
	d['la'] = la
	d['Tb'] = sol[1] * (sol[1]>Tmin) + Tmin * (sol[1]<=Tmin)
	d['Tdm'] = sol[0] #* (sol[0]>Tmin/1e10) + Tmin/1e10 * (sol[0]<=Tmin/1e10)
	d['v'] = sol[2]
	d['u'] = uthf(PROTON, Mdm*GeV_to_mass, d['Tb'], d['Tdm'])
	return d
