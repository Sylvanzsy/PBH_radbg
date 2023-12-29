import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import quad
from scipy import special
from scipy.interpolate import interp1d
from cosmology import *
from tophat import *
from PBHAcret import *
from nhrat import *
import sys
import os
import csv
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import matplotlib.colors as colors
from matplotlib.ticker import AutoMinorLocator
cosmology.setCosmology('planck18')
cosmo = cosmology.getCurrent()



mfunc_fof = mass_function.massFunction(1E12, 0.0, mdef = 'fof', model = 'watson13')



def heat_sub(mdot_, fh=1.0/3.0, er=0.057, A=100):
	return fh*er*A*mdot_/(1.0+A*mdot_) # from the paper




'''
def D(z, O_m = 0.3089):
	O_mz=O_m*(1+z)**3/(O_m*(1+z)**3+(1-O_m))
	O_Lamz=(1-O_m)/(O_m*(1+z)**3+(1-O_m))
	D_z=5*O_mz/2/(O_mz**(4/7)-O_Lamz+(1+O_mz/2)*(1+O_Lamz/70))
	return D_z/(1+z)
'''

def M_char(z_current,delta_crit = 1.686):
	lr=np.logspace(-5,3,200)
	sigma_R=[cosmo.sigma(R=r,z=0) for r in lr]
	R_sigma=interpolate.interp1d(np.log10(sigma_R),np.log10(lr),fill_value='extrapolate')
	sigma_Mchar=delta_crit/cosmo.growthFactor(z_current)
	return 4*np.pi*cosmo.rho_m(0)*1e9*(10**R_sigma(np.log10(sigma_Mchar)))**3

def colfac(z_current,M_h = 1e6):
	delta_crit = 1.686
	R_scale=1.85*(M_h/1e12)**(1/3)
	sigma_R=cosmo.sigma(R=R_scale,z=z_current)
	return special.erfc(delta_crit / (sigma_R *np.sqrt(2)))/2

'''
def Mh_z(z,z_i=300, Mh_i=1e4):
	lz3=np.linspace(z_i,z,z_i-z+1)
	Mh=Mh_i
	for i in range(len(lz3)-1):
		dM_tM_h= H(1/(1+lz3[i]))*2.3*(Mh/1e10)**0.15*((1+lz3[i])/7)**0.75
		Mh+=dM_tM_h*Mh*(TZ(lz3[i+1])-TZ(lz3[i]))
	return Mh
'''

# differential number density of PBH(comoving density)
def n_PBH_IGM(M_c,m_pbh,f_pbh, col_frac,Distr=0,sigM = 0.5, O_m = 0.3153, O_b = 0.0493, h = 0.6736):
	rhodm = rhom(1 , O_m, h) *(O_m-O_b)/O_m
	if Distr==0: #monochromatic
		n_pbh = f_pbh * rhodm/m_pbh/Msun
	elif Distr==1: #lognormal
		psi_m = lambda x: 1 / (np.sqrt(2 * np.pi) * sigM * x) * np.exp(-(np.log(x / M_c)) ** 2 / 2 / sigM ** 2)
		n_pbh = psi_m(m_pbh)*rhodm*f_pbh/m_pbh/Msun
		lm=np.logspace(np.log10(M_c)-2,3,100)
		fac=1/np.trapz([psi_m(m) for m in lm],x=lm)
		n_pbh*=fac  # to account for the numerical factor if distribution function was cutoff at 1000M_sun
	n_pbh*=1-col_frac
	return n_pbh #calculate the comoving number density of PBHs

# assume uniform density, calculate the energy produced per volume per sec from PBH accretion
def QPBH_IGM(m_pbh,z,er=0.057,PBH_halo = 0):
	mdot = bondi_IGM(m_pbh,z, PBH_halo=PBH_halo)
	print("Accretion rate is {:2e}".format(mdot))
	L_pbh =heat_sub(mdot)*mdot*6.7e-16*Msun*m_pbh*(0.1/er)*SPEEDOFLIGHT**2
	return L_pbh


"""
#calculate the light traveling distance from redshift z_i to z_f
def light_dist(z_i,z_f, Om = 0.3089, h = 0.6774, OR = 9.54e-5):
	def integrand(z):
		return 1/((1+z)*H(1/(1+z), Om, h, OR))
	I = quad(integrand, z_f, z_i, epsrel = 1e-8)
	return I[0]*H(1/(1+z_f),Om,h,OR)
"""


# Luminosity distance
def DL_z(z_i,z_f, Om = 0.3153, h = 0.6736, OR = 9.54e-5):
	def integrand(a):
		return SPEEDOFLIGHT/(a**2)/H(a, Om, h, OR)
	I = quad(integrand, 1/(1+z_i), 1/(1+z_f), epsrel = 1e-8)
	return I[0]*(1+z_i)/(1+z_f)

# calculate the power emitted by PBHs accretion IGM
def L_tot_IGM(M_c,f_pbh,z, col_frac,Distr=0,SigmaM=0.5,n_m=50, PBH_halo= 0):
	e_bh = 0
	if Distr==0: # monochromatic
		m_pbh=M_c
		e_bh += QPBH_IGM(m_pbh, z, PBH_halo=PBH_halo )*n_PBH_IGM(M_c,m_pbh,f_pbh, col_frac,Distr = Distr)*(1+z)**3
	elif Distr == 1:
		lmExt = np.logspace(np.log10(M_c)-1,np.log10(M_c)+1,n_m) #cut off the maximum PBH mass at 100 M_sun
		dL_dM=[QPBH_IGM(m,z, PBH_halo=PBH_halo)*n_PBH_IGM(M_c,m,f_pbh,col_frac,Distr=Distr,sigM=SigmaM)*(1+z)**3 for m in lmExt]
		e_bh += np.trapz(dL_dM,x=lmExt)
	# calculate the case for extended mass distribution: Lognormal
	return e_bh

'''

#use the bondi accretion to cross check with PBH accretion IGM
def Q_tot_IGM_bondi(M_c,f_pbh,z_i,z_f,col_frac,Distr=0,SigmaM=0.5,mu=1.22, O_m = 0.3153, O_b = 0.04930, h = 0.6736,n=100,n_m=50):
	lz = np.linspace(z_i, z_f, n)
	e_bh = np.zeros(n)
	n_gas_z= [rhom(1 / (1 + z), O_m, h) *O_b/O_m/mu/PROTON for z in lz] # calculate the average density of the halo, here we add a boost factor to reach an overdensity of 1300
	v_tilda=[c_s(z) for z in lz] #PBH relative velocity in the virialized halo
	if Distr==0: # monochromatic
		m_pbh=M_c
		for i in range(n - 1):
			e_bh += QPBH_halo(m_pbh, n_gas_z[i],v_tilda[i])*n_PBH_IGM(M_c,m_pbh,f_pbh,col_frac,Distr=Distr) * (TZ(lz[i + 1]) - TZ(lz[i]))*(1+z_f)**3#*light_dist(lz[i],z_f)
			#m_pbh += bondi_halo(m_pbh, n_gas_z[i],v_tilda[i]) * (TZ(lz[i + 1]) - TZ(lz[i])) / YR
	elif Distr == 1:
		lmExt = np.logspace(np.log10(M_c)-1,np.log10(M_c)+1,n_m)
		for i in range(n-1):
			Q_t=0
			for j in range(n_m-1):
				Q_t += QPBH_halo(lmExt[j],n_gas_z[i],v_tilda[i])*n_PBH_IGM(M_c,lmExt[j],f_pbh,col_frac,Distr,sigM = SigmaM)*(lmExt[j+1]-lmExt[j])
			#for j in range(n_m):
			#	lmExt[j]+= bondi_halo(lmExt[j], n_gas_z[i],v_tilda[i]) * (TZ(lz[i + 1]) - TZ(lz[i])) / YR
			e_bh += Q_t* (TZ(lz[i + 1]) - TZ(lz[i]))*(1+z_f)**4#*light_dist(lz[i],z_f)
	# calculate the case for extended mass distribution: Lognormal
	return e_bh
	
'''

#calculate the number of given PBH mass in a halo for different PBH mass distribution
def N_PBH_halo(M_c,m_pbh,M_h,f_pbh,Distr=0, sigM = 0.5, O_m = 0.3153, O_b = 0.04930, h = 0.6736,n_m=50):
	M_PBH_tot =  M_h*(O_m-O_b)/O_m*f_pbh
	#if m_pbh > M_PBH_tot:
	#	return 0
	#else:
	if Distr==0: #monochromatic
		N_pbh = M_PBH_tot / m_pbh
	if Distr==1: #lognormal  with some distribution width
		psi_m = lambda x: 1 / (np.sqrt(2 * np.pi) * sigM * x) * np.exp(-(np.log(x / M_c)) ** 2 / 2 / sigM ** 2)
		N_pbh = psi_m(m_pbh)*M_PBH_tot/m_pbh # return the differential number of PBHs at m_pbh
		lm=np.logspace(np.log10(M_c)-1,np.log10(M_c)+1,n_m)
		fac=1/np.trapz([psi_m(m) for m in lm],x=lm)
		N_pbh*=fac  # to account for the normalization factor if distribution function was cutoff at 260M_sun
	return N_pbh

# the luminous power from BH accretion at some fixed point in the DM halos
def QPBH_halo(m_pbh,n,v,er=0.057, r_min = -3):
	lr = np.logspace(r_min,2,20)
	mdot = [bondi_halo(m_pbh,n/3/r**2.2,v) for r in lr]
	dL = [heat_sub(mdot_r)* mdot_r *6.7e-16*Msun*m_pbh*(0.1/er)*SPEEDOFLIGHT**2 for mdot_r in mdot] #integrate over the radius with respect to #density of PBHs
	L_pbh = np.trapz(dL,x=lr)
	return L_pbh

# the luminous power from HMXB accretion at some fixed point in the DM halos
def QHMXB_halo(m_bh,er=0.057):
	L_pbh =heat_sub(1)*6.7e-16*Msun*m_bh*(0.1/er)*SPEEDOFLIGHT**2
	return L_pbh


'''
#Calculate the total amount of heating for Pbhs accreting in halos with monochromatic mass assuming roughly constant number density
def Q_tot_Halo(M_c,f_pbh,z_i,z_f,mu =1.22, Distr=0,SigmaM=0.5,n=100,n_m=100, h = 0.6774, O_m = 0.3089, O_b = 0.048,delta=6.5):
	e_bh = 0
	lz = np.logspace(np.log10(z_i), np.log10(z_f), n) #list of z's
	M_min=1e5
	#M_min=1.54*1e5*(31/(1+z))**2.074
	M_max=1e13
	lm = np.logspace(np.log10(M_min),np.log10(M_max),50) #list of halo mass
	dndM_halo =[[mass_function.massFunction(m, z, mdef = 'fof', model = 'press74',q_out = 'dndlnM')/m* 1/(1+z)**3/MPC**3*h**3 for m in lm] for z in lz ] #calculate the differential number density of the DM halo from halo mass function: dn/dM
	n_gas= [[delta * m * Msun * (O_m-O_b)/O_m/mu/PROTON/(4/3*np.pi*RV(m,z)**3) for m in lm] for z in lz] # calculate the average density of the halo, here we add a boost factor to reach an overdensity of 1300
	vtilda=[[5.4*(m/1e6)**(1/3)*(21/(1+z))**(1/2) for m in lm] for z in lz] #PBH relative velocity in the virialized halo
	# average over the halo mass function
	if Distr == 0:
		for i in range(n - 1):
			m_pbh_0=np.full(len(lm),M_c)
			for k in range(len(lm)-1):
				e_bh += N_PBH_halo(M_c, m_pbh_0[k], lm[k],f_pbh)*QPBH_halo(m_pbh_0[k],n_gas[i][k],vtilda[i][k]) * (TZ(lz[i + 1]) - TZ(lz[i]))*dndM_halo[i][k]*(lm[k+1]-lm[k]) *((1+z_f)/(1+lz[i]))**4 #*light_dist(lz[i],z_f)
			for k in range(len(lm)):
				m_pbh_0[k] += bondi_halo(m_pbh_0[k], n_gas[i][k],vtilda[i][k]) * (TZ(lz[i + 1]) - TZ(lz[i])) / YR
	elif Distr == 1:
		lmExt = np.full((len(lm),n_m),[np.logspace(np.log10(M_c)-2,np.log10(M_c)+2,n_m)])
		lm_ini = np.logspace(np.log10(M_c)-2,np.log10(M_c)+2,n_m)
		for i in range(n-1):
			Q_t=np.zeros(len(lm))
			for k in range(len(lm)): #calculate the heating rate at z for each halo of some mass M_h
				for j in range(n_m-1):
					Q_t[k] += QPBH_halo(lmExt[k][j],n_gas[i][k],vtilda[i][k])*N_PBH_halo(M_c,lm_ini[j],lm[k],f_pbh,Distr,sigM=SigmaM)*(lm_ini[j+1]-lm_ini[j]) #calculate average heating for a given PBH mass, using the initial PBH mass distribution
				for j in range(n_m):
					lmExt[k][j]+= bondi_halo(lmExt[k][j], n_gas[i][k] , vtilda[i][k]) * (TZ(lz[i + 1]) - TZ(lz[i])) / YR #change in mass function of PBHs by accretion in a halo of some mass M_h
			for k in range(len(lm)-1):
				e_bh += Q_t[k] * (TZ(lz[i + 1]) - TZ(lz[i]))*dndM_halo[i][k]*(lm[k+1]-lm[k]) *((1+z_f)/(1+lz[i]))**4 #*light_dist(lz[i],z_f)
	# calculate the case for extended mass distribution: Lognormal
	return e_bh
'''


# calculate the average PBH power from accretion in Halo as a function of redshift z:
def L_tot_Halo1(M_c,f_pbh,z,lM_h,dndM_halo,mu =1.22, Distr=0,SigmaM=0.5,n_m=50, h = 0.6736, O_m = 0.3153, O_b = 0.04930,delta0=6.5, r_min = -3):
	#if z>=100:
	#	M_min= 1.35*1e5/(O_m*h**2/0.15)**0.5 # set the minimum mass to the Jeans limit
	#elif z<100:
	#	M_min=4.54*1e3/((O_m*h**2/0.15)**0.5)/((O_b*h**2/0.022)**0.6)*((1+z)/10)**1.5
	#M_min=1.54*1e5*(31/(1+z))**2.074
	#M_max=1e13
	#lM_h = np.logspace(np.log10(M_min),np.log10(M_max),50) #list of halo mass
	#dndM_halo =[mass_function.massFunction(m, z, mdef = 'fof', model = 'press74',q_out = 'dndlnM')/m /MPC**3*h**3 for m in lM_h] #calculate the comoving differential number density of the DM halo from halo mass function: dn/dM
	if lM_h[-1] == 0:
		return 0
	else:
		n_gas= delta0*200/(mu*PROTON) *rhom(1/(1+z))*O_b/O_m  # calculate the average density of the halo
		vtilda=[5.4*(m/1e6)**(1/3)*(21/(1+z))**(1/2) for m in lM_h] #PBH relative velocity in the virialized halo	# average over the halo mass function
		if Distr == 0:
			Q_t=np.zeros(len(lM_h))
			m_pbh=M_c
			for k in range(len(lM_h)):
				if dndM_halo[k] == 'NaN':
					dndM_halo[k] = 0
				Q_t[k]+= N_PBH_halo(M_c, m_pbh, lM_h[k],f_pbh)*QPBH_halo(M_c,n_gas,vtilda[k],r_min=r_min) *dndM_halo[k]#*light_dist(lz[i],z_f)
			L=np.trapz(Q_t,x=lM_h)
		elif Distr == 1:
			lmExt = np.logspace(np.log10(M_c)-1,np.log10(M_c)+1,n_m)
			Q_t=np.zeros(len(lM_h))
			for k in range(len(lM_h)): #calculate the heating rate at z for each halo of some mass M_h
				if dndM_halo[k] == 'NaN':
					dndM_halo[k] = 0
				dQ_tdm=np.zeros(n_m)
				for j in range(n_m):
					dQ_tdm[j]+= QPBH_halo(lmExt[j],n_gas,vtilda[k],r_min=r_min)*N_PBH_halo(M_c,lmExt[j],lM_h[k],f_pbh,Distr,sigM=SigmaM)*dndM_halo[k]
				Q_t[k] +=np.trapz(dQ_tdm,x=lmExt)#calculate average heating for a given PBH mass, using the initial PBH mass distribution
			L=np.trapz(Q_t,x=lM_h) # average over the halo mass
		return L
	# calculate the case for extended mass distribution: Lognormal









'''
# cumulative flux rate in unit of erg/s/cm^2/sr
def Flux(M_c,f_pbh,z_i,z_f,Distr=0,n=100,n_m=100):
	F_bh = np.zeros(n)
	lz = np.linspace(z_i, z_f, n)
	if Distr==0: # monochromatic
		m_pbh=M_c
		for i in range(n - 1):
			F_bh[i] += QPBH_(m_pbh, lz[i],f_pbh)*n_PBH(M_c,m_pbh,f_pbh,lz[i],Distr)/ (4*np.pi*DL_z(lz[i])**2) #*light_dist(lz[i],z_f)
			m_pbh += mdot(m_pbh, lz[i + 1],f_pbh) * (TZ(lz[i + 1]) - TZ(lz[i])) / YR
	# calculate the case for extended mass distribution: Lognormal
	else:
		lmExt = np.logspace(np.log10(M_c)-2,np.log10(M_c)+2,n_m)
		for i in range(n-1):
			Q_t=0
			for j in range(n_m-1):
				Q_t += QPBH_(lmExt[j],lz[i],f_pbh)*n_PBH(M_c,lmExt[j],f_pbh,lz[i],Distr)*(lmExt[j+1]-lmExt[j])
			for j in range(n_m):
				lmExt[j]+= mdot(lmExt[j], lz[i],f_pbh) * (TZ(lz[i + 1]) - TZ(lz[i])) / YR
			F_bh[i] += Q_t / (4*np.pi*DL_z(lz[i])**2)#*light_dist(lz[i],z_f)
	return F_bh
'''

# include the effect of BH seeding such that it would not under produce BH
def SMBHmass(M_H,z,f_conv=0.1, h = 0.6774, seeding = 0):
	if z <=30: # assume first star forms at around redshift 30 and starts to generate SMBHs immediately
		BH_eff=2*1e-7*(1+z)/10*(0.1/f_conv)*(M_H/1e10)**(2/3)
		M_BH=BH_eff*M_H
		if seeding == 1:
			if BH_eff<=2*1e-6:
				M_BH=max(BH_eff*M_H,100)
		#dn_dMBH = 3/5* lnh
		#dn_dMBH=3/5*mass_function.massFunction(M_H, z, mdef = 'fof', model = 'press74',q_out = 'dndlnM')/M_H/MPC**3*h**3
	else:
		M_BH=0
		dn_dMBH=0
	return M_BH

#calculate the total energy from SMBH in the center of the halo
#lm: mass list of halos
#lnh: number density correpond to halo mass
#z: redshift of interest
def L_tot_SMBH(lm_SMBH,lmh,lnh, er=0.057):
	if lmh[-1] == 0:
		return 0
	else:
		#M_min=1e8
		#M_min=1.54*1e5*(31/(1+z))**2.074
		#M_max=1e13
		#lm = np.logspace(np.log10(M_min),np.log10(M_max),50) #list of halo mass
		Q_t=0
		L_pbh=np.zeros(len(lm_SMBH))
		dL_dM=np.zeros(len(lm_SMBH))
		for j in range(len(lmh)):
			if lnh[j] == 'NaN':
				lnh[j] = 0
			L_pbh[j] +=heat_sub(1)*6.7e-16*Msun*lm_SMBH[j]*(0.1/er)*SPEEDOFLIGHT**2
			dL_dM[j] += L_pbh[j]*lnh[j]
		Q_t+= np.trapz(dL_dM,x=lmh)
		return Q_t



# interpolate the initial mass of stars as a function of the masses of their final state as BH
BHdata=np.genfromtxt('M_BH.dat',delimiter=',')
M_BH_f=interpolate.interp1d(BHdata[:,0],BHdata[:,1], fill_value='extrapolate')




# assume a constant PBH mass function since DM halo captures much more halo during its growth
#calculate separately for pop III and pop I/II remnants
# z: current redshift
# M_BH: total BH mass
# lM_h: array of mass of halos at redshift z
# dndM_halo: halo mass function at redshift z
# HMXB_flag: 0: BH case 1:HMXB case

def L_tot_Halo2(z,M_BH_pop3,M_BH_pop12,lM_h,dndM_halo,mu =1.22, Pop3_IMF = 'Salpter',n_m=20, h = 0.6736, O_m = 0.3153, O_b = 0.04930,delta0=6.5, f_pop12=0.5, f_pop3=0.3,HMXB_flag=0,Duty = 0.1, r_min = -3):
	e_HMXB = 0
	e_bh = 0
	if lM_h[-1] == 0: # check if it is an empty list
		return 0
	else:
		#M_min=max(1.54*1e5*(31/(1+z))**2.074, 4.54*1e3/((O_m*h**2/0.15)**0.5)/((O_b*h**2/0.022)**0.6)*((1+z)/10)**1.5) # cut off at the minimum halo mass that allows for star formation
		#M_max=1e13
		#lM_h = np.logspace(np.log10(M_min),np.log10(M_max),50) #list of halo mass
		#dndM_halo =[mass_function.massFunction(m, z, mdef = 'fof', model = 'press74',q_out = 'dndlnM')/m/MPC**3*h**3 for m in lM_h]  #calculate the differential number density of the DM halo from halo mass function: dn/dM
		n_gas= delta0*200/(mu*PROTON) *rhom(1/(1+z))*O_b/O_m  # calculate the average density of the halo
		vtilda=[5.4*(m/1e6)**(1/3)*(21/(1+z))**(1/2) for m in lM_h]  #PBH relative velocity in the virialized halo
		# average over the halo mass function
		lm_star_pop3 = np.linspace(25,140,n_m)
		lm_abh_pop3 = [M_BH_f(m) for m in lm_star_pop3]
		lm_star_pop12 = np.linspace(25,260,n_m)
		lm_abh_pop12 = [M_BH_f(m) for m in lm_star_pop12]
		Q_t=np.zeros(len(lM_h))
		Q_t_HMXB=np.zeros(len(lM_h))
		if Pop3_IMF == 'Salpter':  # initialize the slope of the mass function based on the IMF shape
			alpha = -2.35
		elif Pop3_IMF == 'TopHeavy':
			alpha = -1.35
		C_pop3 = M_BH_pop3/np.trapz(lm_abh_pop3*lm_star_pop3**alpha,x=lm_star_pop3)  #find the constant in the mass function from the pop3 star remnants
		C_pop12 = M_BH_pop12/np.trapz(lm_abh_pop12*lm_star_pop12**(-2.35),x=lm_star_pop12)  #find the constant in the mass function from the pop1/2 star remnants
		if HMXB_flag == 0:
			for k in range(len(lM_h)): #calculate the heating rate at z for each halo of some mass M_h
				for j in range(n_m-1):
					Q_t[k] += QPBH_halo(lm_abh_pop3[j],n_gas,vtilda[k],r_min=r_min)*C_pop3[k]*lm_abh_pop3[j]*lm_star_pop3[j]**alpha*(lm_star_pop3[j+1]-lm_star_pop3[j])*dndM_halo[k] #calculate average heating for a given BH mass from Pop 3 remnants, using the initial PBH mass distribution
					Q_t[k] += QPBH_halo(lm_abh_pop12[j],n_gas,vtilda[k],r_min=r_min)*C_pop12[k]*lm_abh_pop12[j]*lm_star_pop12[j]**(-2.35)*(lm_star_pop12[j+1]-lm_star_pop12[j])*dndM_halo[k] #calculate average heating for a given BH mass from Pop 2 remnants, using the initial PBH mass distribution
			e_bh += np.trapz(Q_t,x=lM_h)
			return e_bh
		elif HMXB_flag == 1:
			for k in range(len(lM_h)):  # calculate the heating rate at z for each halo of some mass M_h
				for j in range(n_m - 1):
					Q_t_HMXB[k] += QHMXB_halo(lm_abh_pop3[j])*C_pop3[k]*f_pop3/2*lm_abh_pop3[j]*lm_star_pop3[j]**alpha*(lm_star_pop3[j+1]-lm_star_pop3[j])*dndM_halo[k]*Duty #calculate average heating for a given BH mass from Pop 3 remnants, using the initial PBH mass distribution
					Q_t_HMXB[k] += QHMXB_halo(lm_abh_pop12[j])*C_pop12[k]*f_pop12/2*lm_abh_pop12[j]*lm_star_pop12[j]**(-2.35)*(lm_star_pop12[j+1]-lm_star_pop12[j])*dndM_halo[k]*Duty #calculate average heating for a given BH mass from Pop 2 remnants, using the initial PBH mass distribution
			e_HMXB += np.trapz(Q_t_HMXB,x=lM_h)
			# calculate the case for extended mass distribution: Lognormal
			return e_HMXB



'''
f_pbh=1e-4
PsiM=1 # initialize distribution funtion
Sigma_M=0.5
z_ini=100
z_end=6
lz2 = np.linspace(z_ini, z_end, z_ini-z_end+1)
O_m = 0.3153
O_b = 0.04930
h=0.6736

#Find the critical redshift where heating in the halo starts to dominate
PsiM=0
Sigma_M=0.5
Mc=10
lPBHfrac = np.logspace(0,-4,20)
z_ini=50
z_end=6
lz2 = np.logspace(np.log10(z_ini), np.log10(z_end), 50)


L_z0_ABH=[L_tot_Halo2(z,IMF_BH='TopHeavy') for z in lz2]

L_z0_SMBH=[L_tot_SMBH(z) for z in lz2]


def criticalZ(Mc, PsiM):
	z_crit = np.zeros(20)
	for j in range(20):
		L_diff = np.zeros(len(lz2))
		L_z0_Halo=[L_tot_Halo1(Mc,lPBHfrac[j],z,Distr=PsiM,SigmaM=Sigma_M) for z in lz2]
		L_z0_IGM=[L_tot_IGM(Mc,lPBHfrac[j],z,Distr=PsiM,SigmaM=Sigma_M) for z in lz2]
		for i in range(len(lz2)):
			L_diff[i]+= L_z0_Halo[i]+L_z0_ABH[i]+L_z0_SMBH[i]-L_z0_IGM[i]
		z_crit[j] += np.interp(0,L_diff,lz2)
	return z_crit

fig, ax = plt.subplots(constrained_layout=True)

ax.plot(lPBHfrac,criticalZ(10,0),linewidth = 2,label='Mono PBH')
ax.plot(lPBHfrac,criticalZ(10,1),linewidth = 2,label='Lognormal PBH')
ax.legend(fontsize=15)
ax.set_xlabel(r'$f_{PBH}$',fontsize=15)
ax.set_ylabel(r'$z_{crit} $',fontsize=15)
ax.set_xscale('log')
ax.set_yscale('log')

plt.savefig('Z_crit'+str(PsiM)+'.pdf')
plt.close()
'''





if __name__ == "__main__":
# Halo mass range to plot
	lM = np.logspace(6, 16, num=40)
	lM1 = np.logspace(6, 16, num=12)

	# Redshifts to plot
	lz = np.linspace(20,0,40)
	IMF = 'Salpter'
	fit= 'Harikane22'
	#fit = 'Madau14'

	# Calculate SFR for each halo mass and redshift
	SFR_UNIV_Pop3 = np.zeros((len(lz), len(lM)))	#SFR from universe machine model
	SFR_SFRD = np.zeros((len(lz), len(lM)))			#SFR from star formation rate density (SFRD) model


	M_BH_tot_UNIV = np.zeros((len(lz), len(lM)))
	M_HMXB_tot_UNIV = np.zeros((len(lz), len(lM)))
	M_BH_tot = np.zeros((len(lz), len(lM)))
	M_SMBH_tot = np.zeros((len(lz), len(lM)))
	M_HMXB_tot = np.zeros((len(lz), len(lM)))
	rate = np.zeros((len(lz), len(lM)))
	M_h_z = np.zeros((len(lz), len(lM)))
	M_h_z_1 = np.zeros((len(lz), len(lM1)))
	for i in range(len(lz)):
		for j in range(len(lM1)):
			M_h_z_1[i,j] += M_z(lM1[j],lz[i])
	for i in range(len(lz)):
		for j in range(len(lM)):
			M_BH_tot_UNIV[i,j],M_h_z[i,j], M_HMXB_tot_UNIV[i,j]  = M_BH_UNIV(lM[j],lz[i],'TopHeavy',mode=1)


	for i in range(len(lz)):
		for j in range(len(lM)):
			M_SMBH_tot[i,j] += M_SMBH_Mstar(M_star_Mh(M_h_z[i,j],lz[i]),lz[i])
			M_BH_tot[i,j] , M_HMXB_tot[i,j] = M_BH(M_h_z[i,j],lz[i],IMF)
			rate[i,j] += M_SMBH_tot[i,j]/M_BH_tot[i,j]
			SFR_UNIV_Pop3[i,j] += SFR_UNIV(M_h_z[i,j],lz[i]) + SFR_pop3(M_h_z[i,j],lz[i],lz[i])
			SFR_SFRD[i,j] += SFR_pop12(M_h_z[i,j],lz[i],lz[i],fit) + SFR_pop3(M_h_z[i,j],lz[i],lz[i])

	print(rate)

	# make data
	X, Y = np.meshgrid(lM,lz)

	lz1 = np.linspace(21,1,40)
	X1, Y1 = np.meshgrid(lM, lz1)
	# plot
	fig, ax = plt.subplots()

	pcm = ax.pcolormesh( Y1,M_h_z, SFR_UNIV_Pop3,
                       norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=np.min(SFR_UNIV_Pop3), vmax=np.max(SFR_UNIV_Pop3), base=10),
                       cmap='RdBu_r', shading='nearest')
	cbar1 = fig.colorbar(pcm, ax=ax, extend='both')
	cbar1.ax.tick_params(labelsize=12)
	cbar1.set_label(label='SFR ($\mathrm{M}_{\odot}/yr$)', size=15)
	ax.plot(lz1,M_h_z_1,color = 'white',linewidth =0.5)
	ax.annotate('No Halo',xy=(10, 1e14), xycoords='data',fontsize=12,color='black')

	ax.set_title(r'SFR as a function of Halo Mass and Redshift', fontsize = 15)

	ax.set_xticklabels(lz1, fontsize=12)
	ax.set_yticklabels(M_h_z_1, fontsize=12)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylabel('Halo Mass ($\mathrm{M}_{\odot}$)', fontsize = 15)
	ax.set_xlabel('z+1', fontsize = 15)
	ax.set_ylim(1e5, 1e16)
	ax.set_xlim(1, 21)
	ax.set_facecolor('lightgray')

	plt.savefig('SFR_Mh_z.pdf')
	plt.close()


	fig, ax = plt.subplots()

	pcm = ax.pcolormesh( Y,M_h_z, M_BH_tot_UNIV,
                       norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=1, vmax=np.max(M_BH_tot_UNIV), base=10),
                       cmap='RdBu_r', shading='nearest')
	cbar2 =fig.colorbar(pcm, ax=ax, extend='both')
	cbar2.ax.tick_params(labelsize=12)
	cbar2.set_label(label='$\Sigma M_{\mathrm{SRBH}}$ ($\mathrm{M}_{\odot}$)', size = 15)
	ax.plot(lz,M_h_z_1,color = 'white',linewidth =0.5)
	ax.set_title(r'Total SRBH mass as a function of $M_h$ and $z$', fontsize = 15)
	ax.annotate('No Halo',xy=(16, 1e12), xycoords='data',fontsize=12,color='black')

	ax.set_xticklabels(lz, fontsize=12)
	ax.set_yticklabels(M_h_z_1, fontsize=12)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylabel('Halo Mass ($\mathrm{M}_{\odot}$)', fontsize = 15 )
	ax.set_xlabel('z', fontsize = 15 )
	ax.set_ylim(1e5, 1e13)
	ax.set_xlim(6, 20)
	ax.set_facecolor('lightgray')

	plt.savefig('M_BH_Mh_z.pdf')
	plt.close()

	fig, ax = plt.subplots()

	pcm = ax.pcolormesh( Y,M_h_z, M_HMXB_tot_UNIV,
                       norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=1, vmax=np.max(M_BH_tot_UNIV), base=10),
                       cmap='RdBu_r', shading='nearest')
	cbar3 = fig.colorbar(pcm, ax=ax, extend='both')
	cbar3.ax.tick_params(labelsize=12)
	cbar3.set_label(label='$M_{\mathrm{BH}}$ ($\mathrm{M}_{\odot}$)',size = 15)
	ax.plot(lz,M_h_z_1,color = 'white',linewidth =0.5)
	ax.set_title(r'Total HMXB mass as a function of Halo Mass and Redshift', fontsize = 15)
	ax.annotate('No Halo',xy=(16, 1e12), xycoords='data',fontsize=12,color='black')

	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xticklabels(lz, fontsize=12)
	ax.set_yticklabels(M_h_z_1, fontsize=12)
	ax.set_ylabel('Halo Mass ($\mathrm{M}_{\odot}$)', fontsize = 15 )
	ax.set_xlabel('z', fontsize = 15 )
	ax.set_ylim(1e5, 1e13)
	ax.set_xlim(6, 20)
	ax.set_facecolor('lightgray')

	plt.savefig('M_HMXB_Mh_z.pdf')
	plt.close()

	fig, ax = plt.subplots()

	pcm = ax.pcolormesh( Y,M_h_z, M_SMBH_tot,
                       norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=1, vmax=np.max(M_SMBH_tot), base=10),
                       cmap='RdBu_r', shading='nearest')
	cbar4 = fig.colorbar(pcm, ax=ax, extend='both')
	cbar4.ax.tick_params(labelsize=12)
	cbar4.set_label(label='$M_{\mathrm{SMBH}}$ ($\mathrm{M}_{\odot}$)', size=15)
	ax.plot(lz,M_h_z_1,color = 'white',linewidth =0.5)
	ax.set_title(r'SMBH mass as a function of Halo Mass and Redshift', fontsize = 15)
	ax.annotate('No Halo',xy=(16, 1e12), xycoords='data',fontsize=12,color='black')

	ax.set_xticklabels(lz, fontsize=12)
	ax.set_yticklabels(M_h_z_1, fontsize=12)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylabel('Halo Mass ($\mathrm{M}_{\odot}$)', fontsize = 15 )
	ax.set_xlabel('z', fontsize = 15 )
	ax.set_ylim(1e8, 1e13)
	ax.set_xlim(6, 20)
	ax.set_facecolor('lightgray')

	plt.savefig('M_SMBH_Mh_z.pdf')
	plt.close()


