# this python script calculate the spectra from the accretion of BHs formed from different sources.
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
from SMBHgrow import *
import sys
import os
import csv
from colossus.cosmology import cosmology
from colossus.lss import mass_function
cosmology.setCosmology('planck18')
cosmo = cosmology.getCurrent()




# luminosity of a PBH in the band defined by re (photon energy in eV)
# m: PBH mass, n: hydrogen number density in cm^-3
# v: characteristic velocity in km/s
# ne: number of freqeuncy/energy bins
# num=0: energy per unit time, >0: photon number per unit time




def radbg(m, n, v, re=[500, 2e3], ne=10, er=0.057, alpha=0.1, mode=0, cont=0, num=0):
	mdot = mdot_bondi(m, n, v, er, fed=1e8)
	if cont>0:
		fac = 0.075/er
	else:
		fac = 1.0
	mcut = 0.07*alpha *fac
	if num == 2:
		if mdot>mcut:
			lnu, emax = radbg_SMBH(mdot, m, v,re, er) # for a good approximation for slim disk case(mdot >=1)
		else:
			lnu, emax = ADAF(re, m, mdot, er=er, alpha=alpha, mode=mode)
		return lnu
	else:
		if re[-1] == re[0]:
			return 0
		le = np.geomspace(*re, ne + 1)
		if mdot>mcut:
			ll, emax = radbg_SMBH(mdot,m, v,le, er) # for a good approximation for slim disk case(mdot >=1)
		else:
			ll, emax = ADAF(le, m, mdot, er=er, alpha=alpha, mode=mode)
		if num == 1:
			return np.trapz(ll/(le*eV), le)
		elif num == 0:
			return np.trapz(ll, le)

def radbg_SMBH(mdot,M_BH,v, re=[500, 2e3], er=0.057, fed=1.0,mu=1.22):
	Ti = 1.3e3*mdot**0.25/(M_BH/10)**0.25  #eq(3.5), rewrite as a function of accretion rate
	Tmax = 0.488*Ti
	r_out = 1.16*1e10*(M_BH/10)**(2/3)*(v/10)**(-10/3) # ratio of r_out/r_ISCO
	To = Ti * (3/r_out)**0.75#eq (3.8), approximation when r_out>>r_isco
	ca = 1.27e29*M_BH**(5/4)*er/0.057*(1/mu)**0.75*(mdot/2.64*1e7)**0.75 #eq(3.7)
	Lnu = ca*(Tmax/To)**(5./3.)*(re/Tmax)**2 * (re<=To)
	Lnu += ca*(re/Tmax)**(1./3.) * (re>To)*(re<Tmax)
	Lnu += ca*(re/Tmax)**2*np.exp(1.0-re/Tmax) * (re>=Tmax)
	#le = np.logspace(-5,5,30)
	#Lnu_tot = ca*(Tmax/To)**(5./3.)*(le/Tmax)**2 * (le<=To)+ ca*(le/Tmax)**(1./3.) * (le>To)*(le<Tmax)+ ca*(le/Tmax)**2*np.exp(1.0-le/Tmax) * (le>=Tmax)
	#L_Edd = np.trapz(Lnu_tot, x=le)
	#Lnu *= QHMXB_halo(M_BH)/L_Edd
	#print(QHMXB_halo(M_BH)/L_Edd)
	if mdot>fed:
		Lnu = Lnu*fed/mdot # renormalize to Eddington Luminosity if accretion ratio is calculated to be larger
	return Lnu, 10 * Ti





def radbg_IGM(m, z, re=[500, 2e3], ne=10, er=0.057, alpha=0.1, mode=0, cont=0, num=0, PBH_halo = 0):
	mdot = bondi_IGM(m, z, PBH_halo= PBH_halo)

	if num == 2:
		lnu, emax = ADAF(re, m, mdot, er=er, alpha=alpha, mode=mode)
		return lnu
	else:
		if re[-1] == re[0]:
			return 0
		le = np.geomspace(*re, ne+1)
		ll, emax = ADAF(le, m, mdot, er=er, alpha=alpha, mode=mode)
		if num == 1:
			return np.trapz(ll/(le*eV), le)
		elif num == 0:
			return np.trapz(ll, le)

# luminosity/flux from all PBHs in a halo in the band re (photon energy in eV)
# mpbh: PBH mass, mh: halo mass, z: redshift
# mode=0: isothermal distribution of gas
# mode>0: uniform distribution of gas with a characteristic density of
# mode * average gas density in the halo
# nr: number of radius bins (only relevant for mode=0)
# r0: inner edge of PBH distribution in pc
# ctr=0: luminosity, >0: flux at the halo center assuming optically thin
def radhalo(re, mbh, mh, z, mode=0, nr=100, r0=1, Om=0.3153, Ob=0.04930, X=0.76, ctr=0):
	rvir = RV1(mh, z)/PC # virial radius, see cosmology.py
	vcir = Vcir(mh, z)/1e5 # circular velocity, see cosmology.py
	#print(vcir)
	nbh = mh*(Om-Ob)/Om/mbh
	n0 = 200*rhom(1/(1+z))*Ob/Om*X/PROTON # average gas density, see cosmology.py
	if mode==0:
		lr = np.geomspace(r0, rvir, nr+1)
		ln =  n0/3 * (rvir/lr)**2
		#print(ln[0], ln[-1])
		ll = np.array([radbg(mbh, n, vcir, re) for n in ln])
		if ctr>0:
			lum = nbh * np.trapz(ll/lr**2, lr)/(rvir-r0)/(4*np.pi) #np.trapz(np.ones(nr+1), lr)
		else:
			lum = nbh * np.trapz(ll, lr)/(rvir-r0)
	else:
		lum = nbh * radbg(mbh, n0*mode, vcir, re)
	return lum

#calculate the total energy from SMBH in the center of the halo
#aussmu f ~v^-1
# set the inner core radius to be r_core/r_vir = 1e-3, and assume rho ~ r^-2
def Lnu_tot_SMBH(z, lm_SMBH, lmh, lnh, er=0.057,re=[500,2e3],mu=1.22, O_m = 0.3153, O_b = 0.04930, X=0.76,delta0=1,TauH=0):
	if lmh[-1] == 0:
		return 0
	else:
		#M_min=1e8
		#M_min=1.54*1e5*(31/(1+z))**2.074
		#M_max=1e13
		#lm = np.logspace(np.log10(M_min),np.log10(M_max),50) #list of halo mass
		r_vir = [RV1(m,z) for m in lmh]
		n_gas= delta0*200/(mu*PROTON) *rhom(1/(1+z))*O_b/O_m  # calculate the average density of the halo
		vcir = [Vcir(mh, z) / 1e5  for mh in lmh] # circular velocity in units of km/s, see cosmology.py
		tau = np.zeros(len(lmh))
		L_x=0
		L_pbh=np.zeros(len(lmh))
		dL_dM=np.zeros(len(lmh))
		for j in range(len(lmh)):
			lnu, emax = radbg_SMBH(1,lm_SMBH[j],vcir[j], er=er,re=re*(1+z))
			#if len(re) == 1:
			#	fac = QHMXB_halo(lm_SMBH[j]) / re/(1+z) / lnu
			#	print('Normalization factor for {:.2e} mass BH is {:.1f}, at peak freq {:.2e}'.format(lm_SMBH[j], int(fac),emax))
			L_pbh[j]+= lnu
			if TauH == 1:
				tau[j] += sigma_E(re*(1+z)) * n_gas * r_vir[j]
				L_pbh[j]*= np.exp(-tau[j]) #add attenuation factor of exp(-tau), to model the emission of light into IGM
			dL_dM[j] += L_pbh[j]*lnh[j]
		L_x+= np.trapz(dL_dM,x=lmh)
		return L_x



# calculate the average PBH power from accretion in Halo as a function of redshift z:
def Lnu_tot_Halo1(M_c,f_pbh,z,lM_h, dndM_halo,mu =1.22, Distr=0,SigmaM=0.5,n_m=50, O_m = 0.3153, O_b = 0.04930,delta0=1, er=0.057,re=[500,2e3],TauH=0, r_min = -3):
	if lM_h[-1] == 0: # check if it is an empty list
		return 0
	else:
		lr = np.logspace(r_min, 2, 10)
		r_vir = [RV1(m,z) for m in lM_h]
		n_gas= delta0*200/(mu*PROTON) *rhom(1/(1+z))*O_b/O_m  # calculate the average density of the halo
		vtilda=[5.4*(m/1e6)**(1/3)*(21/(1+z))**(1/2) for m in lM_h] #PBH relative velocity in the virialized halo	# average over the halo mass function
		tau = np.zeros(len(lM_h))
		if Distr == 0:
			Q_t=np.zeros(len(lM_h))
			m_pbh=M_c
			for k in range(len(lM_h)):
				dL_dr =[np.sum(radbg(M_c,n_gas/3/r**2.2,vtilda[k], er=er,re=re*(1+z),num=2)) for r in lr]
				dL=np.trapz(dL_dr, lr)
				Q_t[k]+= dL *N_PBH_halo(M_c, m_pbh, lM_h[k],f_pbh) *dndM_halo[k]
				if TauH == 1:
					tau[k] += sigma_E(re*(1+z)) * n_gas * r_vir[k]
					Q_t[k] *= np.exp(-tau[k]) #calculate average heating for a given PBH mass, using the initial PBH mass distribution
			#print(n_gas)
			L_X=np.trapz(Q_t,x=lM_h)
		elif Distr == 1:
			lmExt = np.logspace(np.log10(M_c)-1,np.log10(M_c)+1,n_m)
			Q_t=np.zeros(len(lM_h))
			for k in range(len(lM_h)): #calculate the heating rate at z for each halo of some mass M_h
				dQ_tdm=np.zeros(n_m)
				for j in range(n_m):
					dL_dr =[np.sum(radbg(lmExt[j],n_gas/3/r**2.2,vtilda[k], er=er,re=re*(1+z),num=2)) for r in lr]
					dL=np.trapz(dL_dr, lr)
					dQ_tdm[j]+= dL*N_PBH_halo(M_c,lmExt[j],lM_h[k],f_pbh,Distr,sigM=SigmaM)*dndM_halo[k]
				Q_t[k] +=np.trapz(dQ_tdm,x=lmExt)
				if TauH == 1:
					tau[k] += sigma_E(re*(1+z)) * n_gas * r_vir[k]
					Q_t[k] *= np.exp(-tau[k]) #calculate average heating for a given PBH mass, using the initial PBH mass distribution
			L_X=np.trapz(Q_t,x=lM_h) # average over the halo mass
		return L_X
	# calculate the case for extended mass distribution: Lognormal


# averaged over the density profile of the DM halo
# assume a constant PBH mass function since DM halo captures much more halo during its growth
def Lnu_tot_Halo2(z,M_BH_pop3,M_BH_pop12,lM_h,dndM_halo,mu =1.22, Pop3_IMF = 'Salpter', h = 0.6736, O_m = 0.3153, O_b = 0.04930, delta0=6.5, f_pop12=0.5, f_pop3=0.3,HMXB_flag=0,Duty = 0.1,n_m=30,er=0.057,re=[500,2e3],TauH=0, r_min = -3):
	e_HMXB = 0
	e_bh = 0
	if lM_h[-1] == 0: # check if it is an empty list
		return 0
	else:
		r_vir = [RV1(m,z) for m in lM_h]
		n_gas= delta0*200/(mu*PROTON) *rhom(1/(1+z))*O_b/O_m  # calculate the average density of the halo
		vtilda=[5.4*(m/1e6)**(1/3)*(21/(1+z))**(1/2) for m in lM_h]  #PBH relative velocity in the virialized halo
		tau = np.zeros(len(lM_h))
		# average over the halo mass function
		lm_star_pop3 = np.linspace(20,130,n_m)
		lm_abh_pop3 = [M_BH_f(m) for m in lm_star_pop3]
		lm_star_pop12 = np.linspace(20,260,n_m)
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
			lr = np.logspace(r_min, 2, 10)
			for k in range(len(lM_h)): #calculate the heating rate at z for each halo of some mass M_h
				for j in range(n_m-1):
					dL_3_dr =[np.sum(radbg(lm_abh_pop3[j],n_gas/3/r**2.2,vtilda[k], er=er,re=re*(1+z),num=2)) for r in lr]
					dL_12_dr =[np.sum(radbg(lm_abh_pop12[j],n_gas/3/r**2.2,vtilda[k], er=er,re=re*(1+z),num=2)) for r in lr]
					dL_3=np.trapz(dL_3_dr, lr)
					dL_12=np.trapz(dL_12_dr, lr)
					Q_t[k] += dL_3*C_pop3[k]*lm_abh_pop3[j]*lm_star_pop3[j]**alpha*(lm_star_pop3[j+1]-lm_star_pop3[j])*dndM_halo[k]
					 #calculate average heating for a given BH mass from Pop 3 remnants, using the initial PBH mass distribution
					Q_t[k] += dL_12*C_pop12[k]*lm_abh_pop12[j]*lm_star_pop12[j]**(-2.35)*(lm_star_pop12[j+1]-lm_star_pop12[j])*dndM_halo[k]
					#calculate average heating for a given BH mass from Pop 2 remnants, using the initial PBH mass distribution
				if TauH == 1:
					tau[k] += sigma_E(re*(1+z)) * n_gas * r_vir[k]
					Q_t[k] *= np.exp(-tau[k]) #calculate average heating for a given PBH mass, using the initial PBH mass distribution
			e_bh += np.trapz(Q_t,x=lM_h)
			return e_bh
		elif HMXB_flag == 1:
			for k in range(len(lM_h)):  # calculate the heating rate at z for each halo of some mass M_h
				for j in range(n_m - 1):
					lnu_pop3, emax = radbg_SMBH(1,lm_abh_pop3[j],vtilda[k], er=er,re=re*(1+z))
					lnu_pop12, emax = radbg_SMBH(1,lm_abh_pop12[j],vtilda[k], er=er,re=re*(1+z))
					Q_t_HMXB[k] += lnu_pop3*C_pop3[k]*f_pop3/2*lm_abh_pop3[j]*lm_star_pop3[j]**alpha*(lm_star_pop3[j+1]-lm_star_pop3[j])*dndM_halo[k]*Duty
					#calculate average heating for a given BH mass from Pop 3 remnants, using the initial PBH mass distribution
					Q_t_HMXB[k] += lnu_pop12*C_pop12[k]*f_pop12/2*lm_abh_pop12[j]*lm_star_pop12[j]**(-2.35)*(lm_star_pop12[j+1]-lm_star_pop12[j])*dndM_halo[k]*Duty
					#calculate average heating for a given BH mass from Pop 2 remnants, using the initial PBH mass distribution
				if TauH == 1:
					tau[k] += sigma_E(re*(1+z)) * n_gas * r_vir[k]
					Q_t_HMXB[k] *= np.exp(-tau[k]) #calculate average heating for a given PBH mass, using the initial PBH mass distribution
			e_HMXB += np.trapz(Q_t_HMXB,x=lM_h)
			# calculate the case for extended mass distribution: Lognormal
			return e_HMXB

'''
#use the bondi accretion to cross check with PBH accretion IGM
def Lnu_tot_IGM_bondi(M_c,f_pbh,z,col_frac,Distr=0,SigmaM=0.5,mu=1.22, O_m = 0.3153, O_b = 0.04930, h = 0.6736,n_m=50, er=0.057,re=[500,2e3]):
	n_gas_z= rhom(1 / (1 + z), O_m, h) *O_b/O_m/mu/PROTON # calculate the average density of the halo, here we add a boost factor to reach an overdensity of 1300
	v_tilda=c_s(z)  #PBH relative velocity in the virialized halo
	Q_t=0
	if Distr == 0: # monochromatic
		m_pbh=M_c
		Q_t += radbg(m_pbh, n_gas_z,v_tilda, er=er,re=re*(1+z),num=2)*n_PBH_IGM(M_c,m_pbh,f_pbh,col_frac,Distr=Distr) #*light_dist(lz[i],z_f)
		#m_pbh += bondi_halo(m_pbh, n_gas_z[i],v_tilda[i]) * (TZ(lz[i + 1]) - TZ(lz[i])) / YR
	elif Distr == 1:
		lmExt = np.logspace(np.log10(M_c)-1,np.log10(M_c)+1,n_m)
		for j in range(n_m-1):
			Q_t += radbg(lmExt[j],n_gas_z,v_tilda, er=er,re=re*(1+z),num=2)*n_PBH_IGM(M_c,lmExt[j],f_pbh,col_frac,Distr=Distr,sigM = SigmaM)*(lmExt[j+1]-lmExt[j])
	# calculate the case for extended mass distribution: Lognormal
	return Q_t
	
'''




def Lnu_tot_IGM(M_c,f_pbh,z, col_frac,Distr=0,SigmaM=0.5,mu=1.22, O_m = 0.3153, O_b = 0.04930, h = 0.6736,n_m=50, er=0.057,re=[500,2e3],PBH_halo = 0):
	Q_t=0
	if Distr == 0: # monochromatic
		m_pbh=M_c
		Q_t += radbg_IGM(m_pbh, z, er=er,re=re*(1+z),num=2,PBH_halo = PBH_halo)*n_PBH_IGM(M_c,m_pbh,f_pbh,col_frac,Distr = Distr)*(1+z)**3 #*light_dist(lz[i],z_f)
		#m_pbh += bondi_halo(m_pbh, n_gas_z[i],v_tilda[i]) * (TZ(lz[i + 1]) - TZ(lz[i])) / YR
	elif Distr == 1:
		lmExt = np.logspace(np.log10(M_c)-1,np.log10(M_c)+1,n_m)
		for j in range(n_m-1):
			Q_t += radbg_IGM(lmExt[j],z, er=er,re=re*(1+z),num=2,PBH_halo = PBH_halo)*n_PBH_IGM(M_c,lmExt[j],f_pbh,col_frac, Distr = Distr,sigM = SigmaM)*(lmExt[j+1]-lmExt[j])*(1+z)**3
	# calculate the case for extended mass distribution: Lognormal
	return Q_t


# test the case for the BH accretion spectra
if __name__=="__main__":
	m, n, v = 100, 10, 10
	fac1, fac2, fac3 = 1, 10, 100
	mdot1 = mdot_bondi(m, n/fac1, v)
	mdot2 = mdot_bondi(m, n/fac2, v)
	mdot3 = mdot_bondi(m, n/fac3, v)
	x1, x2 = 1, 1e5
	lnu = np.geomspace(x1, x2, 100)
	ltd1, em_td1 = radbg_SMBH(mdot1,m,v, lnu, er=0.057, fed=1.0,mu=1.22)
	ltd2, em_td2 = radbg_SMBH(mdot2,m,v, lnu, er=0.057, fed=1.0,mu=1.22)
	ltd3, em_td3 = radbg_SMBH(mdot3,m,v, lnu, er=0.057, fed=1.0,mu=1.22)
	ladaf1, em1 = ADAF(lnu, m, mdot1,mode=1)
	ladaf2, em2 = ADAF(lnu, m, mdot2,mode=1)
	ladaf3, em3 = ADAF(lnu, m, mdot3,mode=1)
	print('Maximum temperature: {:.2e},{:.2e},{:.2e}, {:.2e}, {:.2e} {:.2e} eV'.format(em_td1,em_td2,em_td3, em1, em2, em3))
	plt.figure()
	plt.loglog(lnu, ltd1, color = 'c',label='Thin disk')
	plt.loglog(lnu, ltd2, color = 'orange', label='Thin disk')
	plt.loglog(lnu, ltd3, color = 'grey', label='Thin disk')
	plt.loglog(lnu, ladaf1, '--', color = 'c', label=r'ADAF, $\dot{m}='+'{:.2e}'.format(mdot1)+'$')
	plt.loglog(lnu, ladaf2, '--', color = 'orange', label=r'ADAF, $\dot{m}='+'{:.2e}'.format(mdot2)+'$')
	plt.loglog(lnu, ladaf3, '--', color = 'grey', label=r'ADAF, $\dot{m}='+'{:.2e}'.format(mdot3)+'$')
	plt.xlabel(r'$h\nu\ [\rm eV]$')
	plt.ylabel(r'$L_{h\nu}\ [\rm erg\ s^{-1}\ eV^{-1}]$')
	plt.xlim(x1, x2)

	plt.ylim(1e20, 1e38)
	plt.legend()
	#plt.title(r'$m_{}={:.1f}\ {}$, ${}={:.1f}\ {}$'.format(r'{\rm PBH}',m,r'\rm M_{\odot}', r'\tilde{v}',v,r'\rm km\ s^{-1}'))
	plt.title(r'$m_{}={:.1f}\ {}$, $n_{}={:.1e}\ {}$, ${}={:.1f}\ {}$'.format(r'{\rm PBH}',m,r'\rm M_{\odot}', r'{\rm H}',n,r'\rm cm^{-3}',r'\tilde{v}',v,r'\rm km\ s^{-1}'), size=16)
	plt.tight_layout()
	plt.savefig('spectra_m='+str(m)+'_n='+str(n)+'.pdf')
	plt.close()



	m1, m2, m3 = 10, 100, 1000
	v = 10
	x1, x2 = 1, 1e5
	lnu = np.geomspace(x1, x2, 100)
	l_flat1 = np.zeros(100)
	l_flat2 = np.zeros(100)
	l_flat3 = np.zeros(100)
	ltd1, em_td1 = radbg_SMBH(1,m1,v, lnu, er=0.057, fed=1.0,mu=1.22)
	ltd2, em_td2 = radbg_SMBH(1,m2,v, lnu, er=0.057, fed=1.0,mu=1.22)
	ltd3, em_td3 = radbg_SMBH(1,m3,v, lnu, er=0.057, fed=1.0,mu=1.22)
	print('Maximum temperature: {:.2e},{:.2e},{:.2e} eV'.format(em_td1,em_td2,em_td3))

	L_Edd  = 6.7*1e-16 * Msun*SPEEDOFLIGHT**2 * 0.057
	for i in range(100):
		l_flat1[i] += L_Edd /lnu[i] *10
		l_flat2[i] += l_flat1[i] *10
		l_flat3[i] += l_flat2[i] *10
	plt.figure()
	plt.loglog(lnu, ltd1, color = 'c',label='Thin disk, ${:.1f}'.format(m1)+'M_{\odot}$')
	plt.loglog(lnu, ltd2, color = 'orange', label='Thin disk, ${:.1f}'.format(m2)+'M_{\odot}$')
	plt.loglog(lnu, ltd3, color = 'grey', label='Thin disk, ${:.1f}'.format(m3)+'M_{\odot}$')
	plt.loglog(lnu, l_flat1, '--', color = 'c', label=r'Flat, ${:.1f}'.format(m1)+'M_{\odot}$')
	plt.loglog(lnu, l_flat2, '--', color = 'orange', label=r'Flat, ${:.1f}'.format(m2)+'M_{\odot}$')
	plt.loglog(lnu, l_flat3, '--', color = 'grey', label=r'Flat, ${:.1f}'.format(m3)+'M_{\odot}$')
	plt.xlabel(r'$h\nu\ [\rm eV]$')
	plt.ylabel(r'$L_{h\nu}\ [\rm erg\ s^{-1}\ eV^{-1}]$')
	plt.xlim(x1, x2)

	plt.ylim(1e30, 1e42)
	plt.legend()
	#plt.title(r'$m_{}={:.1f}\ {}$, ${}={:.1f}\ {}$'.format(r'{\rm PBH}',m,r'\rm M_{\odot}', r'\tilde{v}',v,r'\rm km\ s^{-1}'))
	plt.title(r'$m_{}={:.1f}\ {}$, $n_{}={:.1e}\ {}$, ${}={:.1f}\ {}$'.format(r'{\rm PBH}',m,r'\rm M_{\odot}', r'{\rm H}',n,r'\rm cm^{-3}',r'\tilde{v}',v,r'\rm km\ s^{-1}'), size=16)
	plt.tight_layout()
	plt.savefig('spectra_Edd.pdf')
	plt.close()