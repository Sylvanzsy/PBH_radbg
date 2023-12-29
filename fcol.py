import matplotlib.pyplot as plt
from scipy import special
from cosmology import *
from txt import *
import matplotlib
from nhrat import *
from AccretSpec import *
#import mpl_toolkits.mplot3d
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import sys
import os
matplotlib.rcParams.update({'font.size': 14})
plt.style.use('tableau-colorblind10')
from scipy.optimize import curve_fit
from hmfunc import *
#from xraylw import *
from colossus.cosmology import cosmology
from colossus.lss import mass_function

# threshold masses for molecular and atomic cooling haloes
Mdown = lambda z: 1.54e5*((1+z)/31)**-2.074
Mup = lambda z: 7.75e6*((1+z)/31.)**-1.5

lls = ['-', '--', '-.', ':', (0,(10,5)), (0,(1,1,3)), (0,(5,1))]
llc = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

# choose a cosmology model
cosname = 'planck18'
cosmo = cosmology.setCosmology(cosname)
h = cosmo.H0/100
Om = cosmo.Om0
Ob = cosmo.Ob0
Or = cosmo.Or0
aeq = Or/Om

# power spectrum
k1, k2, kc = 1e-4, 1e5, 1
lk0 = np.geomspace(k1, kc*0.99, 1000)
lk1 = np.geomspace(kc, k2, 10000)
lk = np.hstack([lk0, lk1])
lPk0 = cosmo.matterPowerSpectrum(lk)

# mass threshold of Pop III star formation under external effects
# j21: LW intensity, vbc: strength of gas dark matter streaming motion
def mth_anna(j21, vbc=0.8):
	logM = 5.562*(1+0.279*j21**0.5)
	s = 0.614*(1-0.56*j21**0.5)
	logM += s*vbc
	return 10**logM

def Q_tot(lz,L_tot_z):
	n=len(lz)
	e_bh = np.zeros(n)
	u_tot=0
	for i in range(n - 1):
		delta_t = TZ(lz[i + 1]) - TZ(lz[i])*dz
		u_tot += (L_tot_z[i]/(1+lz[i])**4+L_tot_z[i+1]/(1+lz[i+1])**4)/2 * delta_t#*light_dist(lz[i],z_f)
		e_bh[i+1] += u_tot*(1+lz[i+1])**4
	return e_bh

# integrate the total energy per volume emitted as a function of redshift z
def IX(lz,Lnu_z,re,ne=8):
	I_z=0
	if re[0]==re[-1]:
		for i in range(len(lz)-1):
			I_z += (Lnu_z[i]/(1+lz[i])**3+Lnu_z[i+1]/(1+lz[i+1])**3)/2*(1+lz[-1])**3*SPEEDOFLIGHT/(4*np.pi)*(TZ(lz[i + 1]) - TZ(lz[i]))*dz*PLANCK/eV # covert from eV-1 to Hz-1
	else:
		dI_z = np.zeros(ne)
		for j in range(ne):
			for i in range(len(lz)- 1):
				dI_z[j] += (Lnu_z[j,i]/(1+lz[i])**3+Lnu_z[j,i+1]/(1+lz[i+1])**3)/2*SPEEDOFLIGHT /(4*np.pi)* (TZ(lz[i + 1]) - TZ(lz[i]))*dz/(180/np.pi)**2 # convert from sr^{-1} to deg^{-2}
		I_z = np.trapz(dI_z,x=re)
	return I_z

# typical LW background intensity produced by stars from Greif & Bromm 2006
j21_z = lambda z: 10**(2-z/5)



if __name__=="__main__":
	lf = [1e-4, 1e-3, 1e-2, 1e-1]#, 1e-5, 1e-6, 1e-8]
	
	# read mass threshold data (produced by virial.py)
	mth = np.array(retxt('mth_z.txt',6))
	mth_func = [interp1d(mth[0], mth[i]) for i in range(1,6)]

	mode = 0
	# mode=1: PS formalism that includes corrections for ellipsoidal dynamics
	# mode=0: original PS formalism
	mfac0 = 1 # normalization of the isocurvature mode from PBHs

	seed=0

	seedstr = ""
	if seedstr == 1:
		seedstr = "_PBHseed"
	# seed=0: only consider the Poisson effect
	# seed=1: include the seed effect (tentative)

	cut=1
	# cut=0: add the isocurvature term at all scales
	# cut=1: truncate the PBH perturbation at scales smaller than mpbh
		
	iso = 0 # =0: consider mode mixing, >0: only isocurvature mode from PBHs
	mix_flag = ""
	if iso > 0:
		mix_flag = "_nomix"



	delta0 = 1 # parameter controls over density factor
	delta_fac = ""
	if delta0 > 1:
		delta_fac = "_overden"


	seedSMBH = 0
	SMBHstr = ""
	if seedSMBH == 1:
		SMBHstr = "_SMBHseed"
	#include BH seeding: 1 or not: 0



	fmth = 0 
	# fmth=0: consider all haloes containing at least one PBH
	# with virial temperatures above 100 K 
	# fmth>0: only consider star forming haloes
	ext = 0 # =0: ignore external effects on halo mass threshold 
	# for star formation, >0: consider external effects
	
	# mass grid for HMF calculation
	lm0 = np.geomspace(1e3, 1e13, 100)
	# CDM power spectrum
	ps0 = [lk, lPk0]
	rep = './' # repository for plots
	if ext>0:
		rep = './ext/'
	if not os.path.exists(rep):
		os.makedirs(rep)
	
	fradbg = 0
	# fradbg=0: calculate radiation backgrounds under given assumptions
	# fradbg=1: make plots from existing radiation background data
	lab = 'mth' # label of the current model
	rmode=0
	# rmode=0: isothermal distribution of gas
	# rmode>0: uniform distribution of gas with a characteristic density of
	# rmode * average gas density in the halo
	
	# existing models 
	# (with mpbh=33, lf = [1e-4, 1e-3, 1e-2, 1e-1], rmode=0, seed=0, cut=1): 
	# no label: iso=fmth=ext=0
	# lab='iso': iso=1, fmth=ext=0
	# lab='mth': iso=fmth=1, ext=0
	# lab='ext': iso=fmth=ext=1
	
	dnu = (13.6-11.2)*eV/PLANCK # freqeuncy interval of LW band
	exray = np.array([5e2, 2e3]) # photon energy range of X-rays 
	lfpbh = [1e-4, 1e-3, 1e-2, 1e-1] # grid of PBH fraction
	nz0 = 5 # number of redshift bins for the integration of LW intensity
	z1, z2 = 4, 100 # redshift range, now try 1000
	nz = int(z2-z1) # number of redshift bins for radiation backgrounds
	lz = np.linspace(z2, z1, nz+1)
	dz = (z2-z1)/nz
	z1_IGM, z2_IGM = 4, 1000 # redshift range, now try 1000
	nz_IGM = int(z2_IGM-z1_IGM) # number of redshift bins for radiation backgrounds
	lz_IGM = np.linspace(z2_IGM, z1_IGM, nz_IGM+1)
	dz_IGM = (z2_IGM-z1_IGM)/nz_IGM
	data0 = []
	data1 = []
	data2 = []

	# main loop starts
	Mc = 10 # PBH mass
	f_pbh = 1e-3
	PsiM = 0 # initialize distribution funtion 0: monochromatic 1:lognormal
	Sigma_M = 0.5
	O_m = 0.3089
	O_b = 0.048
	h = 0.6774
	E_ion = [13.6]
	E_lyman = [10.2]
	alpha_s = 0.5
	E_x = [500, 2e3]
	E_x_hard = [2e3, 1e4]
	IMF_BH1='Salpter'
	IMF_BH2='TopHeavy'
	SFRD_fit1='Madau14'
	SFRD_fit2='Harikane22'
	hubble = 0.6774


	TauH = 0
	# TauH = 0: suppose all photons escape
	# TauH = 1: consider the effect of absorption by neutral hydrogen


	PBH_halo = 1
	# PBH_halo = 0: All PBHs in IGM are not contained in halos
	# PBH_halo = 1: DM Halo clothing

	PBH_flag = ""
	if PBH_halo == 1:
		PBH_flag = "_PBHHalo"



	#need to change the threshold of PBH
	m0 = [max(M_Tvir(1e2, z), Mc / f_pbh * Om / (Om - Ob), 3 * Mc * (1000 / (1 + z))) for z in lz]
	m0_IGM = [max(M_Tvir(1e2, z), Mc / f_pbh * Om / (Om - Ob), 3 * Mc * (1000 / (1 + z))) for z in lz_IGM]
	print(m0_IGM)
	#m0 = [max(M_Tvir(1e2, z), 3 * Mc * (1000 / (1 + z))) for z in lz_IGM]
	#m0 = [3*Mc*(1000/(1+z))for z in lz_IGM]
	halo_handle = "_1e13"
	#halo_handle = "_PBHMixed"
	#halo_handle = "_PBHhalo"

	Xray_mode = 1
	if Xray_mode == 0:
		lE = np.geomspace(*E_x,8)
		Xray_freq = '05_2'
	elif Xray_mode == 1:
		lE = np.geomspace(*E_x_hard,8)
		Xray_freq = '2_10'


	rad_mode = 0
	if rad_mode == 0:
		E_ion = [13.6]
		rad_freq = 'uv'
	elif rad_mode == 1:
		E_ion = [10.2]
		rad_freq = 'alpha'

	#R_min: log power of minimum radius inside halo
	R_min = -3
	R_flag = ""
	if R_min > -3:
		R_flag = "_Router"
	elif R_min < -3:
		R_flag = "_Rinner"


	imf = 'Salpter'
	# imf='TopHeavy'
	f_HMXB_PopIII = 0.3
	f_HMXB_PopI_II = 0.5
	rhoPBH = f_pbh * rhom(1, O_m, h) * (O_m - O_b) / O_m * MPC ** 3 / Msun
	f_star = 0.1
	f_cool = 0.01

	M_0_max = 1e16
	M_max = [M_z(M_0_max, z) for z in lz]

	ps3 = PBH(ps0, Mc, f_pbh, aeq, h, mfac=mfac0, iso=iso, seed=seed, cut=cut)
	fcol_PBH = np.zeros(nz_IGM+1)




	I_z0_IGM =  np.zeros(nz_IGM + 1)
	I_z0_Halo = np.zeros(nz+1)
	I_z0_ABH = np.zeros(nz+1)
	I_z0_ABH_1 = np.zeros(nz+1)
	I_z0_ABH_2 = np.zeros(nz+1)
	I_z0_HMXB = np.zeros(nz+1)
	I_z0_HMXB_1 = np.zeros(nz+1)
	I_z0_HMXB_2 = np.zeros(nz+1)
	I_z0_SMBH = np.zeros(nz+1)
	I_z0_SMBH_1 = np.zeros(nz+1)
	IX_z0_IGM =  np.zeros(nz_IGM + 1)
	IX_z0_Halo = np.zeros(nz+1)
	IX_z0_ABH = np.zeros(nz+1)
	IX_z0_ABH_1 = np.zeros(nz+1)
	IX_z0_ABH_2 = np.zeros(nz+1)
	IX_z0_HMXB = np.zeros(nz+1)
	IX_z0_HMXB_1 = np.zeros(nz+1)
	IX_z0_HMXB_2 = np.zeros(nz+1)
	IX_z0_SMBH = np.zeros(nz+1)
	IX_z0_SMBH_1 = np.zeros(nz+1)
	# save array of BHs
	M_save_pop12 =np.zeros([30,nz+1])
	M_save_pop12_1 = np.zeros([30,nz+1])
	M_save_pop12_2 = np.zeros([30,nz+1])
	M_save_pop3 =np.zeros([30,nz+1])
	M_save_pop3_1 = np.zeros([30,nz+1])
	M_save_pop3_2 = np.zeros([30,nz+1])
	lm_ABH = np.zeros([30,nz+1])
	lm_PBH = np.zeros([30,nz+1])
	lmh_SMBH = np.zeros([30,nz+1])
	M_SMBH = np.zeros([30,nz+1])# mass of SMBH in halo
	lm_SMBH = np.zeros([30,nz+1])# mass of SMBH in halo
	M_XB_pop12 = np.zeros([30,nz+1])# total mass of HMXBs in halo
	M_XB_pop12_1 = np.zeros([30,nz+1])# total mass of HMXBs in halo(assume classic)
	M_XB_pop12_2 = np.zeros([30,nz+1])# total mass of HMXBs in halo(Universe Machine)
	M_XB_pop3 = np.zeros([30,nz+1])# total mass of HMXBs in halo
	M_XB_pop3_1 = np.zeros([30,nz+1])# total mass of HMXBs in halo(assume classic)
	M_XB_pop3_2 = np.zeros([30,nz+1])# total mass of HMXBs in halo(Universe Machine)
	lnh_PBH = np.zeros([30,nz+1])
	lnh_ABH = np.zeros([30,nz+1])
	lnh_SMBH = np.zeros([30,nz+1])
	
	for i in range(nz_IGM+1):
		z = lz_IGM[i]
		print(z)
		fcol_PBH[i] = special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(m0_IGM[i], ps3)) * np.sqrt(2)))
		print('stage 1')
		
	for i in range(nz+1):
		z = lz[i]
		print(z)
		M_h_max = 1e13
		m_ABH = max(1.54*1e5*(31/(1+z))**2.074, m0[i])
		#if m0 >=  M_max[i]:
		lm = np.geomspace(m0[i] , M_h_max , 30)
		#print('stage 1')
		if m_ABH >= M_max[i]:
		#print('stage 2'))
			lm1 = np.zeros(30)
			lm2 = np.zeros(30)
			for j in range(30):
				lm_PBH[j,i]+= lm[j]
		elif M_max[i]<=1e8:
			print('stage 3')
			lm = np.geomspace(m0[i] , M_h_max , 30)
			lm1 = np.geomspace(m_ABH ,M_max[i],30)
			lm2 = np.zeros(30)
			for j in range(30):
				lm_PBH[j,i]+= lm[j]
				lm_ABH[j,i]+= lm1[j]
		else:
			print('stage 4')
			lm = np.geomspace(m0[i] , M_max[i] , 30) # store the temperary list for halo mass for PBHs
			lm1 = np.geomspace(m_ABH ,M_max[i],30)# store the temperary list for halo mass for ABHs
			lm2 = np.geomspace(1e8, M_max[i], 30)# store the temperary list for halo mass for SMBHs
			for j in range(30):
				lm_PBH[j,i]+= lm[j]
				lm_ABH[j,i]+= lm1[j]
				lmh_SMBH[j,i]+=lm2[j]
		print('Max halo mass: {:.2e}'.format(M_max[i]))
		lm_ABH2 = np.geomspace(m_ABH ,M_0_max,30)
		M_BH_z = np.zeros(4)# total mass of BHs in halo
		M_BH_z_1 = np.zeros(4)# total mass of BHs in halo(assume classic)
		M_BH_z_2 = np.zeros(5)# total mass of BHs in halo(Universe Machine)
		hmf3_PBH = halomassfunc(lm, z, ps3, wf3, gf3, mode=mode)
		hmf3_ABH = halomassfunc(lm1, z, ps3, wf3, gf3, mode=mode)
		hmf3_SMBH = halomassfunc(lm2, z, ps3, wf3, gf3, mode=mode)
		# here we write the halo mass function in proper units
		lnh1 = np.array([hmf3_PBH(m) for m in lm]) *(1+z)**3*h ** 3 / MPC ** 3
		lnh2 = np.array([hmf3_ABH(m) for m in lm1]) *(1+z)**3*h**3/MPC**3
		lnh3 = np.array([hmf3_SMBH(m) for m in lm2]) *(1+z)**3 *h**3/MPC**3
		M_star_SMBH=[M_star_Mh(M,z) for M in lm2]
		lm_S =np.array([SMBHmass(Mh,z,seeding = seedSMBH) for Mh in lm2]) #optimistic value of SMBHs
		#print(lm_S)
		for j in range(30):
			lnh_PBH[j,i] += lnh1[j]
			lnh_ABH[j,i] += lnh2[j]
			lnh_SMBH[j,i] += lnh3[j]
			M_SMBH[j,i] += M_SMBH_Mstar(M_star_SMBH[j],z) #return the mass of SMBH for given halo mass at redshift z
			lm_SMBH[j,i] += lm_S[j]
			M_BH_z = M_BH(lm1[j],z, IMF='TopHeavy',fit='Madau14',output=1)
			M_BH_z_1 = M_BH(lm1[j],z, IMF='TopHeavy',fit='Harikane22',output=1)
			#dM_BH_1 = np.array([n_StellarBH(M_star_tot_1[j],mbh,IMF=imf) for mbh in lmExt])
			M_BH_z_2 = M_BH_UNIV(lm_ABH2[j],z, IMF='TopHeavy',mode=1,output=1)
			M_XB_pop12[j,i] = M_BH_z[3]*f_HMXB_PopI_II
			M_XB_pop12_1[j,i] = M_BH_z_1[3]*f_HMXB_PopI_II
			M_XB_pop12_2[j,i] = M_BH_z_2[3]*f_HMXB_PopI_II
			M_XB_pop3[j,i] = M_BH_z[2]*f_HMXB_PopIII
			M_XB_pop3_1[j,i] = M_BH_z_1[2]*f_HMXB_PopIII
			M_XB_pop3_2[j,i] = M_BH_z_2[4]*f_HMXB_PopIII
			M_save_pop12[j,i]= M_BH_z[1]
			M_save_pop12_1[j,i]= M_BH_z_1[1]
			M_save_pop12_2[j,i]= M_BH_z_2[0]
			M_save_pop3[j,i]= M_BH_z[0]
			M_save_pop3_1[j,i]= M_BH_z_1[0]
			M_save_pop3_2[j,i]= M_BH_z_2[1]
			#dM_BH_1 = np.array([n_StellarBH(M_star_tot_1[j],mbh,IMF=imf) for mbh in lmExt])
		#print('$z = $ ' + str(z))
		#print(M_SMBH)




'''

if __name__ == "__main__":
	
	for i in range(nz_IGM + 1):
		if i>0:
			lz0 = lz_IGM[0:i]
			Lnu_z0_IGM = np.zeros(i+1)
			for j in range(len(lz0)):
				print('Source Redshift:')
				print(lz0[j])
				z=lz0[j]
				Lnu_z0_IGM[j] +=  Lnu_tot_IGM(Mc,f_pbh,z, fcol_PBH[j],Distr=PsiM,SigmaM=Sigma_M,re=E_ion/(1+lz0[-1]), PBH_halo= PBH_halo)
			I_z0_IGM[i] += IX(lz0,Lnu_z0_IGM,E_ion)
			#print(I_z0_IGM))
		print("Done with IGM case!")
		
	for i in range(nz + 1):
		if i>0:
			lz0 = lz[0:i]
			Lnu_z0_IGM = np.zeros(i+1)
			Lnu_z0_SMBH = np.zeros(i+1)
			Lnu_z0_SMBH_1 = np.zeros(i+1)
			Lnu_z0_ABH = np.zeros(i+1)
			Lnu_z0_ABH_1 = np.zeros(i+1)
			Lnu_z0_ABH_2 = np.zeros(i+1)
			Lnu_z0_HMXB = np.zeros(i+1)
			Lnu_z0_HMXB_1 = np.zeros(i+1)
			Lnu_z0_HMXB_2 = np.zeros(i+1)
			Lnu_z0_Halo = np.zeros(i+1)
			for j in range(len(lz0)):
				print('Source Redshift:')
				print(lz0[j])
				z=lz0[j]
				Lnu_z0_SMBH[j] +=  Lnu_tot_SMBH(z, lm_SMBH[:,j],lmh_SMBH[:,j],lnh_SMBH[:,j], delta0 = delta0, re=E_ion/(1+lz0[-1]),TauH=TauH)
				Lnu_z0_SMBH_1[j] +=  Lnu_tot_SMBH(z, M_SMBH[:,j],lmh_SMBH[:,j],lnh_SMBH[:,j], delta0 = delta0, re=E_ion/(1+lz0[-1]),TauH=TauH)
				Lnu_z0_ABH[j] +=  Lnu_tot_Halo2(z,M_save_pop3[:,j],M_save_pop12[:,j],lm_ABH[:,j],lnh_ABH[:,j], 
				Pop3_IMF  = IMF_BH2, delta0 = delta0, re=E_ion/(1+lz0[-1]),TauH=TauH, r_min = R_min)
				Lnu_z0_ABH_1[j] +=  Lnu_tot_Halo2(z,M_save_pop3_2[:,j],M_save_pop12_1[:,j],lm_ABH[:,j],lnh_ABH[:,j],
				 Pop3_IMF  = IMF_BH2, delta0 = delta0, re=E_ion/(1+lz0[-1]),TauH=TauH, r_min = R_min)
				Lnu_z0_ABH_2[j] +=  Lnu_tot_Halo2(z,M_save_pop3_1[:,j],M_save_pop12_2[:,j],lm_ABH[:,j],lnh_ABH[:,j], 
				Pop3_IMF  = IMF_BH2, delta0 = delta0, re=E_ion/(1+lz0[-1]),TauH=TauH, r_min = R_min)
				Lnu_z0_HMXB[j] +=  Lnu_tot_Halo2(z,M_XB_pop3[:,j],M_XB_pop12[:,j],lm_ABH[:,j],lnh_ABH[:,j], Pop3_IMF  = IMF_BH2, delta0 = delta0,HMXB_flag= 1, re=E_ion/(1+lz0[-1]),TauH=TauH)
				Lnu_z0_HMXB_1[j] +=  Lnu_tot_Halo2(z,M_XB_pop3_1[:,j],M_XB_pop12_1[:,j],lm_ABH[:,j],lnh_ABH[:,j], Pop3_IMF  = IMF_BH2, delta0 = delta0,HMXB_flag= 1, re=E_ion/(1+lz0[-1]),TauH=TauH)
				Lnu_z0_HMXB_2[j] +=  Lnu_tot_Halo2(z,M_XB_pop3_2[:,j],M_XB_pop12_2[:,j],lm_ABH[:,j],lnh_ABH[:,j], Pop3_IMF  = IMF_BH2, delta0 = delta0,HMXB_flag= 1, re=E_ion/(1+lz0[-1]),TauH=TauH)
				#print('Input parameters')
				#print(lm_PBH[:,j])
				#print(lnh_PBH[:,j])
				Lnu_z0_Halo[j] +=  Lnu_tot_Halo1(Mc,f_pbh,z,lm_PBH[:,j], lnh_PBH[:,j], 
				Distr=PsiM,SigmaM=Sigma_M,delta0=delta0,re=E_ion/(1+lz0[-1]),TauH=TauH, r_min = R_min)
				#print('Output energy')
				#print(Lnu_z0_Halo[j])
			I_z0_Halo[i] += IX(lz0,Lnu_z0_Halo,E_ion)
			I_z0_ABH[i] += IX(lz0,Lnu_z0_ABH,E_ion)
			I_z0_ABH_1[i] += IX(lz0,Lnu_z0_ABH_1,E_ion)
			I_z0_ABH_2[i] += IX(lz0,Lnu_z0_ABH_2,E_ion)
			I_z0_HMXB[i] += IX(lz0,Lnu_z0_HMXB,E_ion)
			I_z0_HMXB_1[i] += IX(lz0,Lnu_z0_HMXB_1,E_ion)
			I_z0_HMXB_2[i] += IX(lz0,Lnu_z0_HMXB_2,E_ion)
			I_z0_SMBH[i] += IX(lz0,Lnu_z0_SMBH,E_ion)
			I_z0_SMBH_1[i] += IX(lz0,Lnu_z0_SMBH_1,E_ion)
			print('Radiation level at z='+str(lz[i]))
			print(I_z0_Halo[i])
			print(I_z0_ABH[i])
			print(I_z0_SMBH_1[i])



	rep_data = './RadFeedBack/'
	totxt(rep_data + 'PBH_J'+rad_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_IGM.txt', list([I_z0_IGM]), 0, 0, 0)
	totxt(rep_data + 'PBH_J'+rad_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_halo.txt', list([I_z0_Halo]), 0, 0, 0)
	totxt(rep_data + 'ABH_J'+rad_freq+'_'+str(Mc)+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', list([I_z0_ABH,I_z0_ABH_1,I_z0_ABH_2]), 0, 0, 0)
	totxt(rep_data + 'HMXB_J'+rad_freq+'_'+str(Mc)+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', list([I_z0_HMXB,I_z0_HMXB_1,I_z0_HMXB_2]), 0, 0, 0)
	totxt(rep_data + 'SMBH_J'+rad_freq+'_'+str(Mc)+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', list([I_z0_SMBH,I_z0_SMBH_1]), 0, 0, 0)




if __name__ == "__main__":

	rho_rad = 1e-21
	I_0_ion=np.zeros(nz+1)
	I_0_ion_1=np.zeros(nz+1)
	I_0=np.zeros(nz+1)
	I_0_1=np.zeros(nz+1)
	for i in range(nz+1):
		I_0_ion[i]+= rho_rad*((1+lz[i])/6)**1.5
		I_0_ion_1[i] = I_0_ion[i]/10
	#minimum to re-ionize IGM, from Madau et al 1999
		I_0[i]+= 9e-23*(1+lz[i])*10.2/13.6
		I_0_1[i] = I_0[i]/10
		#minimum to generate 21cm-line

	rep_data = './RadFeedBack/'
	I_PBH_IGM_z = retxt(rep_data + 'PBH_J'+rad_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_IGM.txt', 1)
	I_PBH_z = retxt(rep_data + 'PBH_J'+rad_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_halo.txt', 1)
	I_ABH_z = retxt(rep_data + 'ABH_J'+rad_freq+'_'+str(Mc)+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', 3)
	I_HMXB_z = retxt(rep_data + 'HMXB_J'+rad_freq+'_'+str(Mc)+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt',3)
	I_SMBH_z = retxt(rep_data + 'SMBH_J'+rad_freq+'_'+str(Mc)+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', 2)
	

	max_ABH = np.zeros(nz+1)
	min_ABH = np.zeros(nz+1)
	max_HMXB = np.zeros(nz+1)
	min_HMXB = np.zeros(nz+1)
	for i in range(nz+1):
		max_ABH[i] += max(I_ABH_z[0][i], I_ABH_z[1][i], I_ABH_z[2][i])
		min_ABH[i] += min(I_ABH_z[0][i], I_ABH_z[1][i], I_ABH_z[2][i])
		max_HMXB[i] += max(I_HMXB_z[0][i], I_HMXB_z[1][i], I_HMXB_z[2][i]) * 100
		min_HMXB[i] += min(I_HMXB_z[0][i], I_HMXB_z[1][i], I_HMXB_z[2][i])
	
	
	plt.figure()
	fig, ax = plt.subplots()

	ax.plot(lz, I_0_ion,'-.',linewidth=4 ,color= 'black', label=r'Reionization Limit')
	ax.plot(lz, I_0,':',linewidth=4 ,color= 'black', label=r'Ly$\alpha$ Limit')

	ax.plot(lz, I_PBH_z[0],linewidth=2,color= 'g',label='PBH Halo')
	ax.plot(lz_IGM, I_PBH_IGM_z[0],linewidth=2,color= 'b',label='PBH IGM')

	#ax.plot(lz, I_ABH_z[0],linewidth=2,color= 'r',label='SRBH')
	#ax.plot(lz, I_ABH_z[1],linewidth=2,color= 'r')
	#ax.plot(lz, I_ABH_z[2],'-.',linewidth=2,color= 'r')
	#ax.plot(lz, I_HMXB_z[0],linewidth=2,color= 'cyan',label='HMXB')
	#ax.plot(lz, I_HMXB_z[1],linewidth=2,color= 'cyan')
	#ax.plot(lz, I_HMXB_z[2],'-.',linewidth=2,color= 'cyan')
	
	ax.plot(lz, max_ABH,linewidth=1,color= 'r',label='SRBH')
	ax.plot(lz, min_ABH,'--',linewidth=1,color= 'r')
	ax.plot(lz, max_HMXB,linewidth=1,color= 'cyan',label='HMXB')
	ax.plot(lz, min_HMXB,'--',linewidth=1,color= 'cyan')
	ax.plot(lz, I_SMBH_z[0],linewidth=2,color= 'y',label='SMBH')
	ax.plot(lz, I_SMBH_z[1],'-.',linewidth=2,color= 'y')


	ax.fill_between(lz, I_0,I_0_1 ,facecolor= 'black', alpha=0.5)
	ax.fill_between(lz, I_0_ion,I_0_ion_1 ,facecolor= 'grey', alpha=0.5)
	ax.axvspan(15, 20, alpha=0.5, color='tab:purple')
	ax.axvspan(4, 6, alpha=0.5, color='chocolate')
	#ax.fill_between(lz, I_ABH_z[0], I_ABH_z[2],facecolor='red',alpha = 0.5)
	ax.fill_between(lz, max_ABH, min_ABH,facecolor='red',alpha = 0.5)
	ax.fill_between(lz, max_HMXB, min_HMXB,facecolor='cyan', alpha=0.5)
	ax.fill_between(lz, I_SMBH_z[0], I_SMBH_z[1],facecolor='yellow', alpha=0.5)
	#ax.fill_between(lz, I_HMXB_z[0], I_HMXB_z[2],facecolor='cyan', alpha=0.5)




	# plt.plot(lz2,rho_rad, linewidth=2, label='Radiation Density')


	ax.legend(fontsize=12,loc= 'lower right')
	ax.set_xlabel(r'$z$',fontsize=15)
	ax.set_ylabel(r'$J_{\mathrm{uv}}(z)[\mathrm{ erg\ s^{-1} cm^{-2} Hz^{-1} sr^{-1}}] $',fontsize=15)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(z1,z2)
	ax.set_ylim(1e-30, 1e-18)

	ax.tick_params(which='major', labelcolor='black', labelsize=12, width=3)

	plt.savefig('J_'+rad_freq+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_topheavy.pdf')
	plt.close()
'''
'''

if __name__=="__main__":


	for i in range(nz_IGM + 1):
		if i>0:
			lz0 = lz_IGM[0:i]
			LX_z0_IGM = np.zeros([8,i])
			for j in range(len(lz0)):
				#print('Source Redshift:')
				#print(lz0[j])
				z=lz0[j]
				for k in range(len(lE)):
					LX_z0_IGM[k, j] +=   Lnu_tot_IGM(Mc,f_pbh,z, fcol_PBH[j],Distr=PsiM,SigmaM=Sigma_M,re=lE[k], PBH_halo= PBH_halo)
			IX_z0_IGM[i] += IX(lz0,LX_z0_IGM,lE)
			#print(I_z0_IGM))

	print("IGM part complete!")
			
	for i in range(nz + 1):
		if i>0:
			lz0 = lz[0:i]
			LX_z0_SMBH = np.zeros([8,i])
			LX_z0_SMBH_1 = np.zeros([8,i+1])
			LX_z0_ABH = np.zeros([8,i+1])
			LX_z0_ABH_1 = np.zeros([8,i+1])
			LX_z0_ABH_2 = np.zeros([8,i+1])
			LX_z0_HMXB = np.zeros([8,i+1])
			LX_z0_HMXB_1 = np.zeros([8,i+1])
			LX_z0_HMXB_2 = np.zeros([8,i+1])
			LX_z0_Halo = np.zeros([8,i+1])
			for j in range(len(lz0)):
				print('Source Redshift:')
				print(lz0[j])
				z=lz0[j]
				for k in range(len(lE)):
					LX_z0_SMBH[k,j] += Lnu_tot_SMBH(z, lm_SMBH[:,j],lmh_SMBH[:,j],lnh_SMBH[:,j], delta0 = delta0, re=lE[k],TauH=TauH)
					LX_z0_SMBH_1[k,j] +=  Lnu_tot_SMBH(z, M_SMBH[:,j],lmh_SMBH[:,j],lnh_SMBH[:,j], delta0 = delta0, re=lE[k],TauH=TauH)
					LX_z0_ABH[k,j] += Lnu_tot_Halo2(z,M_save_pop3[:,j],M_save_pop12[:,j],lm_ABH[:,j],lnh_ABH[:,j],
													Pop3_IMF  = IMF_BH2, delta0 = delta0, re=lE[k],TauH=TauH, r_min = R_min)
					LX_z0_ABH_1[k,j] += Lnu_tot_Halo2(z,M_save_pop3_2[:,j],M_save_pop12_1[:,j],lm_ABH[:,j],lnh_ABH[:,j],
													  Pop3_IMF  = IMF_BH2, delta0 = delta0, re=lE[k],TauH=TauH, r_min = R_min)
					LX_z0_ABH_2[k,j] +=  Lnu_tot_Halo2(z,M_save_pop3_1[:,j],M_save_pop12_2[:,j],lm_ABH[:,j],lnh_ABH[:,j],
													   Pop3_IMF  = IMF_BH2, delta0 = delta0, re=lE[k],TauH=TauH, r_min = R_min)
					LX_z0_HMXB[k,j] += Lnu_tot_Halo2(z,M_XB_pop3[:,j],M_XB_pop12[:,j],lm_ABH[:,j],lnh_ABH[:,j], Pop3_IMF  = IMF_BH2, delta0 = delta0,HMXB_flag= 1, re=lE[k],TauH=TauH)
					LX_z0_HMXB_1[k,j] +=  Lnu_tot_Halo2(z,M_XB_pop3_1[:,j],M_XB_pop12_1[:,j],lm_ABH[:,j],lnh_ABH[:,j], Pop3_IMF  = IMF_BH2, delta0 = delta0,HMXB_flag= 1, re=lE[k],TauH=TauH)
					LX_z0_HMXB_2[k,j] +=  Lnu_tot_Halo2(z,M_XB_pop3_2[:,j],M_XB_pop12_2[:,j],lm_ABH[:,j],lnh_ABH[:,j], Pop3_IMF  = IMF_BH2, delta0 = delta0,HMXB_flag= 1, re=lE[k],TauH=TauH)
					LX_z0_Halo[k,j] +=  Lnu_tot_Halo1(Mc,f_pbh,z,lm_PBH[:,j], lnh_PBH[:,j],
													  Distr=PsiM,SigmaM=Sigma_M,delta0=delta0,re=lE[k],TauH=TauH, r_min = R_min)
			IX_z0_Halo[i] += IX(lz0,LX_z0_Halo,lE)
			IX_z0_ABH[i] += IX(lz0,LX_z0_ABH,lE)
			IX_z0_ABH_1[i] += IX(lz0,LX_z0_ABH_1,lE)
			IX_z0_ABH_2[i] += IX(lz0,LX_z0_ABH_2,lE)
			IX_z0_HMXB[i] += IX(lz0,LX_z0_HMXB,lE)
			IX_z0_HMXB_1[i] += IX(lz0,LX_z0_HMXB_1,lE)
			IX_z0_HMXB_2[i] += IX(lz0,LX_z0_HMXB_2,lE)
			IX_z0_SMBH[i] += IX(lz0,LX_z0_SMBH,lE)
			IX_z0_SMBH_1[i] += IX(lz0,LX_z0_SMBH_1,lE)
			print('Radiation level at z='+str(lz[i]))
			print(IX_z0_Halo[i])
			print(IX_z0_ABH[i])
			print(IX_z0_SMBH_1[i])

	rep_data = './RadFeedBack/'
	totxt(rep_data + 'PBH_I'+Xray_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+SMBHstr+halo_handle+R_flag+PBH_flag+'_IGM.txt', list([IX_z0_IGM]), 0, 0, 0)
	totxt(rep_data + 'PBH_I'+Xray_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+SMBHstr+halo_handle+R_flag+PBH_flag+'_halo.txt', list([IX_z0_Halo]), 0, 0, 0)
	totxt(rep_data + 'ABH_I'+Xray_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+SMBHstr+halo_handle+R_flag+PBH_flag+'.txt', list([IX_z0_ABH,IX_z0_ABH_1,IX_z0_ABH_2]), 0, 0, 0)
	totxt(rep_data + 'HMXB_I'+Xray_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+SMBHstr+halo_handle+R_flag+PBH_flag+'.txt', list([IX_z0_HMXB,IX_z0_HMXB_1,IX_z0_HMXB_2]), 0, 0, 0)
	totxt(rep_data + 'SMBH_I'+Xray_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+SMBHstr+halo_handle+R_flag+PBH_flag+'.txt', list([IX_z0_SMBH,IX_z0_SMBH_1]), 0, 0, 0)

'''
'''
if __name__=="__main__":

	plt.figure()
	fig, ax = plt.subplots()

	Ix_0=np.zeros(nz+1)
	Ix_0_1=np.zeros(nz+1)
	Ix_soft = 2.90*1e-12
	Ix_hard = 6.47*1e-12

	if Xray_mode == 0:
		for i in range(nz+1):
			Ix_0[i]+= Ix_soft
			Ix_0_1[i]+= Ix_soft*1e4
	#soft Xray Background
	elif Xray_mode == 1:
		for i in range(nz+1):
			Ix_0[i]+= Ix_hard
			Ix_0_1[i]+= Ix_hard*1e4
	#hard Xray Background


	
	rep_data = './RadFeedBack/'
	IX_PBH_IGM_z = retxt(rep_data + 'PBH_I'+Xray_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+SMBHstr+halo_handle+R_flag+PBH_flag+'_IGM.txt', 1)
	IX_PBH_z = retxt(rep_data + 'PBH_I'+Xray_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+SMBHstr+halo_handle+R_flag+PBH_flag+'_halo.txt', 1)
	IX_ABH_z = retxt(rep_data + 'ABH_I'+Xray_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+SMBHstr+halo_handle+R_flag+PBH_flag+'.txt', 3)
	IX_HMXB_z = retxt(rep_data + 'HMXB_I'+Xray_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+SMBHstr+halo_handle+R_flag+PBH_flag+'.txt', 3)
	IX_SMBH_z = retxt(rep_data + 'SMBH_I'+Xray_freq+'_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+SMBHstr+halo_handle+R_flag+PBH_flag+'.txt', 2)


	max_ABH = np.zeros(nz+1)
	min_ABH = np.zeros(nz+1)
	max_HMXB = np.zeros(nz+1)
	min_HMXB = np.zeros(nz+1)
	#for i in range(nz_IGM + 1):
	#	IX_PBH_IGM_z[0][i]*= 1e6 #might not need if it is accreting in Halo
	for i in range(nz+1):
		max_ABH[i] += max(IX_ABH_z[0][i], IX_ABH_z[1][i], IX_ABH_z[2][i])
		min_ABH[i] += min(IX_ABH_z[0][i], IX_ABH_z[1][i], IX_ABH_z[2][i])
		max_HMXB[i] += max(IX_HMXB_z[0][i], IX_HMXB_z[1][i], IX_HMXB_z[2][i]) * 100 #to account for full duty cycle
		min_HMXB[i] += min(IX_HMXB_z[0][i], IX_HMXB_z[1][i], IX_HMXB_z[2][i])


	ax.plot(lz, Ix_0,'-.',linewidth=4 ,color= 'black', label='CXB limit')

	ax.plot(lz, IX_PBH_z[0],linewidth=2,color= 'g',label='PBH Halo')
	ax.plot(lz_IGM, IX_PBH_IGM_z[0],linewidth=2,color= 'b',label=' PBH IGM') #$10^6 $

	#ax.plot(lz, IX_HMXB_z[1],linewidth=2,color= 'cyan')
	#ax.plot(lz, IX_ABH_z[1],linewidth=2,color= 'r')
	ax.plot(lz, max_ABH,linewidth=1,color= 'r',label='SRBH')
	ax.plot(lz, min_ABH,'--',linewidth=1,color= 'r')
	ax.plot(lz, max_HMXB,linewidth=1,color= 'cyan',label='HMXB')
	ax.plot(lz, min_HMXB,'--',linewidth=1,color= 'cyan')
	#ax.plot(lz, IX_ABH_z[0],linewidth=2,color= 'r',label='SRBH')
	#ax.plot(lz, IX_ABH_z[2],'-.',linewidth=2,color= 'r')
	#ax.plot(lz, IX_HMXB_z[0],linewidth=2,color= 'cyan',label='HMXB')
	#ax.plot(lz, IX_HMXB_z[2],'-.',linewidth=2,color= 'cyan')
	ax.plot(lz, IX_SMBH_z[0],linewidth=2,color= 'y',label='SMBH')
	ax.plot(lz, IX_SMBH_z[1],'-.',linewidth=2,color= 'y')


	ax.fill_between(lz, max_ABH, min_ABH,facecolor='red',alpha = 0.5)
	ax.fill_between(lz, max_HMXB, min_HMXB,facecolor='cyan', alpha=0.5)
	ax.fill_between(lz, Ix_0, Ix_0_1,facecolor='black',alpha = 0.3)
	#ax.fill_between(lz, IX_ABH_z[0], IX_ABH_z[2],facecolor='red',alpha = 0.5)
	ax.fill_between(lz, IX_SMBH_z[0], IX_SMBH_z[1],facecolor='yellow', alpha=0.5)
	#ax.fill_between(lz, IX_HMXB_z[0], IX_HMXB_z[2],facecolor='cyan', alpha=0.5)

	# plt.plot(lz2,rho_rad, linewidth=2, label='Radiation Density')


	ax.legend(fontsize=12,loc= 'lower right')
	ax.set_xlabel(r'$z$',fontsize=15)
	ax.set_ylabel(r'$I_{2-10\mathrm{keV}}[\mathrm{erg \ s^{-1} cm^{-2}  deg^{-2}}] $',fontsize=15)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(6,z2)
	ax.set_ylim(1e-20, 1e-9)

	ax.set_title('X-ray Background $I_{2-10\mathrm{keV}}$',fontsize=15)
	ax.tick_params(which='major', labelcolor='black', labelsize=12, width=3)

	plt.savefig('Intensity_I'+Xray_freq+'_'+str(PsiM)+'_'+str(f_pbh)+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_topheavy.pdf')
	plt.close()
	
'''




if __name__=="__main__":

	L_z0_IGM = np.zeros(nz_IGM + 1)
	L_z0_Halo = np.zeros(nz + 1)
	L_z0_ABH = np.zeros(nz + 1)
	L_z0_ABH_1 = np.zeros(nz + 1)
	L_z0_ABH_2 = np.zeros(nz + 1)
	L_z0_HMXB = np.zeros(nz + 1)
	L_z0_HMXB_1 = np.zeros(nz + 1)
	L_z0_HMXB_2 = np.zeros(nz + 1)
	L_z0_SMBH = np.zeros(nz + 1)
	L_z0_SMBH_1 = np.zeros(nz + 1)
	# save array of BHs
	M_save_pop12 = np.zeros(30)
	M_save_pop12_1 = np.zeros(30)
	M_save_pop12_2 = np.zeros(30)
	M_save_pop3 = np.zeros(30)
	M_save_pop3_1 = np.zeros(30)
	M_save_pop3_2 = np.zeros(30)
	M_XB_pop12 = np.zeros(30)  # total mass of HMXBs in halo
	M_XB_pop12_1 = np.zeros(30)  # total mass of HMXBs in halo(assume classic)
	M_XB_pop12_2 = np.zeros(30)  # total mass of HMXBs in halo(Universe Machine)
	M_XB_pop3 = np.zeros(30)  # total mass of HMXBs in halo
	M_XB_pop3_1 = np.zeros(30)  # total mass of HMXBs in halo(assume classic)
	M_XB_pop3_2 = np.zeros(30)  # total mass of HMXBs in halo(Universe Machine)

	for i in range(nz_IGM+1):
		z = lz_IGM[i]
		print(z)
		f_col = special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(m0_IGM[i], ps3)) * np.sqrt(2)))
		L_z0_IGM[i] += L_tot_IGM(Mc, f_pbh, z, f_col, Distr=PsiM, SigmaM=Sigma_M, PBH_halo= PBH_halo)
		print('stage 1')

	for i in range(nz + 1):
		z = lz[i]
		print(z)
		m_ABH = max(1.54 * 1e5 * (31 / (1 + z)) ** 2.074, m0[i])
		M_h_max = 1e13
		lm_PBH = np.geomspace(m0[i], M_h_max, 30)
		if m_ABH >= M_max[i]:
			lm_ABH = np.zeros(30)
			lmh_SMBH = np.zeros(30)
			print('stage 2')
		elif M_max[i] <= 1e8:
			lmh_SMBH = np.zeros(30)
			lm_ABH = np.geomspace(m_ABH, M_max[i], 30)
			print('stage 3')
		else:
			lm_ABH = np.geomspace(m_ABH, M_max[i], 30)
			lm_PBH = np.geomspace(m0[i], M_max[i], 30)
			lmh_SMBH = np.geomspace(1e8, M_max[i], 30)
			print('stage 4')
		lm_ABH2 = np.geomspace(m_ABH, M_0_max, 30)
		M_SMBH = np.zeros(len(lmh_SMBH))  # mass of SMBH in halo
		M_BH_z = np.zeros(4)  # total mass of BHs in halo
		M_BH_z_1 = np.zeros(4)  # total mass of BHs in halo(assume classic)
		M_BH_z_2 = np.zeros(5)  # total mass of BHs in halo(Universe Machine)
		hmf3_PBH = halomassfunc(lm_PBH, z, ps3, wf3, gf3, mode=mode)
		hmf3_ABH = halomassfunc(lm_ABH, z, ps3, wf3, gf3, mode=mode)
		hmf3_SMBH = halomassfunc(lmh_SMBH, z, ps3, wf3, gf3, mode=mode)
		lnh_PBH = np.array([hmf3_PBH(m) for m in lm_PBH]) * h ** 3 / MPC ** 3*(1+z)**3
		lnh_ABH = np.array([hmf3_ABH(m) for m in lm_ABH]) * h ** 3 / MPC ** 3*(1+z)**3
		lnh_SMBH = np.array([hmf3_SMBH(m) for m in lmh_SMBH]) * h ** 3 / MPC ** 3*(1+z)**3
		#print(lnh_PBH)
		#print(lm_PBH)
		M_star_SMBH = [M_star_Mh(M, z) for M in lmh_SMBH]
		for j in range(len(lmh_SMBH)):
			M_SMBH[j] += M_SMBH_Mstar(M_star_SMBH[j], z)  # return the mass of SMBH for given halo mass at redshift z
		lm_SMBH = np.array([SMBHmass(Mh, z, seeding=seedSMBH) for Mh in lmh_SMBH])
		for j in range(len(lm_ABH)):
			M_BH_z = M_BH(lm_ABH[j], z, IMF='TopHeavy', fit='Madau14', output=1)
			M_BH_z_1 = M_BH(lm_ABH[j], z, IMF='TopHeavy', fit='Harikane22', output=1)
			# dM_BH_1 = np.array([n_StellarBH(M_star_tot_1[j],mbh,IMF=imf) for mbh in lmExt])
			M_BH_z_2 = M_BH_UNIV(lm_ABH2[j], z, IMF='TopHeavy', mode=1, output=1)
			M_XB_pop12[j] += M_BH_z[3]*f_HMXB_PopI_II
			M_XB_pop12_1[j] += M_BH_z_1[3]*f_HMXB_PopI_II
			M_XB_pop12_2[j] += M_BH_z_2[3]*f_HMXB_PopI_II
			M_XB_pop3[j] += M_BH_z[2]*f_HMXB_PopIII
			M_XB_pop3_1[j] += M_BH_z_1[2]*f_HMXB_PopIII
			M_XB_pop3_2[j] += M_BH_z_2[4]*f_HMXB_PopIII
			M_save_pop12[j]= M_BH_z[1]
			M_save_pop12_1[j]= M_BH_z_1[1]
			M_save_pop12_2[j]= M_BH_z_2[0]
			M_save_pop3[j]= M_BH_z[0]
			M_save_pop3_1[j]= M_BH_z_1[0]
			M_save_pop3_2[j]= M_BH_z_2[1]
			# dM_BH_1 = np.array([n_StellarBH(M_star_tot_1[j],mbh,IMF=imf) for mbh in lmExt])
			# print('$z = $ ' + str(z))
		# print(I_z0_IGM)
		L_z0_ABH[i] += L_tot_Halo2(z, M_save_pop3, M_save_pop12, lm_ABH, lnh_ABH, Pop3_IMF=IMF_BH2, delta0=delta0, r_min = R_min)
		L_z0_ABH_1[i] += L_tot_Halo2(z, M_save_pop3_1, M_save_pop12_1, lm_ABH, lnh_ABH, Pop3_IMF=IMF_BH2, delta0=delta0, r_min = R_min)
		L_z0_ABH_2[i] += L_tot_Halo2(z, M_save_pop3_2, M_save_pop12_2, lm_ABH, lnh_ABH, Pop3_IMF=IMF_BH2, delta0=delta0, r_min = R_min)
		L_z0_HMXB[i] += L_tot_Halo2(z, M_XB_pop3, M_XB_pop12, lm_ABH, lnh_ABH, Pop3_IMF=IMF_BH2, delta0=delta0, HMXB_flag=1)
		L_z0_HMXB_1[i] += L_tot_Halo2(z, M_XB_pop3_1, M_XB_pop12_1, lm_ABH, lnh_ABH, Pop3_IMF=IMF_BH2, delta0=delta0,
								  HMXB_flag=1)
		L_z0_HMXB_2[i] += L_tot_Halo2(z, M_XB_pop3_2, M_XB_pop12_2, lm_ABH, lnh_ABH, Pop3_IMF=IMF_BH2, delta0=delta0,
								  HMXB_flag=1)
		L_z0_Halo[i] += L_tot_Halo1(Mc, f_pbh, z, lm_PBH, lnh_PBH, Distr=PsiM, SigmaM=Sigma_M, delta0=delta0, r_min = R_min)
		L_z0_SMBH[i] += L_tot_SMBH(lm_SMBH, lmh_SMBH, lnh_SMBH)
		L_z0_SMBH_1[i] += L_tot_SMBH(M_SMBH, lmh_SMBH, lnh_SMBH)
	u_z0_IGM = Q_tot(lz_IGM, L_z0_IGM)
	u_z0_Halo = Q_tot(lz, L_z0_Halo)
	u_z0_ABH = Q_tot(lz, L_z0_ABH)
	u_z0_ABH_1 = Q_tot(lz, L_z0_ABH_1)
	u_z0_ABH_2 = Q_tot(lz, L_z0_ABH_2)
	u_z0_HMXB = Q_tot(lz, L_z0_HMXB)
	u_z0_HMXB_1 = Q_tot(lz, L_z0_HMXB_1)
	u_z0_HMXB_2 = Q_tot(lz, L_z0_HMXB_2)
	u_z0_SMBH = Q_tot(lz, L_z0_SMBH)
	u_z0_SMBH_1 = Q_tot(lz, L_z0_SMBH_1)
	print(u_z0_IGM)
	plt.figure()
	#L_tot_0_max = np.zeros(nz+1)
	#L_tot_0_min = np.zeros(nz+1)

	#for i in range(nz+1):
	#	L_tot_0_max[i] += L_z0_Halo[i] + L_z0_ABH[i] + L_z0_SMBH[i] + L_z0_IGM[i]+L_z0_HMXB[i]
	#	L_tot_0_min[i] += L_z0_Halo[i] + L_z0_ABH_2[i] + L_z0_SMBH_1[i] + L_z0_IGM[i]+L_z0_HMXB_2[i]


	#for i in range(nz+1):
	#	L_z0_Halo[i] /= L_tot_0_max[i]
	#	L_z0_IGM[i] /= L_tot_0_max[i]
	#	L_z0_ABH[i] /= L_tot_0_max[i]
	#	L_z0_HMXB[i] /= L_tot_0_max[i]
	#	L_z0_SMBH[i] /= L_tot_0_max[i]
	#	L_tot_0_max[i] /= L_tot_0_max[i]
	#	L_z0_Halo[i] /= L_tot_0_min[i]
	#	L_z0_IGM[i] /= L_tot_0_min[i]
	#	L_z0_ABH_2[i] /= L_tot_0_min[i]
	#	L_z0_HMXB_2[i] /= L_tot_0_min[i]
	#	L_z0_SMBH_1[i] /= L_tot_0_min[i]
	#	L_tot_0_min[i] /= L_tot_0_min[i]


	rep_data = './RadFeedBack/'
	#totxt(rep_data + 'Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_tot_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+'.txt', list([L_tot_0_max,L_tot_0_min]), 0, 0, 0)
	totxt(rep_data + 'PBH_Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_halo_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_IGM.txt', list([L_z0_IGM]), 0, 0, 0)
	totxt(rep_data + 'PBH_Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_halo_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_halo.txt', list([L_z0_Halo]), 0, 0, 0)
	totxt(rep_data + 'ABH_Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', list([L_z0_ABH,L_z0_ABH_1,L_z0_ABH_2]), 0, 0, 0)
	totxt(rep_data + 'HMXB_Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', list([L_z0_HMXB,L_z0_HMXB_1,L_z0_HMXB_2]), 0, 0, 0)
	totxt(rep_data + 'SMBH_Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', list([L_z0_SMBH,L_z0_SMBH_1]), 0, 0, 0)


	totxt(rep_data + 'PBH_Energy_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_halo_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_IGM.txt', list([u_z0_IGM]), 0, 0, 0)
	totxt(rep_data + 'PBH_Energy_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_halo_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_halo.txt', list([u_z0_Halo]), 0, 0, 0)
	totxt(rep_data + 'ABH_Energy_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', list([u_z0_ABH,u_z0_ABH_1,u_z0_ABH_2]), 0, 0, 0)
	totxt(rep_data + 'HMXB_Energy_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', list([u_z0_HMXB,u_z0_HMXB_1,u_z0_HMXB_2]), 0, 0, 0)
	totxt(rep_data + 'SMBH_Energy_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', list([u_z0_SMBH,u_z0_SMBH_1]), 0, 0, 0)
	






if __name__ == "__main__":

	fig, ax = plt.subplots(constrained_layout=True)
	

	rep_data = './RadFeedBack/'
	#L_tot_z = retxt(rep_data +'Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_tot_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+'.txt', 2)
	L_pbh_z = retxt(rep_data + 'PBH_Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_halo_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_halo.txt', 1)
	L_pbh_IGM_z = retxt(rep_data + 'PBH_Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_halo_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_IGM.txt', 1)
	L_ABH_z = retxt(rep_data + 'ABH_Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', 3)
	L_HMXB_z = retxt(rep_data +'HMXB_Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', 3)
	L_SMBH_z = retxt(rep_data + 'SMBH_Power_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', 2)

	#ax.plot(lz, L_tot_z[0], '-', linewidth=2, color='m', label='Total Power')
	#ax.plot(lz, L_tot_z[1], '-', linewidth=2, color='m')

	ax.plot(lz_IGM, L_pbh_IGM_z[0], linewidth=1, color='b', label='PBH IGM Power')
	ax.plot(lz, L_pbh_z[0], linewidth=1, color='g', label='PBH Halo Power')

	ax.plot(lz, L_ABH_z[0], linewidth=1, color='r', label='SRBH Power')
	ax.plot(lz, L_ABH_z[2], linewidth=1, color='r')

	ax.plot(lz, L_HMXB_z[0], linewidth=1, color='cyan', label='HMXB Power')
	ax.plot(lz, L_HMXB_z[2], linewidth=1, color='cyan')
	ax.plot(lz, L_SMBH_z[0], linewidth=1, color='y', label='SMBH Power')
	ax.plot(lz, L_SMBH_z[1], linewidth=1, color='y')


	#ax.fill_between(lz, L_tot_z[0], L_tot_z[1],facecolor='m',alpha = 0.5)
	ax.fill_between(lz, L_ABH_z[0], L_ABH_z[2],facecolor='red',alpha = 0.5)
	ax.fill_between(lz, L_SMBH_z[0], L_SMBH_z[1],facecolor='yellow', alpha=0.5)
	ax.fill_between(lz, L_HMXB_z[0], L_HMXB_z[2],facecolor='cyan', alpha=0.5)


# cosmic age
#def TZ1(z):
#	return 10 ** np.interp(z, lz, np.log10(lt0 / YR))


#def ZT1(T):
#	return np.interp(np.log10(T), np.log10(lt0 / YR), lz0)


	ax.legend(fontsize=12,loc= 'upper right',frameon=False)
	ax.set_xlabel(r'$z$')
	ax.set_ylabel(r'$L(z)/L_{tot}(z=6) $')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(z1, z2)
	#ax.set_ylim(1e-10, 1)

	#secax = ax.secondary_xaxis('top', functions=(TZ1, ZT1))
	#secax.xaxis.set_minor_locator(AutoMinorLocator())
	#secax.set_xlabel('Time [s]')
	plt.savefig('Power_Density_' + str(PsiM) + '_' + str(f_pbh)  +delta_fac+mix_flag+PBH_flag+'_3_topheavy_SMBHseed.pdf')
	plt.close()

	u_z0_tot_max = np.zeros(nz+1)
	u_z0_tot_min = np.zeros(nz+1)
	lyman_alpha = 1216*1e-8
	UV = 912*1e-8
	Freq_lya = SPEEDOFLIGHT/lyman_alpha
	Freq_UV = SPEEDOFLIGHT/UV
	u_limit_lya = [9*1e-23*(1+z)*Freq_lya*4*np.pi/SPEEDOFLIGHT for z in lz]
	u_limit_UV = [1e-21*((1+z)/6)**1.5*Freq_UV*4*np.pi/SPEEDOFLIGHT for z in lz]
	u_limit_UV1 = [1e-21*((1+z)/6)**1.5*Freq_UV*4*np.pi/SPEEDOFLIGHT/10 for z in lz]
	# assume a flat spectrum, plot the upper limit



	fig, ax = plt.subplots(constrained_layout=True)

	u_pbh_IGM_z = retxt(rep_data + 'PBH_Energy_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_halo_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_IGM.txt', 1)
	u_pbh_z = retxt(rep_data + 'PBH_Energy_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_halo_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_halo.txt', 1)
	u_ABH_z = retxt(rep_data + 'ABH_Energy_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', 3)
	u_HMXB_z = retxt(rep_data + 'HMXB_Energy_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', 3)
	u_SMBH_z = retxt(rep_data + 'SMBH_Energy_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_'+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'.txt', 2)


	max_ABH = np.zeros(nz+1)
	min_ABH = np.zeros(nz+1)
	max_HMXB = np.zeros(nz+1)
	min_HMXB = np.zeros(nz+1)
	for i in range(nz+1):
		max_ABH[i] += max(u_ABH_z[0][i], u_ABH_z[1][i], u_ABH_z[2][i])
		min_ABH[i] += min(u_ABH_z[0][i], u_ABH_z[1][i], u_ABH_z[2][i])
		max_HMXB[i] += max(u_HMXB_z[0][i], u_HMXB_z[1][i], u_HMXB_z[2][i]) * 100
		min_HMXB[i] += min(u_HMXB_z[0][i], u_HMXB_z[1][i], u_HMXB_z[2][i])
		u_z0_tot_max[i] += u_pbh_z[0][i] + max_ABH[i] + u_SMBH_z[0][i] + u_pbh_z[0][i] + max_HMXB[i]
		u_z0_tot_min[i] += u_pbh_z[0][i] + min_ABH[i] + u_SMBH_z[0][i] + u_pbh_z[0][i] + min_HMXB[i]


	ax.plot(lz, u_limit_lya,'-.',linewidth=4 ,color= 'black', label=r'Lyman$-\alpha$ Limit')
	ax.plot(lz, u_limit_UV,'--',linewidth=2 ,color= 'black', label=r'Reionization Min')

	#ax.plot(lz, u_z0_tot_max,'-',linewidth=3 ,color= 'm', label='Total')
	#ax.plot(lz, u_z0_tot_min,'-',linewidth=3 ,color= 'm')

	ax.plot(lz, u_pbh_z[0],linewidth=2,color= 'g',label='PBH in Halo')
	ax.plot(lz_IGM, u_pbh_IGM_z[0],linewidth=3,color= 'b',label='PBH in IGM')

	#ax.plot(lz, u_ABH_z[0],linewidth=1,color= 'r',label='SRBH')
	#ax.plot(lz, u_ABH_z[2],'--',linewidth=1,color= 'r')
	#ax.plot(lz, u_HMXB_z[0],linewidth=1,color= 'cyan',label='HMXB')
	#ax.plot(lz, u_HMXB_z[2],'--',linewidth=1,color= 'cyan')
	ax.plot(lz, max_ABH,linewidth=1,color= 'r',label='SRBH')
	ax.plot(lz, min_ABH,'--',linewidth=1,color= 'r')
	ax.plot(lz, max_HMXB,linewidth=1,color= 'cyan',label='HMXB')
	ax.plot(lz, min_HMXB,'--',linewidth=1,color= 'cyan')
	ax.plot(lz, u_SMBH_z[0],linewidth=1,color= 'y',label='SMBH')
	ax.plot(lz, u_SMBH_z[1],'--',linewidth=1,color= 'y')


	ax.fill_between(lz, u_limit_UV, u_limit_UV1,facecolor='black',alpha = 0.4)
	#ax.fill_between(lz, u_z0_tot_max, u_z0_tot_min,facecolor='m',alpha = 0.5)

	ax.fill_between(lz, max_ABH, min_ABH,facecolor='red',alpha = 0.5)
	ax.fill_between(lz, max_HMXB, min_HMXB,facecolor='cyan', alpha=0.3)
	#ax.fill_between(lz, u_ABH_z[0], u_ABH_z[2],facecolor='red',alpha = 0.5)
	#ax.fill_between(lz, u_HMXB_z[0], u_HMXB_z[2],facecolor='cyan',alpha = 0.5)
	ax.fill_between(lz, u_SMBH_z[0], u_SMBH_z[1],facecolor='yellow', alpha=0.5)
	#ax.fill_between(lz, u_z0_tot_HMXB, u_z0_tot_HMXB_2,facecolor='pink', alpha=0.5)


	#plt.plot(lz,np.ones(nz+1)*rho_rad, linewidth=2, label='Radiation Density')


	ax.legend(loc='lower right',fontsize=12)
	ax.set_xlabel(r'$z$',fontsize=15)
	ax.set_ylabel(r'$u(z)[\mathrm{erg \cdot cm^{-3}}] $',fontsize=15)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(6,z2)
	ax.set_ylim(1e-21, 1e-9)

	plt.savefig('Energy_z_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+seedstr+delta_fac+mix_flag+halo_handle+R_flag+PBH_flag+'_3_topheavy_07.pdf')
	plt.close()



'''
	for i in range(len(lfpbh)):
		fpbh = lfpbh[i]
		if fradbg==0:
			func = mth_func[i+1]
			ps3 = PBH(ps0, mpbh, fpbh, aeq, h, mfac=mfac0, iso=iso, seed=seed, cut=cut)
			#yxray = np.zeros(nz+1)
			#ylw = np.zeros(nz+1)
			#print('$f_{pbh} = $ ' + str(fpbh) )
			fcol_PBH = np.zeros(nz+1)
			I_z0_IGM_PBH =  np.zeros(nz + 1)
			for j in range(nz+1):
				z = lz[j]
				if fmth>0:
					m0 = max(func(z), mpbh/fpbh*Om/(Om-Ob))
				else:
					m0 = max(mpbh/fpbh*Om/(Om-Ob), M_Tvir(1e2, z))
				if ext>0:
					m0 = max(m0, mth_anna(j21_z(z)))
				lmh = np.geomspace(m0*h, 1e13*h, 100)
				dl = (1+z)*DZ(z)
				hmf3 = halomassfunc(lm0, z, ps3, wf3, gf3, mode=mode)
				lnh = np.array([hmf3(m) for m in lmh])*h**3/MPC**3
				#print('$z = $ ' + str(z))
				#print(lnh)
				fcol_PBH[j] += special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(M_Tvir(1e2, z), ps3))*np.sqrt(2)))
				#print(fcol_PBH)


				llx = np.array([radhalo(exray*(1+z), mpbh, m/h, z, mode=rmode) for m in lmh])
				# contribution to the cosmic X-ray background (intensity) at z=0
				# by PBH accretion in haloes at redshift bin j
				yxray[j] = np.trapz(lnh*llx, lmh)/(1+z)**2 * abs(DZ(z+dz)-DZ(z)) /(4*np.pi)/(180/np.pi)**2 # DZ(z): co-moving distance, see cosmology.py
				z0 = 13.6/11.2*(1+z)-1
				lz0 = np.linspace(z, z0, nz0)
				lylw = np.zeros(nz0)
				for k in range(nz0):
					z_ = lz0[k]
					hmf3 = halomassfunc(lm0, z_, ps3, wf3, gf3, mode=mode)
					lnh = np.array([hmf3(m) for m in lmh])*h**3/MPC**3
					llw = np.array([radhalo([11.2*(1+z_)/(1+z), 13.6], mpbh, m/h, z_, mode=rmode) for m in lmh])
					lylw[k] = np.trapz(lnh*llw) * SPEEDOFLIGHT*dt_da(1/(1+z_))/(1+z_)**2
					# dt_da(a): see cosmology.py
				# LW background intensity at redshift bin j
				ylw[j] = (1+z)**3 * np.trapz(lylw, lz0) /4/np.pi * 1e21 / dnu
				print('z={}, M_min={:.2e} Msun, F_X={:.2e} erg s^-1 cm^-2 deg^-2, J_21 = {:.2e}'.format(z, m0, yxray[j]*fpbh, fpbh*ylw[j]))
			# write data to file (see txt.py)
			# columns: 1: z, 2: contribution of X-ray intensity (at z=0)
			# 3: LW intensity
			if lab is not '':
				totxt('radbg_fpbh'+str(fpbh)+'_'+lab+'.txt', [lz, yxray*fpbh, ylw*fpbh])
			else:
				totxt('radbg_fpbh'+str(fpbh)+'.txt', [lz, yxray*fpbh, ylw*fpbh])
			# the radiation backgrounds need to be normalized by fpbh
			data0.append([lz, yxray*fpbh, ylw*fpbh])
			data1.append([lz, yxray*fpbh, ylw*fpbh])
			data2.append([lz, yxray*fpbh, ylw*fpbh])
		else:
			# read data from files (see txt.py)
			data0.append(retxt('radbg_fpbh'+str(fpbh)+'.txt', 3))
			#data1.append(retxt('radbg_fpbh'+str(fpbh)+'_'+lab+'.txt', 3))
			data1.append(retxt('radbg_fpbh'+str(fpbh)+'_iso.txt', 3))
			if ext>0:
				data2.append(retxt('radbg_fpbh'+str(fpbh)+'_ext.txt', 3))
			else:
				#data2.append(retxt('radbg_fpbh'+str(fpbh)+'_mth.txt', 3))
				data2.append(retxt('radbg_fpbh'+str(fpbh)+'_mth.txt', 3))
	# main loop ends

	# plot results
	jxmin, jxmax = 2.74e-12, 7.73e-12
	y1, y2, ny = 1e-16, 3e-7, 10
	plt.figure()
	for i in range(4):
		plt.plot(data0[i][0], np.cumsum(data0[i][1]), ls=lls[i], color=llc[i], label=r'$f_{\rm PBH}='+str(lfpbh[i])+'$')
		#if i<3:
		plt.plot(data1[i][0], np.cumsum(data1[i][1]), ls=lls[i], color=llc[i])
		plt.plot(data2[i][0], np.cumsum(data2[i][1]), ls=lls[i], color=llc[i], lw=4.5, alpha=0.5)
		print('f_PBH={}, J_X_max = {:.2e}'.format(lfpbh[i], np.max(np.cumsum(data0[i][1]))))
	plt.fill_between([z1, z2], [jxmin]*2, [jxmax]*2, fc='gray', alpha=0.5, label='CXB')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$J_{[0.5-2\ \rm keV]}\ [\rm erg\ s^{-1}\ cm^{-2}\ deg^{-2}]$')
	plt.yscale('log')
	plt.xlim(z1, z2)
	plt.ylim(y1, y2)
	plt.yticks(np.geomspace(y1, 1e-7, ny))
	plt.legend(ncol=2, loc=1)
	plt.tight_layout()
	plt.savefig(rep+'JX_z.pdf')
	plt.close()
	
	y1, y2, ny = 1e-9, 1e4, 7
	plt.figure()
	for i in range(4):
		plt.plot(data0[i][0], data0[i][2], ls=lls[i], color=llc[i], label=r'$f_{\rm PBH}='+str(lfpbh[i])+'$')
		#if i<3:
		plt.plot(data1[i][0], data1[i][2], ls=lls[i], color=llc[i])
		plt.plot(data2[i][0], data2[i][2], ls=lls[i], color=llc[i], lw=4.5, alpha=0.5)
		print('f_PBH={}, J_21_max = {:.2e}'.format(lfpbh[i], np.max(data0[i][2])))
	plt.plot([z1, z2], [1]*2, 'k', ls=(0,(10,5)), label=r'$k_{\rm H_{2},des}\sim k_{\rm H_{2},form}$')
	plt.plot(lz, 10**(2-lz/5), color='r', alpha=0.7, lw=4.5, label='Stars', ls=(0,(1,1,3)))
	plt.xlabel(r'$z$')
	plt.ylabel(r'$J_{21}$')#\ [10^{-21}\ erg\ s^{-1}\ cm^{-2}\ Hz^{-1}\ sr^{-2}]$')
	plt.yscale('log')
	plt.xlim(z1, z2)
	plt.ylim(y1, y2)
	plt.yticks(np.geomspace(y1, 1e3, ny))
	plt.legend(ncol=2, loc=1)
	plt.tight_layout()
	plt.savefig(rep+'J21_z.pdf')
	plt.close()
'''
'''

if __name__=="__main__":
	mix_flag = "_nomixing"

	MPBH = 10
	lz = np.linspace(z2, z1, 48)
	y1, y2 = 1e-10, 1
	plt.figure(figsize=(7,5))
	plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e5, ps0)) * np.sqrt(2))) for z in lz],marker = 'p',markersize = 15,fillstyle='none', color='b',linestyle = 'none', label=r'$M_{\mathrm{h}} = 10^{5} \rm \ M_{\odot},\Lambda \mathrm{CDM} $ ')
	plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e6, ps0)) * np.sqrt(2))) for z in lz], marker = '^',markersize = 15,fillstyle='none', color='b',linestyle = 'none', label=r'$M_{\mathrm{h}} = 10^{6} \rm \ M_{\odot} $ ')
	plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e8, ps0)) * np.sqrt(2))) for z in lz], marker = 'o',markersize = 15,fillstyle='none', color='b',linestyle = 'none', label=r'$M_{\mathrm{h}} = 10^{8} \rm \ M_{\odot} $ ')

	ps = PBH(ps0, 10, 1e-3, aeq, h, mfac=mfac0, iso=iso, seed=seed, cut=cut)
	plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e5, ps)) * np.sqrt(2))) for z in lz],  marker = 'p',markersize = 12,fillstyle='none', color='r',linestyle = 'none', label=r'$ M_{\rm c} = 1 \rm \ M_{\odot}, f_{\mathrm{PBH}}=0.001$ ')
	plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e6, ps)) * np.sqrt(2))) for z in lz],  marker = '^',markersize = 12,fillstyle='none', color='r',linestyle = 'none')
	plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e8, ps)) * np.sqrt(2))) for z in lz],  marker = 'o',markersize = 12,fillstyle='none', color='r',linestyle = 'none')


	ps_nomix = PBH(ps0, 10, 1e-3, aeq, h, mfac=mfac0, iso=1 , seed=seed, cut=cut)
	plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e5, ps_nomix)) * np.sqrt(2))) for z in lz],  marker = 'p',markersize = 12,fillstyle='none', color='g',linestyle = 'none', label=r'$M_{\rm c} = 1 \rm \ M_{\odot}, f_{\mathrm{PBH}}=0.001, No mixing$ ')
	plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e6, ps_nomix)) * np.sqrt(2))) for z in lz],  marker = '^',markersize = 12,fillstyle='none', color='g',linestyle = 'none')
	plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e8, ps_nomix)) * np.sqrt(2))) for z in lz],  marker = 'o',markersize = 12,fillstyle='none', color='g',linestyle = 'none')


	#ps = PBH(ps0, 1, 1e-1, aeq, h, mfac=mfac0, iso=iso, seed=seed, cut=cut)
	#plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e6, ps)) * np.sqrt(2))) for z in lz], '--', color='g')
	#plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e8, ps)) * np.sqrt(2))) for z in lz], '-.', color='g')


	#ps = PBH(ps0, 100, 1e-1, aeq, h, mfac=mfac0, iso=iso, seed=seed, cut=cut)
	#plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e5, ps)) * np.sqrt(2))) for z in lz], marker = 'p',markersize = 12,fillstyle='none',color='y',linestyle = 'none', label=r'$M_{\rm c} = 100 \rm \ M_{\odot}, f_{\mathrm{PBH}}=0.1 $ ')
	#plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e6, ps)) * np.sqrt(2))) for z in lz], marker = '^',markersize = 12,fillstyle='none', color='y',linestyle = 'none')
	#plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e8, ps)) * np.sqrt(2))) for z in lz],  marker = 'o',markersize = 12,fillstyle='none', color='y',linestyle = 'none')

	#ps = PBH(ps0, 100, 1e-3, aeq, h, mfac=mfac0, iso=iso, seed=seed, cut=cut)
	#plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e5, ps)) * np.sqrt(2))) for z in lz], marker = 'p',markersize = 8,fillstyle='none',color='c',linestyle = 'none', label=r'$M_{\rm c} = 100 \rm \ M_{\odot}, f_{\mathrm{PBH}}=0.001 $ ')
	#plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e6, ps)) * np.sqrt(2))) for z in lz], marker = '^',markersize = 8,fillstyle='none', color='c',linestyle = 'none')
	#plt.plot(lz, [special.erfc(deltac_z(z) / (np.sqrt(sigma2_M(1e8, ps)) * np.sqrt(2))) for z in lz],  marker = 'o',markersize = 8,fillstyle='none', color='c',linestyle = 'none')


	plt.xlabel(r'$z$',fontsize=15)
	plt.ylabel(r'$f_{\mathrm{col}} (>M_{\rm h}) $',fontsize=15)#\ [10^{-21}\ erg\ s^{-1}\ cm^{-2}\ Hz^{-1}\ sr^{-2}]$')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlim(z1, z2)
	plt.ylim(y1, y2)
	plt.legend(loc=0,frameon=False,fontsize=15)
	plt.tight_layout()
	plt.savefig('fcol_z'+str(MPBH)+mix_flag+'.pdf')
	plt.close()

'''

'''
	I_0 = np.zeros(nz+1)
	for i in range(nz+1):
		I_0[i]+= rho_rad

	#totxt(rep + 'IGM_J912.txt', I_z0_IGM, 0, 0, 0)
	#totxt(rep + 'ABH_J912.txt', I_z0_ABH, 0, 0, 0)
	#totxt(rep + 'SMBH_J912.txt', I_z0_SMBH, 0, 0, 0)
	#totxt(rep + 'Halo_J912.txt', I_z0_Halo, 0, 0, 0)

	plt.figure()
	fig, ax = plt.subplots()

	ax.plot(lz, I_0,'-.',linewidth=4 ,color= 'm', label='Reionization Limit')

	ax.plot(lz, I_z0_Halo,linewidth=2,color= 'g',label='PBH Halo Flux')
	ax.plot(lz, I_z0_IGM,linewidth=2,color= 'b',label='PBH IGM Flux')

	ax.plot(lz, I_z0_ABH,linewidth=2,color= 'r',label='ABH Flux')
	ax.plot(lz, I_z0_SMBH,linewidth=2,color= 'y',label='SMBH Flux')

	ax.legend(fontsize=15)
	ax.set_xlabel(r'$z$',fontsize=15)
	ax.set_ylabel(r'$J(z)[erg s^{-1} cm^{-2} Hz^{-1} sr^{-1}] $',fontsize=15)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(z1,z2)
	ax.set_ylim(1e-30, 1e-18)

	ax.tick_params(which='major', labelcolor='black', labelsize=15, width=3)

	plt.savefig('J_912_'+str(Mc)+'_'+str(PsiM)+'_'+str(f_pbh)+'_3.pdf')

	plt.close()
'''


'''

if __name__=="__main__":
	lz2 = np.linspace(z2, z1, nz*5+1)
	imf='Salpter'
	#imf='TopHeavy'
	mpbh = 10
	f_pbh = 1e-3
	f_HMXB_PopIII = 0.3
	f_HMXB_PopI_II = 0.5
	rhoPBH = f_pbh * rhom(1 , O_m, h) *(O_m-O_b)/O_m*MPC**3/ Msun
	rho_pbh=np.zeros(len(lz2))
	rho_pbh_halo=np.zeros(len(lz2))
	rho_ABH =np.zeros(len(lz2))
	rho_ABH_1 =np.zeros(len(lz2))
	rho_ABH_2 =np.zeros(len(lz2))
	rho_HMXB =np.zeros(len(lz2))
	rho_HMXB_1 =np.zeros(len(lz2))
	rho_HMXB_2 =np.zeros(len(lz2))
	rho_SMBH=np.zeros(len(lz2))
	rho_SMBH_1=np.zeros(len(lz2))
	f_star=0.1
	f_cool=0.01
	ps3 = PBH(ps0, mpbh, f_pbh, aeq, h, mfac=mfac0, iso=iso, seed=seed, cut=cut)

	M_0_max=1e17
	M_max=[M_z(M_0_max,z) for z in lz2]



	for i in range(len(lz2)):
		#m0 = 3*Mc*(1000/(1+lz2[i]))
		#m0 = max(M_Tvir(1e2, lz2[i]),mpbh/f_pbh*Om/(Om-Ob))
		m0 = max(M_Tvir(1e2, lz2[i]),3*Mc*(1000/(1+lz2[i])),mpbh/f_pbh*Om/(Om-Ob))
		f_col=special.erfc(deltac_z(lz2[i]) / (np.sqrt(sigma2_M(m0, ps3)) * np.sqrt(2)))
		rho_pbh[i]+=rhoPBH
		rho_pbh_halo[i]+=f_col*rhoPBH
		lm_PBH = np.geomspace(m0 , 1e13 , 30)
		m_ABH = max(1.54*1e5*(31/(1+lz2[i]))**2.074, m0)
		lm_ABH = np.geomspace(m_ABH ,M_max[i],30)
		print('z='+str(lz2[i]))
		print(lm_PBH)
		lm_ABH2 = np.geomspace(m_ABH ,M_0_max,30)
		lmh_SMBH = np.geomspace(1e8, M_max[i] , 30)
		lmExt = np.logspace(np.log10(1.4),2,20)
		M_SMBH = np.zeros(len(lmh_SMBH))# mass of SMBH in halo
		M_BH_z = np.zeros(2)# total mass of BHs in halo
		M_BH_z_1 = np.zeros(2)# total mass of BHs in halo(assume classic)
		M_BH_z_2 = np.zeros(3)# total mass of BHs in halo(Universe Machine)
		hmf3_PBH = halomassfunc(lm_PBH, lz2[i], ps3, wf3, gf3, mode=mode)
		hmf3_ABH = halomassfunc(lm_ABH, lz2[i], ps3, wf3, gf3, mode=mode)
		hmf3_SMBH = halomassfunc(lmh_SMBH, lz2[i], ps3, wf3, gf3, mode=mode)
		lnh_PBH = np.array([hmf3_PBH(m) for m in lm_PBH])/MPC ** 3*h**3
		lnh_ABH = np.array([hmf3_ABH(m) for m in lm_ABH])/MPC**3*h**3
		lnh_SMBH = np.array([hmf3_SMBH(m) for m in lmh_SMBH])/MPC**3*h**3
		M_star_SMBH=[M_star_Mh(M,lz2[i]) for M in lmh_SMBH]
		for j in range(len(lmh_SMBH)):
			M_SMBH[j] += M_SMBH_Mstar(M_star_SMBH[j],lz2[i])
		lm_SMBH =np.array([SMBHmass(Mh,lz2[i],seeding = seedSMBH) for Mh in lmh_SMBH])
		rho_SMBH_1[i] += np.trapz(lm_SMBH*lnh_SMBH*MPC**3, x=lmh_SMBH)
		rho_SMBH[i] += np.trapz(M_SMBH*lnh_SMBH*MPC**3, x=lmh_SMBH)
		M_star_tot=[M_star_Mh(M_h,lz2[i]) for M_h in lm_ABH]
		M_star_tot_1 = [1e5 * M_h / 1e8 * f_star / 0.1 * f_cool / 0.01 for M_h in lm_ABH]
		drho = np.zeros(len(lm_ABH))
		drho_1 = np.zeros(len(lm_ABH))
		drho_2 = np.zeros(len(lm_ABH))
		drho_HMXB = np.zeros(len(lm_ABH))
		drho_HMXB_1 = np.zeros(len(lm_ABH))
		drho_HMXB_2 = np.zeros(len(lm_ABH))
		for j in range(len(lm_ABH)):
			M_BH_z = M_BH(lm_ABH[j],lz2[i], IMF='TopHeavy',fit='Madau14',output=1,HMXB_Lifetime =1e6)
			drho[j] += (M_BH_z[0]+M_BH_z[1])* lnh_ABH[j]
			drho_HMXB[j] += (M_BH_z[2]*f_HMXB_PopIII/2+M_BH_z[3]*f_HMXB_PopI_II/2)* lnh_ABH[j]
			M_BH_z_1 = M_BH(lm_ABH[j],lz2[i], IMF='TopHeavy',fit='Harikane22',output=1,HMXB_Lifetime =1e6)
			#dM_BH_1 = np.array([n_StellarBH(M_star_tot_1[j],mbh,IMF=imf) for mbh in lmExt])
			drho_1[j] += (M_BH_z_1[0]+M_BH_z_1[1])* lnh_ABH[j]
			drho_HMXB_1[j] += (M_BH_z_1[2]*f_HMXB_PopIII/2+M_BH_z_1[3]*f_HMXB_PopI_II/2)* lnh_ABH[j]
			M_BH_z_2 = M_BH_UNIV(lm_ABH2[j],lz2[i], IMF='TopHeavy' ,mode=1,output=1,HMXB_Lifetime =1e6)
			#dM_BH_1 = np.array([n_StellarBH(M_star_tot_1[j],mbh,IMF=imf) for mbh in lmExt])
			drho_2[j] += (M_BH_z_2[0]+M_BH_z_2[1])* lnh_ABH[j]
			drho_HMXB_2[j] += (M_BH_z_2[4]*f_HMXB_PopIII/2+M_BH_z_2[3]*f_HMXB_PopI_II/2)* lnh_ABH[j]
		rho_ABH[i]+= np.trapz(drho*MPC**3, x=lm_ABH)
		rho_ABH_1[i]+= np.trapz(drho_1*MPC**3, x=lm_ABH)
		rho_ABH_2[i]+= np.trapz(drho_2*MPC**3, x=lm_ABH)
		rho_HMXB[i]+= np.trapz(drho_HMXB*MPC**3, x=lm_ABH)
		rho_HMXB_1[i]+= np.trapz(drho_HMXB_1*MPC**3, x=lm_ABH)
		rho_HMXB_2[i]+= np.trapz(drho_HMXB_2*MPC**3, x=lm_ABH)

	rep_data = './RadFeedBack/'
	totxt(rep_data + 'rho2_pbh_'+str(f_pbh)+mix_flag+SMBHstr+halo_handle+PBH_flag+'_halo.txt', list([rho_pbh,rho_pbh_halo]), 0, 0, 0)
	totxt(rep_data + 'rho2_ABH_'+str(f_pbh)+mix_flag+SMBHstr+halo_handle+PBH_flag+'.txt', list([rho_ABH,rho_ABH_1,rho_ABH_2]), 0, 0, 0)
	totxt(rep_data + 'rho2_HMXB_'+str(f_pbh)+mix_flag+SMBHstr+halo_handle+PBH_flag+'.txt', list([rho_HMXB,rho_HMXB_1,rho_HMXB_2]), 0, 0, 0)
	totxt(rep_data + 'rho2_SMBH_'+str(f_pbh)+mix_flag+SMBHstr+halo_handle+PBH_flag+'.txt', list([rho_SMBH,rho_SMBH_1]), 0, 0, 0)



if __name__=="__main__":
#plot the overall BH mass function
	fig, ax = plt.subplots()

	rep_data = './RadFeedBack/'
	rho_pbh_z = retxt(rep_data + 'rho2_pbh_' +str(f_pbh)+mix_flag+SMBHstr+halo_handle+PBH_flag+'_halo.txt', 2)
	rho_ABH_z = retxt(rep_data + 'rho2_ABH_' +str(f_pbh)+mix_flag+SMBHstr+halo_handle+PBH_flag+ '.txt', 3)
	rho_HMXB_z = retxt(rep_data + 'rho2_HMXB_' +str(f_pbh)+mix_flag+SMBHstr+halo_handle+PBH_flag+ '.txt', 3)
	rho_SMBH_z = retxt(rep_data + 'rho2_SMBH_' +str(f_pbh)+mix_flag+SMBHstr+halo_handle+PBH_flag+ '.txt', 2)

	max_ABH = np.zeros(len(lz2))
	min_ABH = np.zeros(len(lz2))
	max_HMXB = np.zeros(len(lz2))
	min_HMXB = np.zeros(len(lz2))
	for i in range(len(lz2)):
		max_ABH[i] += max(rho_ABH_z[0][i], rho_ABH_z[1][i], rho_ABH_z[2][i])
		min_ABH[i] += min(rho_ABH_z[0][i], rho_ABH_z[1][i], rho_ABH_z[2][i])
		max_HMXB[i] += max(rho_HMXB_z[0][i], rho_HMXB_z[1][i], rho_HMXB_z[2][i]) * 100
		min_HMXB[i] += min(rho_HMXB_z[0][i], rho_HMXB_z[1][i], rho_HMXB_z[2][i])

	ax.plot(lz2, rho_pbh_z[0],'-',linewidth=2 ,color= 'b', label='PBH (total)')
	ax.plot(lz2, rho_pbh_z[1],'-',linewidth=2 ,color= 'g', label='PBH (Halo)')
	ax.plot(lz2, rho_ABH_z[0],linewidth=2,color= 'r',label='SRBH, Madau+14;Liu+20')
	ax.plot(lz2, rho_ABH_z[1],'--',linewidth=2,color= 'r',label='SRBH, Harikane+22;Liu+20')
	ax.plot(lz2, rho_ABH_z[2],'-.',linewidth=2,color= 'r',label='SRBH, Zhang+23')
	ax.plot(lz2, rho_HMXB_z[0],linewidth=2,color= 'cyan',label='HMXB, Madau+14;Liu+20')
	ax.plot(lz2, rho_HMXB_z[1],'--',linewidth=2,color= 'cyan',label='HMXB, Harikane+22;Liu+20')
	ax.plot(lz2, rho_HMXB_z[2],'-.',linewidth=2,color= 'cyan',label='HMXB,Zhang+23')
	ax.plot(lz2, max_HMXB,':',linewidth=2,color= 'cyan',label='HMXB, max')
	ax.plot(lz2, rho_SMBH_z[0],linewidth=2,color= 'y',label='SMBH, Zhang+23')
	ax.plot(lz2, rho_SMBH_z[1],'--',linewidth=2,color= 'y',label='SMBH, Jeon+22')


	ax.fill_between(lz2, max_ABH, min_ABH,facecolor='red',alpha = 0.5)
	ax.fill_between(lz2, rho_SMBH_z[0], rho_SMBH_z[1],facecolor='yellow', alpha=0.5)
	ax.fill_between(lz2, max_HMXB, min_HMXB,facecolor='cyan', alpha=0.3)
# cosmic age
#	def TZ1(z):
#		return 10**np.interp(z, lz0, np.log10(lt0/YR))

#	def ZT1(T):
#		return np.interp(np.log10(T),np.log10(lt0/YR), lz0)

	ax.legend(handlelength = 4,fontsize=10,loc= 'upper right')
	ax.set_xlabel(r'$z$')
	ax.set_ylabel(r'$ \rho_{\mathrm{BH}} [\rm M_{\odot}   \mathrm{Mpc}^{-3}] $')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(6,100)
	ax.set_ylim(1e-6, 1e10)

#	secax = ax.secondary_xaxis('top', functions=(TZ1,ZT1))
#	secax.xaxis.set_minor_locator(AutoMinorLocator())
#	secax.set_xlabel('Time [s]')
	plt.savefig('BHDensity_'+str(mpbh)+'_'+str(f_pbh)+mix_flag+SMBHstr+halo_handle+PBH_flag+'.pdf')
	plt.close()


'''