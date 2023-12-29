from tophat import *
from scipy import interpolate
from scipy.interpolate import interp1d

# gaussian distribution of streaming motion velocity
def vdis(v, sigma = 30.):
	return v**2 * np.exp(-v**2*1.5/sigma**2)

# cosmic average number density of halos above the mass threshold by integrating 
# the number density over the distribution of streaming motion velocity
def Nhalo(z, lm0, lv0, mode = 0, Mdm = 0.3, h = 0.6736, sigma = 30., vmax = 5.):
	"""
		lm0: array of mass threshold (corresponding to lv0)
		lv0: array of streaming motion velocity
		mode=0: CDM, mode=1: BDMS
		vmax: boundary of the integration (over the gaussian distribution)
		sigma: rms of the streaming motion velocity at z=1100
	"""
	sel = lv0 < vmax*sigma
	lm = lm0[sel]
	lv = lv0[sel]
	lw = vdis(lv, sigma)
	Mmin = min(np.log10(np.min(lm*h))-1, np.log10(Mup(z))-1)
	if mode == 0:
		hmf_ = hmf.MassFunction()
		hmf_.update(n=0.966, sigma_8=0.829,cosmo_params={'Om0':0.315,'H0':67.74},Mmin=Mmin,Mmax=np.log10(Mup(z))+2,z=z)
	else:
		hmf_ = hmf.wdm.MassFunctionWDM(wdm_mass=Mdm*1e6)
		hmf_.update(n=0.966, sigma_8=0.829,cosmo_params={'Om0':0.315,'H0':67.74},Mmin=Mmin,Mmax=np.log10(Mup(z))+2,z=z)
	intm = np.log10(hmf_.m/h)
	intn = np.log10(hmf_.ngtm)
	nm = interp1d(intm, intn)
	lnpopIII = (10**nm(np.log10(lm))- 10**nm(np.log10(Mup(z))))
	lnpopIII = lnpopIII * (lnpopIII>0)
	out = lnpopIII * lw
	return np.trapz(out, lv)/np.trapz(lw, lv)
		
# wrapper of Nhalo that sets up a grid of streaming motion velocity 
# for integration under certain model parameters
def nh_z(z = 20, m1 = 1e2, m2 = 1e10, mode = 0, Mdm = 0.3, sigma = 8e-20, rat = 1.0, dmax = 2e4, Om = 0.3153, h = 0.6736, fac = 1e-3, vmin = 0.0, beta = .7, sk = False, v1 = 0, v2 = 150, nv = 16, ncore = 8):
	lv = np.linspace(v1, v2, nv) #np.geomspace(v0, v1, nv)
	np_core = int(nv/ncore)
	lpr = [[i*np_core, (i+1)*np_core] for i in range(ncore-1)] + [[(ncore-1)*np_core, nv]]
	print(lpr)
	manager = mp.Manager()
	output = manager.Queue()
	def sess(pr0, pr1):
		lm = []
		for i in range(pr0, pr1):
			init = initial(v0 = lv[i], mode = mode, Mdm = Mdm, sigma = sigma)
			d = Mth_z(z, z, 1, mode = mode, v0 = lv[i], rat = rat, dmax = dmax, fac = fac, beta = beta, sk = sk, init = init, Mdm = Mdm, sigma = sigma)
			lm.append(d[0][0])
		output.put([pr0, lm])
	pros = [mp.Process(target=sess, args=(lpr[k][0], lpr[k][1])) for k in range(ncore)]
	for p in pros:
		p.start()
	for p in pros:
		p.join()
	out = [output.get() for p in pros]
	out.sort()
	lm = np.hstack([x[1] for x in out])
	nh = Nhalo(z, np.array(lm), lv, mode, Mdm)
	return nh

# scan the parameter space of BDMS to calculate the cosmic average number 
# density of halos above the mass threshold
def nh_para(m1 = -4, m2 = 2, s1 = -1, s2 = 6, z = 20, dmax = 2e4, nbin = 2, fac = 1e-3, rat = 1.0, beta = .7, sk = False, ncore = 8):
	lm = np.logspace(m1, m2, nbin)
	ls = np.logspace(s1, s2, nbin)
	X, Y = np.meshgrid(lm, ls, indexing = 'ij')
	#lMh = np.zeros(X.shape)
	lnh = np.array([[nh_z(z, Mdm = m, sigma = s*1e-20, mode=1, rat=rat, fac=fac, dmax=dmax, beta=beta, sk=sk, ncore=ncore) for m in lm] for s in ls]).T
	return X, Y*1e-20, lnh

#total stellar mass as a function of the DM halo and redshift, in units of solar mass
#Based on the fitting formula from Behroozi et al. 2013
def M_star_Mh(M_h, z):
	a = 1/(z+1)
	v=np.exp(-4*a**2)
	M_1 = 10**(11.514+(-1.793*(a-1)+(-0.251*z))*v)
	epsilon = 10**(-1.777+(-0.006*(a-1)+0*z)*v)
	alpha_z = -1.412+(0.731*(a-1))*v
	delta_z = 3.508+(2.608*(a-1)+(-0.043*z))*v
	gamma_z = 0.316+(1.319*(a-1)+0.279*z)*v
	x=np.log10(M_h)-np.log10(M_1)
	def f_x(x,alpha,del_z,gamma):
		return - np.log10(10**(alpha*x)+1)+del_z *((np.log10(1+np.exp(x)))**gamma)/(1+np.exp(10**(-x)))
	M_star = epsilon*M_1*10**(f_x(x,alpha_z,delta_z,gamma_z)-f_x(0,alpha_z,delta_z,gamma_z))
	return M_star

# SMBH mass as a function of stellar mass and buldge mass
# From Zhang et al. 2023
def M_SMBH_Mstar(M_star,z):
	a = 1/(z+1)
	f_z = (z+2)/(2*z+2)
	beta_BH = 8.343+(-0.173)*(a-1)+0.044*z
	gamma_BH = 1.028+0.036*(a-1)+(0.052)*z
	M_bulge = f_z*M_star
	M_SMBH = 10**beta_BH*(M_bulge/1e11)**gamma_BH
	return M_SMBH

#caculate the star formation rate from the UNIVERSE MACHINE (Behroozi et al.(2020))
def SFR_UNIV(M_h,z):
	if M_h<=1.54*1e5*(31/(1+z))**2.074:
		return 0
	else:
		a = 1/(z+1)
		M_200= 1.64*1e12/((a/0.378)**(-0.142)+(a/0.378)**(-1.79))
		v_peak = 200*(M_h/M_200)**(1/3)
		V = 10**(2.289+1.548*(a-1)+1.218*np.log(1+z)-0.087*z)
		epsilon = 10**(0.556-0.994*(a-1)-0.042*np.log(1+z)+0.418*z)
		alpha = -3.907+32.223*(a-1)+20.241*np.log(1+z)-2.193*z
		beta = 0.329+2.342*(a-1)+0.492*z
		v=v_peak/V
		SFR_z = epsilon/(v**alpha+v**beta)
		if SFR_z <= 1e-20:
			SFR_z=0
		#print('SFR for '+str(M_h)+' solar mass halo at redshift  '+str(z)+' is '+str(SFR_z)+' solar mass/yr')
		return SFR_z

#caculate the mass growth of the halo by accretion from the UNIVERSE MACHINE (Behroozi et al.(2013))
#M_0 is mass of the halo at redshift 0
# it trace back to the mass of progenitor halo at redshift z
def M_z(M_0,z):
	a = 1/(z+1)
	M_13_0 = 10**13.276
	a_0 = 0.205-np.log10((10**9.649/M_0)**0.18+1)
	def g(a):
		return 1+np.exp(-4.651*(a-a_0))
	f_M0_z=np.log10(M_0/M_13_0)*g(1)/g(a)
	M_13=10**13.276*(1+z)**3*(1+z/2)**(-6.11)*np.exp(-0.503*z)
	M=M_13*10**f_M0_z
	#print('Progenitor for '+str(M_0)+' at redshift '+str(z)+' is '+str(M)+' solar mass')
	return M

# from the volume star formation density to calculate the star formation for a give halo at z
def SFR_pop12(M_h,z,z_f, fit = 'Madau14'):
	V_c = M_h*Msun/rhom(1)/MPC**3#*(1+z_f)**3  # comoving volume in units of MPC:
	if fit == 'Madau14':
		rho_SFR_popI_II = 0.015*(1+z)**2.7/(1+((1+z)/2.9)**5.6)
	elif fit == 'Harikane22':
		rho_SFR_popI_II = 1/(61.7*(1+z)**(-3.13)+10**(0.22*(1+z))+2.4*(10**((1+z)/2-3)))
	return V_c*rho_SFR_popI_II

def SFR_pop3(M_h,z,z_f):
	V_c = M_h*Msun/rhom(1)/MPC**3 #*(1+z_f)**3  # comoving volume in units of MPC
	rho_SFR_popIII = 756.7*(1+z)**(-5.92)/(1+((1+z)/12.83)**(-8.55))
	return V_c*rho_SFR_popIII



# interpolate the initial mass of stars as a function of the masses of their final state as BH
BHdata=np.genfromtxt('M_BH.dat',delimiter=',')
M_BH_f=interpolate.interp1d(BHdata[:,1],BHdata[:,0], fill_value='extrapolate')


#calculate the BH mass inside halo with M_h at z=0
# mode: 0: overall formation rate from UniverseMachine 1:Include PopIII formation rate
def M_BH_UNIV(M_h,z_f,IMF,mode=1, output=0,HMXB_Lifetime =1e6):
	if IMF == 'Salpter':
		alphaIII =-2.35
	if IMF == 'TopHeavy':
		alphaIII =-1.35
	alphaI_II =-2.35
	if z_f>=50:
		if output == 0:
			return 0,0,0
		if output == 1:
			return 0,0,0,0,0
	else:
		lz = np.linspace(50,z_f,int(50-z_f)*2+1)
		lm_h = [M_z(M_h,z) for z in lz]
		lm_abh_3 = np.linspace(M_BH_f(25),M_BH_f(140),20)
		lm_abh_12 = np.linspace(M_BH_f(25),M_BH_f(260),20)
		lm_star_3 = np.linspace(25,140,20)
		lm_star_12 = np.linspace(25,260,20)
		T = [TZ(z)/YR for z in lz] # number of year btw
		f_loss_3 = (140 ** (alphaIII + 2) -  25 ** (alphaIII+ 2)) / (260 ** (alphaIII + 2) - 0.1 ** (alphaIII + 2))
		f_rem_3 = np.trapz(lm_abh_3*lm_star_3**alphaIII,x=lm_star_3)/((260 ** (alphaIII + 2) - 0.1 ** (alphaIII + 2))/ (alphaIII+2))
		f_loss_12 = (260 ** (alphaI_II + 2) -  25 ** (alphaI_II+ 2)) / (260 ** (alphaI_II + 2) - 0.1 ** (alphaI_II + 2))
		f_rem_12 = np.trapz(lm_abh_12*lm_star_12**alphaI_II,x=lm_star_12)/((260 ** (alphaI_II + 2) - 0.1 ** (alphaI_II + 2))/ (alphaI_II+2))
		SFR_Mh = np.zeros(len(lz))
		for i in range(len(lz)):
			SFR_Mh[i] += SFR_UNIV(lm_h[i],lz[i])*f_rem_12*f_loss_12
		#print(SFR_Mh)
		M_ABH = np.trapz(SFR_Mh,x=T)
		#print('Total BH mass for halo with'+str(lm_h[-1])+' solar mass has a total BH mass of '+str(M_ABH)+' at redshift '+ str(z_f))
		if mode == 1:
			SFR_Mh_pop3 = [SFR_pop3(lm_h[-1],z,z_f)*f_rem_3*f_loss_3  for z in lz]
			M_ABH_pop3 = np.trapz(SFR_Mh_pop3,x=T)
			M_ABH_tot=M_ABH+ M_ABH_pop3
			if len(SFR_Mh)>1:
				SFR_HMXB_pop12 = np.gradient(SFR_Mh,T)*HMXB_Lifetime
				SFR_HMXB_pop3 =np.gradient(SFR_Mh_pop3,T)*HMXB_Lifetime
			elif len(SFR_Mh)==1:
				SFR_HMXB_pop12 = SFR_Mh
				SFR_HMXB_pop3 = SFR_Mh_pop3
			M_HMXB_pop3 = np.trapz(SFR_HMXB_pop3, x=T)
			M_HMXB_pop12 = np.trapz(SFR_HMXB_pop12, x=T)
			M_HMXB_pop3*=(M_HMXB_pop3>0)
			M_HMXB_pop12*=(M_HMXB_pop12>0) # to avoid the case of negative BH mass
			M_HMXB_tot = M_HMXB_pop12+M_HMXB_pop3
		if output == 0:
			return M_ABH_tot, lm_h[-1],M_HMXB_tot
		elif output == 1:
			return np.array([M_ABH,M_ABH_pop3,lm_h[-1],M_HMXB_pop12,M_HMXB_pop3])

# Calculate the BH mass, including the mass for active HMXBs for given halo mass and SFRD at redshift z
# 	output: 0: total BH mass; 1: black hole mass as remnants of Pop III stars/ Pop I/II stars separately
def M_BH(M_h,z_f,IMF, fit = 'Madau14', output= 0, HMXB_Lifetime=1e6):
	if IMF == 'Salpter':
		alphaIII =-2.35
	if IMF == 'TopHeavy':
		alphaIII =-1.35
	alphaI_II =-2.35
	if z_f>=50:
		if output == 0:
			return 0
		if output == 1:
			return 0,0,0,0
	else:
		lz = np.linspace(50,z_f,int(50-z_f)+1)
		lm_abh_12 = np.linspace(M_BH_f(25),M_BH_f(260),20)
		lm_abh_3 = np.linspace(M_BH_f(25),M_BH_f(140),20)
		lm_star_12= np.linspace(25,260,20)
		lm_star_3 = np.linspace(25,140,20)
		T = [TZ(z)/YR for z in lz] # number of year btw
		f_loss_3 = (140 ** (alphaIII + 2) -  25 ** (alphaIII+ 2)) / (260 ** (alphaIII + 2) - 0.1 ** (alphaIII + 2))
		f_rem_3 = np.trapz(lm_abh_3*lm_star_3**alphaIII,x=lm_star_3)/((260 ** (alphaIII + 2) - 0.1 ** (alphaIII + 2))/ (alphaIII+2))
		f_loss_12 = (260 ** (alphaI_II + 2) -  25 ** (alphaI_II+ 2)) / (260 ** (alphaI_II + 2) - 0.1 ** (alphaI_II + 2))
		f_rem_12 = np.trapz(lm_abh_12*lm_star_12**alphaI_II,x=lm_star_12)/((260 ** (alphaI_II + 2) - 0.1 ** (alphaI_II + 2))/ (alphaI_II+2))
		if output == 0:
			SFR_Mh = [SFR_pop12(M_h,z,z_f,fit)*f_rem_12*f_loss_12+SFR_pop3(M_h,z,z_f)*f_rem_3*f_loss_3  for z in lz]
			if len(SFR_Mh)>1:
				SFR_HMXB = np.gradient(SFR_Mh,T)*HMXB_Lifetime
			elif len(SFR_Mh)==1:
				SFR_HMXB = SFR_Mh
			M_ABH = np.trapz(SFR_Mh,x=T)
			M_HMXB = np.trapz(SFR_HMXB,x=T)
			M_HMXB*=(M_HMXB>0)
			return M_ABH,M_HMXB
		elif output == 1:
			SFR_Mh_pop3 = [SFR_pop3(M_h,z,z_f)*f_rem_3*f_loss_3  for z in lz]
			SFR_Mh_pop12 = [SFR_pop12(M_h,z,z_f,fit)*f_rem_12*f_loss_12  for z in lz]
			if len(SFR_Mh_pop12)>1: #set the initial condition at the beginning of the star formation
				SFR_HMXB_pop12 = np.gradient(SFR_Mh_pop12,T)*HMXB_Lifetime
				SFR_HMXB_pop3 =np.gradient(SFR_Mh_pop3,T)*HMXB_Lifetime
			elif len(SFR_Mh_pop12)==1:
				SFR_HMXB_pop12 = SFR_Mh_pop12
				SFR_HMXB_pop3 = SFR_Mh_pop3
			M_HMXB_pop3 = np.trapz(SFR_HMXB_pop3,x=T)
			M_HMXB_pop12 = np.trapz(SFR_HMXB_pop12,x=T)
			M_HMXB_pop3*=(M_HMXB_pop3>0)
			M_HMXB_pop12*=(M_HMXB_pop12>0) # to avoid the case of negative BH mass
			M_ABH_pop3 = np.trapz(SFR_Mh_pop3,x=T)
			M_ABH_pop12 = np.trapz(SFR_Mh_pop12,x=T)
			return np.array([M_ABH_pop3,M_ABH_pop12, M_HMXB_pop3,M_HMXB_pop12])


#calculate the total mass of active HMXBs from the


'''
if __name__ == '__main__':
	rep = 'Nhrat_test/'
	ncore = 4
	nbin = 32
	z = 20
	dmax = delta0 * 100
	rat = 1.
	fac = 1e-3
	beta = 0.7
	sk = False
	if not os.path.exists(rep):
		os.makedirs(rep)
	d0 = nh_z(z=z, dmax=dmax, fac=fac, rat=rat, beta=beta, sk=sk, ncore=ncore)
	totxt(rep+'nh_ref_z'+str(z)+'.txt',[[d0]],0,0,0)
	X, Y, Mh = nh_para(z=z, dmax=dmax, nbin=nbin, fac=fac, rat=rat, beta=beta, sk=sk, ncore=ncore)
	totxt(rep+'nh_z'+str(z)+'.txt',Mh,0,0,0)
	totxt(rep+'X_z'+str(z)+'.txt',X,0,0,0)
	totxt(rep+'Y_z'+str(z)+'.txt',Y,0,0,0)
	lowb = 1e-8
	fh = Mh/d0
	fh = fh + lowb*(fh<lowb)
	plt.contourf(X, Y, np.log10(fh), nbin*2)
	cb = plt.colorbar()
	cb.set_label(r'$f_{h}$',size=14)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$m_{\chi}c^{2}\ [\mathrm{GeV}]$')
	plt.ylabel(r'$\sigma_{1}\ [\mathrm{cm^{2}}]$')
	plt.tight_layout()
	plt.savefig(rep+'fhMap.pdf')
	plt.close()
'''
'''

if __name__ == '__main__':
	lM_h = np.logspace(7,13,7)
	lz = np.linspace(6,30,50)
	M_SMBH_ = [[ M_H * 2 * 1e-7 * (1 + z) / 10  * (M_H / 1e10) ** (2 / 3) for z in lz] for M_H in lM_h]
	M_star_ = [[1e5 * M_H / 1e8 for z in lz]for M_H in lM_h]

	M_star = [[M_star_Mh(M,z)for z in lz]for M in lM_h]
	M_SMBH = np.zeros([len(lz),len(lM_h)])
	for i in range(len(lz)):
		for j in range(len(lM_h)):
			M_SMBH[i,j] += M_SMBH_Mstar(M_star[j][i],lz[j])

	plt.plot(lz,M_star[0],color = 'r',label = '$10^{7}\rm \ M_{\odot}$')
	plt.plot(lz,M_star[1],color = 'g',label = '$10^{8}\rm \ M_{\odot}$')
	plt.plot(lz,M_star[2],color = 'b',label = '$10^{9}\rm \ M_{\odot}$')
	plt.plot(lz,M_star[3],color = 'm',label = '$10^{10}\rm \ M_{\odot}$')
	plt.plot(lz,M_star[4],color = 'y',label = '$10^{11}\rm \ M_{\odot}$')
	plt.plot(lz,M_star[5],color = 'c',label = '$10^{12}\rm \ M_{\odot}$')
	plt.plot(lz,M_star[6],color = 'k',label = '$10^{13}\rm \ M_{\odot}$')
	plt.plot(lz,M_star_[0],'-.',color ='r')
	plt.plot(lz,M_star_[1],'-.',color ='g')
	plt.plot(lz,M_star_[2],'-.',color ='b')
	plt.plot(lz,M_star_[3],'-.',color ='m')
	plt.plot(lz,M_star_[4],'-.',color ='y')
	plt.plot(lz,M_star_[5],'-.',color ='c')
	plt.plot(lz,M_star_[6],'-.',color ='k')
	plt.legend(fontsize=12)
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(1e2, 1e11)
	plt.xlabel(r'$z$',fontsize=12)
	plt.ylabel(r'$M_{\star}\ [\mathrm{M}_{\odot}]$',fontsize=12)
	plt.tight_layout()
	plt.savefig('Mstar_z.pdf')
	plt.close()


	plt.plot(lz,M_SMBH[:,0],color = 'r',label = '$10^{7} \rm \ M_{\odot}$')
	plt.plot(lz,M_SMBH[:,1],color = 'g',label = '$10^{8}\rm \ M_{\odot}$')
	plt.plot(lz,M_SMBH[:,2],color = 'b',label = '$10^{9} \rm \ M_{\odot}$')
	plt.plot(lz,M_SMBH[:,3],color = 'm',label = '$10^{10}\rm \ M_{\odot}$')
	plt.plot(lz,M_SMBH[:,4],color = 'y',label = '$10^{11}\rm \ M_{\odot}$')
	plt.plot(lz,M_SMBH[:,5],color = 'c',label = '$10^{12}\rm \ M_{\odot}$')
	plt.plot(lz,M_SMBH[:,6],color = 'k',label = '$10^{13}\rm \ M_{\odot}$')
	plt.plot(lz,M_SMBH_[0],'-.',color ='r')
	plt.plot(lz,M_SMBH_[1],'-.',color ='g')
	plt.plot(lz,M_SMBH_[2],'-.',color ='b')
	plt.plot(lz,M_SMBH_[3],'-.',color ='m')
	plt.plot(lz,M_SMBH_[4],'-.',color ='y')
	plt.plot(lz,M_SMBH_[5],'-.',color ='c')
	plt.plot(lz,M_SMBH_[6],'-.',color ='k')


	plt.legend(fontsize=12)
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(1e2, 1e11)
	plt.xlabel(r'$z$',fontsize=12)
	plt.ylabel(r'$M_{\mathrm{SMBH}}\ [\mathrm{M}_{\odot}]$',fontsize=12)
	plt.tight_layout()
	plt.savefig('M_SMBH_z.pdf')
	plt.close()
	



	lM_h = np.logspace(6,13,8)
	lz = np.linspace(6,30,25)
	IMF = 'Salpter'

	M_ABH= [[M_BH(M,z,IMF)for z in lz]for M in lM_h]

	plt.plot(lz, M_ABH[0], label='$10^{6}\rm \ M_{\odot}$')
	plt.plot(lz, M_ABH[1], color='r', label='$10^{7}\rm \ M_{\odot}$')
	plt.plot(lz, M_ABH[2], color='g', label='$10^{8}\rm \ M_{\odot}$')
	plt.plot(lz, M_ABH[3], color='b', label='$10^{9}\rm \ M_{\odot}$')
	plt.plot(lz, M_ABH[4], color='m', label='$10^{10}\rm \ M_{\odot}$')
	plt.plot(lz, M_ABH[5], color='y', label='$10^{11}\rm \ M_{\odot}$')
	plt.plot(lz, M_ABH[6], color='c', label='$10^{12}\rm \ M_{\odot}$')
	plt.plot(lz, M_ABH[7], color='k', label='$10^{13}\rm \ M_{\odot}$')

	plt.legend(fontsize=12,frameon=False)
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(1e2, 1e11)
	plt.xlabel(r'$z$', fontsize=12)
	plt.ylabel(r'$M_{\mathrm{BH}}\ [\mathrm{M}_{\odot}]$', fontsize=12)
	plt.tight_layout()
	plt.savefig('M_BH_z.pdf')
	plt.close()
'''

#write a file of text



