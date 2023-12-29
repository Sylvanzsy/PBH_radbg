from txt import *
from scipy.optimize import root
kB = 1.3806e-16 # Boltzmann constant

def xh(X = 0.75):
	return 4.0*X/(1.0+3.0*X)

# reaction rate coefficients (in cgs units)
def rates(J_21, T, nh, nh2, z, T0, fnth):
	k = np.zeros(34+10+2+1+3+2,dtype='float')
	Tcmb = T0*(1+z)
	tt = T**0.5
	t5fac = 1.0/(1.0+(T/1.e5)**0.5)
	logT = np.log10(T)
	#nh = n[0]+n[1]+n[2]+2.0*(n[3]+n[4])
	LJ = 4.8e19*(T/nh)**0.5
	NH2 = nh2*LJ
	# Recombination: Hep from Black (1981, MNRAS 197, 553) Hp and Hepp from Cen (1992, ApJS 78, 341)
	if T>5e3 or 1: 
		k5a = 1.5e-10*T**-0.6353 
	else:
		k5a = 0.0
	if T>10**3.6: # 
		k5b = 1.9e-3*(T**-1.5)*np.e**(-4.7e5/T)*(1.0+0.3*np.e**(-9.4e4/T))
	else:
		k5b = 0.0
	k[4] = k5a + k5b # He+ + e- = He dielectronic recombination of HeII (Black 1992)
	k[3] = 8.4e-11/tt*(T/1.e3)**-0.2/(1.0+(T/1.e6)**0.7) # H+ + e- = H + hnu
	k[5] = 3.36e-10/tt*((T/1.e3)**-0.2)/(1.0+(T/1.e6)**0.7) # He++ + e- = He+ + hnu (Cen 1992)
	# Collisional ionization: from Black, with Cen fix at high-T):
	if T>10**2.8:# H + e- = H+ + 2e-
		k[0] = 5.85e-11*tt*np.e**(-157809.1/T)*t5fac 
	else:
		k[0] = 0.0
	if T>1e3: # He + e- = He+ + 2e- (Cen 1992)
		k[1] = 2.38e-11*tt*np.e**(-285335.4/T)*t5fac
	else:
		k[1] = 0.0
	if T>10**3.4: # He+ + e- = He++ + 2e- (Cen 1992)
		k[2] = 5.68e-12*tt*np.e**(-631515.0/T)*t5fac
	else:
		k[2] = 0.0
	# Additional
	#if T<=4000.0: # H + H+ = H2+ + hnu ???
	#	k[6] = 1.38e-23*(T**1.845)
	#else:
	#	if T<1e5:
	#		k[6] = -6.157e-17 + 3.255e-20*T - 4.152e-25*T**2
	#	else:
	#		k[6] = 0.0
	alpha = -19.38-1.523*logT+1.118*logT**2-0.1269*logT**3 # GP98
	k[6] = 10**alpha
	if k[6]<=0.0:
		k[6] = 0.0
	k[7] = 6.4e-10 # H2+ + H = H2 + H+
	# Abel et al. 1997
	if T<6000 or 1: # H + e- = H- + hnu
		k[8] = 1.429e-18*T**0.762*T**(0.1523*logT)*T**(-3.274e-2*logT**2) 
	else:
		k[8] = 3.802e-17*T**(0.1998*logT)*10**(4.04e-5*logT**6-5.45e-3*logT**4)
	#k[9] = 1.3e-9 # H + H- = H2 + e-
	if T<300:
		k[9] = 1.5e-9
		#logT2 = logT-2
		#k[9] = 10**(-14.4-0.15*logT2**2-7.9e-3*logT2**4) # H + H- = H2 + e- CC11
	else:
		k[9] = 4e-9*T**-0.17 # GP98
	#k[10] = 1.68e-8*(T/300.0)**-0.29 # e- + H2+ = 2H ???
	k[10] = 2e-7*T**-0.5 # GP98
	k[11] = 5.0e-6/tt # H2+ + H- = H2 + H Too efficient???
	#k[12] = 7.0e-7/tt # H- + H+ = 2H  
	k[12] = 5.7e-6/tt + 6.8e-8 - 9.2e-11*tt + 4.4e-13*T
	if T>10**2.2: # e- + H2 = H- + H   ?
		k[13] = (2.7e-8/(T*tt))*np.e**(-43000.0/T)*t5fac #!!!
	else:
		k[13] = 0.0
	if T<=10**2.2: # H2 + H = 3H, H2 + H2 = H2 + H + H   ???
		k[14], k[15] = 0.0, 0.0 #!!!
	else:
		x = logT-4.0
		ncr_1 = 10**(4.0-0.416*x-0.327*x**2)
		ncr_2 = 10**(4.13-0.968*x+0.119*x**2)
		kh_1 = 3.52e-9*np.e**(-43900.0/T)
		kh_2 = 5.48e-9*np.e**(-53000.0/T)
		if T>=7390:
			kl_1 = 6.11e-14*np.e**(-29300.0/T)
		else:
			if T>10**2.6:
				kl_1 = 2.67e-15*np.e**(-(6750.0/T)**2)
			else:
				kl_1 = 0.0
		if T>=7291:
			kl_2 = 5.22e-14*np.e**(-32200.0/T)
		else:
			if T>10**2.8:
				kl_2 = 3.17e-15*np.e**(-(4060.0/T)-(7500.0/T)**2)
			else:
				kl_2 = 0.0
		k[14] = kh_1*(kl_1/kh_1)**(1.0/(1.0+nh/ncr_1))
		k[15] = kh_2*(kl_2/kh_2)**(1.0/(1.0+nh/ncr_2))
	# Shapiro and Kang
	#if T>10**2 : # H2 + H+ = H2+ + H
	#	k[16] = 2.4e-9*np.e**(-21200.0/T) # !!!
	#else:
	#	k[16] = 0.0
	if T<=1e4: # GP98
		k[16] = 3e-10*np.exp(-21050/T)
	else:
		k[16] = 1.5e-10*np.exp(-14000/T)
	#k[16] = np.exp(-33.081+6.3e-5*T-2.35e4/T-1.87e-9*T**2) # H2 + H+ = H2+ + H, CC11
	if T>10**2.6: # H2 + e- = 2H + e-
		k[17] = 4.38e-10*np.e**(-102000.0/T)*T**0.35 # !!!
	else:
		k[17] = 0.0
	if T>10**1.6: # H- + e- = H + 2e-
		k[18] = 4.e-12*T*np.e**(-8750.0/T)
	else:
		k[18] = 0.0
	if T>10**1.6: # H- + H = 2H + e-
		k[19] = 5.3e-20*T**2.17*np.e**(-8750.0/T)
	else:
		k[19] = 0.0
	#if T>1e4: # H- + H+ = H2+ + e-   ?
	#	k[20] = 1.e-8*T**-0.4
	#else:
	#	if T>10**1.8:
	#		k[20] = 4.e-4*T**-1.4*np.e**(-15100.0/T)
	#	else:
	#		k[20] = 0.0
	if T<=8000: # H- + H+ = H2+ + e- GP98
		k[20] = 6.9e-9*T**-0.35
	else:
		k[20] = 9.6e-7*T**-0.9
	# Radiative processes
	k[21] = 5.16e-12*J_21 # H + hnu = H+ + e-
	k[22] = 7.9e-12*J_21
	k[23] = 1.9e-13*J_21
	k[24] = 2.0e-11*J_21
	k[25] = 4.8e-12*J_21
	k[26] = 1.2e-11*J_21
	#J_LW = 1e4
	if NH2<=1:
		xshield = 1.0
	else:
		xshield = NH2**-0.75
	k[27] = 1.2e-12*J_21*xshield
	# Three-body reations from Palla et al.,1983,ApJ 271,632
	k[28] = 5.5e-29/T # H + H + H = H2 + H
	k[29] = 0.125*k[28] # H + H + H2 = H2 + H2
# HD from Galli 1998
	k[30]=3.7e-10*T**0.28*np.e**(-43.0/T) # D + H+ = D+ + H
	k[31]=3.7e-10*T**0.28 # D+ + H = D + H+
	k[32]=2.1e-9 # D+ + H2 = H+ + HD
	k[33]=1.e-9*np.e**(-464.0/T) # HD + H+ = H2 + D+

	k[34]=1.036e-11/((T/107.7)**0.5 * (1+(T/107.7)**0.5)**0.612 * (1+(T/1.177e7)**0.5)**1.388)
	# Li+ + e- = Li + hnu
	k[35]=6.3e-6*T**-0.5-7.6e-9+2.6e-10*T**0.5+2.7e-14*T # Li- + H+ = Li + H
	k[36]=6.1e-17*T**0.58*np.exp(-T/17200) # Li + e- = Li- + hnu
	k[37]= 1.0/(5.6e19*T**-0.15+7.2e15*T**1.21) # Li(2S) + H = LiH + hnu
	k[38]=4.0e-10 # Li- + H = LiH + e-
	k[39]=2.0e-11 # LiH + H = Li + H2
	k[40]=10**(-22.4+0.999*np.log10(T)-0.351*np.log10(T)**2) # Li+ + H = LiH+ + hnu
	k[41]=4.8e-14*T**-0.49 # Li + H+ = LiH+ + hnu
	k[42]=3.8e-7*T**-0.47 # LiH+ + e- = Li + H
	k[43]=3.0e-10 # LiH+ + H = Li+ + H2
	k[44]=0.0#2.5e-40*T**7.9*np.exp(-T/1210)
	k[45]=0.0#1.7e-13*T**-0.051*np.exp(-T/282000)
	k[46]=0.0#4.e-10

	S2 = 1.0/(1.0+3.0*np.exp(-1.85*1.6*10**-12/T/kB))
	k[37]=k[37]*S2 + (1.9e-14*T**-0.34+2.0e-16*T**0.18*np.exp(-T/5100.0))*(1.0-S2)
	if k[35]<=0:
		k[35] = 0.0

	# CMB (GP98)
	k[47] = 0.0 #2.41e15*Tcmb**1.5 * np.exp(-39472/Tcmb)*8.76e-11*(1+z)**-0.58 # H + gamma = H+ + e-
	if z>0:
		k[48] = 1.1e-1*Tcmb**2.13* np.exp(-8823/Tcmb) # H- + gamma = H + e-
		k[48] += 8e-8*Tcmb**1.3*np.exp(-2.3e3/Tcmb) *fnth # H- + gamma = H + e- CC11, non-thermal photons
		k[49] = (20*Tcmb**1.59* np.exp(-82000/Tcmb) + 1.63e7*np.exp(-32400/Tcmb))*0.5 # H2+ + gamma = H + H+ GP98, old
		k[50] = 90.0*Tcmb**1.48*np.exp(-335000/Tcmb) # H2+ + gamma = 2H+ + e- GP98, old
		k[51] = 2.9e2*Tcmb**1.56*np.exp(-178500/Tcmb) # H2 + gamma = H2+ + e-
	#k[51] = 3.0659e9*np.exp(-1.9e5/T) # CC11
	
	#k[51] = 1.3e3*Tcmb**1.45*np.exp(-60500/Tcmb) # Li + gamma = Li+ + e-
	#k[52] = 1.8e2*Tcmb**1.4*np.exp(-8100/Tcmb) # Li- + gamma = Li + e-
	#k[53] = 8.3e4*Tcmb**0.3*np.exp(-29000/Tcmb) # LiH + gamma = Li + H
	#k[54] = 7e2*np.exp(-1900/Tcmb) # LiH+ + gamma = Li+ + H
	return k

# evolve abundances with the implicit Euler method
def chemistry1(T, nin, dt0, epsH, J_21, Ns, xnh, xnhe, xnd, xnli, Cr0, Ds0, z = 5, T0 = 2.726, nmax = 100, Tcut=300, fnth=0):
	"""
		T: temperature
		nin: input number densities
		dt0: overall timestep
		epsH: maximum changes of the abundances of key species (e-, H2 and HD)
		Ns: number of species
		xnh, xnhe, xnd, xnli: number densities of H, He, D, Li nuclei
		Cr0, Ds0: formation and destruction rates from the last timestep
		nmax: set the minimum timestep to dt0/nmax
		Tcut: use equilibrium abundances for H2+ and H- when Tcmb(z)<Tcut
		fnth: contribution of non-thermal CMB photons
	"""
	total = 0
	out = np.zeros(Ns,dtype='float')
	dt_cum = 0.0
	dt = dt0
	Cr, Ds = np.zeros(Ns,dtype='float'), np.zeros(Ns,dtype='float')
	ny = nin
	while dt_cum<dt0:
		k = rates(J_21, T, xnh, nin[3], z, T0, fnth)
		Cr[5]=(k[21]+k[47])*ny[0]+k[22]*ny[6]+k[23]*ny[7]+(k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5] #+ k[48]*ny[2] + k[50]*ny[4] + k[51]*ny[3]
		Ds[5]=k[3]*ny[1]+k[4]*ny[7]+k[5]*ny[8]
		if (dt*abs(Cr[5]-Ds[5]*ny[5])>epsH*abs(ny[5])) and (ny[5]>0) and dt>dt0/nmax:
			dt = dt/2.0 #epsH*ny[5]/abs(Cr[5]-Ds[5]*ny[5])#
			continue
		#if ny[5]<=1e-4*xnh and T<2e4:
		for i in [3, 11]:
			if (dt*abs(Cr0[i]-Ds0[i]*ny[i])>epsH*abs(ny[i])) and (ny[i]/xnh>1e-10):
				dt = max(epsH*abs(ny[i]/(Cr0[i]-Ds0[i]*ny[i])),dt0/nmax) #dt/2.0
		if dt>1.e5*3.14e7:
			dt = 1.e5*3.14e7
		if dt + dt_cum>dt0:
			dt = dt0 - dt_cum
			dt_cum = dt0
		else:
			dt_cum += dt
		Cr[5]=k[21]*ny[0]+k[22]*ny[6]+k[23]*ny[7]+(k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5] #+k[48]*ny[2] + k[50]*ny[4] + k[51]*ny[3]# +k[24]*ny[2]+k[26]*ny[3]
		Ds[5]=k[3]*ny[1]+k[4]*ny[7]+k[5]*ny[8]
		ny[5]=(ny[5]+Cr[5]*dt)/(1.e0+Ds[5]*dt)

		Cr[0]=k[3]*ny[1]*ny[5] + k[48]*ny[2] + k[49]*ny[4] + 2*k[50]*ny[4];
		Ds[0]=k[0]*ny[5]+k[21]+k[47];
		ny[0]=(ny[0]+Cr[0]*dt)/(1.e0+Ds[0]*dt);

		Cr[1]=k[0]*ny[5]*ny[0] + (k[21]+k[47])*ny[0] + k[49]*ny[4];
		Ds[1]=k[3]*ny[5];
		ny[1]=(ny[1]+Cr[1]*dt)/(1.e0+Ds[1]*dt);

		Cr[6]=k[4]*ny[7]*ny[5];
		Ds[6]=k[1]*ny[5]+k[22];
		ny[6]=(ny[6]+Cr[6]*dt)/(1.e0+Ds[6]*dt);

		Cr[7]=(k[1]*ny[5]+k[22])*ny[6]+k[5]*ny[5]*ny[8];
		Ds[7]=(k[2]+k[4])*ny[5]+k[23];
		ny[7]=(ny[7]+Cr[7]*dt)/(1.e0+Ds[7]*dt);

		Cr[8]=k[2]*ny[7]*ny[5]+k[23]*ny[7];
		Ds[8]=k[5]*ny[5];
		ny[8]=(ny[8]+Cr[8]*dt)/(1.e0+Ds[8]*dt);
#/**** calculate equilibrium abundance for H- *********************/
		XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] + k[48] +k[11]*ny[4];
		if T0*(1+z)<Tcut:
			#print(XNUM1, XDENOM1)
			if XDENOM1>1e-52:
				ny[2]=XNUM1/XDENOM1
			#else:
			#	print('Weird production of H-.')
		else:
			Cr[2] = XNUM1
			Ds[2] = XDENOM1
			ny[2] = (ny[2]+Cr[2]*dt)/(1.e0+Ds[2]*dt);
#/**** calculate equilibrium abundance for H2+ ********************/
		XNUM2=(k[6]*ny[0] + k[16]*ny[3]+k[20]*ny[2])*ny[1] + k[26]*ny[3] +k[51]*ny[3];
		XDENOM2=k[7]*ny[0]+k[10]*ny[5] + k[49] + k[50] +k[11]*ny[2]+k[25];
		if T0*(1+z)<Tcut:
			if XDENOM2>1e-52:
				ny[4]=XNUM2/XDENOM2
			#else:
			#	print('Weird production of H2+.')
		else:
			Cr[4] = XNUM2
			Ds[4] = XDENOM2
			ny[4] = (ny[4]+Cr[4]*dt)/(1.e0+Ds[4]*dt);
		Ds[3]=k[13]*ny[5]+k[14]*ny[0]+k[15]*ny[3]+k[16]*ny[1]+k[17]*ny[5]+k[26]+k[27];
		Cr[3]=k[7]*ny[4]*ny[0]+k[9]*ny[2]*ny[0]#+k[11]*ny[4]*ny[2];
		ny[3]=(ny[3]+Cr[3]*dt)/(1.e0+Ds[3]*dt);
		tform, tdest = 0.0, 0.0
		if Cr[3]>1e-52:
			tform=ny[3]/Cr[3];
		if Ds[3]>1e-52:
			tdest=1.e0/Ds[3];

		ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
		if (ny[0] < 0.e0):
			ny[0]=0.e0;
		ny[6]=xnhe-ny[7]-ny[8];
		if (ny[6] < 0.e0):
			ny[6]=0.e0;
		ny[7]=xnhe-ny[6]-ny[8]
		if (ny[7]<0.0):
			ny[7]=0.0
		ny[8] = xnhe-ny[6]-ny[7]
		if ny[8]<0.0:
			ny[8]=0.0
		ny[1]=ny[5]+ny[2]-ny[4]-ny[7]-2.e0*ny[8];
		if (ny[1] < 0.e0):
			ny[1]=0.e0;
		ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
		if (ny[0] < 0.e0):
			ny[0]=0.e0;

		XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] + k[48] +k[11]*ny[4];
		ny[2]=XNUM1/XDENOM1;

		for i in range(9):
			if ny[i]<1e-30:
				ny[i] = 0.0

		# D network
		Cr[10]=k[30]*ny[9]*ny[1]+k[33]*ny[11]*ny[1]
		Ds[10]=k[3]*ny[5]+k[31]*ny[0]+k[32]*ny[3]
		ny[10]=min((ny[10]+Cr[10]*dt)/(1.0+Ds[10]*dt),xnd)
		Cr[11]=k[32]*ny[10]*ny[3]
		Ds[11]=k[33]*ny[1]
		ny[11]=min((ny[11]+Cr[11]*dt)/(1.0+Ds[11]*dt),xnd)
		ny[9]=xnd-ny[11]-ny[10]

		# Li network
		#Cr[12] = k[34]*ny[13]*ny[5]+k[42]*ny[15]*ny[5]+k[35]*ny[14]*ny[1]+k[39]*ny[16]*ny[0]
		#Ds[12] = k[41]*ny[1]+k[36]*ny[5]+k[37]*ny[0]+(k[44]+k[45])*ny[s1]+k[46]*ny[2]
		#ny[12] = min((ny[12]+Cr[12]*dt)/(1.0+Ds[12]*dt),xnli)

		#Cr[13] = k[43]*ny[0]*ny[15]+(k[44]+k[45])*ny[12]*ny[1]
		#Ds[13] = k[34]*ny[5]+k[40]*ny[1]
		#ny[13] = min((ny[13]+Cr[13]*dt)/(1.0+Ds[13]*dt),xnli)

		#Cr[15] = k[40]*ny[0]*ny[13]+k[41]*ny[1]*ny[12]
		#Ds[15] = k[43]*ny[0]+k[42]*ny[5]
		#ny[15] = min((ny[15]+Cr[15]*dt)/(1.0+Ds[15]*dt),xnli)

		#Cr[16] = k[37]*ny[12]*ny[0]+k[38]*ny[14]*ny[0]+k[46]*ny[2]*ny[12]
		#Ds[16] = k[39]*ny[0]
		#ny[16] = min((ny[16]+Cr[16]*dt)/(1.0+Ds[16]*dt),xnli)

		#Cr[14] = k[36]*ny[12]*ny[5]
		#Ds[14] = k[35]*ny[1]+k[38]*ny[0]
		#ny[14] = min((ny[14]+Cr[14]*dt)/(1.0+Ds[14]*dt),xnli)

		#ny[12] = min(max(0,xnli-ny[15]-ny[13]-ny[14]-ny[16]),xnli)
		#ny[13] = min(max(0,xnli-ny[12]-ny[14]-ny[15]-ny[16]),xnli)
		#ny[14] = min(max(0,xnli-ny[12]-ny[13]-ny[15]-ny[16]),xnli)
		#ny[15] = min(max(0,xnli-ny[12]-ny[13]-ny[14]-ny[16]),xnli)
		#ny[16] = min(max(0,xnli-ny[12]-ny[14]-ny[15]-ny[13]),xnli)

		for i in range(17):
			if ny[i]<1e-30:
				ny[i] = 0.0
		#dt = min(dt*2.0, dt0 - dt_cum)
		#print(dt)
		dt = dt0
		total += 1
	tform, tdest = 0.0, 0.0
	return [ny, dt_cum, total, tform, tdest, Cr, Ds]


