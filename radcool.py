# cooling rates (in erg/s/cm^3), all quantities in cgs units
from cosmology import *
#from numba import njit
kB = BOL

#@njit
def LambdaBre(T, nhII, nheII, nheIII, ne): #LAM(1)
	if T>5.e3:
		gff = 1.1+0.34*np.e**(-(5.5-np.log10(T))**2.0/3.0)
		L = 1.42e-27*gff*T**0.5*(nhII+nheII+4.0*nheIII)*ne
	else:
		L = 1.e-52
	return L
#@njit
def LambdaIC(T, z=20, ne=1e4):
	return ne*5.41e-36*(1+z)**4*T #LAM(11)
#@njit
def LambdaHeI(T, n, ne=1e4): #LAM(3)
	if T>8.e3:
		return n*ne*9.38e-22*T**0.5*np.e**(-285335.4/T)/(1+(T/10**5)**0.5)
	else:
		return 1.e-52
#@njit
def LambdaHeII(T, n, ne=1e4):
	if T>1.e4:
		L1 = n*ne*4.95e-22*T**0.5*np.e**(-631515.0/T)/(1+(T/10**5)**0.5) #LAM(4)
		L2 = 0.0 #n*ne**2*5.01e-27*T**(-0.1687)*np.e**(-55338/T)/(1+(T/10**5)**0.5) #???
	else:
		L1, L2 = 1.e-52, 0.0
	if T>5.e3:
		L3 = n*ne*1.55e-26*T**0.3647 #LAM(6)
		LD = n*ne*1.24e-13*np.e**(-470000.0/T)*(1.0+0.3*np.e**(-94000.0/T))/T**1.5 #??? LAM(8)
	else:
		L3, LD = 1.e-52, 1.e-52
	if T>8.e3:
		L4 = n*ne*5.54e-17*T**(-0.397)*np.exp(-473638.0/T)/(1+(T/10**5)**0.5) #LAM(10)
	else:
		L4 = 1.e-52
	return L1+L2+L3+L4+LD
#@njit
def LambdaHeIII(T, n, ne=1e4):
	if T>5.e3:
		L1 = n*ne*3.48e-26*T**0.5*(T/1000)**(-0.2)/(1+(T/10**6)**0.7) #LAM(7)
	else:
		L1 = 1.e-52
	if T>8.e3:
		L2 = 0.0 #n*ne**2*9.10e-27*T**(-0.1687)*np.e**(-13179/T)/(1+(T/10**5)**0.5) #???
	else:
		L2 = 0.0
	return L1+L2
#@njit
def LambdaHI(T, n, ne=1e4):
	if T>5.e3:
		L1 = n*ne*1.27e-21*T**0.5*np.e**(-157809.1/T)/(1.0+(T/10**5)**0.5) #LAM(2)
		L2 = n*ne*7.5e-19*np.e**(-118348.0/T)/(1.0+(T/10**5)**0.5) #LAM(9)
		return L1+L2
	else:
		return 2.e-52
#@njit
def LambdaHII(T, n, ne=1e4): #LAM(5)
	if T>5.e3:
		return n*ne*8.70e-27*T**0.5*(T/1000.0)**(-0.2)/(1.0+(T/1e6)**0.7)
	else:
		return 1.e-52
#@njit
def LambdaHD(T, n, nh0, nh=1e8):
	aa, bb, omeg, phi, c1, c2, d1, d2 = -26.2982, -0.215807, 2.03657, 4.63258, 0.283978, -1.27333, -2.08189, 4.66288
	y = np.log10(nh) #np.log10(np.min([np.max([nh,1]),1e8]))
	x = np.log10(T)#np.min([np.max([np.log10(T),np.log10(30)]),np.log10(3)+3])
	w = 0.5*y+aa*x**bb-(0.25*y**2+(c1*y+c2)*np.sin(omeg*x+phi)+(d1*y+d2))**0.5
	LHD = 10**w
	if 1: #y!=np.log10(nh) or x!=np.log10(T):
		w = -55.5725+56.649*np.log10(T)-37.9102*(np.log10(T))**2+12.698*(np.log10(T))**3-2.02424*(np.log10(T))**4+0.122393*(np.log10(T))**5
		LTE = 10**w	
		g10 = (4.4e-12+3.6e-13*T**0.77)/1.27
		g21 = (4.1e-12+2.1e-13*T**0.92)/1.27
		E10, E21 = 128.0, 255.0
		LOW = 2*g10*E10*kB*np.exp(-E10/T)+5.0/3.0*g21*E21*kB*np.exp(-E21/T)
		LOW = LOW*nh0
		LHD = LOW*LTE/(LOW+LTE)
	return LHD*n
#@njit
def LambdaLiH(T, n, nh=1e3):
	a = np.zeros(5)#,dtype='f')
	a[0], a[1], a[2], a[3], a[4] = -31.47, 8.817, -4.144, 0.8292, -0.04996
	Llow = 10**np.array([a[x]*(np.log10(T))**x for x in range(len(a))]).sum()*nh
	a = np.zeros(7)#,dtype='f')
	a[0], a[1], a[2], a[3], a[4], a[5], a[6] = -31.894, 34.3512, -31.0805, 14.9459, -3.72318, 0.455555, -0.0216129
	LLTE = 10**np.array([a[x]*(np.log10(T))**x for x in range(len(a))]).sum()
	#ncrit = LLTE/Llow
	LLi = LLTE*Llow/(LLTE+Llow)
	return n*LLi
#@njit
def LambdaH2(T, nh2, nh):
	WH2 = -103.0+97.59*np.log10(T)-48.05*(np.log10(T))**2+10.8*(np.log10(T))**3-0.9032*(np.log10(T))**4
	if WH2>-35:
		LOW = 10**WH2
	else:
		LOW = 0.0
	LOW = LOW*nh
	T3 = T/1000.0
	RLTE = (9.5e-22*T3**3.76*np.e**(-(0.13/T3)**3)/(1.0+0.12*T3**2.1)+3.e-24*np.e**(-0.51/T3))
	VLTE = (6.7e-19*np.e**(-5.86/T3)+1.6e-18*np.e**(-11.7/T3))
	LTE = RLTE+VLTE
	#XNCRIT = nh*(LTE/LOW)
	#LH2 = LTE/(1.0+XNCRIT/nh)
	LH2 = LOW*LTE/(LTE+LOW)
	return nh2*LH2
	
# dT/dt determined by the cooling rate
# ntot: overall number density, n: number densities of different species
#@njit
def cool(T, ntot, n, J_21, z, gamma, X, T0):
	Tcmb = T0*(1+z)
	#if T<Tcmb:
	#	return 0.0
	mH = PROTON*4.0/(1.0+3.0*X)
	nh = n[0]+n[1]+n[2]+2.0*(n[3]+n[4])
	ny = n
	Gam = 0.0 #J_21*(5.1e-23*n[0] + 1.2e-22*n[6] + 2.5e-24*n[7])
	L = np.zeros(4)
	"""
	L = np.zeros(11,dtype='float')
	if T>5e3:
		gff=1.1+0.34*np.exp(-(5.5-np.log10(T))**2/3.0)
		L[0] = 1.42e-27*gff*np.sqrt(T)*(ny[1]+ny[7]+4.0*ny[8])*ny[5]
	else:
		L[0] = 1.e-42
	T5 = T/1.e5
	XH1 = np.sqrt(T)/(1.0+np.sqrt(T5))
	if T>5e3:
		L[1]=1.27e-21*XH1*np.exp(-157809.1/T)*ny[0]*ny[5]
	else:
		L[1]=1.e-42
	if T>8e3:
		L[2]=9.38e-22*XH1*np.exp(-285335.4/T)*ny[6]*ny[5]
	else:
		L[2]=1.e-42
	if T>1e4:
		L[3]=4.95e-22*XH1*np.exp(-631515.0/T)*ny[7]*ny[5]
	else:
		L[3]=1.e-42
	T3=T/1.e3
	T6=T/1.e6
	XH4 = 1.0/(1.e0+T6**0.7)
	if T>5e3:
		L[4]=8.7e-27*XH4*np.sqrt(T)*T3**(-0.2)*ny[1]*ny[5]
	else:
		L[4]=1.e-42
	if T>5e3:
		L[5]=1.55e-26*T**(0.3647)*ny[7]*ny[5]
	else:
		L[5]=1.e-42
	if T>5e3:
		L[6]=3.48e-26*XH4*np.sqrt(T)*T3**(-0.2)*ny[8]*ny[5]
	else:
		L[6]=1.e-42
	XH5=np.exp(-470000.e0/T)*(1.e0+0.3e0*np.exp(-94000.e0/T))
	if T>5e3:
		L[7]=1.2e-13*XH5*T**(-1.5)*ny[7]*ny[5]
	else:
		L[7]=1.e-42
	XH2=1.0/(1.0+np.sqrt(T5))
	if T>5e3:
		L[8]=7.5e-19*XH2*np.exp(-118348.0/T)*ny[0]*ny[5]
	else:
		L[8]=1.e-42
	XH3=XH2*T**(-0.397)
	if T>8e3:
		L[9]=5.54e-17*XH3*np.exp(-473638.0/T)*ny[7]*ny[5]
	else:
		L[9]=1.e-42
	L[10]=5.4e-36*(1.e0+z)**4*ny[5]*(T-Tcmb)
	"""
	L[1] = LambdaIC(T, z, n[5]) - LambdaIC(Tcmb, z, n[5]) 
	if 1: #T>=Tcmb:
		L[0] = LambdaBre(T, n[1], n[7], n[8], n[5]) 
		L[2] = LambdaHI(T, n[0], n[5]) + LambdaHII(T, n[1], n[5]) 
		L[3] = LambdaHeI(T, n[6], n[5]) + LambdaHeII(T, n[7], n[5]) + LambdaHeIII(T, n[8], n[5]) 
		LH2, LHD, LLiH = 0.0, 0.0, 0.0
	if T<=2e4 and T>=Tcmb:
		nhd = n[11]# 0.01*xnd 
		LH2 = LambdaH2(T, n[3], n[0]) - LambdaH2(Tcmb, n[3], n[0]) 
		LHD = LambdaHD(T, nhd, n[0], nh) - LambdaHD(Tcmb, nhd, n[0], nh) 
		LLiH = LambdaLiH(T, n[16], n[0]) - LambdaLiH(Tcmb, n[16], n[0])
	Lam = np.sum(L)+LH2+LHD+LLiH-Gam
	expansion = 0.0#- 3*kB*T*Hubble(z)
	out = (-Lam/ntot + expansion)*(gamma-1.0)/BOL
	return out





