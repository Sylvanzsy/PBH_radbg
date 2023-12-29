from cosmology import *
from txt import *
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.mplot3d
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from scipy.interpolate import interp1d
#plt.style.use('test2')
#plt.style.use('tableau-colorblind10')

def sigma_E0(E, s0=6.06e-16, E0=13.6/2):
	y = E/E0
	return s0*y**-1.5*(1+y**0.5)**-4
# eq(3.1)

iondata = np.log10(retxt('Olive_ion.txt',2))
interion = interp1d(*iondata)

def sigma_E(E, func=interion, E1=30, E2=2e9, sigma_0=6.06e-16, E0=13.6/2):
	lE = E*(E1<=E)*(E<=E2) + E2*(E>E2) + E1*(E<E1)
	s = sigma_E0(E, sigma_0, E0)*(E<E1) + PROTON/(10**func(np.log10(lE)))*(E>=E1)
	return s

# also include the case for E>30, refer to Olive(2014)
'''''
Sedata = np.array(retxt('Se.txt',2))
interSe = interp1d(np.log10(Sedata[0])+3, Sedata[1])

def Se(E, func=interSe, E1=1e4, E2=3e6, dE=0.03):
	lE = E*(E1<=E)*(E<=E2) + E2*(E>E2) + E1*(E<E1)
	slope2 = (func(np.log10(E2))-func(np.log10(E2)-dE))/dE
	slope1 = (func(np.log10(E1)+dE)-func(np.log10(E1)))/dE
	s = func(np.log10(lE))*(E<=E2)*(E>=E1)
	s += (func(np.log10(E1)) + slope1*np.log10(E/E1)) * (E<E1)
	s += (func(np.log10(E2)) + slope2*np.log10(E/E2)) * (E>E2)
	s[s<0] = 0
	return s
'''''
def mdot_bondi(m, n, v, er=0.057, mu=1.22, fed=1.0):
	return min(2.64e-7*m*(er/0.1)*(n/0.93)*mu/(v/10)**3, fed)
# equation (2.3), M_edd as the largest rate
# assume non-rotating Schwarzchild BH, which give 0.057 as the radiative efficiency, and L=epsilon_o M^{dot}

def thin_disk(nu, m, n, v, er=0.057, fed=1.0):
	mdot = mdot_bondi(m, n, v, er, fed=1e8)
	Ti = 53.3*(n/0.93)**0.25/(v/10)**0.75  #eq(3.5)
	Tmax = 0.488*Ti
	To = 6.13e-5*(n/0.93)**0.25/m**0.5*(v/10)**1.75 #eq (3.8)
	ca = 1.27e29*m**2*(n/0.93)**0.75/(v/10)**2.25*er/0.057 #eq(3.7)
	Lnu = ca*(Tmax/To)**(5./3.)*(nu/Tmax)**2 * (nu<=To)
	Lnu += ca*(nu/Tmax)**(1./3.) * (nu>To)*(nu<Tmax)
	Lnu += ca*(nu/Tmax)**2*np.exp(1.0-nu/Tmax) * (nu>=Tmax)
	if mdot>fed:
		Lnu = Lnu*fed/mdot # renormalize to Eddington Luminosity if accretion ratio is calculated to be larger
	return Lnu, 10*Ti
	
def te_func(taues, alphac):
	return ((4/taues**(1.0/alphac)-3.0)**0.5-1.0)/8.0
#eq(3.22)
	
def ac_func(ac, te, m, mdot, alpha, beta, rmin):
	rat = 1.78e-5*te**-8.15*alpha**-0.5*beta*(1-beta)**-1.5*rmin**0.75*m**0.5*mdot**-0.25 #eq(3.24)-1
	Cf = 20.2*alpha**0.5*(1-beta)**-0.5*rmin**1.25/te*m**0.5*mdot**-0.75 #eq(3.24)-2
	acn = 1.0-np.log((1-ac)*rat+1.0)/np.log(Cf)  #eq(3.23)
	return acn
	
def ADAF(nu, m, mdot, alpha=0.1, beta=10/11., delta=0.3, rmin=3.0, er=0.057, eps=1e-2, nmax=100, mode=0, cont=0):
	madaf = 1
	if cont>0:
		fac = 0.075/er
	else:
		fac = 1.0
	if mode>0:
		madaf = 0.1*alpha**2 *fac
	meadaf = 1e-3*alpha**2 *fac
	#if mdot<madaf:
	Ac = 1.1*(mdot<=meadaf) #A_c^(1/7)
	Ac += 1.1*(mdot>meadaf)#*(mdot<madaf)
	te = 0.17/Ac*delta**(1.0/7.)*alpha**(3/14.)/(1-beta)**(1/14.)
	te *= rmin**(3./28.)*m**(1./14.)/mdot**(5./28.) # eq(3.21) for both eADAF and ADAF regime
	#else:
	A = 1+4*te+16*te**2 # amplification factor
	taues = 12.4*mdot/alpha/rmin**0.5 # optical depth btw eq(3.14) and eq(3.15)
	ac = -np.log(taues)/np.log(A) # eq(3.14)
	if mdot>madaf: # LHAF regime
		ac = 0.75 # initial guess
		for i in range(nmax):
			te = te_func(taues, ac) # eq(3.22)
			acnew = ac_func(ac, te, m, mdot, alpha, beta, rmin) #eq(3.23)
			if abs(ac-acnew)<eps:
				break
			else:
				ac = 0.5*(acnew+ac)# average new and old value, to get convergence by iterative method
		if i==nmax-1:
			print('Te does not converge!')
	alphac = ac
	nup = 1.83e-2/(alpha/0.1)**0.5*((1-beta)*11)**0.5
	nup *= te**2/(rmin/3.0)**1.25/m**0.5*(mdot*1e8)**0.75 # eq(3.9)
	Lp = 5.06e38/alpha*(1.0-beta)*m*mdot**1.5*te**5/rmin**0.5 #eq(3.11)
	Power = nup*Lp#*(0.71+1/(1-alphac)*((6.25*1e7*(te*5.93)/(nup*2.417*1e14/1e12))**(1-alphac)-1)) #eq(3.12) and (3.15)
	L_pbh = heat_sub(mdot) * mdot * 1.44 * 1e17 * m * (0.1 / er) * SPEEDOFLIGHT ** 2
	normFac = L_pbh / Power # normalize the total power from synchrotron (as the dominant source) to total emission power
	#print('For {} BH with mdot {}, the normalization factor is',m,mdot)
	#print(normFac)
	Lnu = Lp*normFac*(nu/nup)**-alphac #eq(3.13)
	return Lnu, 3*te*0.511e6 # upper bound for frequency(3k T_e = theta_e * m_e c^2)
	
def outflow(m, n, v, ei=13.6, ne=100, l=1.0, fh=1.0/3.0, rmin=3, fv=0.2, s=0.7, er=0.057):
	mdot = mdot_bondi(m, n, v, er)
	rb = GRA*m*Msun/(v*1e5)**2  # Bondi radius, assume ISM at rest, v in units of km/s
	rs = 2*GRA*m*Msun/SPEEDOFLIGHT**2 #shwarzschild radius
	rat = rb/rs
	rout =100*rs #10*(rb*rs)**0.5, choose the outer radius as 100r_s
	eout = 782e3/(rmin/3.0)*(fv/0.1)**2 #rmin(r_in) as the inner radius, eq(3.32)
	le = np.geomspace(ei, eout, ne+1) #generate an array of energies
	de = np.ones(ne+1)*le # eq(3.33)
	#lstop = Se(le)*n*l*PC
	#de[de>lstop] = lstop[de>lstop]
	dmax = 2**0.5*l*PC*(v*1e5)**2/(GRA*m*Msun)
	D = min(1, dmax) #eq(3.38) duty cycle
	fe = 3.54e22*s*(2.35e8)**s*m**2*(n/0.93)/(v/10)**3*D/(rout/rs)**s*fv**(2*s)/le**(s+1)#eq(3.35)
	intg = fh*de*fe # eq(3.34)
	y = np.trapz(intg, le) # do the integration over E
	ltot = mdot*6.7e-16*0.1/er*m*Msun*SPEEDOFLIGHT**2 # eddington luminosity
	return y/ltot

def heat_sub(mdot_, fh=1.0/3.0, er=0.057, A=100):
	return fh*er*A*mdot_/(1.0+A*mdot_) # from the paper

def heat_subgrid(m, n, v, fh=1.0/3.0, er=0.057, A=100):
	mdot = mdot_bondi(m, n, v, er)
	return er*A*mdot/(1.0+A*mdot) # from the paper

# calculate the heat at different limit
def heat(m, n, v, ei=13.6, ne=100, l=1.0, fh=1.0/3.0, er=0.057, alpha=0.1, fo=0, mode=1, cont=0):
	mdot = mdot_bondi(m, n, v, er)
	print(mdot)
	if cont>0:
		fac = 0.075/er
	else:
		fac = 1.0
	mcut = 0.07*alpha *fac
	if mdot>mcut:
		a, emax = thin_disk(ei, m, n, v, er)
		le = np.geomspace(ei, emax, ne+1)
		ll, emax = thin_disk(le, m, n, v, er)
	else:
		a, emax = ADAF(ei, m, mdot, er=er, alpha=alpha, mode=mode)
		le = np.geomspace(ei, emax, ne+1)
		ll, emax = ADAF(le, m, mdot, er=er, alpha=alpha, mode=mode)
	tau = sigma_E(le)*n*l*PC # eq(3.2)
	#intg = ll*fh*(1.0-np.exp(-tau)) #eq(3.3)
	intg = ll #eq(3.3)
	y = np.trapz(intg, le) #do the integration
	ltot = mdot*6.7e-16*0.1/er*m*Msun*SPEEDOFLIGHT**2  # eddington luminosity
	eff = y/ltot # same in the outflow part
	if fo>0:
		madaf = 0.1*alpha**2
		if mdot>madaf and mdot<=mcut:
			eout = outflow(m, n, v, er=er)
			eff += eout
			#print('Outflow heating efficiency: {:.2e}'.format(eout))
	return eff



'''
m, n, v = 100, 1e5, 10
mdot = mdot_bondi(m, n, v)
print(mdot)
fac1, fac2, fac3 = 10000, 100000, 1e6
x1, x2 = 10, 1e4
lnu = np.geomspace(x1, x2, 100)
ltd, em0 = thin_disk(lnu, m, n, v)
ladaf1, em1 = ADAF(lnu, m, mdot_bondi(m,n/fac1, v),mode=1)
ladaf2, em2 = ADAF(lnu, m, mdot_bondi(m, n/fac2, v),mode=1)
ladaf3, em3 = ADAF(lnu, m, mdot_bondi(m,n/fac3, v),mode=1)
print('Maximum temperature: {:.2e}, {:.2e}, {:.2e} {:.2e} eV'.format(em0, em1, em2, em3))
plt.figure()
plt.loglog(lnu, ltd, label='Thin disk, $\dot{m}='+'{:.2e}'.format(mdot)+'$')
plt.loglog(lnu, ladaf1, '--', label=r'LHAF, $\dot{m}='+'{:.2e}'.format(mdot_bondi(m,n/fac1, v))+'$')
plt.loglog(lnu, ladaf2, '-.', label=r'ADAF, $\dot{m}='+'{:.2e}'.format(mdot_bondi(m,n/fac2, v))+'$')
plt.loglog(lnu, ladaf3, ':', label=r'eADAF, $\dot{m}='+'{:.2e}'.format(mdot_bondi(m,n/fac3, v))+'$')
plt.xlabel(r'$h\nu\ [\rm eV]$')
plt.ylabel(r'$L_{h\nu}\ [\rm erg\ s^{-1}\ eV^{-1}]$')
plt.xlim(x1, x2)
plt.legend()
#plt.title(r'$m_{}={:.1f}\ {}$, ${}={:.1f}\ {}$'.format(r'{\rm PBH}',m,r'\rm M_{\odot}', r'\tilde{v}',v,r'\rm km\ s^{-1}'))
plt.title(r'$m_{}={:.1f}\ {}$, $n_{}={:.1e}\ {}$, ${}={:.1f}\ {}$'.format(r'{\rm PBH}',m,r'\rm M_{\odot}', r'{\rm H}',n,r'\rm cm^{-3}',r'\tilde{v}',v,r'\rm km\ s^{-1}'), size=16)
plt.tight_layout()
plt.savefig('spectra.pdf')
plt.close()
'''

'''
fh0 = heat(m, n, v)
fh1 = heat(m, n/fac1, v)
fh2 = heat(m, n/fac2, v)
fh3 = heat(m, n/fac3, v)
fhs0 = heat_subgrid(m, n, v)
fhs1 = heat_subgrid(m, n/fac1, v)
fhs2 = heat_subgrid(m, n/fac2, v)
fhs3 = heat_subgrid(m, n/fac3, v)
print('Heating efficiency: {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(fh0, fh1, fh2, fh3))
print('Subgrid heating efficiency: {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(fhs0, fhs1, fhs2, fhs3))



if __name__=="__main__":
	fac = 0.66
	nn = 1000
	n1, n2 = 1e-1, 1e5 # different number density input
	ln = np.geomspace(n1, n2, nn+1) # generate an array
	lfh = np.array([heat(m, x, v) for x in ln])
	lfh1 = np.array([heat(m, x, v, mode=1) for x in ln])
	lfhs = np.array([heat_subgrid(m, x, v) for x in ln])
	lfho = np.array([outflow(m, x, v) for x in ln])
	plt.figure()
	plt.loglog(ln, lfh, label='Radiative transfer w/o LHAF')
	plt.loglog(ln, lfh1, ':', label='Radiative transfer w/ LHAF')
	#plt.loglog(ln, lfhs, '-.', label=r'Subgrid, $f_{\rm abs}=1.0$')
	plt.loglog(ln, lfhs*fac, '--', label=r'Sub-grid, $f_{\rm abs}='+r'{}$'.format(fac))
	plt.loglog(ln, lfho, ':', label='Outflow')
	plt.xlabel(r'$n_{\rm H}\ [\rm cm^{-3}]$')
	plt.ylabel(r'$\epsilon_{\rm heat}\equiv f_{h}f_{\rm abs}\epsilon_{\rm EM}$')
	plt.xlim(n1, n2)
	plt.legend()
	plt.title(r'$m_{}={:.1f}\ {}$, ${}={:.1f}\ {}$, $l=1\ \rm pc$, ${}$'.format(r'{\rm PBH}',m,r'\rm M_{\odot}', r'\tilde{v}',v,r'\rm km\ s^{-1}', 'f_{h}=1/3'), size=16)
	#plt.title(r'$m_{}={:.1f}\ {}$, $n_{}={:.1e}\ {}$, ${}={:.1f}\ {}$'.format(r'{\rm PBH}',m,r'\rm M_{\odot}', r'{\rm H}',n,r'\rm cm^{-3}',r'\tilde{v}',v,r'\rm km\ s^{-1}'))
	plt.tight_layout()
	plt.savefig('eheat_n.pdf')
	plt.close()
	
	rat = np.sum(lfh*ln**0.2)/np.sum(lfhs*ln**0.2)
	ratm = np.min(lfh/lfhs)
	print('Overall/minimum ratio (FRT/Sub-grid): {:.2e}/{:.2e}'.format(rat, ratm))
	print(rat/3.0, ratm/3.0)
	print('Difference within {:.2e} (w/o LHAF)'.format(np.max(np.abs(1.0-lfh/(lfhs*fac)))))
	print('Difference within {:.2e} (w/ LHAF)'.format(np.max(np.abs(1.0-lfh1/(lfhs*fac)))))

	
	lE = np.geomspace(13.6, 1e10, 100)
	ls = sigma_E(lE, interion)
	lam = PROTON/ls

	ref = 10**iondata
	plt.figure()
	ax = plt.subplot(111)
	plt.scatter(*ref, marker='^', label='Olive+2014')
	plt.loglog(lE, lam, 'k-', label='Interpolation')
	plt.xlabel(r'$E\ [\rm eV]$')
	plt.ylabel(r'$m_{\rm p}/\sigma\ [\rm g\ cm^{-2}]$')
	plt.xlim(10, 1e10)
	plt.ylim(1e-7, 1e3)
	plt.legend()
	ax.yaxis.set_major_locator(plt.LogLocator(10, numticks=101))
	ax.xaxis.set_major_locator(plt.LogLocator(10, numticks=91))
	plt.tight_layout()
	plt.savefig('lam_E.pdf')
	plt.close()

	lse = Se(lE, interSe)
	plt.figure()
	ax = plt.subplot(111)
	plt.scatter(Sedata[0]*1e3, Sedata[1], marker='^', label='Bailey+2019')
	plt.plot(lE, lse, 'k-', label='Interpolation')
	plt.xscale('log')
	plt.xlabel(r'$E\ [\rm eV]$')
	plt.ylabel(r'$S_{\rm e}\ [\rm ev\ cm^{2}]$')
	plt.xlim(10, 1e10)
	plt.ylim(0, 10)
	plt.legend()
	#ax.yaxis.set_major_locator(plt.LogLocator(10, numticks=101))
	#ax.xaxis.set_major_locator(plt.LogLocator(10, numticks=91))
	plt.tight_layout()
	plt.savefig('Se_E.pdf')
	plt.close()
	"""
	
'''
'''
if __name__=="__main__":
	fac = 0.66
	nm = 1000
	n,v = 1000,10 # different number density input
	m1, m2 = 1,200
	er_0 = 0.057
	lm = np.geomspace(1, 200, nm+1) # generate an array
	lfh = np.array([heat(m, n, v,er=er_0 ) for m in lm])
	lfh1 = np.array([heat(m, n, v,er=er_0 , mode=1) for m in lm])
	lfhs = np.array([heat_subgrid(m, n, v, fh=1/3, er=er_0 , A=100) for m in lm])
	lfho = np.array([outflow(m, n, v) for m in lm])
	plt.figure()
	plt.loglog(lm, lfh, label='Radiative transfer w/o LHAF')
	plt.loglog(lm, lfh1, ':', label='Radiative transfer w/ LHAF')
	#plt.loglog(ln, lfhs, '-.', label=r'Subgrid, $f_{\rm abs}=1.0$')
	plt.loglog(lm, lfhs*fac, '--', label=r'Sub-grid, $f_{\rm abs}='+r'{}$'.format(fac))
	plt.loglog(lm, lfho, ':', label='Outflow')
	plt.xlabel(r'$m_{\rm PBH}\ [\rm M_{\odot}]$')
	plt.ylabel(r'$\epsilon_{\rm heat}\equiv f_{h}f_{\rm abs}\epsilon_{\rm EM}$')
	plt.xlim(m1, m2)
	plt.legend()
	plt.title(r'$n_{}={:.1f}\ {}$, ${}={:.1f}\ {}$, $l=1\ \rm pc$, ${}$'.format(r'{\rm H}',n,r'\rm cm^{-3}', r'\tilde{v}',v,r'\rm km\ s^{-1}', 'f_{h}=1/3'), size=16)
	#plt.title(r'$m_{}={:.1f}\ {}$, $n_{}={:.1e}\ {}$, ${}={:.1f}\ {}$'.format(r'{\rm PBH}',m,r'\rm M_{\odot}', r'{\rm H}',n,r'\rm cm^{-3}',r'\tilde{v}',v,r'\rm km\ s^{-1}'))
	plt.tight_layout()
	plt.savefig('eheat_mPBH.pdf')
	plt.close()
'''
