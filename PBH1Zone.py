from bdms import * # module for baryon-dark matter scattering
from radcool import * # cooling rates
import chemi as chemi1 # chemical network
#import cheminet as chemi2
from txt import * # file IO
import os
import multiprocessing as mp # https://docs.python.org/3/library/multiprocessing.html
import time
#import hmf # module for halo mass function, https://github.com/halomod/hmf
#import hmf.wdm
#from hmf import wdm
#from numba import njit # to make python code faster, https://numba.pydata.org/
from injection import *
from tophat import *
import matplotlib
import numpy as np
from scipy import interpolate
#import mpl_toolkits.mplot3d
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('tableau-colorblind10')


proton = PROTON/GeV_to_mass
electron = ELECTRON/GeV_to_mass


def PBHcool(m_h, Tdm, Tb, v_rel, rhob, rhodm, Mdm, sigma, gamma, X, m_pbh, v_pbh,f_pbh, mu=1.22,er=0.057, Om=0.3153, Ob = 0.04930):
	xh = 4*X/(1+3*X)
	n_pbh = f_pbh * rhodm/(1-f_pbh) / m_pbh / Msun
	n_H=rhob/mu/PROTON
	rad=np.linspace(0.01,0.1,19) # ratio of r/R_vir
	v_pbh=v_pbh/1e5
	del_rad=0.005
	L_pbh=0
	M_c=0.1*m_h*Ob/Om*X
	for i in range(19):
		L_pbh+=heat_subgrid(m_pbh, n_H/3/rad[i]**2, v_pbh)*mdot_bondi(m_pbh, n_H/3/rad[i]**2, v_pbh)*6.7e-16*m_pbh*(0.1/er)*Msun*SPEEDOFLIGHT**2*n_pbh/3/rad[i]**2*rad[i]**2*del_rad
	Q_PBH=L_pbh*3*m_h*Ob/(rhob*Om)*mu*PROTON/M_c*(gamma-1.0)/BOL
	#a = 1/(1+z)
	rho = rhob + rhodm
	#rhob = Ob/Om * rhom(a, Om, h)
	QH = Q_IDMB(rhob, v_rel, Tdm, Tb, Mdm*GeV_to_mass, PROTON, sigma, gamma)*xh
	QHe = Q_IDMB(rhob, v_rel, Tdm, Tb, Mdm*GeV_to_mass, 4*PROTON, sigma, gamma)*(1-xh)
	dTdm = (QH+QHe)
	#rhodm = (Om-Ob)/Om * rhom(a, Om, h)
	QH = Q_IDMB(rhodm, v_rel, Tb, Tdm, PROTON, Mdm*GeV_to_mass, sigma, gamma)*xh
	QHe = Q_IDMB(rhodm, v_rel, Tb, Tdm, 4*PROTON, Mdm*GeV_to_mass, sigma, gamma)*(1-xh)
	dTb = (QH+QHe+Q_PBH) #+ GammaC(1/a-1, Om, Ob, OR, h, X, T0 = T0)*(T0/a-Tb)
	DH = drag(rho, v_rel, Tb, Tdm, PROTON, Mdm*GeV_to_mass, sigma)
	DHe = drag(rho, v_rel, Tb, Tdm, 4*PROTON, Mdm*GeV_to_mass, sigma)
	dv = - (xh*DH + (1-xh)*DHe)
	return [dTdm, dTb, dv]

def PBHheating(m_h,rhob, gamma, X, m_pbh, v_pbh, f_pbh, mu=1.22,er=0.057, Om=0.3089, Ob = 0.048):
	xh = 4*X/(1+3*X)
	n_pbh = f_pbh * (Om-Ob)*m_h/Om/m_pbh*0.1
	n_H=rhob/mu/PROTON
	v_pbh=v_pbh/1e5
	M_c=0.1*m_h*Ob/Om*X*Msun
	L_pbh=heat_subgrid(m_pbh, n_H, v_pbh)*mdot_bondi(m_pbh, n_H, v_pbh)*6.7e-16*m_pbh*(0.1/er)*Msun*SPEEDOFLIGHT**2*n_pbh
	return L_pbh*mu*PROTON/M_c*(gamma-1.0)/BOL

def PBHcool_(m_h, Tdm, Tb, v_rel, rhob, rhodm, Mdm, sigma, gamma, X, m_pbh, v_pbh,f_pbh, mu=1.22,er=0.057, Om=0.3153, Ob = 0.04930):
	xh = 4*X/(1+3*X)
	n_pbh = f_pbh * (Om-Ob)*m_h/Om/m_pbh*0.1
	n_H=rhob/mu/PROTON
	v_pbh=v_pbh/1e5
	M_c=0.1*m_h*Ob/Om*X*Msun
	L_pbh=heat_subgrid(m_pbh, n_H, v_pbh)*mdot_bondi(m_pbh, n_H, v_pbh)*6.7e-16*m_pbh*(0.1/er)*Msun*SPEEDOFLIGHT**2*n_pbh
	Q_PBH=L_pbh*mu*PROTON/M_c*(gamma-1.0)/BOL
	#a = 1/(1+z)
	rho = rhob + rhodm
	#rhob = Ob/Om * rhom(a, Om, h)
	QH = Q_IDMB(rhob, v_rel, Tdm, Tb, Mdm*GeV_to_mass, PROTON, sigma, gamma)*xh
	QHe = Q_IDMB(rhob, v_rel, Tdm, Tb, Mdm*GeV_to_mass, 4*PROTON, sigma, gamma)*(1-xh)
	dTdm = (QH+QHe)
	#rhodm = (Om-Ob)/Om * rhom(a, Om, h)
	QH = Q_IDMB(rhodm, v_rel, Tb, Tdm, PROTON, Mdm*GeV_to_mass, sigma, gamma)*xh
	QHe = Q_IDMB(rhodm, v_rel, Tb, Tdm, 4*PROTON, Mdm*GeV_to_mass, sigma, gamma)*(1-xh)
	dTb = (QH+QHe+Q_PBH) #+ GammaC(1/a-1, Om, Ob, OR, h, X, T0 = T0)*(T0/a-Tb)
	DH = drag(rho, v_rel, Tb, Tdm, PROTON, Mdm*GeV_to_mass, sigma)
	DHe = drag(rho, v_rel, Tb, Tdm, 4*PROTON, Mdm*GeV_to_mass, sigma)
	dv = - (xh*DH + (1-xh)*DHe)
	return [dTdm, dTb, dv]

# cooling timescale, mode=0: CDM, mode=1: include BDMS
def coolt_PBH(Mh,Tb_old, Tdm_old, v_old, nb, nold, rhob_old, rhodm_old, z, mode,m_pbh,v_pbh,f_pbh, Mdm, sigma, gamma, Om=0.3153, Ob = 0.04930, h = 0.6736, X = 0.75, J_21 = 0, T0 = 2.726, vmin = 0.0):
	"""
		Tb_old: gas temperature
		Tdm_old: DM temperature
		v_old: baryon-DM relative velocity
		nb: overall number density of particles
		nold: number densities of different species (an array of 17 elements)
		rhob_old: mass density of gas
		rhodm_old: mass density of DM
		gamma: adiabatic index
		J_21: strength of the LW background
		X: primordial hydrogen mass fraction
	"""
	xh = 4*X/(1+3*X)
	dTs_dt = [0]*3
	if mode!=0:
		uth = (Tb_old*BOL/PROTON+Tdm_old*BOL/(Mdm*GeV_to_mass))**0.5
		v = max(v_old, uth*vmin)
		dTs_dt = PBHcool_(Mh,Tdm_old, Tb_old, v, rhob_old, rhodm_old, Mdm, sigma, gamma, X, m_pbh, v_pbh,f_pbh)
		dTb_dt = cool(Tb_old, nb, nold, J_21, z, gamma, X, T0) + dTs_dt[1] #eq(1)
	else:
		dTb_dt = cool(Tb_old, nb, nold, J_21, z, gamma, X, T0) + dTs_dt[1] + PBHheating(Mh, rhob_old, gamma, X, m_pbh, v_pbh, f_pbh) #eq(1)
	if abs(dTb_dt) <= Tb_old/TZ(0):
		return TZ(0)
	else:
		return -Tb_old/dTb_dt



def PBHevolve(Mh = 1e6, zvir = 20, z0 = z0_default, v0 = 30, mode = 0, fac = 1.0, Mdm = 0.3, sigma = 8e-20, num = int(1e3), epsT = 1e-3, epsH = 1e-2, dmax = 18*np.pi**2, gamma = 5/3, X = 0.75, D = 2.38e-5, Li = 4.04e-10, T0 = 2.726, Om = 0.3153, Ob = 0.04930, h = 0.6736, dtmin = YR, J_21=0.0, Tmin = 1., vmin = 0.0, nmax = int(1e6), init =init, hat=1, fnth=0.17, m_pbh=30, v_pbh=10,f_pbh=1e-3):
	"""
		Mh, zcir: halo mass and redshift
		tpost: duration of the run after virialization in unit of 1/H(a)
		num: set maximum timestep
		epsT, epsH: maximum changes of temperature and abundances
		dmax: maximum overdensity
		D/Li: primordial abundance of D/Li nuclei (with respect to H nuclei)
		dtmin: initial timestep
		nmax: set the timestep to smaller values at the early stage for stability
		init: initial condition data
		fnth: contribution of non-thermal CMB photons
	"""
	#start = time.time()
	#print(Mdm, sigma, init['Tb'], init['Tdm'], init['vbdm'])
	xh = 4.0*X/(1.0+3.0*X)
	xhe, xd, xli = 1-xh, D, Li
	refa = np.array([xh]*6+[xhe]*3+[xd]*3+[xli]*5)
	t0 = TZ(z0)
	t1 = TZ(zvir)
	tpost = min(fac/H(1/(1+zvir), Om, h), TZ(0)-t1)
	tmax = t1 + tpost
	dt0 = (tmax-t0)/num
	dt0_ = (tmax-t0)/nmax
	#print('Time step: {} yr'.format(dt0/YR))
	if hat>0:
		rhodm_z = lambda x: rho_z(x, zvir, dmax, Om, h)*(Om-Ob)/Om*(1-f_pbh)
		rhob_z = lambda x: rho_z(x, zvir, dmax, Om, h)*Ob/Om
	else:
		rhodm_z = lambda x: rhom(1/(1+x), Om, h)*(Om-Ob)/Om*(1-f_pbh)
		rhob_z = lambda x: rhom(1/(1+x), Om, h)*Ob/Om
	Ns = len(init['X'])
	lz = [z0]
	lt = [t0]
	lTb = [max(init['Tb'], Tmin)]
	lTdm = [max(init['Tdm'], Tmin/1e10)]
	lv = [init['vbdm']]
	lrhob = [rhob_z(z0)]
	lrhodm = [rhodm_z(z0)]
	lX = [[x] for x in init['X']]
	t_cum, count, total = t0, 0, 0
	tag0 = 0
	tag1 = 0
	TV = Tvir(Mh, zvir, delta0)
	VV = Vcir(Mh, zvir, delta0)
	tag2 = 0
	if lTb[0]>TV:
		tag2 = 1
	Tb_V = TV
	pV = []
	pV_pri = []
	para = [Mdm, sigma, gamma, Om, Ob, h, X, J_21, T0, vmin]
	tcool = 0.0
	tffV = 1/H(1/(1+zvir), Om, h) #tff(zvir, dmax)
	#yy = np.zeros(Ns, dtype='float')
	tagt = 0
	while t_cum < tmax or z>zvir:
		if count==0:
			z = z0
			dt_T = dtmin
			yy = np.array([x[count] for x in lX])
			mgas = mmw(yy[5], yy[7], X)*PROTON
			nb = lrhob[0]/mgas
			nold = yy * refa
			Tb_old = lTb[count]
			Tdm_old = lTdm[count]
			v_old = lv[count]
			rhob_old, rhodm_old = lrhob[count], lrhodm[count]
			dlnrho_dt_old = Dlnrho(t0, t0+dtmin/2.0, zvir, dmax, hat=hat)
			dTs_dt_old = [0]*3
			dTs_dt = [0]*3
			if mode!=0:
				V_pbh=Vcir(Mh,z0,delta0)
				dTs_dt_old = PBHcool_(Mh, Tdm_old, Tb_old, v_old, rhob_old, rhodm_old, Mdm, sigma, gamma, X, m_pbh, V_pbh ,f_pbh)
				dTb_dt_old = cool(max(Tb_old, 10*Tmin), nb, nold*nb, J_21, z0, gamma, X, T0) \
						+ dTs_dt_old[1] + dlnrho_dt_old*(gamma-1)*Tb_old
			else:
				V_pbh=Vcir(Mh,z0,delta0)
				dTb_dt_old = cool(max(Tb_old, 10 * Tmin), nb, nold * nb, J_21, z0, gamma, X, T0) \
							 + dTs_dt_old[1] + dlnrho_dt_old * (gamma - 1) * Tb_old+ PBHheating(Mh, rhob_old, gamma, X, m_pbh, V_pbh, f_pbh)
			dTdm_dt_old = dTs_dt_old[0] + dlnrho_dt_old*(gamma-1)*Tdm_old
			dv_dt_old = dTs_dt_old[2] + dlnrho_dt_old*(gamma-1)*v_old/2
		else:
			if (t_cum-t0)/(t1-t0)<0.01:
				dt_T = dt0_
			else:
				dt_T = dt0
			if abs(dTb_dt_old*dt_T)>epsT*Tb_old and Tb_old>10*Tmin:
				dt_T = epsT*Tb_old/abs(dTb_dt_old)
				#if dt_T < dt0_ and Tb_old<=Tmin*100:
				#	dt_T = dt0_
		if dt_T + t_cum>t1 and tagt==0:
			dt_T = t1 - t_cum
			tagt = 1
		if dt_T + t_cum>tmax:
			dt_T = tmax - t_cum

		if count==0:
			Cr0, Ds0 = np.zeros(Ns,dtype='float'), np.zeros(Ns,dtype='float')
			abund0 = chemi1.chemistry1(Tb_old, nold*nb, dt_T, epsH, J_21, Ns, xh*nb, xhe*nb, xd*nb, xli*nb, Cr0, Ds0, z = z, T0 = T0, fnth=fnth)
			Cr0, Ds0 = abund0[5], abund0[6]
		else:
			Cr0, Ds0 = abund[5], abund[6]

		nold = yy * refa
		abund = chemi1.chemistry1(Tb_old, nold*nb, dt_T, epsH, J_21, Ns, xh*nb, xhe*nb, xd*nb, xli*nb, Cr0, Ds0, z = z, T0 = T0, fnth=fnth)
		#abund = chemi2.chemistry1(Tb_old, nold*nb, dt_T, epsH, J_21, Ns, xh*nb, xhe*nb, xd*nb, xli*nb, z = z, T0 = T0)
		nold = abund[0]/nb
		for x in range(Ns):
			if refa[x]!=0:
				yy[x] = nold[x]/refa[x]
			else:
				yy[x] = 0.0
		mgas = mmw(yy[5], yy[7], X)*PROTON
		#if count<10:
		#	print(nold, Tb_old)
		t_cum += abund[1]
		z = ZT(t_cum)
		dlnrho_dt = Dlnrho(t_cum, t_cum + abund[1]/2.0, zvir, dmax, hat=hat)
		uth = (Tb_old*BOL/PROTON+Tdm_old*BOL/(Mdm*GeV_to_mass))**0.5
		if mode!=0: #and (Tb_old>Tdm_old or v_old>vmin*uth):
			v_pbh=Vcir(Mh,z,delta0)
			dTs_dt = PBHcool_(Mh, Tdm_old, Tb_old, v_old, rhob_old, rhodm_old, Mdm, sigma, gamma, X, m_pbh, v_pbh, f_pbh)
			dTb_dt = cool(max(Tb_old, 10 * Tmin), nb, nold * nb, J_21, z, gamma, X, T0) + dTs_dt[1]
		else:
			v_pbh=Vcir(Mh,z,delta0)
			dTb_dt = cool(Tb_old, nb, nold, J_21, z, gamma, X, T0) + PBHheating(Mh, rhob_old, gamma, X, m_pbh, v_pbh, f_pbh)+ dTs_dt[1]
		if tag0==0:
			dTb_dt += dlnrho_dt*(gamma-1)*Tb_old
		dTdm_dt = dTs_dt[0] + dlnrho_dt*(gamma-1)*Tdm_old
		dv_dt = dTs_dt[2] + dlnrho_dt*(gamma-1)*v_old/2
		Tb_old = max(Tb_old + (dTb_dt + dTb_dt_old)*abund[1]/2.0, Tmin)
		Tdm_old = max(Tdm_old + (dTdm_dt + dTdm_dt_old)*abund[1]/2.0, 0.0)
		v_old = max(v_old + (dv_dt + dv_dt_old)*abund[1]/2.0, vmin*uth)
		dTb_dt_old = dTb_dt
		dTdm_dt_old = dTdm_dt
		dv_dt_old = dv_dt
		if tag0==0:
			rhob_old = rhob_z(z)
		rhodm_old = rhodm_z(z)
		nb = rhob_old/mgas
		#print(z, nb, mgas)
		#total += abund[4]
		total += abund[2]
		count += 1
		if tag2==1:
			if Tb_old<TV:
				tag2 = 0
		#if max(Tb_old, Tdm_old) > TV and tag0==0 and tag2==0:
		#	Tb_old = TV
		#	tag0 = 1
		if t_cum>=t1 and tag1==0 and tag0==0 and hat>0:
			v_pbh=Vcir(Mh, z, delta0)
			pV_pri = [nold[3]/refa[3], nold[11]/refa[11], nold[5]/refa[5], Tb_old, v_old]
			Tb_V = Tb_old
			Tb_old = TV #max(TV, Tb_old)
			Tdm_old = TV #max(TV, Tdm_old)
			pV = [Tb_old, Tdm_old, v_old, nb, nold*nb, rhob_old, rhodm_old, z]
			tcool = coolt_PBH(Mh, *pV, mode, m_pbh, v_pbh,f_pbh, *para)
			#v_old = max(VV, v_old)
			tag1 = 1
			#print('x_H2 = {}'.format(nold[3]/refa[3]))
		if (count%10==0)or(t_cum>=tmax):
			lt.append(t_cum)#[count] = t_cum
			lTb.append(Tb_old)#[count] = Told
			lTdm.append(Tdm_old)#[count] = nb
			lv.append(v_old)
			lrhob.append(rhob_old)
			lrhodm.append(rhodm_old)
			lz.append(z)
			for x in range(Ns):
				if refa[x]!=0:
					lX[x].append(nold[x]/refa[x])#[count] = newX[x]
				else:
					lX[x].append(0.0)
	d = {}
	d['t'] = np.array(lt)/YR/1e6
	d['z'] = np.array(lz)
	d['Tb'] = np.array(lTb)
	d['Tdm'] = np.array(lTdm)
	d['v'] = np.array(lv)
	d['rho'] = np.array(lrhob) + np.array(lrhodm)
	d['nb'] = np.array(lrhob)/mgas
	d['X'] = np.array(lX) # abundances
	d['rat'] = Tb_old/TV
	d['rat0'] = tpost/(t1 + tpost)
	d['s'] = int(tpost/tmax > Tb_old/TV)
	d['Tvir'] = TV
	d['TbV'] = Tb_V
	d['rat1'] = Tb_V/TV
	d['rat2'] = tcool/tffV
	d['m'] = M_T(Tb_V/d['rat0'], zvir, dmax)
	d['pV'] = pV  # state of the system at virialization
	d['pV_pri'] = pV_pri # important quantities at virialization
	d['para'] = para
	#end = time.time()
	#print(t_cum-t1)
	#print('Time taken: {} s'.format(end-start))
	#print(count, total)
	return d


# mass threshold for efficient cooling, defined by tcool/tff = rat
# sk=True: modify the baryon-DM relative velocity from virialization
def Mth_z_PBH(m_pbh, f_pbh, z1, z2, nzb = 10, m1 = 1e2, m2 = 1e10, nmb = 100, mode = 0, z0 = z0_default, v0 = 30, Mdm = 0.3, sigma = 8e-20, rat = 1.0, dmax = 18*np.pi**2, Om = 0.3153, h = 0.6736, fac = 1e-3, vmin = 0.0, beta = 0.7, sk = False, init = init):
	m0 = (m1*m2)**0.5
	lz = np.linspace(z1, z2, nzb)
	#lz = np.logspace(np.log10(z1), np.log10(z2), nzb)
	out = []
	lxh2 = []
	lxhd = []
	lxe = []
	lTb = []
	lvr = []
	for z in lz:
		mmax = Mup(z)*10
		mmin = M_T(200, z, delta0, Om)
		lm = np.logspace(np.log10(mmin), np.log10(mmax), nmb)
		d = PBHevolve(m0, z, z0, v0, mode, Mdm = Mdm, sigma = sigma, dmax = dmax, Om = Om, h = h, fac = fac, init = init, m_pbh = m_pbh ,f_pbh = f_pbh)
		tffV = tff(z, dmax)
		#tffV = 1/H(1/(1+z), Om, h)
		#lT = [Tvir(m, z, delta0) for m in lm]
		red = (dmax/delta0)**0.5
		lT = [Tvir(m/red, z, dmax) for m in lm]
		lvv = [Vcir(m, z, delta0) for m in lm]
		if mode!=0 and sk:
			pV = d['pV'][3:]
			lv = np.zeros(nmb)
			for i in range(nmb):
				uth = (lT[i]*BOL/PROTON+lT[i]*BOL/(Mdm*GeV_to_mass))**0.5
				dvdt = PBHcool_(lm[i], lT[i], lT[i], lvv[i], *pV[-3:-1], Mdm, sigma, d['para'][2], d['para'][6], m_pbh, lvv[i], f_pbh)[2]
				vf = max(lvv[i] + dvdt * tffV, uth*vmin)
				lv[i] = 0.5 * ((vf*lvv[i])**0.5 + d['pV'][2])
			lt0 = [coolt_PBH(mh, T, T, v, *pV, mode, m_pbh, v_pbh,f_pbh, *d['para'])/tffV for mh, T, v, v_pbh in zip(lm, lT, lv, lvv)]
		else:
			pV = d['pV'][2:]
			lt0 = [coolt_PBH(mh, T, T, *pV, mode, m_pbh, v_pbh, f_pbh, *d['para'])/tffV for mh, T, v_pbh in zip(lm, lT, lvv)]
		lt00 = np.array(lt0)
		ltt0 = lt00[lt00>0]
		if ltt0 is []:
			print(Mdm, sigma, 'Heating!')
			return []
		else:
			imax = lt0.index(np.max(lt00))
			if imax<nmb-1:
				lt00 = np.array(lt0[imax:])
				ltt0 = lt00[lt00>0]
			imin = lt0.index(np.min(lt00))
			if imin==0:
				print(Mdm, sigma, 'lower bound')
				mth = mmin
			else:
				#if imax<=imin:
				#print(lt0)
				lt0 = np.array(lt0[imax:imin+1])
				lm = lm[imax:imin+1]
				#else:
					#lt0 = np.array(lt0[imin:imax+1])
					#lm = lm[imin:imax+1]
				lt = lt0[lt0>0]
				lm = lm[lt0>0]
				if np.min(np.log10(lt))>=np.log10(rat):
					mth = np.max(lm)
					print(Mdm, sigma, 'Upper bound')
				elif np.max(np.log10(lt))<=np.log10(rat):
					mth = np.min(lm)
				else:
					rat_m = interp1d(np.log10(lt), np.log10(lm))
					mth = 10**rat_m(np.log10(rat))
		#if mode!=0:
		mth0 = mth
		mth = mth * (1+beta*d['pV'][2]**2/Vcir(mth, z, delta0)**2/(dmax/delta0)**(2/3))**(3/2)
		print(Mdm, sigma, mth/1e6, mth0/1e6, z)
		out.append(mth)
		lxh2.append(d['pV_pri'][0])
		lxhd.append(d['pV_pri'][1])
		lxe.append(d['pV_pri'][2])
		lTb.append(d['pV_pri'][3])
		lvr.append(d['pV_pri'][4])
	return [np.array(out), lz, lxh2, lxhd, lxe, lTb, lvr]



#another method to calculate m_th by comparing H2 cooling timescale with the hubble time scale(Trenti & Stiavelli 2009)


def fh2_pbh(fpbh, alp=0.15, fref=0.1, rref=3):
	return max(rref*(fpbh/fref)**alp, 1.0)

# PBH heating w/ respect to density
def QPBH(mh, O_b,O_dm, n_H,m_pbh, v_pbh,f_pbh,X=0.75,er=0.057,mu=1.22):
	n_pbh = f_pbh * O_dm*mh/(O_b+O_dm)/m_pbh
	v_pbh=v_pbh/1e5
	rad = np.linspace(0.01, 1, 100)
	del_rad = 0.01
	L_pbh=0
	M_c=mh*O_b/(O_b+O_dm)*X
	for i in range(100):
		L_pbh += heat_subgrid(m_pbh, n_H / 3 / rad[i] ** 2, v_pbh) *mdot_bondi(m_pbh, n_H/3/rad[i]**2,v_pbh)*6.7e-16*m_pbh*(0.1/er)*Msun*SPEEDOFLIGHT**2* n_pbh / rad[i] ** 2 * rad[i] ** 2 * del_rad
	Q_PBH=L_pbh*PROTON*mmw(X=X)/M_c
	return Q_PBH

Mdown = lambda z: 1.54e5*((1+z)/31)**-2.074
# assume uniform density
def QPBH_(mh, O_b,O_dm, n_H,m_pbh, v_pbh,f_pbh,X=0.75,er=0.057,mu=1.22):
	xh = 4 * X / (1 + 3 * X)
	n_pbh = f_pbh * O_dm*mh/(O_b+O_dm)/m_pbh
	v_pbh=v_pbh/1e5
	M_c=mh*O_b/(O_b+O_dm)*X
	L_pbh =heat_subgrid(m_pbh, n_H, v_pbh) *mdot_bondi(m_pbh, n_H, v_pbh)*6.7e-16*m_pbh*(0.1/er)*Msun*SPEEDOFLIGHT**2*n_pbh
	Q_PBH=L_pbh*PROTON*mmw(X=X)/M_c
	return Q_PBH

# solve for fraction of fh2
def trat(m,z,dmax,m_pbh,f_pbh,flag,delt=6.5,Om = 0.3089, Ob = 0.048, X=0.76 ,h = 0.6774,gamma=5.0/3,T0=2.73):
	T_halo = Tvir(m, z, 200, Om=Om, h=h)
	n_H = rhom(1/(1+z), Om, h)*delt*dmax*Ob/Om*X/PROTON
	fmax = 3.5 / 1e4 * (T_halo / 1000) ** 1.52
	v_pbh =Vcir(m ,z, dmax)
	ny = np.zeros(17)
	ny[0] = n_H
	if flag==1:
		ny[3] = n_H * fmax*fh2_pbh(f_pbh)
		lamb = cool(T_halo, n_H / 0.93, ny, 0, z, gamma, X, T0)
		Qdot = 1/(gamma-1)*BOL*lamb + QPBH_(m,Ob,Om-Ob,n_H,m_pbh,v_pbh,f_pbh)
		t_dyn=tff(z, delt*dmax, Om, h)
	else:
		ny[3] = n_H * fmax
		lamb = cool(T_halo, n_H / 0.93, ny, 0, z, gamma, X, T0)
		Qdot = 1/(gamma-1)*BOL*lamb
		t_dyn=tff(z, delt*dmax, Om, h)
	return abs(-1/(gamma-1)*BOL*T_halo/Qdot/t_dyn)

def Mmin_z(z1, z2, nzb = 10, m_pbh=33, f_pbh=1e-3, flag=0, dmax = 18*np.pi**2, Om = 0.3153, Ob = 0.04930, h = 0.6736):
	lz=np.linspace(z1,z2,nzb)
	mth=[]
	for i in range(nzb):
		m1, m2 = Mdown(lz[i]) * 0.1, Mdown(lz[i]) * 10
		lm=np.logspace(np.log10(m1),np.log10(m2),50)
		f=[]
		for j in range(50):
			f.insert(i,trat(lm[j],lz[i],dmax,m_pbh,f_pbh,flag,Om=Om,Ob=Ob,h=h))
		ind = np.argmax(f)
		if f[ind] < 1:
			mth.insert(i,lm[ind])
		else:
			func = interp1d(np.log10(f[ind:]), np.log10(lm[ind:]))
			mth.insert(i,10**func(0))
	return lz,mth

'''
tag = 0
v0 = 0
rat = 1.
nbin = 32 #64
ncore = 4
dmax = delta0*100  # typical overdensity for the star forming cloud
fac = 1e-3
beta = 0.7
sk = False
Mdm =  0.3 #0.001
sigma = 8e-20 #1e-17
init0 = initial(v0=v0, mode=0, Mdm=Mdm, sigma=sigma)
init1 = initial(v0=v0, mode=1, Mdm=Mdm, sigma=sigma)

z0_default = 300
x0_default = [1., 5e-4, 2.5e-19, 2e-11, 3e-16] + \
			 [5e-4, 1.0, 4.7e-19, 0.0] + \
			 [1.0, 5e-4, 8.4e-11] + \
			 [1.0, 1e-4, 0, 0, 1e-14]


if tag==0:
	#d_ = Mth_z(5, 40, 36, mode = 0, v0 = v0, rat = rat, dmax = dmax , fac = fac, beta = beta, sk = sk, init = init1)
	#totxt('Mthz_BDMS'+str(v0)+'.txt',d_,0,0,0)
	#d4 = Mth_z_PBH(33, 1e-4, 5, 40, 36, mode = 0, v0 = v0, rat = rat, dmax = dmax, fac = fac, beta = beta, sk = sk, init = init1)
	#totxt('Mthz_BDMS_PBH4'+str(v0)+'.txt',d4,0,0,0)
	#d3 = Mth_z_PBH(33, 1e-3, 5, 40, 36, m1 = 1e5, m2 = 1e9, nmb= 100, mode = 0, v0 = v0, rat = rat, dmax = dmax, fac = fac, beta = beta, sk = sk, init = init1)
	#totxt('Mthz_BDMS_PBH3'+str(v0)+'.txt',d3,0,0,0)
	#d2 = Mth_z_PBH(33, 1e-2, 5, 40, 36, m1 = 1e5, m2 = 1e8, nmb= 150, mode = 0, v0 = v0, rat = rat, dmax = dmax, fac = fac, beta = beta, sk = sk, init = init1)
	#totxt('Mthz_BDMS_PBH2'+str(v0)+'.txt',d2,0,0,0)
	#d1 = Mth_z_PBH(33, 1e-1, 5, 40, 36, m1 = 1e5, m2 = 1e8, nmb= 150, mode = 0, v0 = v0, rat = rat, dmax = dmax, fac = fac, beta = beta, sk = sk, init = init1)
	#totxt('Mthz_BDMS_PBH1'+str(v0)+'.txt',d1,0,0,0)
	d1e2 = Mth_z_PBH(100, 1e-3, 5, 40, 36, m1 = 1e5, m2 = 1e8, nmb= 150, mode = 0, v0 = v0, rat = rat, dmax = dmax, fac = fac, beta = beta, sk = sk, init = init1)
	totxt('Mthz_BDMS_PBH3_M100'+str(v0)+'.txt',d1e2,0,0,0)

	
lm_, lz_, lxh2_, lxhd_, lxe_, lTb_, lvr_ = np.array(retxt('Mthz_BDMS'+str(v0)+'.txt',7,0,0))
lm4, lz4, lxh2_4, lxhd4, lxe4, lTb4, lvr4 = np.array(retxt('Mthz_BDMS_PBH4'+str(v0)+'.txt',7,0,0))
lm3, lz3, lxh2_3, lxhd3, lxe3, lTb3, lvr3 = np.array(retxt('Mthz_BDMS_PBH3'+str(v0)+'.txt',7,0,0))
lm2, lz2, lxh2_2, lxhd2, lxe2, lTb2, lvr2 = np.array(retxt('Mthz_BDMS_PBH2'+str(v0)+'.txt',7,0,0))
lm1, lz1, lxh2_1, lxhd1, lxe1, lTb1, lvr1 = np.array(retxt('Mthz_BDMS_PBH1'+str(v0)+'.txt',7,0,0))
lm1e2, lz1e2, lxh2_1e2, lxhd1e2, lxe1e2, lTb1e2, lvr1e2 = np.array(retxt('Mthz_BDMS_PBH3_M100'+str(v0)+'.txt',7,0,0))
#lm = mth_stm(lm, lz, v0, beta = beta0)
plt.figure()

plt.plot(lz_, lm_, '--',linewidth=2, label='BDMS')
plt.plot(lz4, lm4, linewidth=1,label='$M_{PBH}=33 M_{sun},f_{PBH}=1e-4$')
plt.plot(lz3, lm3, linewidth=1,label='$M_{PBH}=33 M_{sun},f_{PBH}=1e-3$')
plt.plot(lz2, lm2,':', linewidth=2, label='$M_{PBH}=33 M_{sun},f_{PBH}=1e-2$')
plt.plot(lz1, lm1,'-.', linewidth=2, label='$M_{PBH}=33 M_{sun},f_{PBH}=1e-1$')
plt.plot(lz1e2, lm1e2,'k-.', label='$M_{PBH}=100 M_{sun},f_{PBH}=1e-3$')



if tag==0:
	lzTS1,lmTS1 = Mmin_z(10, 60, 51,  m_pbh=33, f_pbh=1e-1, flag=1, dmax = dmax)
	lzTS2,lmTS2 = Mmin_z(10, 60, 51,  m_pbh=33, f_pbh=1e-2, flag=1, dmax = dmax)
	lzTS3,lmTS3 = Mmin_z(10, 60, 51,  m_pbh=33, f_pbh=1e-3, flag=1, dmax = dmax)
	lzTS4,lmTS4 = Mmin_z(10, 60, 51,  m_pbh=33, f_pbh=1e-4, flag=1, dmax = dmax)
	#lzTS0,lmTS0 = Mmin_z(10, 60, 51,  flag=0, dmax = dmax)

plt.plot(lzTS1, lmTS1,':', linewidth=2, label='$M_{PBH}=33 M_{sun},f_{PBH}=1e-1$')
plt.plot(lzTS2, lmTS2,'-.', linewidth=2, label='$M_{PBH}=33 M_{sun},f_{PBH}=1e-2$')
plt.plot(lzTS3, lmTS3,'k-.', label='$M_{PBH}=33 M_{sun},f_{PBH}=1e-3$')
plt.plot(lzTS4,lmTS4,'.', label='$M_{PBH}=33 M_{sun},f_{PBH}=1e-4$')
#plt.plot(lzTS0,lmTS0,'--', label='$\Lambda-CDM$(TS15)')

#plt.plot(lz, Mdown(lz), 'k-.', label='Trenti & Stiavelli (2009)')
plt.fill_between([15,20],[1e4,1e4],[3e8,3e8], facecolor='gray', label='EDGES')


plt.legend()
plt.xlabel(r'$z_{\mathrm{vir}}$')
plt.ylabel(r'$M_{\mathrm{th}}\ [M_{\odot}]$')
#plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.xlim(5, 40)
plt.ylim(1e5, 1e7)
plt.savefig('Mth_z_'+str(v0)+'.pdf')
plt.close()


'''