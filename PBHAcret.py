import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cosmology import *
from tophat import *
import sys
import os
import csv

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('tableau-colorblind10')



def c_s(z,beta=1.72,z_dec=130):
    if z >6:
        return 5.7*((1+z)/1000)**(1/2)/(((1+z_dec)/(1+z))**beta+1)**(1/beta/2)
    elif z<=6:
        return 5.7*(20000/2730)**0.5 # T_gas rise to ~20000K
# sound speed in km/s

#data=np.genfromtxt('vrelz.dat',delimiter=',')
#vrel_z=interpolate.interp1d(data[:,0],data[:,1], fill_value='extrapolate')
vrel_z = lambda z: min(1,z/1e3) *30
# v_rel as a function of z, data from fig 2 in https://iopscience.iop.org/article/10.1086/587831/pdf

def v_eff(z):
    M=vrel_z(z)/c_s(z)
    if M>1:
        return np.array([c_s(z) * (16 / np.sqrt(2 * np.pi) * M ** 3) ** (1 / 6) ,
                                   c_s(z) * (M / (np.sqrt(2 / np.pi) * np.log( 2/np.exp(1) * M)) ** (1 / 3) )])
    elif M<=1:
        return np.array([c_s(z)*np.sqrt(1+M**2)]*2)
'''
    if alphav == 3 or f_pbh == 1: # low efficient acretion rate mdot<1 (spherical disk)
        return c_s(z)*((16/np.sqrt(2*np.pi)*M**3)**(1/6)*(M>=1)+np.sqrt(1+M**2)*(M<1))
    elif alphav == 6: # efficient accretion rate mdot>1 (thin disk)
        return c_s(z)*(M/(np.sqrt(2/np.pi)*np.abs(np.log(2/np.exp(1)*M))**(1/3))*(M>=1)+np.sqrt(1+M**2)*(M<1))
'''
# effective velocity in km/s"


#describe halo accretion with PBH, stops growing when all DM has been accreted
def M_H(z,M_PBH,phi=3):
    #if z<=phi*1000-1:
    #    z=phi*1000-1
    return phi*M_PBH*1000/(1+z)
# in terms of solar mass



# return the accretion eigenvalue for PBHs in IGM
def lam(M_PBH,z,PBH_halo = 0,x_e=1e-3,alpha = 2.2):
    beta_hat=M_PBH/1e4*(((z+1)/1000)**1.5)/(c_s(z)/5.74)**3*(0.257+1.45*(x_e/0.01)*((1+z)/1000)**2.5)
    if PBH_halo == 0:
        x_cr=(-1+np.sqrt(1+beta_hat))/beta_hat
        lam1=np.exp(9/2/(3+beta_hat**0.75))*x_cr**2
        print("Accretion factor is {:2e}".format(lam1))
        return lam1
    elif PBH_halo > 0:
        kappa = 0.22*(1+z)/1000*M_H(z,M_PBH)**(2/3)*c_s(z)**(-2)
        if kappa >=2:
            x_cr=(-1+np.sqrt(1+beta_hat))/beta_hat
            lam1=np.exp(9/2/(3+beta_hat**0.75))*x_cr**2
            print("Accretion as pt mass! Kappa is {:2e}".format(kappa))
        elif kappa < 2:
            p=2-alpha
            #p = 3-alpha
            beta_h=kappa**(p/(1-p))*beta_hat
            x_cr=(-1+np.sqrt(1+beta_h))/beta_h
            Upsilson=(1+10*beta_h)**0.1*np.exp(2-kappa)*(kappa/2)**2
            #x_cr *= (kappa/2)**(p/(1-p))
            lam1=Upsilson**(p/(1-p))*np.exp(9/2/(3+beta_h**0.75))*x_cr**2
        print("Accretion factor is {:2e}".format(lam1))
        return lam1, kappa

# here we assume that f_PBH<1 for halo case, and f_pbh=1 for non-halo case(but there could have PBH clustering)


# return the dimensionless accretion rate: M_BH/M_edd for PBH in IGM
def bondi_IGM(M_PBH,z, PBH_halo = 0):
    v_eff_z = v_eff(z)
    print(v_eff_z)
    #md=np.zeros(2)
    if PBH_halo == 0:
        return 1.8*1e-3*lam(M_PBH,z, PBH_halo=PBH_halo)*((1+z)/1000)**3*M_PBH/(v_eff_z[0]/5.74)**3
    elif PBH_halo == 1:
        lam1, kappa = lam(M_PBH,z, PBH_halo=PBH_halo)
        if kappa >=2:
            return   min(1.8*1e-3  * lam1 * ((1+z)/1000) * M_PBH/(v_eff_z[0]/5.74)**3 ,1.8*1e-3 * lam1 * ((1+z)/1000) * M_PBH/(v_eff_z[1]/5.74)**3)
        else:
            return 1.8*1e-3 * lam1 * ((1+z)/1000)**3 *M_PBH/(v_eff_z[0]/5.74)**3
           # return 0.016 * lam1 * ((1+z)/1000) *M_PBH/(v_eff_z[0]/5.74)**3
    #md[1] += 1.8*1e-3*lam(M_PBH,z, PBH_halo=PBH_halo)*((1+z)/1000)**3*M_PBH/(v_eff(z)[1]/5.74)**3
    #if min(md)>=1:
    #    return md[1]
    #elif max(md)<=1:
    #    return md[0]
    #else:
    #    return min(md)

# return the accretion rate in IGM
def accret_IGM(M_i,z_i,z_end,n=200):
    lz=np.linspace(z_i,z_end,n)
    M=np.zeros(n)
    M[0]=M_i
    for i in range(n-1):
        Mpbh_dot = 0.002*bondi_IGM(M_i,lz[i+1])*M_i/1e6
        M_i+=Mpbh_dot*(TZ(lz[i+1])-TZ(lz[i]))/YR
        M[i+1]+=M_i
    return M

# return the accretion rate at some point in the DM halo(assume eddington-limited)
def bondi_halo(m, n, v, er=0.057, mu=1.22,fed=1.0):
	return min(2.64e-7*m*(er/0.1)*(n/0.93)*mu/(v/10)**3,fed)


# return the accretion of a single PBHs at some point in the halo
def accret_halo(M_i, z_i, z_end, n_gas, vtilda, n=1000):
    lz = np.linspace(z_i, z_end, n)
    M = np.zeros(n)
    M[0] = M_i
    for i in range(n - 1):
        Mpbh_dot = 0.002*bondi_halo(M_i,n_gas, vtilda)*M_i/1e6
        M_i += Mpbh_dot * (TZ(lz[i + 1]) - TZ(lz[i])) / YR
        M[i + 1] += M_i
        print(M_i)
    return M

#print(accret_halo(100,300,10,1e3,10,n=100))

#print(accret_IGM(1000,300,10,n=100))


if __name__=="__main__":
    z_ini=100
    z_end=0
    lz3=np.linspace(z_ini,z_end,200)
    c_s_inf = [c_s(z) for z in lz3]
    v_tilda = [v_eff(z) for z in lz3]
    '''
    plt.figure()


    plt.plot(lz3,c_s_inf,label='$c_s$')
    plt.plot(lz3,[vrel_z(z) for z in lz3],label='$v_{rel}$')
    plt.plot(lz3,v_tilda,label='$v_{eff}$')


    plt.xlabel(r'$z$')
    plt.ylabel(r'$v()km/s$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.5,1e1)
    plt.legend()
    plt.savefig('v_eff.pdf')
    plt.close()
    '''
    '''
    M_c=100 #critical mass of PBHs
    sigM=0.5 #distribution width
    psi_m = lambda x: 1/(np.sqrt(2*np.pi)*sigM*x)*np.exp(-(np.log(x/M_c))**2/2/sigM**2)
    fpbh=0.1

    z_ini=100
    z_end=0
    lz1=np.linspace(z_ini,z_end,200)
    lt1=[TZ(z) for z in lz1]
    M100=accret_IGM(100,z_ini,z_end,n=200)
    M1e3=accret_IGM(1e3,z_ini,z_end,n=200)
    M1e4=accret_IGM(1e4,z_ini,z_end,n=200)
    M1e5=accret_IGM(1e5,z_ini,z_end,n=200)
    M1e6=accret_IGM(1e6,z_ini,z_end,n=200)


    plt.figure()

    plt.plot(lz1, M100, '--',linewidth=2, label='$100 \rm M_{\odot}$')
    plt.plot(lz1, M1e3, linewidth=1,label='$M_{PBH}=10^3 \rm M_{\odot}$')
    plt.plot(lz1, M1e4, linewidth=1,label='$M_{PBH}=10^4 \rm M_{\odot}$')
    plt.plot(lz1, M1e5,':', linewidth=2, label='$M_{PBH}=10^5 \rm M_{\odot}$')
    plt.plot(lz1, M1e6,'-.', linewidth=2, label='$M_{PBH}=10^6 \rm M_{\odot}$')

    plt.legend()
    plt.xlabel(r'$z$')
    plt.ylabel(r'$M_{\mathrm{PBH}}\ [\rm M_{\odot}]$')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.xlim(100,6)
    plt.ylim(1e2, 1e7)
    plt.savefig('Bondi.pdf')
    plt.close()

    plt.figure()

    lm=np.logspace(0,np.log10(200),20)
    mf6=[accret_IGM(m,100,6)[-1] for m in lm]
    mf15=[accret_IGM(m,100,15)[-1] for m in lm]
    mf30=[accret_IGM(m,100,30)[-1] for m in lm]

    plt.plot(lm,mf6,label='$z_{end}=6$')
    plt.plot(lm,mf15,label='$z_{end}=15$')
    plt.plot(lm,mf30,label='$z_{end}=30$')


    plt.xlabel(r'$M_{z=z_{ini}}\ [\rm M_{\odot}]$')
    plt.ylabel(r'$M_{z=z_{end}}\ [\rm M_{\odot}]$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('finalmass.pdf')
    plt.close()

    plt.figure()

    eps=1e-3
    lm1=np.logspace(0,4,40)
    mf6=[accret_IGM(m,100,6)[-1] for m in lm1]
    mf15=[accret_IGM(m,100,15)[-1] for m in lm1]
    mf30=[accret_IGM(m,100,30)[-1] for m in lm1]

    mfz6=interpolate.interp1d(lm1,mf6,fill_value='extrapolate')
    mfz15=interpolate.interp1d(lm1,mf15,fill_value='extrapolate')
    mfz30=interpolate.interp1d(lm1,mf30,fill_value='extrapolate')

    dmf_dmi6 = lambda x: (mfz6(x+eps)-mfz6(x-eps))/2/eps
    dmf_dmi15 = lambda x: (mfz15(x+eps)-mfz15(x-eps))/2/eps
    dmf_dmi30 = lambda x: (mfz30(x+eps)-mfz30(x-eps))/2/eps

    plt.plot(mf6,[psi_m(m)/dmf_dmi6(m) for m in lm1],label='$z_{end}=6$')
    plt.plot(mf15,[psi_m(m)/dmf_dmi15(m) for m in lm1],label='$z_{end}=15$')
    plt.plot(mf30,[psi_m(m)/dmf_dmi30(m) for m in lm1],label='$z_{end}=30$')
    plt.plot(lm1,[psi_m(m) for m in lm1],label='initial')

    plt.legend()
    plt.xlim(1,1e4)
    plt.ylim(1e-7,1)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\psi (M)$')
    plt.xlabel(r'$M_{PBH}\ [\rm M_{\odot}]$')
    plt.savefig('distribution.pdf')
    plt.close()


    plt.figure()
    '''

    '''
    z_ini=10
    z_end=0
    lz1=np.linspace(z_ini,z_end,200)
    lt1=[TZ(z) for z in lz1]
    M10=accret_halo(10,z_ini,z_end,1e4,10,n=200)
    M100=accret_halo(100,z_ini,z_end,1e4,10,n=200)
    M1e3=accret_halo(1e3,z_ini,z_end,1e4,10,n=200)
    M1e4=accret_halo(1e4,z_ini,z_end,1e4,10,n=200)


    plt.figure()

    plt.plot(lz1, M10, '--',linewidth=2, label='$10 M_{sun}$')
    plt.plot(lz1, M100, '--',linewidth=2, label='$100 M_{sun}$')
    plt.plot(lz1, M1e3, linewidth=1,label='$M_{PBH}=10^3 M_{sun}$')
    plt.plot(lz1, M1e4, linewidth=1,label='$M_{PBH}=10^4 M_{sun}$')

    plt.legend()
    plt.xlabel(r'$z$')
    plt.ylabel(r'$M_{\mathrm{PBH}}\ [\rm M_{\odot}]$')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.xlim(100,1)
    plt.ylim(1e1, 1e7)
    plt.savefig('Bondi_halo.pdf')
    plt.close()

    plt.figure()
'''

'''
    z_ini=1000
    z_end=10
    lz1=np.linspace(z_ini,z_end,1000)
    fpbh=0.1

    mdotz1=[bondi(1,z,fpbh) for z in lz1]
    mdotz1e1=[bondi(10,z,fpbh) for z in lz1]
    mdotz1e2=[bondi(100,z,fpbh) for z in lz1]
    mdotz1e3=[bondi(1000,z,fpbh) for z in lz1]

    plt.plot(lz1,mdotz1,label='$m_i =1 \rm M_{\odot}$')
    plt.plot(lz1,mdotz1e1,label='$m_i =10 \rm M_{\odot}$')
    plt.plot(lz1,mdotz1e2,label='$m_i =100 \rm M_{\odot}$')
    plt.plot(lz1,mdotz1e3,label='$m_i =1000 \rm M_{\odot}$')


    plt.xlabel(r'$z$')
    plt.ylabel(r'$\dot{m}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(3e-4,3e2)
    plt.xlim(3,5000)
    plt.legend()
    plt.savefig('accretrat_'+str(fpbh)+'.pdf')
    plt.close()

'''