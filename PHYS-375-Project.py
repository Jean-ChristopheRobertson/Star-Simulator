import numpy as np
import matplotlib.pyplot as plt
import os
import time

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!NUMBERS!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~~Constants (SI units)~~
Mp = 1.6726219e-27     ## Mass of proton
Me = 9.10938356e-31    ## Mass of electron
kB = 1.38064852e-23    ## Boltzmann constant
hbar = 1.054571800e-34 ## Reduced Planck constant (h-bar)
rad = 7.5657e-16       ## Radiation constant
sig = 5.670373e-8      ## Stefan-Boltzmann constant
G = 6.67408e-11        ## Newton's constant
ai = 5.0/3.0           ## Adiabatic Index (constant at 5/3??)
c = 299792458.0        ## Speed of light
R_sun = 695700000.0    ## Solar radius
M_sun = 1.989e30       ## Solar mass
L_sun = 3.828e26       ## Solar luminosity
pi = np.pi             ## Ratio of a circle's circumference to its diameter

##~~Composition (take these as constants??)~~
X = 0.7381                      ## Hydrogen fraction
Y = 0.2485                      ## Helium fraction
Z = 0.0134                      ## Metal fraction
mu = 1.0/(2.0*X+0.75*Y+0.5*Z)   ## Mean molecular mass
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!NUMBERS!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!FUNCTIONS!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Specific energy generation; sum of PP-chain & CNO-cycle
def eps(rho, T):

    PPC = 1.07e-7*(rho/10.0**5)*X*X*(T/10.0**6)**4
    CNO = 8.24e-26*(rho/10.0**5)*X*0.03*X*(T/10.0**6)**(19.9)
    return np.array([PPC + CNO, PPC, CNO])

## Scattering(?) Rossaland mean opacity
def Kes(rho, T):

    return 0.02*(1.0 + X)

## Free-free Rossaland mean opacity
def Kff(rho, T):

    return 1.0e24*(Z + 0.0001)*(rho/(10.0**3))**(0.7)*T**(-3.5)

## H- Rossaland mean opacity
def KHm(rho, T):

    return 2.5e-32*(Z/0.02)*(rho/(10.0**3))**(0.5)*T**9

## Total opacity
def Kap(rho, T):

    MaxKap = max([Kes(rho, T), Kff(rho, T)])
    TotKap = (1.0/KHm(rho, T) + 1.0/MaxKap)**(-1.0)
    return np.array([TotKap, Kes(rho, T), Kff(rho, T), KHm(rho, T)])

## Equation of State, i.e. Pressure as a function of temperature & density
def EOS(rho, T):

    Pdeg = (((3.0*pi*pi)**(2.0/3.0))/5.0)*(hbar*hbar/Me)*(rho/Mp)**(5.0/3.0)
    Pigl = (rho*kB*T)/(mu*Mp)
    Prad = (1.0/3.0)*rad*T**4

    #Pigl = 0.0
    #Pdeg = 0.0
    #Prad = 0.0
    return np.array([Pdeg + Pigl + Prad, Pdeg, Pigl, Prad])
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!FUNCTIONS!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!EQUATIONS OF STELLAR STRUCTURE!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Derivative of Pressure wrt radius -> Hydrostatic Equilibrium
def dPdr(rho, M, r):

    return -G*M*rho/(r*r)

## Partial derivative of Pressure wrt density
def dPdrho(rho, T):

    degen = (((3*pi*pi)**(2.0/3.0))/3.0)*((hbar*hbar)/(Me*Mp))*(rho/Mp)**(2.0/3.0)
    thermal = (kB*T)/(mu*Mp)

    #thermal = 0.0
    #degen = 0.0
    return degen + thermal

## Partial derivative of Pressure wrt temperature
def dPdT(rho, T):

    thermal = (rho*kB)/(mu*Mp)
    radiative = (4.0/3.0)*(rad*T*T*T)

    #thermal = 0.0
    #radiative = 0.0
    return thermal + radiative

## Temperature gradient
def dTdr(P, rho, T, M, L, r):

    kap = Kap(rho, T)
    diff = (3.0*kap[0]*rho*L)/(16.0*pi*rad*c*T*T*T*r*r)
    conv = (1.0-1.0/ai)*(T/P)*(G*M*rho)/(r*r)
    
    return -1.0*min(diff, conv)

## Mass continuity
def dMdr(rho, r):

    return 4.0*pi*rho*r*r

## Derivative of density wrt radius
def drhodr(P, rho, T, M, L, r):

    TempGrad = dTdr(P, rho, T, M, L, r)
    return -((G*M*rho)/(r*r) + dPdT(rho, T)*TempGrad)/(dPdrho(rho, T))

## Luminosity Equation
def dLdr(rho, T, r):

    enerate = np.array(eps(rho, T))
    return 4*pi*rho*r*r*enerate

## Gradient of optical depth (not sure this is needed)
def dTaudr(rho, T):
    
    kap = Kap(rho, T)
    return (kap[0])*rho

## Opacity proxy
def deltau(P, rho, T, M, L, r):

    return (Kap(rho, T)[0])*rho*rho/(abs(drhodr(P, rho, T, M, L, r)))
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!EQUATIONS OF STELLAR STRUCTURE!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!BUNCHA RUNGE-KUTTA COEFFICIENTS!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## RK45 coefficients from numerical recipes textbook (Cash/Karp parameters)
a = np.array([1/5, 3/10, 3/5, 1, 7/8])
b = np.array([1/5,
              3/40, 9/40,
              3/10, -9/10, 6/5,
              -11/54, 5/2, -70/27, 35/27,
              1631/55296, 175/512, 575/13824, 44275/110592, 253/4096])
c4 = np.array([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4])
c5 = np.array([37/378, 0, 250/621, 125/594, 0, 512/1771])

## RKF45 coefficients
####a = np.array([1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0])
####b = np.array([1.0/4.0,
####              3.0/32.0, 9.0/32.0,
####              1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0,
####              439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0,
####              -8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0])
####c4 = np.array([25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4101.0, -1.0/5.0, 0.0])
####c5 = np.array([16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0])
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!BUNCHA RUNGE-KUTTA COEFFICIENTS!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!DIFFERENTIAL EQUATION INTEGRATOR!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~~Derivatives Array~~
np.seterr(invalid="raise")

## Takes current solutions for stellar parameters to calculate and return their derivatives with the same structure
def derivs(sols, r):

    Pgrad = dPdr(sols[1], sols[3], r)
    Dgrad = drhodr(sols[0], sols[1], sols[2], sols[3], sols[4], r)
    Tgrad = dTdr(sols[0], sols[1], sols[2], sols[3], sols[4], r)
    Mgrad = dMdr(sols[1], r)
    Lgrad = (dLdr(sols[1], sols[2], r)[0])
    Ograd = dTaudr(sols[1], sols[2])
    return np.array([Pgrad, Dgrad, Tgrad, Mgrad, Lgrad, Ograd])

##~~The DEs solver itself, Terror of the East~~
## Takes central T and rho and solves the system of coupled DE's of stellar structure using Runge-Kutta 4-5.
## Has option to save intermediate values in lists to be used for plotting, which defaults to false.
def DESolve(Tc, rhoc):

    ##print('rho =',rhoc)
    ##~~Initial Conditions~~
    h = 1.0                         ## Initial step size
    r = 0.01*h                      ## Initial radius; r_0 << scale of star
    S = 1.0                         ## "Safety Factor"..."Within a few percent of unity"
    Pc = (EOS(rhoc, Tc)[0])         ## Centre Pressure (from EOS)
    Mc = (4.0/3.0)*pi*(r*r*r)*rhoc  ## Centre Mass
    Lc = Mc*(eps(rhoc, Tc)[0])      ## Centre Luminosity
    Tauc = r*rhoc*(Kap(rhoc, Tc)[0])## Centre Optical Depth

    ## Solutions array; it holds all the answers
    ## sols[0] = Pressure
    ## sols[1] = Density
    ## sols[2] = Temperature
    ## sols[3] = Mass
    ## sols[4] = Luminosity
    ## sols[5] = Optical Depth
    sols = np.array([Pc, rhoc, Tc, Mc, Lc, Tauc])
    lastep = sols
    
    ##~~Error Tolerance & Halting Criteria~~
    tol = 1e-15              ## Sets error tolerance <-- reduce by an order of magnitude if you get invalid values
    maxmass = 10**3*M_sun   ## Maximum mass as an alternate halting criteria
    ZeroDepth = 1e-5        ## Sets halting criteria of integration

    ## Initial opacity proxy
    proxy = deltau(sols[0], sols[1], sols[2], sols[3], sols[4], r)
    
    ## Lists for storing intermediate values
    P = []
    M = []
    D = []
    L = []
    T = []
    O = []

    ## Random, useful values to hold onto
    radii = []
    kappa = []
    LumGrad = []
    logder = []

    ## Pressure Components
    Pigl = []
    Pdeg = []
    Prad = []

    ## Components of energy generation
    LumPPC = []
    LumCNO = []

    ## Components of the oppacity
    KES = []
    KFF = []
    KHM = []
    
    count = 0
    ##~~Main DE Loop~~
    while sols[3] < maxmass and proxy > ZeroDepth:

        try:
            ##~~Runge's Kuttas~~
            k1 = h*derivs(sols, r)
            k2 = h*derivs(sols + b[0]*k1, r + a[0]*h)
            k3 = h*derivs(sols + b[1]*k1 + b[2]*k2, r + a[1]*h)
            k4 = h*derivs(sols + b[3]*k1 + b[4]*k2 + b[5]*k3, r + a[2]*h)
            k5 = h*derivs(sols + b[6]*k1 + b[7]*k2 + b[8]*k3 + b[9]*k4, r + a[3]*h)
            k6 = h*derivs(sols + b[10]*k1 + b[11]*k2 + b[12]*k3 + b[13]*k4 + b[14]*k5, r + a[4]*h)

        except FloatingPointError:
            break
        
        ##~~RK4 & RK5~~
        sols4 = sols + c4[0]*k1 + c4[1]*k2 + c4[2]*k3 + c4[3]*k4 + c4[4]*k5 + c4[5]*k6
        sols5 = sols + c5[0]*k1 + c5[1]*k2 + c5[2]*k3 + c5[3]*k4 + c5[4]*k5 + c5[5]*k6

        ## Checking the second halting criteria (the opacity proxy)    
        proxy = deltau(sols[0], sols[1], sols[2], sols[3], sols[4], r)
        
        ##~~Updating Step Size~~  
        error = abs(sols5 - sols4)  ## Estimate of Error
        target = tol*abs(sols4 + k1)+5## Target Error

        ## Positive gap is good
####        gap = target - error
####
####        del_1 = error[np.argmin(gap)]
####        del_0 = target[np.argmax(gap)]
        
        ## Storing the ratio of error/target in relerr
        relerr = []
        for i in range(4):
            
            if error[i] == 0.0:
                
                relerr.append(0.0)
                
            else:
                
                relerr.append(abs(error[i]/target[i]))

        ## Taking the maximum value; i.e. the error from the "worst-offender equation"
        del_1 = error[np.argmax(relerr)]
        del_0 = target[np.argmax(relerr)]
        
        if abs(del_0) >= abs(del_1):

            ## Error small enough; increase step size & move on
            if del_1 != 0.0:

                count = 0
                h = S*h*(abs(del_0/(2.0*del_1)))**(0.2)
####                if h >= hmax:
####                    h = hmax
                
            ## Appending to lists for plotting and such
            P.append(sols[0])
            D.append(sols[1])
            T.append(sols[2])
            M.append(sols[3])
            L.append(sols[4])
            O.append(sols[5])
            radii.append(r)

            KapVec = Kap(sols[1], sols[2])
            LumVec = dLdr(sols[1], sols[2], r)
            PreVec = EOS(sols[1], sols[2])

            ## Oppacity & Luminosity Gradient
            kappa.append(KapVec[0])
            LumGrad.append(LumVec[0])
            
            ## Pressure Components
            Pdeg.append(PreVec[1])
            Pigl.append(PreVec[2])
            Prad.append(PreVec[3])

            ## Components of energy generation
            LumPPC.append(LumVec[1])
            LumCNO.append(LumVec[2])

            ## Components of the oppacity
            KES.append(KapVec[1])
            KFF.append(KapVec[2])
            KHM.append(KapVec[3])

            if r > 10.0:
                
                if P[-1]/P[-2] == 1 or T[-1]/T[-2] == 1:
                    
                    if P[-1]/P[-2] == 1 and T[-1]/T[-2] == 1:

                        logder.append(1)

                    else:

                        logder.append(0)

                else:
                    
                    logder.append(np.log(P[-1]/P[-2])/np.log(T[-1]/T[-2]))
            
            r += h
            sols = sols4
            
        else:
            
            ## Error too large; lower step size & retry step
            if del_1 != 0.0:

                count += 1
                h = S*h*(abs(del_0/(2.0*del_1)))**(0.25)
####                print('Error is too large, recalculating')
####                print('h = %4.2e' %(h))
####                print('# of failed steps:',count)

        if len(radii)%10000 == 0:

            print('Current radius: %4.2e km' %(r/1e3))

    ##~~Checking for Surface Parameters of the Star
    ## Finding index where the surface is located
    TauInf = O[-1]
    StarDex = np.abs(O - (TauInf - 2.0/3.0)).argmin()

    ## Saving relavent parameters for generating Mais Sequence
    RStar = radii[StarDex]
    TStar = T[StarDex]
    LStar = L[StarDex]
    MStar = M[StarDex]

    ## Trimming arrays of intermediate values to only hold values from the centre to the surface
    P = P[:StarDex]
    D = D[:StarDex]
    T = T[:StarDex]
    M = M[:StarDex]
    L = L[:StarDex]
    O = O[:StarDex]
    radii = radii[:StarDex]
    kappa = kappa[:StarDex]
    LumGrad = LumGrad[:StarDex]
    logder = logder[:StarDex]

    Pdeg = Pdeg[:StarDex]
    Pigl = Pigl[:StarDex]
    Prad = Prad[:StarDex]

    ## Components of energy generation
    LumPPC = LumPPC[:StarDex]
    LumCNO = LumCNO[:StarDex]

    ## Components of the oppacity
    KES = KES[:StarDex]
    KFF = KFF[:StarDex]
    KHM = KHM[:StarDex]


    ## Setting up the tuple of values
    star_attrs = [RStar, TStar, LStar, MStar]
    intermediate_vals = (P, D, T, M, L, O, np.array(radii), kappa, LumGrad, Pdeg, Pigl, Prad, LumPPC, LumCNO, KES, KFF, KHM, logder)

    Pres, Dens, Temp, Mass, Lumi, Opti, Radi, Kapp, LumGrad, Pdeg, Pigl, Prad, LumPPC, LumCNO, KES, KFF, KHM, LnRa = intermediate_vals

    return star_attrs, intermediate_vals
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!DIFFERENTIAL EQUATION SOLVER!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!TUNING BOUNDARY CONDITIONS!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Define f as function of rho given T
def f(rho, T):

    SurfVals1, intermediate_vals = DESolve(T, rho)
    RStar, TStar, LStar, MStar = SurfVals1

    ## Ol' St. Bo
    LStBo = 4.0*pi*sig*RStar*RStar*TStar**(4.0)
    return (LStar-LStBo)/((LStar*LStBo)**(0.5))

## Function returns a central density given a central temperature by finding the root of f
def Bisect(T):
    
    ## rho rho rho your boat
    rho_max = 5.0e5
    rho_min = 3.0e2
    rho_cen = 0.5*(rho_max + rho_min)
    rho_pre = rho_min
    rho_act = rho_cen
    trial = f(rho_cen, T)

    effs = []
    rhos = []
    
    ## Prints message saying whether or not there is a root between the two
    ## endpoints by checking the sign of the product of f(rho_min) & f(rho_max)
####    if f(rho_max,T)*f(rho_min,T) < 0.0:
####        
####        print('There is a root of f between',rho_min,'and',rho_max)
####
####    else:
####        
####        print('There is no root of f between',rho_min,'and',rho_max)

    ## this flag will check whether or not rho converged before f
    flag = False

    ## Tolerance for this bisection method
    BiTol = 0.05
    while abs(trial) >= BiTol:

        ## Evaluates the trial solution through the function f
        trial = f(rho_cen, T)
        ##print(trial)
        
        ## Checking to see if rho converged
        if (abs(rho_pre - rho_cen) == 0.0):

            flag = True
            break

        effs.append(trial)
        rhos.append(rho_cen)
            
        rho_pre = rho_cen
        if trial < 0.0:

            rho_min = rho_cen
            rho_cen = 0.5*(rho_max + rho_min)
            
        elif trial > 0.0:

            rho_max = rho_cen
            rho_cen = 0.5*(rho_max + rho_min)

    effs = np.array(effs)
    return rhos[np.argmin(abs(effs))], flag
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!TUNING BOUNDARY CONDITIONS!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!CREATING STARS!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Function to go through the whole process, from start to finish, with only one star
## Given Central Temp, return Surface Temp, Luminosity, & Radius, kwarg for individual plots
def OneStar(T, rhoc, plots=False):

    if rhoc == 0:
        
        rho, flag = Bisect(T)

    else:

        flag = False
        rho = rhoc

    print(rho)
    SurfVals1, intermediate_vals = DESolve(T, rho)
    Pres, Dens, Temp, Mass, Lumi, Opti, Radi, Kapp, LumGrad, Pdeg, Pigl, Prad, LumPPC, LumCNO, KES, KFF, KHM, LnRa = intermediate_vals
    RStar, TStar, LStar, MStar = SurfVals1
    
    ## If rho converged, but not f, sets T to be such that it matches Stefan-Boltzmann w/ L and R given
    if flag:

        oldTStar = TStar
        TStar = (LStar/(4.0*sig*pi*RStar*RStar))**(0.25)

    ## Making plots of stellar structure of individual stars
    if plots:
        
        a, b = Truism(LnRa)

        ##~~Plot of Pressure, Density, Temperature, Mass & Luminosity~~
        plt.plot(Radi/RStar,Pres/max(Pres),'m-')
        plt.plot(Radi/RStar,Dens/max(Dens),'b-')
        plt.plot(Radi/RStar,Temp/max(Temp),'r-')
        plt.plot(Radi/RStar,Mass/max(Mass),'k-')
        plt.plot(Radi/RStar,Lumi/max(Lumi),'g-')
        for i in range(len(a)):
            
            plt.axvspan(Radi[a[i]]/RStar,Radi[b[i]]/RStar,color='y', alpha=0.5,lw=0)
        
        plt.ylabel(r'$\frac{\rho}{\rho_c},\frac{P}{P_c},\frac{T}{T_c},\frac{M}{M_*},\frac{L}{L_*}$',size=25)
        plt.xlabel(r'$\frac{r}{R_*}$',size=25)
        plt.title(r'$Stellar\ Properties\ (T_c = 7.5\times 10^6K,\ M_* = %4.2f M_{\odot})$' %(MStar/M_sun),size=29, y=1.02)

####        plt.title(r'$Stellar\ Properties\ (T_c = %4.1e,\ M_* = %4.2f M_{\odot})$' %(T,MStar/M_sun),size=29, y=1.02)
####        plt.title(r'$T_c\ =\ %4.3e\mathrm{K},\ \rho_c\ =\ %4.3e\mathrm{kg \cdot m^{-3}},\ T_*\ =\ %4.0f\mathrm{K},$' %(T, rho, TStar)\
####                  + '\n' +\
####                  r'$M_*\ =\ %4.2f M_{\odot},\ R_*\ =\ %4.2f R_{\odot},\ L_*\ =\ %4.2f L_{\odot}$'\
####                  %(MStar/M_sun, RStar/R_sun, LStar/L_sun),size=25)
        
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.legend([r'$P$',
                    r'$\rho$',
                    r'$T$',
                    r'$M$',
                    r'$L$'],\
                   fontsize=15,loc='best')
        plt.savefig('SingleStar.png')
        plt.clf()
        plt.show()
        
        ## Plots of Pressure Components
        plt.plot(Radi/RStar,Pres/max(Pres),'k-')
        plt.plot(Radi/RStar,Pigl/max(Pres),'r--')
        plt.plot(Radi/RStar,Pdeg/max(Pres),'b--')
        plt.plot(Radi/RStar,Prad/max(Pres),'m--')
        for i in range(len(a)):
            
            plt.axvspan(Radi[a[i]]/RStar,Radi[b[i]]/RStar,color='y', alpha=0.5,lw=0)
        
        plt.ylabel(r'$\frac{P}{P_c}$',size=25)
        plt.xlabel(r'$\frac{r}{R_*}$',size=25)
        plt.title(r'$Pressure\ Components\ (T_c = 3.0\times 10^7K,\ M_* = %4.2f M_{\odot})$' %(MStar/M_sun),size=29, y=1.02)
####        plt.title(r'$T_c\ =\ %4.3e\mathrm{K},\ \rho_c\ =\ %4.3e\mathrm{kg \cdot m^{-3}},\ T_*\ =\ %4.0f\mathrm{K},$' %(T, rho, TStar)\
####                  + '\n' +\
####                  r'$M_*\ =\ %4.2f M_{\odot},\ R_*\ =\ %4.2f R_{\odot},\ L_*\ =\ %4.2f L_{\odot}$'\
####                  %(MStar/M_sun, RStar/R_sun, LStar/L_sun),size=25)
        
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.legend([r'$Total$',
                    r'$Thermal$',
                    r'$Degeneracy$',
                    r'$Radiation$'],\
                   fontsize=15,loc='best')
        plt.savefig('SingleStar.png')
        plt.clf()

        plt.show()

	##~~Opacity Plot~~        
        plt.semilogy(Radi/RStar, Kapp,'k-')
        plt.semilogy(Radi/RStar, KES,'r--')
        plt.semilogy(Radi/RStar, KFF,'b--')
        plt.semilogy(Radi/RStar, KHM,'m--')
        for i in range(len(a)):

            plt.axvspan(Radi[a[i]]/RStar,Radi[b[i]]/RStar,color='y', alpha=0.5,lw=0)

        plt.ylabel(r'$\kappa$', size=25)
        plt.xlabel(r'$\frac{r}{R_*}$', size=25)
        plt.title(r'$Opacity\ for\ Star\ (T_c = 7.5\times 10^6K,\ M_* = %4.2f M_{\odot})$' %(MStar/M_sun), size=29, y=1.02)
        plt.xlim(0,1)
        plt.ylim(-1,10**10)
        plt.legend([r'$\kappa$',
                    r'$\kappa_{es}$',
                    r'$\kappa_{ff}$',
                    r'$\kappa_{H^-}$'],\
                   fontsize=15,loc='best')

        plt.savefig('kappa.png')
        plt.clf()

        plt.show()

        ##~~Energy Generation Equation~~            
        plt.plot(Radi/RStar,LumGrad,'k-')
        plt.plot(Radi/RStar,LumPPC,'r--')
        plt.plot(Radi/RStar,LumCNO,'b--')
        for i in range(len(a)):

            plt.axvspan(Radi[a[i]]/RStar,Radi[b[i]]/RStar,color='y', alpha=0.5,lw=0)

        plt.ylabel(r'$\frac{dL}{dr}$', size=25)
        plt.xlabel(r'$\frac{r}{R_*}$', size=25)
        plt.title(r'$Luminosity\ Gradient\ (T_c = 7.5\times 10^6K,\ M_* = %4.2f M_{\odot})$' %(MStar/M_sun), size=29, y=1.02)
        plt.xlim(0.0,1.0)
        plt.legend([r'$Total$',
                    r'$PP Chain$',
                    r'$CNO Cycle$'],\
                   fontsize=15,loc='best')

        plt.savefig('dLdr.png')
        plt.clf()
        plt.show()

        ##~~Logarithmic Differential Ratios~~
        plt.plot(Radi/RStar,LnRa)
        plt.plot([0.0,1.0],[5/2,5/2])
        for i in range(len(a)):

            plt.axvspan(Radi[a[i]]/RStar,Radi[b[i]]/RStar,color='y', alpha=0.5,lw=0)

        plt.title(r'$Convective\ Stability\ Criterion\ (T_c = 7.5\times 10^6K,\ M_* = %4.2f M_{\odot})$' %(MStar/M_sun),size=29, y=1.02)
        plt.xlabel(r'$\frac{r}{R_*}$',size=25)
        plt.ylabel(r'$\frac{d(\mathrm{ln}P)}{d(\mathrm{ln}T)}$',size=25)
        plt.legend([r'$\frac{d(\mathrm{ln}P)}{d(\mathrm{ln}T)}$',
                    r'$\left(1-\frac{1}{\gamma}\right)^{-1}=\frac{5}{2}$'],\
                   fontsize=15,loc='best')
        plt.show()


    print('Radius =',RStar/R_sun,r'$R_{\odot}$')
    print('Mass =',MStar/M_sun,r'$M_{\odot}$')
    print('Luminosity =',LStar/L_sun,r'$L_{\odot}$')
    print('Surface Temperature =',TStar,'K')
    
    return [RStar, LStar, TStar, MStar, rho]




##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!GENERATING MAIN SEQUENCES!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Calls OneStar for an array of Central Temps, hopefully a main sequence arises from this
def MainSequence(Tcs):

    count = 0
    #Tcs = np.linspace(T_min, T_max, N)
##    RStar = []
##    LStar = []
##    TStar = []
##    MStar = []
##    rho = []
##
##    for T in Tcs:
##
##        SurfVals = OneStar(T,0)
##        RStar.append(SurfVals[0])
##        LStar.append(SurfVals[1])
##        TStar.append(SurfVals[2])
##        MStar.append(SurfVals[3])
##        rho.append(SurfVals[4])
##        count += 1

    ##~~Reads BC's from a .dat file~~

    ##~~Reading data from .dat file to plot all at once~~
    ## Without Radiation Pressure ~ 1/4
    Tcs = []
    current_dir = os.getcwd()

    rhocsRad = []
    RStarRad = []
    LStarRad = []
    TStarRad = []
    MStarRad = []
    data_file = open(current_dir + '/WithoutRadiation.dat', 'r')    
    line = data_file.readline()
    while line:
        
        data = line[:-1]
        data = data.split(",")
        Tcs.append(float(data[0]))
        rhocsRad.append(float(data[1]))
        RStarRad.append(float(data[2]))
        LStarRad.append(float(data[3]))
        TStarRad.append(float(data[4]))
        MStarRad.append(float(data[5]))
        line = data_file.readline()
        count += 1
        print('Done', count,'of', 2000,'Stars.')        

    data_file.close()
    RStarRad = np.array(RStarRad)
    LStarRad = np.array(LStarRad)
    TStarRad = np.array(TStarRad)
    MStarRad = np.array(MStarRad)
    rhoRad = np.array(rhocsRad)


    ## Without Degeneracy Pressure ~ 2/4
    rhocsDeg = []
    RStarDeg = []
    LStarDeg = []
    TStarDeg = []
    MStarDeg = []
    data_file = open(current_dir + '/WithoutDegeneracy.dat', 'r')    
    line = data_file.readline()
    while line:
        
        data = line[:-1]
        data = data.split(",")
        rhocsDeg.append(float(data[1]))
        RStarDeg.append(float(data[2]))
        LStarDeg.append(float(data[3]))
        TStarDeg.append(float(data[4]))
        MStarDeg.append(float(data[5]))
        line = data_file.readline()
        count += 1
        print('Done', count,'of', 2000,'Stars.')        

    data_file.close()
    RStarDeg = np.array(RStarDeg)
    LStarDeg = np.array(LStarDeg)
    TStarDeg = np.array(TStarDeg)
    MStarDeg = np.array(MStarDeg)
    rhoDeg = np.array(rhocsDeg)

    ## Without Thermal Pressure ~ 3/4
    rhocsIgl = []
    RStarIgl = []
    LStarIgl = []
    TStarIgl = []
    MStarIgl = []
    data_file = open(current_dir + '/WithoutThermal.dat', 'r')    
    line = data_file.readline()
    while line:
        
        data = line[:-1]
        data = data.split(",")
        rhocsIgl.append(float(data[1]))
        RStarIgl.append(float(data[2]))
        LStarIgl.append(float(data[3]))
        TStarIgl.append(float(data[4]))
        MStarIgl.append(float(data[5]))
        line = data_file.readline()
        count += 1
        print('Done', count,'of', 2000,'Stars.')        

    data_file.close()
    RStarIgl = np.array(RStarIgl)
    LStarIgl = np.array(LStarIgl)
    TStarIgl = np.array(TStarIgl)
    MStarIgl = np.array(MStarIgl)
    rhoIgl = np.array(rhocsIgl)

    ## ALL STEAM AHEAD ~ 4/4
    rhocsAll = []
    RStarAll = []
    LStarAll = []
    TStarAll = []
    MStarAll = []
    data_file = open(current_dir + '/UnalteredMainSequence.dat', 'r')    
    line = data_file.readline()
    while line:
        
        data = line[:-1]
        data = data.split(",")
        rhocsAll.append(float(data[1]))
        RStarAll.append(float(data[2]))
        LStarAll.append(float(data[3]))
        TStarAll.append(float(data[4]))
        MStarAll.append(float(data[5]))
        line = data_file.readline()
        count += 1
        print('Done', count,'of', 2000,'Stars.')        

    data_file.close()
    RStarAll = np.array(RStarAll)
    LStarAll = np.array(LStarAll)
    TStarAll = np.array(TStarAll)
    MStarAll = np.array(MStarAll)
    rhoAll = np.array(rhocsAll)


    #/IMPORTANT PART

    Tcs = np.array(0.5e7)    

    MassEffects = np.linspace(0.08, 100.0, 10000.0)
    RadEffects = MasRad(MassEffects)
    LumEffects = MasLum(MassEffects)
    
    plt.loglog(TStarAll, LStarAll/L_sun, marker='.', linestyle='', color='k')
    plt.loglog(TStarIgl, LStarIgl/L_sun, marker='.', linestyle='', color='r')
    plt.loglog(TStarDeg, LStarDeg/L_sun, marker='.', linestyle='', color='m')
    plt.loglog(TStarRad, LStarRad/L_sun, marker='.', linestyle='', color='g')

    plt.xlim(10**5,10**3)
    plt.title(r'$Main\ Sequences\ with\ Different\ Pressures$', size=30)
    plt.xlabel(r'$\mathrm{log_{10}}(\frac{T_*}{\mathrm{K}})$', size=30)
    plt.ylabel(r'$\mathrm{log_{10}}\left(\frac{L}{L_{\odot}}\right)$', size=30)
    plt.legend([r'$Unaltered$',r'$w/out\ Thermal$',r'$w/out\ Degeneracy$',r'$w/out\ Radiation$'],\
               fontsize=15,loc='best')    
    plt.show()

    plt.loglog(MassEffects, RadEffects)
    plt.loglog(MStarAll/M_sun, RStarAll/R_sun, color='k', marker='.', linestyle='')
    plt.loglog(MStarIgl/M_sun, RStarIgl/R_sun, color='r', marker='.', linestyle='')
    plt.loglog(MStarDeg/M_sun, RStarDeg/R_sun, color='m', marker='.', linestyle='')
    plt.loglog(MStarRad/M_sun, RStarRad/R_sun, color='g', marker='.', linestyle='')
    
    plt.title(r'$Relation\ Between\ Mass\ and\ Radius\ with\ Different\ Pressures$',size=30)
    plt.xlabel(r'$\mathrm{log_{10}}\left(\frac{M}{M_{\odot}}\right)$',size=30)
    plt.ylabel(r'$\mathrm{log_{10}}\left(\frac{R}{R_{\odot}}\right)$',size=30)
    plt.legend([r'$Empirical\ Relation$',
                r'$Unaltered$',r'$w/out\ Thermal$',r'$w/out\ Degeneracy$',r'$w/out\ Radiation$'],\
               fontsize=15,loc='best')
    plt.show()

    plt.loglog(MassEffects, LumEffects)
    plt.loglog(MStarAll/M_sun, LStarAll/L_sun, color='k', marker='.', linestyle='')
    plt.loglog(MStarIgl/M_sun, LStarIgl/L_sun, color='r', marker='.', linestyle='')
    plt.loglog(MStarDeg/M_sun, LStarDeg/L_sun, color='m', marker='.', linestyle='')
    plt.loglog(MStarRad/M_sun, LStarRad/L_sun, color='g', marker='.', linestyle='')
    
    plt.title(r'$Relation\ Between\ Mass\ and\ Luminosity\ with\ Different\ Pressures$',size=30)
    plt.xlabel(r'$\mathrm{log_{10}}\left(\frac{M}{M_{\odot}}\right)$',size=30)
    plt.ylabel(r'$\mathrm{log_{10}}\left(\frac{L}{L_{\odot}}\right)$',size=30)
    plt.legend([r'$Empirical\ Relation$',
                r'$Unaltered$',r'$w/out\ Thermal$',r'$w/out\ Degeneracy$',r'$w/out\ Radiation$'],\
               fontsize=15,loc='best')
    plt.show()

####    current_dir = os.getcwd()
####    file = open(current_dir + '/WithoutRadiation.dat', 'w')
####
####    ## Writes to the file in the forms 'Tc, rhoc' on each line
####    file.write('CenTemp, CenDens, Radiius, TotLumi, SurfTem, TotMass')
####    for i in range(len(Tcs)):
####        
####        a = '{:7.5e}'.format(Tcs[i])
####        b = '{:7.5e}'.format(rho[i])
####        c = '{:7.5e}'.format(RStar[i])
####        d = '{:7.5e}'.format(LStar[i])
####        e = '{:7.5e}'.format(TStar[i])
####        f = '{:7.5e}'.format(MStar[i])
####        
####        file.write(a + ', ' + b + ', ' + c + ', ' + d + ', ' + e + ', ' + f + '\n')
####
####    file.close()

##~~Textbook Empirical Relations Between Stellar Properties~~
## textbook function for L(M)
def MasLum(Ms):
	
	holder = []
	for M in Ms:
            
		if M <= 0.7:
                    
			holder.append(0.35*(M)**(2.62))
			
		else:
                    
			holder.append(1.02*(M)**(3.92))

	return np.array(holder)

## textbook function for R(M)
def MasRad(Ms):
	
	holder = []
	for M in Ms:
            
		if M <= 1.66:
                    
			holder.append(1.06*(M)**(0.945))
			
		else:
                    
			holder.append(1.33*(M)**(0.555))

	return np.array(holder)

##~~The Name is Due to Legacy Reasons~~
def Truism(array):

    a = []
    b = []

    ## Checks for grouped values
    Group = False
    for i in range(len(array)):

        if abs(array[i]-2.5) < 1e-5:

            if not Group:

                Group = True
                a.append(i)

        else:

            if Group:

                Group = False
                b.append(i - 1)
                if a[-1] == b[-1]:

                    b.remove(b[-1])
                    a.remove(a[-1])

        if Group and i == len(array)-1:

            b.append(i)

    return [a, b]


## Density for T_c = 3e7:
## 3368.869487766642
## Density for T_c = 7.5e6:
## 71330.84338390958

Tcs = np.linspace(7.5e6, 3e7, 500)
t1 = time.time()
OneStar(1.85e7, 0.0, plots=True)
##MainSequence(Tcs)
t0 = time.time()
##print('Approximately ','{:4.2f}'.format((t0-t1)/3600), 'hours to generate Main Sequence of %5i Stars.' %(len(Tcs)))



