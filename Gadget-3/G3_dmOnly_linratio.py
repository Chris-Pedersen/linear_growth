import numpy as np
import matplotlib.pyplot as plt
import MAS_library as MASL
import Pk_library as PKL
import readsnap
import sys

'''
Script to calculate the power spectrum from a Gadget-3 snapshot
and divide it by the linear theory power spectrum previously calculcated
in reps. Requires Pylians package.
'''

z=int(sys.argv[1])

## Gadget-3 directory
directory="L60_dmonly/"
directory2="L300_dmonly/"
linearString="LinearPower/1pAmp_"

## Linear theory file names
linearTot=linearString+"Pcb_rescaled_z"+str(z)+".0000.txt"

def getG3power(sim,kmax):
   if z==99:
      snap=sim+"/output/ics"
   elif z==49:
      snap=sim+"/output/snapdir_000/PART_000"
   elif z==9:
      snap=sim+"/output/snapdir_001/PART_001"
   elif z==4:
      snap=sim+"/output/snapdir_002/PART_002"
   elif z==3:
      snap=sim+"/output/snapdir_003/PART_003"
   elif z==2:
      snap=sim+"/output/snapdir_005/PART_005"
   else:
      print("Don't have data for that redshift")
      quit()

   head     = readsnap.snapshot_header(snap)
   rho_crit=2.77536627e11
   BoxSize=head.boxsize/1e3 #Mpc/h
   Nall=head.nall
   Masses=head.massarr*1e10 #Msun/h
   Omega_m=head.omega_m
   Omega_l=head.omega_l
   redshift=head.redshift

   ## Pylian params
   grid=512
   ptypes=[1]
   MAS      = 'CIC'
   do_RSD   = False
   axis     = 0
   threads = 1
   assert (z-redshift)<0.001, "Redshift requested: %s, redshift found: %s" % (z,redshift)
   Omega_cdm=Nall[1]*Masses[1]/BoxSize**3/rho_crit
   Omega_b=Omega_m-Omega_cdm

   ## Calculate fractions
   f_b=Omega_b/(Omega_cdm+Omega_b)
   f_c=Omega_cdm/(Omega_cdm+Omega_b)

   ## CDM
   delta = MASL.density_field_gadget(snap, ptypes, grid, MAS, do_RSD, axis)
   delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
   print("Mean after normalising = %.2e" % np.mean(delta,dtype=np.float64))
   Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads)
   # Calculate power
   k_sim      = Pk.k3D
   Pk_sim     = Pk.Pk[:,0]
   Nmodes1D = Pk.Nmodes1D


   return k_sim[np.where(k_sim<kmax)], Pk_sim[np.where(k_sim<kmax)], BoxSize

k_sim,cdm, BoxSize=getG3power(directory,5)
k_simB, cdmB, BoxSize2=getG3power(directory2,0.4)

## Interpolate linear theory results
linearDataTotal=np.loadtxt(linearTot)
linearInterpTot=np.interp(k_sim,linearDataTotal[:,0],linearDataTotal[:,1])
linearInterpTotB=np.interp(k_simB,linearDataTotal[:,0],linearDataTotal[:,1])

## Plot
plt.figure(figsize=[8,6])
plt.semilogx(k_sim,cdm/linearInterpTot,marker="o",ms=3.5,mew=0,color="blue",linestyle="None",alpha=0.8,label="L=%s" % BoxSize)
plt.semilogx(k_simB,cdmB/linearInterpTotB,marker="o",ms=3.5,mew=0,color="red",linestyle="None",alpha=0.8,label="L=%s" % BoxSize2)
plt.ylabel("Gadget-3/CAMB")
plt.text(min(k_simB),1.13,"z=%d" % z,fontweight="bold")
plt.tick_params(axis="both",which="both",direction="in",right=True,top=True)
plt.tight_layout()
plt.xlabel("k h/Mpc")
plt.legend(loc="best")
plt.axhline(1.0,color="gray",linestyle="dashed")
plt.axhspan(0.99,1.01,color="gray",alpha=0.15)
plt.ylim(0.85,1.15)
plt.show("hold")
