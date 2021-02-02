from reciprocalspaceship.algorithms.scale_merged_intensities import _french_wilson_posterior_quad
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data_fw_mcmc = pd.read_csv("fw_mcmc_data.csv")
I,SigI,Sigma,J,SigJ,F,SigF,Centric = data_fw_mcmc.to_numpy(np.float64).T

Centric = Centric.astype(np.bool)
rs_J,rs_SigJ,rs_F,rs_SigF = _french_wilson_posterior_quad(I, SigI, Sigma, Centric)

#All the plots
plt.plot(J, rs_J, 'k.', label='Acentric')
plt.plot(J[Centric], rs_J[Centric], 'r.', label='Centric')
plt.xlabel("MCMC J")
plt.ylabel("rs J")
plt.legend()
plt.savefig("J.png")

plt.figure()
plt.plot(F, rs_F, 'k.', label='Acentric')
plt.plot(F[Centric], rs_F[Centric], 'r.', label='Centric')
plt.xlabel("MCMC F")
plt.ylabel("rs F")
plt.legend()
plt.savefig("F.png")

plt.figure()
plt.plot(I/SigI, 100.*(rs_F - F)/F, 'k.', alpha=0.1, label='Acentric')
plt.plot((I/SigI)[Centric], (100.*(rs_F - F)/F)[Centric], 'r.', alpha=0.1, label='Centric')
plt.legend()
plt.xlabel('Iobs/SigIobs')
plt.ylabel('Percent Error (F)')
plt.savefig("Ferr.png")

plt.figure()
plt.plot(I/SigI, 100.*(rs_J - J)/J, 'k.', alpha=0.1, label='Acentric')
plt.plot((I/SigI)[Centric], (100.*(rs_J - J)/J)[Centric], 'r.', alpha=0.1, label='Centric')
plt.legend()
plt.xlabel('Iobs/SigIobs')
plt.ylabel('Percent Error (J)')
plt.savefig("Jerr.png")

plt.figure()
plt.hist((100.*(rs_J - J)/J)[~Centric], 100, color='k', label='Acentric', alpha=0.4)
plt.hist((100.*(rs_J - J)/J)[Centric], 100, color='r', label='Centric', alpha=0.4)
plt.legend()
plt.xlabel('Percent Error (J)')
plt.savefig("J_hist.png")

plt.figure()
plt.hist((100.*(rs_F - F)/F)[~Centric], 100, color='k', label='Acentric', alpha=0.4)
plt.hist((100.*(rs_F - F)/F)[Centric], 100, color='r', label='Centric', alpha=0.4)
plt.legend()
plt.xlabel('Percent Error (F)')
plt.savefig("F_hist.png")

plt.figure()
plt.hist((100.*(rs_SigF - SigF)/SigF)[~Centric], 100, color='k', label='Acentric', alpha=0.4)
plt.hist((100.*(rs_SigF - SigF)/SigF)[Centric], 100, color='r', label='Centric', alpha=0.4)
plt.legend()
plt.xlabel('Percent Error (SigF)')
plt.savefig("SigF_hist.png")

plt.figure()
plt.hist((100.*(rs_SigJ - SigJ)/SigJ)[~Centric], 100, color='k', label='Acentric', alpha=0.4)
plt.hist((100.*(rs_SigJ - SigJ)/SigJ)[Centric], 100, color='r', label='Centric', alpha=0.4)
plt.legend()
plt.xlabel('Percent Error (SigJ)')
plt.savefig("SigJ_hist.png")

plt.figure()
plt.plot((I/SigI)[~Centric], (100.*(rs_SigF - SigF)/SigF)[~Centric], 'k.', label='Acentric', alpha=0.1)
plt.plot((I/SigI)[Centric], (100.*(rs_SigF - SigF)/SigF)[Centric], 'r.', label='Centric', alpha=0.1)
plt.legend()
plt.ylabel('Percent Error (SigF)')
plt.xlabel('I/Sigma')
plt.savefig("SigF_hist.png")

plt.show()
