import tike.ptycho
import numpy as np
import matplotlib.pyplot as plt
import importlib

for module in [tike, np]:
    print("{} is version {}".format(module.__name__, module.__version__))

amplitude = plt.imread("../tests/data/Cryptomeria_japonica-0128.tif") / 255
phase = plt.imread("../tests/data/Bombus_terrestris-0128.tif") / 255 * np.pi
#np.min(phase), np.max(phase)

original = amplitude * np.exp(1j * phase)
#tike.plot_phase(original)
print(original.shape)


pw = 15 # probe width
weights = tike.ptycho.gaussian(pw, rin=0.8, rout=1.0)
probe = weights * np.exp(1j * weights * 0.2)
#tike.plot_complex(probe)



v, h = np.meshgrid(
    np.linspace(0, amplitude.shape[0]-pw, 24, endpoint=False),
    np.linspace(0, amplitude.shape[0]-pw, 24, endpoint=False),
    indexing='ij'
    )


# Then what we see at the detector is the wave propagation
# of the near field wavefront
data = tike.ptycho.simulate(data_shape=np.ones(2, dtype=int)*pw*3,
                            probe=probe, v=v, h=h,
                            psi=original, pad=False)


# Start with a guess of all zeros for psi
new_psi = np.ones(original.shape, dtype=complex)

iterations =  40

for i in range(iterations):
    new_psi = tike.ptycho.reconstruct(data=data,
                                      probe=probe, v=v, h=h,
                                      psi=new_psi,
                                      algorithm='sharp', #algorithm='grad',
                                      niter=1, rho=0.5, gamma=0.25)

tike.plot_phase(new_psi)




