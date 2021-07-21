import numpy as np
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt
from matplotlib import rc

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 12
})

sr = 44100
signal_length = 1

plt.style.use('dark_background')

fig = plt.figure(figsize=[16, 9])

axs = fig.subplots(3, 1, sharex=True)
x = np.linspace(0, 0.2, num=int(sr * signal_length))
a = np.sin(x * 440 * 2 * np.pi)
b = np.cos(x * 440 * 2 * np.pi)

for ind in range(3):
    axs[ind].grid(axis='y', c='gray', ls='--', lw=0.5)
    axs[ind].set_ylim([-1.1, 1.1])
    axs[ind].set_xlim([0, 0.02])
    axs[ind].set_facecolor('0.07')
    axs

axs[0].plot(x, a)
axs[0].set_title('$\displaystyle f(t) = \sin(440 * 2 \pi t) $')
axs[1].plot(x, b)
axs[1].set_title('$\displaystyle g(t) = \cos(440 * 2 \pi t) $')
axs[2].set_ylim([-1.5, 1.5])
axs[2].plot(x, (a + b))
axs[2].set_title('$\displaystyle f(t) + g(t) = \sin(440 * 2 \pi t) + \cos(440 * 2 \pi t) $')
axs[2].set_xlabel(r'Time (\textit{s})')

fig.show()