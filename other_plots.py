import numpy as np
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt
from matplotlib import rc

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 16
})

sr = 44100
signal_length = 1

plt.style.use('dark_background')

fig = plt.figure(figsize=[16, 9])

axs = fig.subplots(3, 1, sharex=True)


for ind in range(3):
    axs[ind].grid(axis='y', c='gray', ls='--', lw=0.5)
    axs[ind].set_ylim([-1.1, 1.1])
    axs[ind].set_xlim([0, 0.02])
    axs[ind].set_facecolor('0.07')

x = np.linspace(0, 0.2, num=int(sr * signal_length))
a = (np.sin(2 * np.pi * (110 * x + 0)))
b = (np.sin(2 * np.pi * (110 * x + 0.25)))
c = (np.sin(2 * np.pi * (110 * x + 0.66)))

axs[0].plot(x, a, c='cyan')
axs[0].set_title('$\displaystyle \sin(440 * 2 \pi t) $')
axs[1].plot(x, b, c='goldenrod')
axs[1].set_title('$\displaystyle \sin(440 * 2 \pi t + \pi / 2 ) = \cos(440 * 2 \pi t) $')
axs[2].set_ylim([-1.5, 1.5])
axs[2].plot(x, c, c='yellow')
axs[2].set_title('$\displaystyle \sin(440 * 2 \pi t + 4 \pi / 3)$')
axs[2].set_xlabel(r'Time (\textit{s})')

# fig.show()
fig.savefig(f'./out_imgs/other_1.png', pad_inches=0.1, bbox_inches='tight')

fig = plt.figure(figsize=[16, 9])
axs = fig.subplots(3, 1, sharex=True)

for ind in range(3):
    axs[ind].grid(axis='y', c='gray', ls='--', lw=0.5)
    axs[ind].set_ylim([-1.1, 1.1])
    axs[ind].set_xlim([0, 0.02])
    axs[ind].set_facecolor('0.07')


x = np.linspace(0, 0.2, num=int(sr * signal_length))
a = np.sin(x * 440 * 2 * np.pi)
b = np.cos(x * 440 * 2 * np.pi)

axs[0].plot(x, a, c='cyan')
axs[0].set_title('$\displaystyle f(t) = \sin(440 * 2 \pi t) $')
axs[1].plot(x, b, c='goldenrod')
axs[1].set_title('$\displaystyle g(t) = \cos(440 * 2 \pi t) $')
axs[2].set_ylim([-1.5, 1.5])
axs[2].plot(x, (a + b), c='yellow')
axs[2].set_title('$\displaystyle f(t) + g(t) = \sin(440 * 2 \pi t) + \cos(440 * 2 \pi t) $')
axs[2].set_xlabel(r'Time (\textit{s})')


# fig.show()
fig.savefig(f'./out_imgs/other_2.png', pad_inches=0.1, bbox_inches='tight')

plt.clf()


fig = plt.figure(figsize=[8, 3])
axs = fig.subplots(1, 1)
axs.grid(axis='y', c='gray', ls='--', lw=0.5)
axs.set_ylim([-1.1, 1.1])
axs.set_xlim([0, 0.01])
axs.set_facecolor('0.07')
x = np.linspace(0, 0.1, num=int(sr * signal_length))
a = np.sin(x * 440 * 2 * np.pi) ** 2
hline = np.zeros(a.shape)
axs.fill_between(x, hline, a, color='darkslateblue')
axs.plot(x, a, c='white')
axs.set_title('$\displaystyle \sin^2(440 \cdot 2 \pi t) $')
axs.set_xlabel(r'Time (\textit{s})')
# fig.show()

# fig.show()
fig.savefig(f'./out_imgs/other_3.png', pad_inches=0.1, bbox_inches='tight')

plt.clf()

x = np.linspace(0, 1, num=int(sr * signal_length))
theta = np.linspace(50, 1000, num=500)
guitar_sr, aguitar_note = wavfile.read('aguitar_110hz.wav')
assert guitar_sr == sr
aguitar_note = aguitar_note[:, 0] / np.max(np.abs(aguitar_note))
aguitar_note_crop = aguitar_note[5000:5000 + int(sr * signal_length)]
fig = plt.figure(figsize=[16, 9])
axs = fig.subplots(2, 1)
axs[0].plot(x, aguitar_note_crop, c='cyan')
axs[0].set_xlim([0, signal_length])
axs[0].set_ylim([-1, 1])
axs[0].set_xlabel(r'Time (\textit{s})')
axs[0].set_ylabel('Amplitude')
axs[0].grid(axis='y', c='gray', ls='--')

def ip(inp, f, p):
    return np.sum(np.sin(2 * np.pi * (f * x + p)) * inp) / (len(inp) * 0.5)

real = [ip(aguitar_note_crop, f, 0.25) for f in theta]
imag = [ip(aguitar_note_crop, f, 0) for f in theta]

axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Intensity')
axs[1].grid(axis='y', c='gray', ls='--')
axs[1].set_xlim([theta[0], theta[-1]])
axs[1].set_ylim([-0.125, 0.125])
axs[1].set_xticks([100, 200, 300, 400, 500, 600, 700, 800, 900])
fig.savefig(f'./out_imgs/other_4.png', pad_inches=0.1, bbox_inches='tight')
axs[1].plot(theta, imag, c='goldenrod', label=r'$\displaystyle \hat{f_1}(\theta) = \langle f, \sin(2\pi\theta) \rangle$')
axs[1].legend()
fig.savefig(f'./out_imgs/other_5.png', pad_inches=0.1, bbox_inches='tight')
axs[1].plot(theta, real, c='cornflowerblue', label=r'$\displaystyle \hat{f_2}(\theta) = \langle f, \cos(2\pi\theta) \rangle$')
axs[1].legend()
fig.savefig(f'./out_imgs/other_6.png', pad_inches=0.1, bbox_inches='tight')


