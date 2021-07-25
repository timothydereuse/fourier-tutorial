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

plt.style.use('dark_background')


def inner_product_figure(func_a, func_b, signal_ylim=None, signal_xlim=None, sr=44100, signal_length=1,
    title_a="Signal A", title_b="Signal B", title_ab='A * B', title="", caption=""):

    x = np.linspace(0, signal_length, num=int(sr * signal_length))
    hline = np.zeros(int(sr * signal_length))

    if not callable(func_a):
        signal_a = func_a 
    else:
        signal_a = func_a(x)

    if not callable(func_b):
        signal_b = func_b 
    else:
        signal_b = func_b(x)

    if signal_ylim == None:
        signal_ylim = [-1.1, 1.1]
    if signal_xlim == None:
        signal_xlim = [0, 0.1]

    ab = signal_a * signal_b

    thresh_mult = np.stack([ab, hline], 1)
    ab_above_zero = np.max(thresh_mult, 1)
    ab_below_zero = np.min(thresh_mult, 1)

    dot_result = np.trapz(ab, x)
    dot_below = np.trapz(ab_below_zero, x)
    dot_above = np.trapz(ab_above_zero, x)

    fig = plt.figure(figsize=[16, 9])
    fig.suptitle(title)

    gs = fig.add_gridspec(3, 2, hspace=0, wspace=0.13, width_ratios=[3, 1])

    axs_left = []
    axs_left.append(fig.add_subplot(gs[0, 0]))
    axs_left.append(fig.add_subplot(gs[1, 0], sharex=axs_left[0]))
    axs_left.append(fig.add_subplot(gs[2, 0], sharex=axs_left[0]))

    axs_right = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1:3, 1])]

    # fig.subplots_adjust(right=0.8)
    # gs = left_fig.add_gridspec(3, hspace=0)
    # axs_left = gs.subplots(sharex=True, sharey=False)

    for ind in range(3):
        # axs_left[ind].axhline(y=0, color='gray', linestyle='--')
        axs_left[ind].grid(axis='y', c='gray', ls='--', lw=0.5)
        axs_left[ind].set_ylim(signal_ylim)
        axs_left[ind].set_facecolor('0.07')
        plt.setp(axs_left[ind].get_yticklabels(), visible=False)
        axs_left[ind].get_yaxis().set_ticklabels([])



    axs_left[0].plot(x, signal_a, color='cyan')
    axs_left[0].set_xlim(signal_xlim)
    axs_left[0].set_ylabel(title_a, rotation=90)
    plt.setp(axs_left[0].get_xticklabels(), visible=False)

    axs_left[1].plot(x, signal_b, color='goldenrod')
    axs_left[1].set_ylabel(title_b, rotation=90)
    plt.setp(axs_left[1].get_xticklabels(), visible=False)

    axs_left[2].plot(x, ab)
    axs_left[2].set_ylabel(title_ab, rotation=90)
    axs_left[2].set_xlabel(r"Time (\textit{s})")
    axs_left[2].set_ylim(signal_ylim)


    axs_left[2].fill_between(x, hline, np.max(thresh_mult, 1), color='darkslateblue')
    axs_left[2].fill_between(x, hline, np.min(thresh_mult, 1), color='firebrick')

    # RIGHT SUBFIGURE

    # gs = right_fig.add_gridspec(2, hspace=0, wspace=0)
    # axs_right = gs.subplots()

    axs_right[0].axis('off')

    dots_normalized = np.array([dot_above, dot_below, dot_result]) / signal_length

    axs_right[1].set_ylim([-0.55, 0.55])
    axs_right[1].set_xticklabels(['Above', 'Below', r'$\displaystyle \langle f, g \rangle $'], rotation = 45)
    axs_right[1].grid(axis='y', c='gray', ls='--')
    axs_right[1].set_facecolor('0.07')
    axs_right[1] = plt.bar(
        [1, 2, 3],
        dots_normalized,
        color=['darkslateblue', 'firebrick', 'white'],
        tick_label=['Above', 'Below', r'$\displaystyle \langle f, g \rangle $'],
    )

    tx = (
        f'{caption}\n'
        r'\begin{tabular}{ l  r }'
        r'Area above $\displaystyle f(t)g(t)  $ & \texttt{ ' f'{dots_normalized[0]:10.4}' r'} \\ '
        r'Area below $\displaystyle f(t)g(t)  $ & \texttt{ ' f'{dots_normalized[1]:10.4}' r'} \\ '
        r'$\displaystyle \langle f, g \rangle $ \ & \texttt{ ' f'{dots_normalized[2]:10.4}' r'} \\ '
        r'\end{tabular}'
    )


    plt.text(0.70, 0.7, tx, fontsize=15, transform=fig.transFigure)

    return fig


img_ind = 0
def save_img(figdef):
    # yeah, yeah, it's a hack to save time
    global img_ind
    fig = inner_product_figure(**figdef)
    fig.savefig(f'./out_imgs/{img_ind}.png', pad_inches=0.1, bbox_inches='tight')
    img_ind += 1

sr = 44100
signal_length = 1

figdef = {
    'sr': sr,
    'signal_length': signal_length,
    'title_a': "$\displaystyle f(t)$",
    'title_b': "$\displaystyle g(t)$",
    'title_ab': "$\displaystyle f(t)g(t)$",
    'caption': '',
    'signal_ylim': [-1.1, 1.1],
    'signal_xlim': [0, 0.3],
    'func_a': lambda z: (np.sin(120 * 2 * np.pi * z)),
    'func_b': lambda z: (np.sin(122 * 2 * np.pi * z)),
}

a = np.zeros(sr * signal_length)
a[9000:20000] = 0.7
# a = np.convolve(a, np.ones(500), 'same') / 500
b = np.zeros(sr * signal_length)
b[15000:27000] = 0.7


figdef['func_a'] = a
figdef['func_b'] = b
figdef['signal_xlim'] = [0, 1]
save_img(figdef)

figdef['func_b'] = -1 * b
save_img(figdef)

b = np.zeros(sr * signal_length)
b[25000:35000] = -0.7
figdef['func_b'] = b
save_img(figdef)

a[5000:36000] = 0.7
figdef['func_a'] = a
b[5000:10000] = 0.7
b[10000:15000] = -0.5
b[15000:20000] = 0.4
b[20000:25000] = -0.8
b[25000:27000] = 0.2
b[27000:32000] = -0.55
b[32000:36000] = 0.87
figdef['func_b'] = b
save_img(figdef)

a[10000:20000] = np.linspace(-1, 1, 10000)
a[15000:30000] = np.linspace(-1, 1, 15000)
figdef['func_a'] = a

b = np.convolve(b, np.ones(1000), 'same') / 1000
b[22222] = -1
b[30123] = 0
b = np.convolve(b, np.ones(1000), 'same') / 1000
b = np.convolve(b, np.ones(1000), 'same') / 1000
b[34000:39000] = np.linspace(1, -1, 5000)
b = np.convolve(b, np.ones(1000), 'same') / 1000
figdef['func_b'] = b
save_img(figdef)

figdef['func_a'] = b
save_img(figdef)

figdef['func_a'] = -b
save_img(figdef)

# SINGLE SINUSOIDS

figdef['signal_xlim'] = [0, 0.025]
figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} $\n' 
                '$\displaystyle g(t): 440 \mathrm{Hz} $\n')
figdef['func_a'] = lambda z: (np.sin(440 * 2 * np.pi * z))
figdef['func_b'] = lambda z: (np.sin(440 * 2 * np.pi * z))
save_img(figdef)

figdef['signal_xlim'] = [0, 0.25]
save_img(figdef)

figdef['signal_xlim'] = [0, 0.025]
figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} $\n' 
                '$\displaystyle g(t): 220 \mathrm{Hz} $\n')
figdef['func_a'] = lambda z: (np.sin(440 * 2 * np.pi * z))
figdef['func_b'] = lambda z: (np.sin(220 * 2 * np.pi * z))
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} $\n' 
                '$\displaystyle g(t): 110 \mathrm{Hz} $\n')
figdef['func_a'] = lambda z: (np.sin(440 * 2 * np.pi * z))
figdef['func_b'] = lambda z: (np.sin(110 * 2 * np.pi * z))
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} $\n' 
                '$\displaystyle g(t): 445 \mathrm{Hz} $\n')
figdef['func_a'] = lambda z: (np.sin(440 * 2 * np.pi * z))
figdef['func_b'] = lambda z: (np.sin(445 * 2 * np.pi * z))
save_img(figdef)

figdef['signal_xlim'] = [0, 0.1]
save_img(figdef)

figdef['signal_xlim'] = [0, 0.25]
save_img(figdef)

figdef['signal_xlim'] = [0, 0.025]
figdef['caption'] = ('$\displaystyle f(t): 123 \mathrm{Hz} $\n' 
                '$\displaystyle g(t): 456 \mathrm{Hz} $\n')
figdef['func_a'] = lambda z: (np.sin(123 * 2 * np.pi * z))
figdef['func_b'] = lambda z: (np.sin(456 * 2 * np.pi * z))
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): 190.538 \mathrm{Hz} $\n' 
                '$\displaystyle g(t): 426.943 \mathrm{Hz} $\n')
figdef['func_a'] = lambda z: (np.sin(190.538 * 2 * np.pi * z))
figdef['func_b'] = lambda z: (np.sin(426.943 * 2 * np.pi * z))
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): 303.707 \mathrm{Hz} $\n' 
                '$\displaystyle g(t): 808.909 \mathrm{Hz} $\n')
figdef['func_a'] = lambda z: (np.sin(303.707 * 2 * np.pi * z))
figdef['func_b'] = lambda z: (np.sin(808.909 * 2 * np.pi * z))
save_img(figdef)

# SUMS OF SINUSOIDS
figdef['signal_xlim'] = [0, 0.025]
figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} + 220 \mathrm{Hz}$\n' 
                '$\displaystyle g(t): 440 \mathrm{Hz} $\n')
figdef['func_a'] = lambda z: (np.sin(440 * 2 * np.pi * z) + np.sin(220 * 2 * np.pi * z)) / 2
figdef['func_b'] = lambda z: (np.sin(440 * 2 * np.pi * z))
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} + 220 \mathrm{Hz} + 179 \mathrm{Hz}$\n' 
                '$\displaystyle g(t): 440 \mathrm{Hz} $\n')
figdef['func_a'] = lambda z: (np.sin(440 * 2 * np.pi * z) + np.sin(220 * 2 * np.pi * z)) / 2
figdef['func_b'] = lambda z: (np.sin(440 * 2 * np.pi * z))
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} + 220 \mathrm{Hz} + 179 \mathrm{Hz}$\n' 
                '$\displaystyle g(t): 110 \mathrm{Hz} $\n')
figdef['func_a'] = lambda z: (np.sin(440 * 2 * np.pi * z) + np.sin(220 * 2 * np.pi * z) + np.sin(179 * 2 * np.pi * z)) / 3
figdef['func_b'] = lambda z: (np.sin(110 * 2 * np.pi * z))
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} + 220 \mathrm{Hz} + 179 \mathrm{Hz}$\n' 
                '$\displaystyle g(t): 110 \mathrm{Hz} $\n')
figdef['func_b'] = lambda z: (np.sin(179 * 2 * np.pi * z))
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} + \mathrm{Noise}$\n' 
                '$\displaystyle g(t): 440 \mathrm{Hz} $\n')
x = np.linspace(0, signal_length, num=sr * signal_length)
a = np.sin(440 * 2 * np.pi * x) * 0.5
num_noise_pts = 8000
noise_pts = np.random.uniform(-1, 1, num_noise_pts)
noise = np.interp(x, np.linspace(0, signal_length, num=num_noise_pts), fp=noise_pts)
a = a + (noise / 2)

wavfile.write('slighty_noisy_sine.wav', sr, a)
figdef['func_a'] = a
figdef['func_b'] = lambda z: (np.sin(440 * 2 * np.pi * z))
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): \mathrm{Noise}$\n' 
                '$\displaystyle g(t): 440 \mathrm{Hz} $\n')
x = np.linspace(0, signal_length, num=sr * signal_length)
a = np.sin(440 * 2 * np.pi * x) * 0.2
num_noise_pts = 11000
noise_pts = np.random.uniform(-1, 1, num_noise_pts)
noise = np.interp(x, np.linspace(0, signal_length, num=num_noise_pts), fp=noise_pts)

wavfile.write('very_noisy_sine.wav', sr, a)
figdef['func_a'] = noise
figdef['func_b'] = lambda z: (np.sin(440 * 2 * np.pi * z))
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} + \mathrm{Noise}$\n' 
                '$\displaystyle g(t): 440 \mathrm{Hz} $\n')
a = np.sin(440 * 2 * np.pi * x)
a = a * 0.2 + (noise * 0.8)

figdef['func_a'] = a
save_img(figdef)

figdef['caption'] = ('$\displaystyle f(t): 440 \mathrm{Hz} + \mathrm{Noise}$\n' 
                '$\displaystyle g(t): 445 \mathrm{Hz} $\n')
figdef['func_b'] = lambda z: (np.sin(445 * 2 * np.pi * z))
save_img(figdef)

 
guitar_sr, aguitar_note = wavfile.read('aguitar_110hz.wav')
assert guitar_sr == sr
aguitar_note = aguitar_note[:, 0] / np.max(np.abs(aguitar_note))

figdef['signal_length'] = 0.7
aguitar_note_crop = aguitar_note[5000:5000 + int(sr * 0.7)]

figdef['signal_xlim'] = [0, 0.1]
figdef['caption'] = ('$\displaystyle f(t):$ Guitar playing A2 (110 Hz)\n' 
                '$\displaystyle g(t): 110 \mathrm{Hz} $\n')
figdef['func_a'] = aguitar_note_crop
figdef['func_b'] = lambda z: (np.sin(110 * 2 * np.pi * z))

save_img(figdef)

# I HAVE LIED TO YOU: PHASE PROBLEMS

figdef['signal_length'] = 1
figdef['signal_xlim'] = [0, 0.025]
figdef['caption'] = ('$\displaystyle g(t): 110 \mathrm{Hz}, \phi = 0.33 (2 \pi) $\n' 
                '$\displaystyle g(t): 110 \mathrm{Hz}, \phi = 0 $\n')
figdef['func_a'] = lambda z: (np.sin(2 * np.pi * (110 * z + 0.33)))
figdef['func_b'] = lambda z: (np.sin(2 * np.pi * (110 * z + 0)))
save_img(figdef)

figdef['caption'] = ('$\displaystyle g(t): 110 \mathrm{Hz}, \phi = 0.5 (2 \pi) $\n' 
                '$\displaystyle g(t): 110 \mathrm{Hz}, \phi = 0 $\n')
figdef['func_a'] = lambda z: (np.sin(2 * np.pi * (110 * z + 0.5)))
figdef['func_b'] = lambda z: (np.sin(2 * np.pi * (110 * z + 0)))
save_img(figdef)

figdef['caption'] = ('$\displaystyle g(t): 110 \mathrm{Hz}, \phi = 0.25 (2 \pi) $\n' 
                '$\displaystyle g(t): 110 \mathrm{Hz}, \phi = 0 $\n')
figdef['func_a'] = lambda z: (np.sin(2 * np.pi * (110 * z + 0.25)))
figdef['func_b'] = lambda z: (np.sin(2 * np.pi * (110 * z + 0)))
save_img(figdef)

