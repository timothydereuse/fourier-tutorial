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

plt.style.use('dark_background')


def inner_product_figure(func_a, func_b, func_c, signal_ylim=None, signal_xlim=None, sr=44100, signal_length=1,
    title_a="Input Signal", title_b="Test Signal 1", title_c="Test Signal 2", title_ab='A * B', title_ac='A * C', title="", caption=""):

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

    if not callable(func_c):
        signal_c = func_c
    else:
        signal_c = func_c(x)

    if signal_ylim == None:
        signal_ylim = [-1.1, 1.1]
    if signal_xlim == None:
        signal_xlim = [0, 0.1]

    ac = signal_a * signal_c 
    ab = signal_a * signal_b

    ac_thresh_mult = np.stack([ac, hline], 1)
    ac_above_zero = np.max(ac_thresh_mult, 1)
    ac_below_zero = np.min(ac_thresh_mult, 1)

    ab_thresh_mult = np.stack([ab, hline], 1)
    ab_above_zero = np.max(ab_thresh_mult, 1)
    ab_below_zero = np.min(ab_thresh_mult, 1)

    ac_dot_result = np.trapz(ac, x)
    ac_dot_below = np.trapz(ac_below_zero, x)
    ac_dot_above = np.trapz(ac_above_zero, x)

    ab_dot_result = np.trapz(ab, x)
    bc_dot_below = np.trapz(ab_below_zero, x)
    bc_dot_above = np.trapz(ab_above_zero, x)

    fig = plt.figure(figsize=[16, 9])
    fig.suptitle(title)

    gs = fig.add_gridspec(5, 2, hspace=0, wspace=0.13, width_ratios=[3, 1])

    axs_left = []
    axs_left.append(fig.add_subplot(gs[0, 0]))
    axs_left.append(fig.add_subplot(gs[1, 0], sharex=axs_left[0]))
    axs_left.append(fig.add_subplot(gs[2, 0], sharex=axs_left[0]))
    axs_left.append(fig.add_subplot(gs[3, 0], sharex=axs_left[0]))
    axs_left.append(fig.add_subplot(gs[4, 0], sharex=axs_left[0]))

    axs_right = [fig.add_subplot(gs[0:2, 1]), fig.add_subplot(gs[2:5, 1])]

    # fig.subplots_adjust(right=0.8)
    # gs = left_fig.add_gridspec(3, hspace=0)
    # axs_left = gs.subplots(sharex=True, sharey=False)

    for ind in range(5):
        # axs_left[ind].axhline(y=0, color='gray', linestyle='--')
        axs_left[ind].grid(axis='y', c='gray', ls='--', lw=0.5)
        axs_left[ind].set_ylim(signal_ylim)
        axs_left[ind].set_facecolor('0.07')


    axs_left[0].plot(x, signal_a)
    axs_left[0].set_xlim(signal_xlim)
    axs_left[0].set_ylabel(title_a)
    plt.setp(axs_left[0].get_xticklabels(), visible=False)

    axs_left[1].plot(x, signal_b)
    axs_left[1].set_ylabel(title_b)
    plt.setp(axs_left[1].get_xticklabels(), visible=False)

    axs_left[3].plot(x, signal_c)
    axs_left[3].set_ylabel(title_c)
    plt.setp(axs_left[3].get_xticklabels(), visible=False)

    axs_left[2].plot(x, ab)
    axs_left[2].set_ylabel(title_ab)

    axs_left[2].fill_between(x, hline, np.max(ab_thresh_mult, 1), color='cornflowerblue')
    axs_left[2].fill_between(x, hline, np.min(ab_thresh_mult, 1), color='goldenrod')
    plt.setp(axs_left[2].get_xticklabels(), visible=False)

    axs_left[4].plot(x, ac)
    axs_left[4].set_ylabel(title_ac)
    axs_left[4].set_xlabel(r"Time (\textit{s})")

    axs_left[4].fill_between(x, hline, np.max(ac_thresh_mult, 1), color='cornflowerblue')
    axs_left[4].fill_between(x, hline, np.min(ac_thresh_mult, 1), color='goldenrod')


    # RIGHT SUBFIGURE

    # gs = right_fig.add_gridspec(2, hspace=0, wspace=0)
    # axs_right = gs.subplots()

    axs_right[0].axis('off')
    # axs_right[1].axis('off')


    phase = np.arctan2(ab_dot_result, ac_dot_result) / (2 * np.pi)
    magnitude = np.sqrt(ab_dot_result ** 2 + ac_dot_result ** 2)

    dots_normalized = np.array([ab_dot_result, ac_dot_result, magnitude, phase]) / signal_length

    axs_right[1].set_ylim([-0.5, 0.55])
    axs_right[1].set_xticklabels(['1', '2'], rotation = 45)
    axs_right[1].grid(axis='y', c='gray', ls='--')
    axs_right[1].set_facecolor('0.07')
    axs_right[1] = plt.bar(
        [1, 2, 3, 4],
        dots_normalized,
        color=['teal', 'red','white', 'gray' ],
        tick_label=[
            '$\displaystyle \\langle f,\\ g_1 \\rangle $',
            '$\displaystyle \\langle f,\\ g_2 \\rangle $',
            'Amplitude',
            'Phase'],
    )

    tx = (
        f'{caption}\n'
        r'\begin{tabular}{ l  r }'
        r'$\displaystyle \langle f, g_1 \rangle $ \ & \texttt{ ' f'{dots_normalized[0]:10.4}' r'} \\ '
        r'$\displaystyle \langle f, g_2 \rangle $ \ & \texttt{ ' f'{dots_normalized[1]:10.4}' r'} \\ '
        r'Magnitude \ & \texttt{ '                               f'{dots_normalized[2]:10.4}' r'} \\ '
        r'Phase & \texttt{ '                                     f'{dots_normalized[3]:10.4}' r'} * $\displaystyle 2\pi $ \\ '
        r'\end{tabular}'
    )

    plt.text(0.70, 0.68, tx, fontsize=15, transform=fig.transFigure)

    return fig


img_ind = 0
def save_img(figdef):
    # yeah, yeah, it's a hack to save time
    global img_ind
    fig = inner_product_figure(**figdef)
    fig.savefig(f'./out_imgs/double_{img_ind}.png', pad_inches=0, bbox_inches='tight')
    img_ind += 1

sr = 44100
signal_length = 1

figdef = {
    'sr': sr,
    'signal_length': signal_length,
    'title_a': "$\displaystyle f(t)$",
    'title_b': "$\displaystyle g_1(t)$",
    'title_c': "$\displaystyle g_2(t)$",
    'title_ab': "$\displaystyle f(t)g_1(t)$",
    'title_ac': "$\displaystyle f(t)g_2(t)$",

    'caption': '',
    'signal_ylim': [-1.1, 1.1],
    'signal_xlim': [0, 0.0],
    'func_a': lambda z: (np.sin(2 * np.pi * (440 * z + 0.25))),
    'func_b': lambda z: (np.cos(440 * 2 * np.pi * z)),
    'func_c': lambda z: (np.sin(440 * 2 * np.pi * z)),
}

figdef['caption'] = (
    '$\displaystyle f(t): 440 \mathrm{Hz} $\n' 
    '$\displaystyle g_1(t): 440 \mathrm{Hz},\ \phi = 0 $\n'
    '$\displaystyle g_2(t): 440 \mathrm{Hz},\ \phi = 0.25 (2 \pi) $\n'
    )
figdef['signal_xlim'] = [0, 0.05]
save_img(figdef)
