"""Figure style for IEEE Transactions (TCOM / TWC) - used by every figure script of this project.

Conventions followed (IEEE author guidelines + what these two journals print):
  size      single column 3.5 in, double column 7.16 in; the PDF canvas has exactly that width (no tight cropping), so included at
            width=3.5in nothing is rescaled and 8 pt in the script is 8 pt on the page
  type      Times New Roman, mathematics in STIX (Times-compatible); 8 pt labels and ticks, 7.5 pt legend; no figure titles
            (the caption is typeset by LaTeX)
  frame     full box, ticks pointing inwards on all four sides, thin dotted grid behind the data
  series    identified by colour AND marker AND line style, so the figure survives greyscale printing and colour-vision
            deficiency; bars additionally carry hatching. Hollow markers; the proposed scheme is the only filled one.
  colour    one fixed colour per method in every figure. The order below passed the palette validator of the dataviz skill on a
            white surface (lightness band, chroma, colour-vision separation of neighbours, normal-vision floor); the one
            contrast warning (aqua, 2.8:1) is covered by the marker / line-style encoding.
  legend    always present for two or more series, boxed, inside the axes
  output    vector PDF with embedded TrueType fonts (pdf.fonttype 42) + a 600 dpi PNG for a quick look
One y-axis per plot: two quantities of different scale go to two panels, never to a twin axis."""
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

COL1, COL2 = 3.5, 7.16
INK, GRID = '#000000', '#b4b4b4'
PAL = ['#e34948', '#2a78d6', '#008300', '#4a3aa7', '#eb6834', '#1baf7a']       # validated order: red, blue, green, violet, orange, aqua
RAMP = ['#86b6ef', '#5598e7', '#256abf', '#104281']                            # one hue, light -> dark: ordered categories (validated --ordinal)
LONG, DDOT = (0, (6, 1.6)), (0, (4, 1.2, 1, 1.2, 1, 1.2))
# ONE comparison set for every figure (user, 2026-10-07): the same schemes in the learning curves and in the system-level plots.
# key: (legend label, colour, line style, marker, filled marker, hatch). Learners and rules own a colour; the two references that are
# not competitors of the scheduler (grid search over the structure parameters of our own policy family = offline-tuned reference, NOT a global optimum, HEVC = conventional codec) are in ink.
STYLE = {
    'prop': ('Proposed', PAL[0], '-', 'o', True, ''),
    'ppo': ('H-PPO', PAL[1], '--', 's', False, '////'),
    'lyap': ('Drift-plus-penalty', PAL[2], '-.', '^', False, '\\\\\\\\'),
    'fixm': ('Fixed level', PAL[3], ':', 'D', False, 'xxxx'),
    'd3qn': ('D3QN', PAL[4], LONG, 'v', False, '....'),
    'td3': ('TD3', PAL[5], DDOT, 'X', False, '++'),
    'exh': ('Grid search', INK, (0, (4, 2)), '*', False, ''),
    'hevc': ('HEVC inter', INK, (0, (1, 1.5)), 'x', False, ''),
}


def use():
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'], 'mathtext.fontset': 'stix',
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
        'axes.linewidth': 0.6, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
        'axes.grid': True, 'grid.color': GRID, 'grid.linestyle': ':', 'grid.linewidth': 0.45, 'axes.axisbelow': True,
        'lines.linewidth': 1.2, 'lines.markersize': 4.6, 'lines.markeredgewidth': 0.9,
        'legend.frameon': True, 'legend.fancybox': False, 'legend.edgecolor': INK, 'legend.framealpha': 1.0, 'legend.borderpad': 0.35,
        'legend.handlelength': 2.6, 'legend.labelspacing': 0.28, 'legend.handletextpad': 0.5, 'legend.columnspacing': 1.0,
        'hatch.linewidth': 0.5, 'patch.linewidth': 0.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600, 'savefig.bbox': 'standard',
        'figure.dpi': 150, 'axes.unicode_minus': False,
    })


def fig(ncols=1, width=COL1, height=2.3, **kw):
    """the canvas is exactly `width` wide (no tight cropping), so \\includegraphics[width=3.5in] prints the fonts at their nominal size"""
    f, ax = plt.subplots(1, ncols, figsize=(width, height), layout='constrained', **kw)
    f.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    return f, ax


def kw(key, label=True):
    """plot keyword arguments of a method"""
    lab, c, ls, m, filled, _ = STYLE[key]
    return dict(color=c, linestyle=ls, marker=m, markerfacecolor=c if filled else 'white', markeredgecolor=c, label=lab if label else None)


def line(ax, x, y, key, label=True, **over):
    k = kw(key, label); k.update(over)
    return ax.plot(x, y, **k)[0]


def band(ax, x, lo, hi, key):
    ax.fill_between(x, lo, hi, color=STYLE[key][1], alpha=0.16, linewidth=0)


def ref(ax, y, text, x=0.985, va='bottom', ha='right', ls=(0, (4, 2))):
    """horizontal reference in ink (a requirement or a bound, not a series): line + label on the line"""
    ax.axhline(y, color=INK, linestyle=ls, linewidth=0.7, zorder=1.5)
    ax.annotate(text, xy=(x, y), xycoords=('axes fraction', 'data'), xytext=(0, 2 if va == 'bottom' else -2.5), textcoords='offset points',
                ha=ha, va=va, fontsize=7.5)


def hline(ax, y, label, ls=(0, (4, 2))):
    """horizontal benchmark in ink that goes into the legend (a scheme without a learning curve)"""
    return ax.axhline(y, color=INK, linestyle=ls, linewidth=0.9, zorder=1.5, label=label)


def level(ax, y, key):
    """a scheme without a learning curve as a horizontal line in its own colour and line style (legend entry as in the other figures)"""
    lab, c, ls, m, filled, _ = STYLE[key]
    return ax.plot([0, 1], [y, y], transform=ax.get_yaxis_transform(), color=c, linestyle=ls, linewidth=1.0, marker=m, markevery=[1],
                   markerfacecolor=c if filled else 'white', markeredgecolor=c, clip_on=False, zorder=2.5, label=lab)[0]


def legend_fig(f, ax, order, ncol=4):
    """one boxed legend above all panels of a multi-panel figure"""
    h, l = ax.get_legend_handles_labels(); want = [STYLE[k][0] if k in STYLE else k for k in order]; idx = [l.index(w) for w in want if w in l]
    lg = f.legend([h[i] for i in idx], [l[i] for i in idx], loc='outside upper center', ncol=ncol, handlelength=2.3)
    lg.get_frame().set_linewidth(0.5)
    return lg


def panels(axs, labels=None):
    """sub-figure labels (a), (b), ... centred under each panel, as IEEE prints them"""
    for i, ax in enumerate(axs):
        ax.annotate(labels[i] if labels else f'({chr(97 + i)})', xy=(0.5, 0), xycoords='axes fraction', xytext=(0, -27), textcoords='offset points',
                    ha='center', va='top', fontsize=8)


def legend(ax, **over):
    k = dict(loc='best'); k.update(over)
    lg = ax.legend(**k); lg.get_frame().set_linewidth(0.5)
    return lg


def legend_top(ax, ncol=2, order=None):
    """boxed legend above the axes, exactly as wide as the axes: for plots whose curves leave no free corner"""
    h, l = ax.get_legend_handles_labels()
    if order:
        want = [STYLE[k][0] if k in STYLE else k for k in order]; idx = [l.index(w) for w in want if w in l]
        h, l = [h[i] for i in idx], [l[i] for i in idx]
    lg = ax.legend(h, l, loc='lower left', bbox_to_anchor=(0.0, 1.025, 1.0, 0.1), mode='expand', ncol=ncol, borderaxespad=0.0, handlelength=2.3)
    f = ax.figure; w, ht = f.get_size_inches(); f.set_size_inches(w, ht + 0.145 * -(-len(l) // ncol) + 0.11)    # the plot area keeps its height
    lg.get_frame().set_linewidth(0.5)
    return lg


def save(f, name, out=None):
    out = out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    os.makedirs(out, exist_ok=True)
    f.savefig(os.path.join(out, name + '.pdf')); f.savefig(os.path.join(out, name + '.png'))
    plt.close(f)
    return os.path.join(out, name + '.pdf')
