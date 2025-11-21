# %% Load
import pypulseq as pp

seq1, seq2, seq3 = pp.Sequence(), pp.Sequence(), pp.Sequence()
seq1.read('gre_fa10.seq')
seq2.read('gre_fa20.seq')
seq3.read('gre_label_softdelay.seq')

# %% Standard plot
seq1.plot(time_range=[0.0, 0.012])

# %% Standard plot + hairline
seq1.plot(time_range=[0.0, 0.012], show_guides=True)

# %% Stacked plot
seq1.plot(time_range=[0.0, 0.012], stacked=True)

# %% Stacked plot + hairline
seq1.plot(time_range=[0.0, 0.012], stacked=True, show_guides=True)

# %% Standard overlaid plot
fig = seq1.plot(time_range=[0.0, 0.012], plot_now=False)
seq2.plot(time_range=[0.0, 0.012], overlay=fig)

# %% Standard overlaid plot + hairline
fig = seq1.plot(time_range=[0.0, 0.012], plot_now=False)
seq2.plot(time_range=[0.0, 0.012], overlay=fig, show_guides=True)

# %% Stacked overlaid plot
fig = seq1.plot(time_range=[0.0, 0.012], stacked=True, plot_now=False)
seq2.plot(time_range=[0.0, 0.012], overlay=fig)

# %% Stacked overlaid plot + hairline
fig = seq1.plot(time_range=[0.0, 0.012], stacked=True, plot_now=False)
seq2.plot(time_range=[0.0, 0.012], overlay=fig, show_guides=True)

# %% Soft delay extensions
import numpy as np

seq3.plot(label='lin', time_range=np.array([0, 1]) * 20e-3, time_disp='ms')
