import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Heatmaps of the exact-MLE couplings J and fields h for one temperature,
# saved as Exact_hJ_<T>K.png next to Exact_h.npy / Exact_J.npy.

parser = argparse.ArgumentParser(description='Plot J and h of the exact Ising model.')
parser.add_argument('--T', type=int, required=True, help='Temperature in K')
parser.add_argument('--res_dir', type=str, default='results/MinimalModelModels', help='Folder containing DHFR_<T>K')
parser.add_argument('--labels', type=str, default='data/MinimalModelData/labels_min.npy', help='Variable names')
args = parser.parse_args()

folder = f"{args.res_dir}/DHFR_{args.T}K"
h = np.load(f"{folder}/Exact_h.npy")
J = np.load(f"{folder}/Exact_J.npy")
labels = [str(l) for l in np.load(args.labels)]
n = len(h)
assert J.shape == (n, n) and len(labels) == n

cmap = plt.get_cmap('RdBu_r').copy()
cmap.set_bad('0.85')  # diagonal of J (not a parameter)

# Layout in inches so the h column has exactly the row height of J.
S, left, bottom, top, gap, right = 6.5, 2.7, 2.7, 1.3, 0.35, 1.6
cw = S / n
W = left + S + gap + cw + right
H = bottom + S + top
fig = plt.figure(figsize=(W, H))

axJ = fig.add_axes([left / W, bottom / H, S / W, S / H])
axh = fig.add_axes([(left + S + gap) / W, bottom / H, cw / W, S / H])
caxJ = fig.add_axes([left / W, (bottom + S + 0.25) / H, S / W, 0.15 / H])
caxh = fig.add_axes([(left + S + gap + cw + 0.2) / W, bottom / H, 0.15 / W, S / H])

vJ = np.abs(J).max()
imJ = axJ.imshow(np.ma.masked_where(np.eye(n, dtype=bool), J), cmap=cmap, vmin=-vJ, vmax=vJ)
axJ.set_xticks(range(n), labels, rotation=90, fontsize=8)
axJ.set_yticks(range(n), labels, fontsize=8)
cb = fig.colorbar(imJ, cax=caxJ, orientation='horizontal')
caxJ.xaxis.set_ticks_position('top')
caxJ.xaxis.set_label_position('top')
cb.set_label('Couplings $J_{ij}$')

vh = np.abs(h).max()
imh = axh.imshow(h[:, None], cmap=cmap, vmin=-vh, vmax=vh, aspect='auto')
axh.set_xticks([0], ['$h_i$'])
axh.set_yticks([])
fig.colorbar(imh, cax=caxh).set_label('Fields $h_i$')

for ax in (axJ, axh):
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

fig.suptitle(f'Exact Ising model, DHFR {args.T}K', y=1 - 0.3 / H)
out = f"{folder}/Exact_hJ_{args.T}K.png"
fig.savefig(out, dpi=200)
print(f"Saved → {out}")
