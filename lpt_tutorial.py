#@me SETTINGS =========================
LSD = 3035

# On fir the files are under /scratch/ssiegel/chime/lpt/lsq_mfilt.  On a laptop,
# point this at the folder you downloaded them into (see "Where the files are").
DATA_ROOT = "/mnt/scratch/mseth/chime_data/n2_j1634/"
# Runs from before the noise option have no suffix.  Newer runs end in "_thermal"
# (thermal noise only) or "_sky" (with the noise that scales with the sky).
RUN_SUBDIR = "band_full/mask_{mask}/fir513_ew22_native"
MASKS = ["none", "full"]

# Time of the J1634+44 pulse on each day (UTC), from the CHIME/Pulsar detections.
# On a day that is not listed here, the J1634+44 section is skipped.
J1634_UTC = {
    2389: "2020-05-25 08:23:46",
    2418: "2020-06-23 06:26:15",
    2672: "2021-03-03 13:52:08",
    3035: "2022-02-28 14:04:55",
    3915: "2024-07-25 04:18:38",
}
# Approximate position of J1634+44 (degrees).  Only used to centre the search.
J1634_RA_DEC = (248.52, 44.8)

import pdb
import os
from datetime import datetime, timezone

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from caput.astro.skyfield import skyfield_wrapper
from ch_ephem import sources
from ch_ephem.observers import chime
from IPython.display import display
from matplotlib.colors import LogNorm

from ch_pipeline.core import containers

LAT = np.radians(chime.latitude)
plt.rcParams["figure.dpi"] = 90
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

#@me WHERE THE FILES ARE =========================

run_dirs = {m: os.path.join(DATA_ROOT, str(LSD), RUN_SUBDIR.format(mask=m)) for m in MASKS}


def run_file(mask, prefix):
    return os.path.join(run_dirs[mask], f"{prefix}_lsd_{LSD}.h5")


for m in MASKS:
    print(f"mask = {m}:  {run_dirs[m]}")
    for fn in sorted(os.listdir(run_dirs[m])):
        if fn.endswith(".h5"):
            print(f"    {fn:45s} {os.path.getsize(os.path.join(run_dirs[m], fn)) / 1e9:8.3f} GB")

# The optional large files: used only if they are there
HAVE_EACH_BAND = all(os.path.exists(run_file(m, "transient_each_band_maxx")) for m in MASKS)
HAVE_MFILT_BANDS = os.path.exists(run_file("full", "transient_mfilt_bands"))
print("6 MHz band maps:", "yes" if HAVE_EACH_BAND else "no (spectra by sub-band only)")
print("per-band matched filter file:", "yes" if HAVE_MFILT_BANDS else "no")

#@me CONTAINERS & MPI =========================
cand = {
    m: containers.TransientCandidateMap.from_file(run_file(m, "transient_candidates"), distributed=True)
    for m in MASKS
}

c = cand["full"]
snr_mpi = c.snr[:]
print(type(snr_mpi).__name__, "with", c.comm.size, "process")
print("global shape:   ", snr_mpi.global_shape)
print("split along axis", snr_mpi.axis, f"({c.snr.attrs['axis'][snr_mpi.axis]})")
print("local shape:    ", snr_mpi.local_shape, " local offset:", snr_mpi.local_offset)
print("local_array is a", type(snr_mpi.local_array).__name__)

#@me WHAT IS IN THE CANDIDATE MAP =========================

print("axes (index_map):")
for name, axis in c.index_map.items():
    print(f"    {name:6s} {axis.shape}  {axis.dtype}  first values: {axis[:3]}")
print("\ndatasets:")
for name, dset in c.datasets.items():
    print(f"    {name:16s} {dset.attrs['axis']}  {dset.shape}  {dset.dtype}")
print("\nattributes:")
for key in ("lsd", "tag", "ew_used", "pol_used", "duration_unit_seconds", "time_axis", "reference_freq"):
    print(f"    {key:22s} {c.attrs.get(key)}")

el = c.index_map["el"][:]
ra = c.ra[:]
x = c.index_map["x"][:]
dra = np.median(np.diff(ra))
dt = c.attrs.get("duration_unit_seconds", 86164.0905 / 360 * dra)
ha_map = c.datasets["ha"][:]
dec_map = c.datasets["dec"][:]


def ra_to_utc(r):
    unix = chime.lsd_to_unix(LSD + r / 360.0)
    return datetime.fromtimestamp(unix, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def utc_to_ra(utc):
    unix = datetime.strptime(utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    return 360.0 * (chime.unix_to_lsd(unix) - LSD)


def el_to_dec(y):
    return np.degrees(np.arcsin(np.clip(y, -1, 1)) + LAT)


def dec_to_el(d):
    return np.sin(np.radians(d) - LAT)


print(f"LSD {LSD} runs from {ra_to_utc(0)} to {ra_to_utc(360)} UTC")
print(f"{el.size} values of telescope-y, {ra.size} start times ({dt:.2f} s), {x.size} values of telescope-x")
print(f"x spans {x[0]:+.4f} to {x[-1]:+.4f}, i.e. +/-{np.degrees(np.arcsin(x[-1])):.1f} deg of hour angle on the equator")

# Plain numpy arrays for the rest of the notebook
snr = {m: cand[m].snr[:].local_array[0] for m in MASKS}
weight = {m: cand[m].weight[:].local_array[0] for m in MASKS}
best = {m: {k: cand[m].datasets[k][:].local_array[0] for k in ("amplitude", "x_index")} for m in MASKS}

pdb.set_trace()
#@me THE NOISE IN mJY =========================

fig, ax = plt.subplots(figsize=(10, 4))
for m in MASKS:
    #@me Where weight is nonzero, calculate sigma = 1/sqrt(weight) *1000 -> std in mJy. Where weight is zero, replace with np.nan. 
    sigma = np.where(weight[m] > 0, 1 / np.sqrt(np.where(weight[m] > 0, weight[m], 1)), np.nan) * 1e3  # mJy
    #@me Mask to identify which best['x_index'] values are on the centre of the beam. 
    centre_x = best[m]["x_index"] == x.size // 2 
    #@me For each tel-y value there's 8668 values for the x position in the beam that gave best SNR (27 possible)
    #@me For a given tel-y, look at those. Flag which ones are in the centre of the beam. then get the median noise for those values. 
    #@me AKA the median noise at that tel-y ONLY doing median of times when highest SNR was in centre of beam. 
    med_c = [np.nanmedian(sigma[e][centre_x[e]]) if centre_x[e].any() else np.nan for e in range(el.size)]
    ax.semilogy(el, med_c, label=f"mask = {m}, best x at the beam centre")
    #@me Now just doing: for a given tel-y, the median noise for all times throughout the day. 
    med_all = [np.nanmedian(row) if np.isfinite(row).any() else np.nan for row in sigma]
    ax.semilogy(el, med_all, ls="--", label=f"mask = {m}, all samples")
    i0 = np.argmin(np.abs(el)) #@me central tel-y value. 
    print(f"mask = {m}: noise at telescope-y = 0, beam centre: {med_c[i0]:.2f} mJy")
ax.set(xlabel="telescope-y", ylabel="noise of one sample [mJy]", ylim=(0.5, 200))
ax.legend()

#@me loading in noise scale file. 
#@me noise_scale = weights (expected noise) / measured noise. Shape: (2, 1024, 1, 1217) -> (pol, freq, ew, el)
#@me Right now, NO INFO ABOUT X. Divides the sky into horizontal strips w/ fringes defined by 22m baseline, one at each tel-y. 
#@me For each strip, shows how much noiser it is than expected at 1024 frequencies. 
#@me But we don't know where in the strip was noisier and when. 
ns = containers.HybridVisNoiseScale.from_file(run_file("full", "noise_scale"), distributed=True)
print("noise_scale:", {k: v.shape for k, v in ns.index_map.items()})
for name, dset in ns.datasets.items():
    print(f"    {name:8s} {dset.attrs['axis']}  {dset.shape}")

# The noise changes a lot with both frequency and telescope-y, so look at all of it
fns = ns.index_map["freq"]["centre"]
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for k, (ax, ip) in enumerate(zip(axes, range(ns.scale.shape[0]))):
    scale = ns.scale[:].local_array[ip, :, 0]  # [freq, el], 22 m
    im = ax.imshow(
        np.where(scale > 0, scale, np.nan).T, origin="lower", aspect="auto", interpolation="none",
        extent=[fns[0], fns[-1], ns.index_map["el"][0], ns.index_map["el"][-1]], norm=LogNorm(vmin=0.5, vmax=5),
    )
    ax.set(ylabel="telescope-y", title=f"noise scale, {ns.index_map['pol'][ip]}, 22 m", xlim=(fns.min(), fns.max()))
    ax.tick_params(labelbottom=True)
    plt.colorbar(im, ax=ax, pad=0.01, label="measured / expected\nnoise variance")
axes[-1].set_xlabel("frequency [MHz]")
plt.tight_layout()

pdb.set_trace()

#@me WHERE THE BRIGHT SOURCES ARE =========================

SOURCES = {
    "Sun": skyfield_wrapper.ephemeris["sun"],
    "Cas A": sources.CasA,
    "Cyg A": sources.CygA,
    "Tau A": sources.TauA,
    "Vir A": sources.VirA,
    "Her A": sources.source_dictionary["HERCULES_A"],
    "Hya A": sources.source_dictionary["HYDRA_A"],
    "3C 123": sources.source_dictionary["3C_123"],
    "3C 295": sources.source_dictionary["3C_295"],
    "3C 286": sources.source_dictionary["3C_286"],
}
BELOW_POLE = ["Cas A", "Cyg A"]

marks = []  # (name, ra, el)
src_pos = {}
for name, body in SOURCES.items():
    #@me Getting coords for all bright sources on that day. 
    #@me Also setting up labels for the plot (labeling Cas A & Cyg A as below pole, labeling Galactic centre)
    r, d = chime.object_coords(body, chime.lsd_to_unix(LSD + 0.5), deg=True)
    r, d = chime.object_coords(body, chime.lsd_to_unix(LSD + r / 360), deg=True)
    src_pos[name] = (r % 360, d)
    marks.append((name, r % 360, dec_to_el(d)))
    if name in BELOW_POLE:
        marks.append((f"{name} (below pole)", (r + 180) % 360, np.sin(np.radians(d) + LAT)))
# Galactic centre (J2000), close to the southern horizon
marks.append(("Gal. centre", 266.40, dec_to_el(-28.94)))


def galactic_to_equatorial(l, b=0.0):
    """J2000 RA and dec (degrees) of galactic coordinates (degrees)."""
    ra_gp, dec_gp, l_ncp = np.radians([192.85948, 27.12825, 122.93192])
    l, b = np.radians(l), np.radians(b)
    sin_dec = np.sin(dec_gp) * np.sin(b) + np.cos(dec_gp) * np.cos(b) * np.cos(l_ncp - l)
    y = np.cos(b) * np.sin(l_ncp - l)
    xx = np.cos(dec_gp) * np.sin(b) - np.sin(dec_gp) * np.cos(b) * np.cos(l_ncp - l)
    return np.degrees(ra_gp + np.arctan2(y, xx)) % 360, np.degrees(np.arcsin(sin_dec))

#@me Trace the Galactic plane (b = 0, all longitudes l): convert to RA & dec.
#@me Then convert to tel-y when transiting CHIME. To plot Galactic plane across RA & Y-tel. 
gp_ra, gp_dec = galactic_to_equatorial(np.linspace(0, 360, 721))
order = np.argsort(gp_ra)
gp_ra, gp_el = gp_ra[order], dec_to_el(gp_dec[order])

# Expected position of the J1634+44 pulse in the map (NaN if there was no pulse
# that day, which leaves it off the maps)
HAVE_J1634 = LSD in J1634_UTC #@me Check if the day we're looking at is also a day when CHIME saw J1634+44 & saved data.
if HAVE_J1634:
    ra_j = utc_to_ra(J1634_UTC[LSD]) #@me What RA, HA, dec and y-tel did CHIME see J1634 at? (Based on other data)
    ha_j = ((ra_j - J1634_RA_DEC[0] + 180) % 360) - 180
    dec_j = np.radians(J1634_RA_DEC[1])
    el_j = np.cos(LAT) * np.sin(dec_j) - np.sin(LAT) * np.cos(dec_j) * np.cos(np.radians(ha_j))
    print(f"J1634+44 pulse at {J1634_UTC[LSD]} UTC: RA {ra_j:.2f} deg, hour angle {ha_j:+.2f} deg, telescope-y {el_j:.3f}")
else:
    ra_j = el_j = np.nan
    print(f"no CHIME/Pulsar detection of J1634+44 listed for LSD {LSD}")


def absmax_reduce(img, max_rows=600, max_cols=1600):
    """Shrink a map for display, keeping the value of largest magnitude in each block.

    A full map has ~1200 x ~8700 samples, far more than the pixels on the screen, and
    matplotlib would otherwise drop most single-sample features when it resamples.
    """
    img = np.asarray(img, dtype=np.float32)
    ke = max(1, int(np.ceil(img.shape[0] / max_rows)))
    kr = max(1, int(np.ceil(img.shape[1] / max_cols)))
    if ke == 1 and kr == 1:
        return img
    ne, nr = -(-img.shape[0] // ke) * ke, -(-img.shape[1] // kr) * kr
    pad = np.full((ne, nr), np.nan, dtype=img.dtype)
    pad[: img.shape[0], : img.shape[1]] = img
    blocks = pad.reshape(ne // ke, ke, nr // kr, kr).transpose(0, 2, 1, 3).reshape(ne // ke, nr // kr, -1)
    empty = np.all(np.isnan(blocks) | (blocks == 0), axis=-1)  # no data in the block
    blocks = np.nan_to_num(blocks)
    idx = np.argmax(np.abs(blocks), axis=-1)
    out = np.take_along_axis(blocks, idx[..., np.newaxis], axis=-1)[..., 0]
    return np.where(empty, np.nan, out)


def plot_map(ax, img, title, vmin=0, vmax=10, log=False, ra_axis=ra, el_axis=el, marks_on=True, xlabel=True,
             cmap="viridis"):
    """Image of a map on (telescope-y, ra), with the bright sources and J1634+44 marked."""
    d = np.median(np.diff(ra_axis))
    norm = LogNorm(vmin=max(vmin, 1), vmax=vmax) if log else None
    im = ax.imshow(
        absmax_reduce(img), origin="lower", aspect="auto", interpolation="none", cmap=cmap,
        extent=[ra_axis[0], ra_axis[-1] + d, el_axis[0], el_axis[-1]],
        norm=norm, vmin=None if log else vmin, vmax=None if log else vmax,
    )
    if marks_on:
        inside = (gp_ra >= ra_axis[0]) & (gp_ra <= ra_axis[-1])
        ax.plot(np.where(inside, gp_ra, np.nan), gp_el, "w--", lw=0.8, alpha=0.7)
        for name, r, y in marks + [("J1634+44", ra_j, el_j)]:
            if ra_axis[0] <= r <= ra_axis[-1] and el_axis[0] <= y <= el_axis[-1]:
                color = "r" if name == "J1634+44" else "w"
                ax.plot(r, y, "o", mfc="none", mec=color, ms=16, mew=1.5)
                # Keep the labels inside the frame
                right = r > ra_axis[0] + 0.85 * (ra_axis[-1] - ra_axis[0])
                top = y > el_axis[0] + 0.9 * (el_axis[-1] - el_axis[0])
                ax.annotate(
                    name, (r, y), xytext=(-12 if right else 12, -14 if top else 0), textcoords="offset points",
                    color=color, fontsize=11, fontweight="bold", va="top" if top else "center",
                    ha="right" if right else "left",
                )
    ax.set(ylabel="telescope-y", title=title, xlim=(ra_axis[0], ra_axis[-1] + d), ylim=(el_axis[0], el_axis[-1]))
    ax.tick_params(labelbottom=True)
    if xlabel:
        ax.set_xlabel("RA at the start [deg]")
    return im

#@me THE NOISE THAT SCALES WITH THE SKY =========================

sky = containers.HybridVisSkyNoise.from_file(run_file("full", "sky_noise"))
print("sky_noise:", {k: v.shape for k, v in sky.index_map.items()})
print("datasets:", list(sky.datasets.keys()))

sk_centre = sky.index_map["freq"]["centre"][:] #@me(67) List of 67 frequencies from 401-797 (MHz)
sk_width = sky.index_map["freq"]["width"][:] #@me(67) Frequency separation: 6 (MHz) between every channel. 
blk = np.asarray(sky.index_map["block"][:], dtype=float) #@me (24) RA at the start of each block (RA divided into 24 blocks) 
blk_edges = np.r_[blk, blk[-1] + np.median(np.diff(blk))] 

#@me std of residual = kappa * unfiltered visibility, kappa_coh is common in all parts of the 6MHz band (all frequencies brightening by same factor)
k2 = sky.datasets["kappa2_coh_block"][:, 0, :]  # [band, block], 22 m
k2w = sky.datasets["kappa2_coh_block_weight"][:, 0, :]
#@me Replaces zero weight with nan, where there's nonzero weight, calculate sqrt(kappa_coh^2)=kappa_coh
kappa = np.where(k2w > 0, np.sqrt(np.clip(k2, 0, None)), np.nan) #@me (67, 24) -> (freq, RA). 

# The sub-bands, from the band groups file (only the small datasets)
with h5py.File(run_file("full", "transient_band_groups_x0"), "r") as f:
    grp_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["index_map/group"][:]]
    grp_used = f["band_used"][:]

# Night: the Sun is below the horizon more than its half-day arc from transit
sun_ra, sun_dec = src_pos["Sun"]
ha_set = np.degrees(np.arccos(np.clip(-np.tan(LAT) * np.tan(np.radians(sun_dec)), -1, 1)))
rr = np.linspace(blk_edges[0], blk_edges[-1], 2000)
night = np.abs(((rr - sun_ra + 180) % 360) - 180) > ha_set

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
im = axes[0].pcolormesh(blk_edges, np.r_[sk_centre - sk_width / 2, sk_centre[-1] + sk_width[-1] / 2],
                        kappa * 1e3, norm=LogNorm(vmin=0.5, vmax=10), cmap="magma")
axes[0].set(ylabel="frequency [MHz]", title=f"kappa_coh for each 6 MHz band and block of RA, LSD {LSD}, mask = full")
axes[0].tick_params(labelbottom=True)
plt.colorbar(im, ax=axes[0], pad=0.01, label="kappa_coh x 1e3")
for name, used in zip(grp_names, grp_used):
    if name == "all":
        continue
    w = np.where(used[:, None], k2w, 0)
    kg = np.sqrt(np.clip(np.sum(w * k2, axis=0) / np.maximum(w.sum(axis=0), 1e-30), 0, None))
    kg[w.sum(axis=0) == 0] = np.nan
    axes[1].step(blk_edges, np.r_[kg, kg[-1]] * 1e3, where="post", label=name)
axes[1].fill_between(rr, 0, 1, where=night, transform=axes[1].get_xaxis_transform(), color="0.9", zorder=0)
axes[1].set(xlabel="RA at the start of the block [deg]", ylabel="kappa_coh x 1e3", yscale="log", ylim=(0.3, 20),
            title="weighted over the bands of each sub-band (grey: Sun below the horizon)")
axes[1].legend(ncol=5, fontsize=8)
plt.colorbar(im, ax=axes[1], pad=0.01).ax.set_visible(False)  # keep the two panels the same width
plt.tight_layout()

#@me THE CANDIDATE MAPS =========================

def two_maps(imgs, titles, **kw):
    fig, axes = plt.subplots(len(imgs), 1, figsize=(16, 5.5 * len(imgs)), sharex=True)
    axes = np.atleast_1d(axes)
    for k, (ax, img, title) in enumerate(zip(axes, imgs, titles)):
        im = plot_map(ax, img, title, xlabel=(k == len(imgs) - 1), **kw)
        plt.colorbar(im, ax=ax, pad=0.01, label="S/N")
    plt.tight_layout()


masked = {m: np.where(weight[m] > 0, snr[m], np.nan) for m in MASKS}
two_maps([masked[m] for m in MASKS], [f"LSD {LSD}, mask = {m}" for m in MASKS], vmin=1, vmax=1000, log=True)

pdb.set_trace()

two_maps([masked[m] for m in MASKS], [f"LSD {LSD}, mask = {m}" for m in MASKS], vmin=0, vmax=8)

pdb.set_trace()

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.logspace(0, 4, 161)
for m in MASKS:
    ax.hist(snr[m][weight[m] > 0], bins=bins, histtype="step", label=f"mask = {m}")
ax.set(xscale="log", yscale="log", xlabel="best S/N", ylabel="number of (telescope-y, start time)")
ax.legend()
for m in MASKS:
    v = snr[m][weight[m] > 0]
    print(f"mask = {m}: valid fraction {np.mean(weight[m] > 0):.3f}, above 8: {np.mean(v > 8):.4f}, above 12: {np.mean(v > 12):.4f}")

pdb.set_trace()
#@me HOW IT DEPENDS ON FREQUENCY =========================

groups = {m: containers.TransientBandGroupMap.from_file(run_file(m, "transient_band_groups_maxx"), distributed=True) for m in MASKS}

g = groups["full"]
print("groups:", [str(n) for n in g.index_map["group"]])
for name, dset in g.datasets.items():
    print(f"    {name:16s} {dset.attrs['axis']}  {dset.shape}")
centre = g.index_map["freq"]["centre"][:]
width = g.index_map["freq"]["width"][:]
group_range = {}
for k, name in enumerate(g.index_map["group"]):
    used = g.datasets["band_used"][k]
    group_range[str(name)] = ((centre - width / 2)[used].min(), (centre + width / 2)[used].max())
    print(f"    {name:8s}: {used.sum():2d} bands, {group_range[str(name)][0]:.0f}-{group_range[str(name)][1]:.0f} MHz")

# Keep the axes of the group maps (they should be the same as the candidate map)
g_el = g.index_map["el"][:]
g_ra = g.ra[:]
gsnr = {m: {name: groups[m].snr[:].local_array[0, k] for k, name in enumerate(groups[m].index_map["group"])} for m in MASKS}
gxi = {m: {name: groups[m].datasets["x_index"][:].local_array[0, k] for k, name in enumerate(groups[m].index_map["group"])}
       for m in MASKS}
g_x = g.index_map["x"][:]
g_ha = g.datasets["ha"][:]  # [el, x]
g_dec = g.datasets["dec"][:]
if HAVE_EACH_BAND:
    each = {m: containers.TransientBandGroupMap.from_file(run_file(m, "transient_each_band_maxx"), distributed=True)
            for m in MASKS}
    esnr = {m: each[m].snr[:].local_array[0] for m in MASKS}  # band, el, ra
    e_centre = each["full"].index_map["freq"]["centre"][:]
    e_width = each["full"].index_map["freq"]["width"][:]

fig, ax = plt.subplots(figsize=(13, 4)) if HAVE_EACH_BAND else (None, None)
for m in MASKS if HAVE_EACH_BAND else []:
    has_data = esnr[m] != 0
    frac = [np.mean(esnr[m][b][has_data[b]] > 5) if has_data[b].any() else np.nan for b in range(e_centre.size)]
    ax.semilogy(e_centre, frac, "o-", ms=4, label=f"mask = {m}")
if HAVE_EACH_BAND:
    for lo, hi, label in ((470, 608, "UHF TV"), (617, 652, "5G DL"), (663, 698, "5G UL"), (698, 758, "LTE 700")):
        ax.axvspan(lo, hi, color="0.85", zorder=0)
        ax.text(0.5 * (lo + hi), 1.0, label, ha="center", va="top", fontsize=9, transform=ax.get_xaxis_transform())
    ax.set(xlabel="band centre [MHz]", ylabel="fraction of map with S/N > 5", title=f"each 6 MHz band, LSD {LSD}")
    ax.legend()
else:
    print("no 6 MHz band files; skipped")

gnames = list(gsnr["full"])
fig, ax = plt.subplots(figsize=(10, 4))
for k, m in enumerate(MASKS):
    frac = [np.mean(gsnr[m][n][gsnr[m][n] != 0] > 5) for n in gnames]
    ax.bar(np.arange(len(gnames)) + 0.4 * (k - 0.5), frac, width=0.4, label=f"mask = {m}")
    print(f"mask = {m:4s}: " + ", ".join(f"{n} {f:.4f}" for n, f in zip(gnames, frac)))
ax.set_xticks(np.arange(len(gnames)))
ax.set_xticklabels([f"{n}\n{group_range[n][0]:.0f}-{group_range[n][1]:.0f}" for n in gnames])
ax.set(yscale="log", ylabel="fraction of map with S/N > 5")
ax.legend()

# The cleanest sub-band, with the masks
frac_full = {n: np.mean(gsnr["full"][n][gsnr["full"][n] != 0] > 5) for n in gnames if n != "all"}
clean = min(frac_full, key=frac_full.get)
print(f"cleanest sub-band with the masks: {clean}")

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, 30, 151)
for m in MASKS:
    for name, ls in (("all", "-"), (clean, "--")):
        v = gsnr[m][name][gsnr[m][name] != 0]
        ax.hist(v, bins=bins, histtype="step", ls=ls, label=f"{name}, mask = {m}")
ax.set(yscale="log", xlabel="S/N", ylabel="number")
ax.legend()

ig = np.round((ra - g_ra[0]) / dra).astype(int)
has = (ig >= 0) & (ig < g_ra.size)
has[has] &= np.abs(g_ra[ig[has]] - ra[has]) < 0.5 * dra
ie = np.array([int(np.argmin(np.abs(g_el - v))) for v in el])
same_el = np.abs(g_el[ie] - el) < 1e-6
m = "full"
sel = (weight[m] > 0) & same_el[:, None] & has[None, :]
a = gsnr[m]["all"][np.ix_(ie, np.where(has, ig, 0))][sel]
b = snr[m][sel]
print(f"{sel.sum()} samples compared")
print(f"median |difference| {np.median(np.abs(a - b)):.3g}, fraction within 1%: {np.mean(np.abs(a - b) <= 1e-2 * np.abs(b) + 1e-3):.3f}")

#@me THE MAPS IN UNITS OF NOISE =========================

groups0 = {m: containers.TransientBandGroupMap.from_file(run_file(m, "transient_band_groups_x0"), distributed=True) for m in MASKS}
print("x0 maps at telescope-x =", groups0["full"].attrs["x_reduce"])

gsnr0 = {m: {str(n): groups0[m].snr[:].local_array[0, k] for k, n in enumerate(groups0[m].index_map["group"])} for m in MASKS}
g0_el = groups0["full"].index_map["el"][:]
g0_ra = groups0["full"].ra[:]

bins = np.linspace(-8, 8, 161)
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for col, m in enumerate(MASKS):
    for n in gsnr0[m]:
        v = gsnr0[m][n][gsnr0[m][n] != 0]
        ax[col].hist(v, bins=bins, histtype="step", density=True, label=n, lw=1.5 if n == "all" else 0.8)
    ax[col].plot(bins, np.exp(-0.5 * bins**2) / np.sqrt(2 * np.pi), "k--", label="N(0, 1)")
    ax[col].set(yscale="log", ylim=(1e-6, 1), xlabel="S/N at telescope-x = 0", title=f"mask = {m}")
ax[1].legend(fontsize=8, ncol=2)
plt.tight_layout()

for m in MASKS:
    print(f"mask = {m}: robust std of the S/N (1 for pure noise): "
          + ", ".join(f"{n} {1.4826 * np.median(np.abs(gsnr0[m][n][gsnr0[m][n] != 0])):.2f}" for n in gsnr0[m]))

fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
for k, (ax, m) in enumerate(zip(axes, MASKS)):
    im = plot_map(ax, gsnr0[m]["all"], f"all bands, telescope-x = 0, mask = {m}", vmin=-5, vmax=5,
                  ra_axis=g0_ra, el_axis=g0_el, cmap="RdBu_r", xlabel=(k == 1))
    plt.colorbar(im, ax=ax, pad=0.01, label="S/N")
plt.tight_layout()

#@me J1634+44 =========================
def zoom_index(e, t, half_el=40, half_ra=60, shape=None):
    n_el, n_ra = shape if shape is not None else (el.size, ra.size)
    return slice(max(e - half_el, 0), min(e + half_el + 1, n_el)), slice(max(t - half_ra, 0), min(t + half_ra + 1, n_ra))


def describe(mm, e, t):
    ix = best[mm]["x_index"][e, t]
    return (
        f"S/N {snr[mm][e, t]:6.1f}  amp {best[mm]['amplitude'][e, t]:6.2f} Jy  "
        f"y {el[e]:+.3f}  x {x[ix]:+.4f}  RA start {ra[t]:.2f}  ({ra_to_utc(ra[t])} UTC)  "
        f"src RA {(ra[t] - ha_map[e, ix]) % 360:.2f}  dec {dec_map[e, ix]:.1f}"
    )


peak = {}
if HAVE_J1634:
    near = (np.abs(el - el_j) < 0.03)[:, None] & (np.abs(ra - ra_j) < 1.0)[None, :]
    for m in MASKS:
        e, t = np.unravel_index(np.argmax(np.where(near & (weight[m] > 0), snr[m], -np.inf)), snr[m].shape)
        peak[m] = (e, t)
        print(f"mask = {m:4s}: {describe(m, e, t)}")

if HAVE_J1634:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for col, m in enumerate(MASKS):
        e, t = peak[m]
        se, st = zoom_index(e, t)
        plot_map(axes[0, col], snr[m][se, st], f"J1634+44, mask = {m}", vmax=np.nanmax(snr[m][se, st]),
                 ra_axis=ra[st], el_axis=el[se], marks_on=False)
        axes[0, col].plot(ra[t], el[e], "r+", ms=14)
        axes[1, col].plot(ra[st], snr[m][e, st], "k-", lw=1)
        axes[1, col].set(xlabel="RA at the start [deg]", ylabel="best S/N", title=f"along telescope-y = {el[e]:+.3f}")
    plt.tight_layout()

def at_group_pixel(e, t):
    """Indices in the group maps of a sample of the candidate map."""
    return int(np.argmin(np.abs(g_el - el[e]))), int(np.argmin(np.abs(g_ra - ra[t])))


def plot_spectrum(ax, eg, tg, mm, label=None, **kw):
    """S/N in each sub-band (and each 6 MHz band, if available) at one sample of the group maps."""
    if HAVE_EACH_BAND:
        spec = esnr[mm][:, eg, tg]
        ok = spec != 0
        ax.errorbar(e_centre[ok], spec[ok], xerr=e_width[ok] / 2, yerr=1, fmt="none", color="0.6", lw=0.8)
    subs = [n for n in gsnr[mm] if n != "all"]
    lo = np.array([group_range[n][0] for n in subs])
    hi = np.array([group_range[n][1] for n in subs])
    val = np.array([gsnr[mm][n][eg, tg] for n in subs])
    ok = val != 0
    ax.errorbar(0.5 * (lo + hi)[ok], val[ok], xerr=0.5 * (hi - lo)[ok], yerr=1, fmt="o", ms=4, label=label, **kw)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="frequency [MHz]", ylabel="S/N in the sub-band (best x)", xlim=(398, 800))


def edge_marks(ax, r, y):
    """Mark a position with small triangles outside the frame, so the source itself stays visible."""
    kw = dict(color="r", ms=7, clip_on=False, transform=ax.transData)
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    ax.plot(r, y1, "v", **kw)
    ax.plot(r, y0, "^", **kw)
    ax.plot(x0, y, ">", **kw)
    ax.plot(x1, y, "<", **kw)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def band_panels(eg, tg, mm, title, half_el=25, half_ra=40):
    """Maps around one sample: all bands (large) and each sub-band, on one colour scale."""
    se = slice(max(eg - half_el, 0), min(eg + half_el + 1, g_el.size))
    st = slice(max(tg - half_ra, 0), min(tg + half_ra + 1, g_ra.size))
    ext = [g_ra[st][0], g_ra[st][-1], g_el[se][0], g_el[se][-1]]
    vmax = max(5.0, np.nanmax(gsnr[mm]["all"][se, st]))
    subs = [n for n in gsnr[mm] if n != "all"]

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(3, 6)
    axes = [fig.add_subplot(gs[:, :3])] + [fig.add_subplot(gs[k // 3, 3 + k % 3]) for k in range(len(subs))]
    for k, (ax, n) in enumerate(zip(axes, ["all"] + subs)):
        lo, hi = group_range[n]
        img = gsnr[mm][n][se, st]
        im = ax.imshow(np.where(img != 0, img, np.nan), origin="lower", aspect="auto", interpolation="none",
                       extent=ext, vmin=0, vmax=vmax)
        edge_marks(ax, g_ra[tg], g_el[eg])
        ax.set_title(f"{n} ({lo:.0f}-{hi:.0f} MHz)", fontsize=12 if k == 0 else 9)
        if k == 0:
            ax.set(xlabel="RA at the start [deg]", ylabel="telescope-y")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, pad=0.01, label="S/N (best x)")
    fig.suptitle(title)


if HAVE_J1634:
    fig, ax = plt.subplots(figsize=(12, 4))
    for k, m in enumerate(MASKS):
        eg, tg = at_group_pixel(*peak[m])
        plot_spectrum(ax, eg, tg, m, color=f"C{k}", label=f"mask = {m}")
        print(f"mask = {m:4s}: " + ", ".join(f"{n}: {gsnr[m][n][eg, tg]:5.1f}" for n in gsnr[m]))
    ax.set_title(f"J1634+44 on LSD {LSD}, at the peak")
    ax.legend()
    band_panels(*at_group_pixel(*peak["full"]), "full", "J1634+44, mask = full")


m = "full"
bands_file = run_file(m, "transient_mfilt_bands")
if HAVE_J1634 and not HAVE_MFILT_BANDS:
    print("no per-band matched filter file; skipped")
if HAVE_J1634 and HAVE_MFILT_BANDS:
    with h5py.File(bands_file, "r") as f:
        b_el = f["index_map/el"][:]
        b_ra = f["index_map/ra"][:]
    e, t = peak[m]
    eb, tb = int(np.argmin(np.abs(b_el - el[e]))), int(np.argmin(np.abs(b_ra - ra[t])))
    one = containers.TransientMatchedFilter.from_file(
        bands_file, el_sel=slice(eb, eb + 1), ra_sel=slice(tb, tb + 1), distributed=True
    )
    for name, dset in one.datasets.items():
        print(f"    {name:10s} {dset.attrs['axis']}  {dset.shape}")
    ix = best[m]["x_index"][e, t]
    amp_b = one.amplitude[:].local_array[0, 0, :, 0, 0, ix]
    w_b = one.weight[:].local_array[0, 0, :, 0, 0, ix]
    b_centre = one.index_map["freq"]["centre"][:]

    fig, ax = plt.subplots(figsize=(12, 4))
    ok = w_b > 0
    ax.errorbar(b_centre[ok], amp_b[ok], yerr=1 / np.sqrt(w_b[ok]), fmt="o", ms=3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="band centre [MHz]", ylabel="amplitude [Jy]", title=f"J1634+44, mask = {m}, x = {x[ix]:+.4f}")

#@me OTHER BRIGHT CANDIDATES =========================

TRACKS = {"Sun": 0.03, "Cyg A": 0.02, "Cas A": 0.02}  # half-width in telescope-y
SUN_RA_WIDTH = 20.0  # degrees either side of the Sun's transit, masked at every telescope-y


def track(name, ra_axis):
    """Telescope-y of a source at each start time, and whether it is above the horizon."""
    sra, sdec = src_pos[name]
    h = np.radians(ra_axis - sra)
    d = np.radians(sdec)
    y = np.cos(LAT) * np.sin(d) - np.sin(LAT) * np.cos(d) * np.cos(h)
    up = np.sin(LAT) * np.sin(d) + np.cos(LAT) * np.cos(d) * np.cos(h) > 0
    return y, up


track_mask = np.zeros((g_el.size, g_ra.size), dtype=bool)
for name, half in TRACKS.items():
    y_t, up = track(name, g_ra)
    track_mask |= (np.abs(g_el[:, None] - y_t[None, :]) < half) & up[None, :]
track_mask |= (np.abs(((g_ra - src_pos["Sun"][0] + 180) % 360) - 180) < SUN_RA_WIDTH)[None, :]
print(f"fraction of the map masked: {track_mask.mean():.3f}")

fig, ax = plt.subplots(figsize=(16, 5.5))
im = plot_map(ax, np.where(track_mask, np.nan, gsnr["full"]["all"]),
              "all bands, mask = full, with the tracks of the Sun, Cyg A and Cas A removed (white)",
              vmin=0, vmax=8, ra_axis=g_ra, el_axis=g_el)
plt.colorbar(im, ax=ax, pad=0.01, label="S/N")

CAND_GROUP = "all"
# Candidates within this many degrees of RA of the transit of the Sun, Cas A or Cyg A
# are tagged at any telescope-y: their residuals spread over many rows
BRIGHT_WIDTH = {"Sun": 20.0, "Cas A": 12.0, "Cyg A": 12.0}

# Radio sources brighter than CAT_MIN_JY at 600 MHz.  A candidate within CAT_DY in
# telescope-y and CAT_DRA in RA of the transit of one of these is tagged with it.
CAT_MIN_JY, CAT_DY, CAT_DRA = 5.0, 0.01, 15.0
catalog = []
try:
    from ch_util import fluxcat

    for n in fluxcat.FluxCatalog.keys():
        try:
            s600 = float(fluxcat.FluxCatalog[n].predict_flux(600.0))
        except Exception:
            continue
        if np.isfinite(s600) and s600 >= CAT_MIN_JY:
            c_ = fluxcat.FluxCatalog[n]
            catalog.append((n, c_.ra % 360, c_.dec, s600, dec_to_el(c_.dec)))
except ImportError:
    print("ch_util.fluxcat is not available, so candidates are not matched to the catalogue")
print(f"{len(catalog)} catalogue sources above {CAT_MIN_JY:g} Jy at 600 MHz")


def find_peaks(img, valid, n, ra_axis, el_axis, sep_el=0.02, sep_ra=2.0):
    order = np.argsort(np.where(valid, img, -np.inf), axis=None)[::-1][: 2000 * n]
    found = []
    for flat in order:
        e, t = np.unravel_index(flat, img.shape)
        if any(abs(el_axis[e] - el_axis[e2]) < sep_el and abs(ra_axis[t] - ra_axis[t2]) < sep_ra for e2, t2 in found):
            continue
        found.append((e, t))
        if len(found) == n:
            break
    return found


def cat_match(r, y):
    """The brightest catalogue source transiting near (r, y), or None."""
    near = [c for c in catalog if abs(c[4] - y) < CAT_DY and abs(((r - c[1] + 180) % 360) - 180) < CAT_DRA]
    return max(near, key=lambda c: c[3]) if near else None


def near_tag(r, y):
    """J1634+44, or the transit (in RA only) of the Sun, Cas A or Cyg A."""
    if abs(y - el_j) < 0.03 and abs(r - ra_j) < 2:
        return "J1634+44"
    names = []
    for name, width in BRIGHT_WIDTH.items():
        sra = src_pos[name][0]
        if abs(((r - sra + 180) % 360) - 180) < width:
            names.append(name)
        elif name in BELOW_POLE and abs(((r - sra) % 360) - 180) < width:
            names.append(f"{name} (below pole)")
    return ", ".join(names)


def candidate_table(mm, name, found):
    """A table of candidates in the map of group `name`, indexed by their position in `found`."""
    rows = []
    for e, t in found:
        ix = gxi[mm][name][e, t]
        r, y = g_ra[t], g_el[e]
        row = {
            "S/N": float(gsnr[mm][name][e, t]),
            "telescope-y": y,
            "telescope-x": g_x[ix],
            "RA start": r,
            "UTC": ra_to_utc(r),
            "source RA": (r - g_ha[e, ix]) % 360,
            "source dec": g_dec[e, ix],
        }
        if name == "all":
            ec, tc = int(np.argmin(np.abs(el - y))), int(np.argmin(np.abs(ra - r)))
            row["amp [Jy]"] = float(best[mm]["amplitude"][ec, tc])
        row["y frac"] = frac6[t]
        row["sub-bands > 5"] = sum(int(gsnr[mm][n][e, t] > 5) for n in gsnr[mm] if n != "all")
        row["near transit of"] = near_tag(r, y)
        c_ = cat_match(r, y)
        row["catalogue source"] = c_[0] if c_ else ""
        row["S600 [Jy]"] = c_[3] if c_ else np.nan
        row["its transit - RA [deg]"] = ((c_[1] - r + 180) % 360) - 180 if c_ else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.index.name = "#"
    return table.round({"S/N": 1, "telescope-y": 3, "telescope-x": 4, "RA start": 2, "source RA": 2,
                        "source dec": 1, "amp [Jy]": 2, "y frac": 2, "S600 [Jy]": 1, "its transit - RA [deg]": 1})


m = "full"
cmap_ = gsnr[m][CAND_GROUP]
valid = (cmap_ != 0) & ~track_mask
frac6 = np.sum(valid & (cmap_ > 6), axis=0) / np.maximum(valid.sum(axis=0), 1)
peaks = find_peaks(cmap_, valid, 100, g_ra, g_el)
table = candidate_table(m, CAND_GROUP, peaks)
untagged = table[(table["near transit of"] == "") & (table["catalogue source"] == "")]

print(f"{CAND_GROUP}, mask = {m}, without the bright-source tracks, the 20 brightest:")
display(table.head(20).fillna(""))
print("the 10 brightest not near a transit of the Sun, Cas A, Cyg A or a catalogue source, or J1634+44:")
display(untagged.head(10).fillna(""))

PICK = list(untagged.index[:3])


def show_candidate(k, mm="full", half_ra_deg=15.0, half_el=0.2):
    e, t = peaks[k]
    r0, y0 = g_ra[t], g_el[e]
    display(table.loc[[k]].fillna(""))
    display(pd.DataFrame({n: [float(gsnr[mm][n][e, t])] for n in gsnr[mm]}, index=["S/N in each sub-band"]).round(1))

    # Where it is in the whole day
    fig, ax = plt.subplots(figsize=(16, 4))
    plot_map(ax, np.where(track_mask, np.nan, cmap_), f"{CAND_GROUP}, mask = {mm}: #{k} (red circle)",
             vmin=0, vmax=8, ra_axis=g_ra, el_axis=g_el)
    ax.plot(r0, y0, "o", mfc="none", mec="r", ms=22, mew=2)

    # A wider view, with the catalogue sources that transit in it
    st = (g_ra > r0 - half_ra_deg) & (g_ra < r0 + half_ra_deg)
    se = (g_el > y0 - half_el) & (g_el < y0 + half_el)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    plot_map(ax[0], cmap_[np.ix_(se, st)], f"#{k}, {CAND_GROUP}", vmin=0, vmax=max(8, gsnr[mm][CAND_GROUP][e, t] / 3),
             ra_axis=g_ra[st], el_axis=g_el[se], marks_on=False)
    ax[0].plot(r0, y0, "o", mfc="none", mec="r", ms=18, mew=1.5)
    for c_ in catalog:
        if abs(((c_[1] - r0 + 180) % 360) - 180) < half_ra_deg and abs(c_[4] - y0) < half_el:
            ax[0].plot(c_[1], c_[4], "s", mfc="none", mec="w", ms=10)
            ax[0].annotate(f"{c_[0]} {c_[3]:.0f} Jy", (c_[1], c_[4]), xytext=(6, 6), textcoords="offset points",
                           color="w", fontsize=8)
    ax[1].plot(g_ra[st], cmap_[e, st], "k-", lw=0.8)
    c_ = cat_match(r0, y0)
    if c_ is not None:
        ax[1].axvline(c_[1], color="C1", ls="--", label=f"transit of {c_[0]}")
        ax[1].legend(fontsize=8)
    ax[1].set(xlabel="RA at the start [deg]", ylabel="best S/N", title=f"along telescope-y = {y0:+.3f}")
    plot_spectrum(ax[2], e, t, mm, color="C0")
    plt.tight_layout()

    band_panels(e, t, mm, f"#{k}: all bands and each sub-band")
    plt.show()


for k in PICK:
    show_candidate(k)