import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.time import Time
import astropy.units as u
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import curve_fit

def expfit(t, tau, A):
    return A*np.exp(-np.abs(t)/tau)

def gaussfit(t, sigma, A):
    return A*np.exp(-t**2/(2*sigma**2))

def gaussfitEff(t, sigma):
    return np.exp(-t**2/(2*sigma**2))

def gaussfitC(t, sigma, A, C):
    return A*np.exp(-t**2/(2*sigma**2)) + C

def lorentzian( x, gam, a ):
    return a * gam**2 / ( gam**2 + x **2)


# Load data
d = np.load('../Spectra/Burstspectra_FAST.npz', allow_pickle=True, encoding='latin1')
#d = np.load('../Spectra/Burstspectra_59611.npz', allow_pickle=True, encoding='latin1')

specs = d['spectra']
specsNoise = d['spectra_off']
ts = d['tburst_unix']
onfreqs = d['onfreqs']
freq = d['F_MHz']*u.MHz

# Plot dynamic spectra
plt.figure(figsize=(18, 12))

ilim = np.argwhere(freq.value>1100).squeeze()[0]
for i in range(len(specs)):
    spec1 = specs[i]
    gi = onfreqs[i]#[:-1]
    gi = gi[gi>ilim]
    ti = ts[i]
    specplot = spec1[gi]
    specplot = 0.15*specplot / np.std(specplot)
    if len(specplot) > corrmin:
        plt.plot(freq[gi], specplot+ti/60., color='k', alpha=0.2)

plt.xlabel('frequency (MHz)', fontsize=16)
plt.ylabel('time (min)', fontsize=16)
plt.xlim(1100, 1500)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# Correlate all pairs of spectra
N = len(ts)
dts = np.zeros((N, N))
bad = []

for i in range(N):
    ti = ts[i]
    for j in range(N):
        tj = ts[j]
        dts[i,j] = tj-ti

pairs = np.argwhere((abs(dts)<30000.) & (abs(dts)>0.1))

corrs = []
dtcorrs = []
corrs2D = []

# restrict frequency range
ilim = np.argwhere( (freq.value>1200) & (freq.value<1500)).squeeze()

# minimum number of overlapping frequencies (change to fraction of total)
corrmin = 200
compute_err = 1
Niter = 10
errs = []

# perform correlation, loop over pairs, compute errors
for k in range(pairs.shape[0]):
    i = pairs[k, 0]
    j = pairs[k, 1]
    of1 = onfreqs[i]
    of2 = onfreqs[j]
    of = np.intersect1d(of1, of2)
    of = of[ (of > ilim[0]) & (of < ilim[-1])]
    
    if len(of) > corrmin:
        spec1 = specs[i][of]
        spec2 = specs[j][of]
        spec1[spec1>10] = 0
        spec2[spec2>10] = 0

        N1 = specsNoise[i][of]
        N2 = specsNoise[j][of]  
        dt = dts[i,j]

        if compute_err:
            csims = np.zeros(Niter)
            for ii in range(Niter):
                spec1sim = spec1 + np.random.normal(size=len(spec1), scale=np.std(N1))
                spec2sim = spec2 + np.random.normal(size=len(spec2), scale=np.std(N2))
                std1 = np.sqrt(np.std(spec1)**2.0 - np.std(N1)**2.0)
                std2 = np.sqrt(np.std(spec2)**2.0 - np.std(N2)**2.0)
                c = np.mean((spec1sim-np.mean(spec1sim))*(spec2sim-np.mean(spec2sim))) / (std1*std2)
                csims[ii] = c
            err = np.std(csims)
            errs.append(err)
        
        std1 = np.sqrt(np.std(spec1)**2.0 - np.std(N1)**2.0)
        std2 = np.sqrt(np.std(spec2)**2.0 - np.std(N2)**2.0)
        c = np.mean((spec1-np.mean(spec1))*(spec2-np.mean(spec2))) / (std1*std2)

        dtcorrs.append(dt)
        corrs.append(c)

        norm = len(spec1)
        pad1 = np.zeros(specs.shape[1] - spec1.shape[0]) + np.mean(spec1)
        pad2 = np.zeros(specs.shape[1] - spec2.shape[0]) + np.mean(spec2)

        spec1 = np.concatenate((spec1, pad1))
        spec2 = np.concatenate((spec2, pad2))
        ccorr = np.fft.ifft(np.fft.fft(spec1-np.mean(spec1)) * 
                             np.fft.fft(spec2-np.mean(spec2)).conj() )
        ccorr = ccorr / (std1*std2) / (norm)
        ccorr = np.fft.fftshift(ccorr)
        corrs2D.append(ccorr)
    else:
        print("{0}, {1} don't overlap in frequency".format(i,j))
    
dtplot = np.linspace(0,max(dtcorrs)/60.,1000)
    
Ei = ~np.isnan(corrs)
dtcorrs = np.array(dtcorrs)[Ei]
corrs = np.array(corrs)[Ei]
errs = np.array(errs)[Ei]
corrs2D = np.array(corrs2D)[Ei]

plt.figure(figsize=(10,6))
if compute_err:
    plt.errorbar(abs(dtcorrs)/60., corrs, yerr=errs, marker='o', markersize=3, 
                 linestyle='None', alpha=0.2, color='k')
else:
    plt.scatter(dtcorrs/60., corrs)


plt.xlabel("time (min)", fontsize=14)
plt.ylabel("correlation (arb)", fontsize=14)

# Fit 1D ACF in time
xfit = abs(dtcorrs)/60.
yfit = corrs
yerr = errs
p0 = [ 5., 0.9]
pEff, pEffcov = curve_fit(gaussfit, xfit, yfit, p0=p0)
print(pEff[1])
prefactor = np.sqrt(2*np.log(2))
tsEff = prefactor * pEff[0]
terrEff = prefactor * np.sqrt(pEffcov[0][0])
print(tsEff, terrEff)

plt.plot(dtplot, gaussfit(dtplot, pEff[0], pEff[1]), linestyle='dotted', color='tab:red')
plt.xlim(-5,30)
plt.ylim(-0.5, 1.5)
plt.show()

# Compute 2D ACF, in uniform grid

ntbin = 100
tbin = 0.25

corr_regular = np.zeros(2*ntbin+1)
corrs2D_regular = np.zeros( (2*ntbin+1, corrs2D.shape[1]))
errs_regular = np.zeros(2*ntbin+1)
badindeces = []
npairs = []

for i in range(2*ntbin+1):
    tlow = i*tbin - ntbin*tbin
    thigh = (i+1)*tbin -0.001 -ntbin*tbin
    
    indeces_np = np.argwhere((dtcorrs/60.>tlow) & (dtcorrs/60. < thigh))
    if len(indeces_np) > 0:
        indeces = indeces_np.squeeze()
        corr_i = corrs[indeces]
        errs_i = errs[indeces]
        weightedmean = np.nansum(corr_i/errs_i**2) / np.nansum(1/errs_i**2)
        #weightedmean = np.mean(corr_i)
        std = np.min(errs_i)
        corr_regular[i] = weightedmean
        errs_regular[i] = std
        
        corr2D_i = corrs2D[indeces]
        if len(indeces_np) > 1:
            #print(corr2D_i.shape, errs_i.shape)
            weighted2D = np.nansum(corr2D_i/(errs_i[:,np.newaxis]**2), axis=0) / np.nansum(1/errs_i**2)
        else:
            #print(corr2D_i.shape, errs_i)
            weighted2D = np.nansum(corr2D_i/(errs_i**2), axis=0) / (1/errs_i**2)
            weighted2D = corr2D_i
        corrs2D_regular[i] = weighted2D
    else:
        badindeces.append(i)
    npairs.append(len(indeces_np))

badindeces = np.array(badindeces)
dt_regular = np.arange(2*ntbin+1)*tbin
dt_regular = dt_regular - np.mean(dt_regular)

if len(badindeces) > 0:
    dt_regular = np.delete(dt_regular, badindeces)
    corr_regular = np.delete(corr_regular, badindeces)
    errs_regular = np.delete(errs_regular, badindeces)

p0 = [ 16., 0.8, 0.1]
pEff2, pEffcov2 = curve_fit(gaussfitC, dt_regular, corr_regular, p0=p0)
prefactor = np.sqrt(2*np.log(2))
tsEff = prefactor * pEff2[0]
terrEff = prefactor * np.sqrt(pEffcov2[0][0])
print(tsEff, terrEff)

plt.figure(figsize=(7,4))
plt.errorbar(dt_regular, corr_regular, yerr=errs_regular, marker='o', markersize=3, 
             linestyle='None', alpha=0.4, color='k')


plt.plot(dtplot, gaussfit(dtplot, pEff[0], pEff[1]), linestyle='dotted', color='tab:red')
plt.plot(dtplot, gaussfitC(dtplot, *pEff2), linestyle='dotted', color='tab:blue')

plt.ylim(-0.1, 1.1)

plt.xlabel("time (min)", fontsize=14)
plt.ylabel("correlation (norm.)", fontsize=14)
plt.show()


# Plot 2D ACF, and 1D slices
slices = 0
tlim = 20.
flim = 5.

corrhilim = 1.2
corrlolim = -0.2

fscint = 0.36

midt = C0.shape[0]//2
tslice = slice(midt-1, midt+2)
midf = C0.shape[1]//2
fslice = slice(midf-1, midf+2)

taxis = np.linspace(-ntbin*tbin, (ntbin-1)*tbin, C0.shape[0])
faxis = np.linspace(-ndf*dfbin, (ndf)*dfbin, C0.shape[1], endpoint=False)
if slices:
    CT = C0[:,fslice].mean(-1)
    CF = C0[tslice].mean(0)
else:
    CT = C0[:,midf]
    CF = C0[midt]
    
fig = plt.figure(figsize=(8,8))
plt.subplots_adjust(hspace=0.05, wspace=0.05)
ax = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
axt = plt.subplot2grid((4, 4), (0, 0), colspan=3)
axf = plt.subplot2grid((4, 4), (1, 3), rowspan=3)


ax.imshow(C0.T, aspect='auto', cmap='viridis', vmin=-0.2, vmax=1.2,
           extent=[-ntbin*tbin, (ntbin-1)*tbin, -ndf*dfbin, (ndf+1)*dfbin])
ax.set_xlabel(r'$\Delta t$ (minutes)', fontsize=16)
ax.set_ylabel(r'$\Delta \nu$ (MHz)', fontsize=16)
ax.set_ylim(-flim,flim)
ax.set_xlim(-tlim, tlim)


axt.errorbar(dtcorrs/60., corrs, yerr=errs, marker='o', markersize=3, 
             linestyle='None', alpha=0.05, color='tab:blue')

axt.plot(taxis, CT, color='k')
axf.plot(CF, faxis, color='k')

axt.set_ylim(corrlolim, corrhilim)

axf.set_ylim(-flim,flim)
axt.set_xlim(-tlim,tlim)
axt.plot(taxis, gaussfitC(taxis, *pEff2), linestyle='--', color='tab:orange')
axf.plot(expfit(faxis, fscint, 1), faxis, linestyle='--', color='tab:orange')
axt.set_xticks([])
axf.set_yticks([])
axt.set_ylabel(r'$R(\Delta t, \Delta \nu=0)$', fontsize=16)
axf.set_xlabel(r'$R(\Delta \nu, \Delta t=0)$', fontsize=16)

plt.show()


# Plot secondary spectrum

mask = 1
dfbin = (freq[1] - freq[0]).value
ndf = corrs2D_regular.shape[1]//2

C = np.roll(corrs2D_regular, 1, axis=0)
C[np.isnan(C)] = 0.
C0 = np.copy(C)

if mask:
    taperlength_t = int(C.shape[0]//2)
    taperlength_f = int(C.shape[1]//2)
    window = np.ones_like(C)

    window[:taperlength_t] *= np.hanning(2*taperlength_t)[:taperlength_t, np.newaxis]
    window[-taperlength_t:] *= np.hanning(2*taperlength_t)[-taperlength_t:, np.newaxis]
    window[:, :taperlength_f] *= np.hanning(2*taperlength_f)[np.newaxis, :taperlength_f]
    window[:, -taperlength_f:] *= np.hanning(2*taperlength_f)[np.newaxis, -taperlength_f:]
    C = C * window    

C = np.fft.fftshift(C)

S = np.fft.fft2(C)
S = np.abs(S)#**2.0
S = np.fft.fftshift(S)

bintau = 12
Sbin = S.reshape(S.shape[0], S.shape[1]//bintau, bintau).mean(-1)
print(Sbin.shape)
Splot = np.log10(Sbin)
Splot = Splot - np.median(Splot, axis=0)

vmax = np.max(Splot)
#vmin = vmax-1.5
vmin = vmax - 4.
maxft = (1 / (2*tbin*u.minute)).to(u.mHz)
maxtau = (1 / (2*dfbin*u.MHz)).to(u.microsecond)

vmin = np.median(Splot) - 0.1
vmax = vmin + 1.6
eta = 1.8
ftrange = np.linspace(-3, 3, 1000)

plt.figure(figsize=(6, 8))
plt.imshow(Splot.T, aspect='auto', vmin=vmin, vmax=vmax, origin='lower', #cmap='magma',
          extent=[-maxft.value, maxft.value, -maxtau.value, maxtau.value])
plt.ylabel(r'$\tau$ ($\mu$s)', fontsize=16)
plt.xlabel(r'$f_{D}$ (mHz)', fontsize=16)
plt.plot(ftrange, eta*ftrange**2, color='w', linestyle='--', alpha=0.7
        )
plt.ylim(-maxtau.value, maxtau.value)
plt.xlim(-8, 8)

plt.show()


