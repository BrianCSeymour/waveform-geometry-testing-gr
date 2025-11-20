import jax.numpy as jnp
import jax 
from jax import grad, vmap
import numpy as np
from jimgw.waveform import RippleIMRPhenomD, RippleIMRPhenomPv2
from jimgw.detector import H1, L1, V1
jax.config.update("jax_enable_x64", True)

import scipy.interpolate as interp
import scipy.integrate as integ
import scipy.linalg as sla
import pycbc.conversions

from functools import partial


import astropy.units as u
from astropy import constants as const

Ms = (u.Msun * const.G / const.c**3 ).si.value

import matplotlib as mpl
from matplotlib.legend_handler import HandlerLine2D, HandlerPatch



# ------------ Fisher Stuff --------

def read_mag(freq, fileName):
    f_tf, mag_tf = np.loadtxt(fileName, unpack=True)
    
    idx = jnp.where(f_tf>0)
    mag_tf = mag_tf[idx]
    f_tf = f_tf[idx]

    mag_func = interp.interp1d(jnp.log(f_tf), jnp.log(mag_tf), kind='linear', bounds_error=False, fill_value=np.inf)
    mag_out = jnp.exp(mag_func(jnp.log(freq)))
    return mag_out


def innprod(hf1, hf2, psd, freqs):
    prod = 2. * jax.scipy.integrate.trapezoid( (jnp.conj(hf1) * hf2 + hf1 * jnp.conj(hf2)) / psd , freqs)
    return prod

@jax.jit
def fish(freqs, dh, par, idx_par, psd, log_flag):
    n_pt = len(freqs)
    n_dof = len(idx_par)

    dh_arr = jnp.zeros([n_dof, n_pt], dtype=jnp.complex128)

    # Convert idx_par to a list for static looping
    idx_list = list(idx_par.keys())
    for idx in idx_list:
        idx_position = idx_par[idx]
        dh_arr = dh_arr.at[idx_position, :].set(dh[idx])

        # Use jax.lax.cond for conditional multiplication
        dh_arr = dh_arr.at[idx_position, :].set(
            jax.lax.cond(
                log_flag[idx],
                lambda x: x * par[idx],
                lambda x: x,
                dh_arr[idx_position, :]
            )
        )

    gamma = jnp.zeros([n_dof, n_dof], dtype=jnp.float64)

    # Use static loops
    for i in range(n_dof):
        for j in range(i, n_dof):
            gamma = gamma.at[i, j].set(
                jnp.real(innprod(dh_arr[i, :], dh_arr[j, :], psd, freqs))
            )
        for j in range(i):
            gamma = gamma.at[i, j].set(jnp.conj(gamma[j, i]))

    return gamma

@jax.jit
def bias_innerprod(freqs, dh, par, Dh, idx_par, psd, log_flag):
    # Initialize a zero array for bias with the correct length
    n_dof = len(idx_par)
    bias = jnp.zeros(n_dof, dtype=jnp.float64)
    
    # Loop through parameters in idx_par, avoiding the need for sorting
    for param, index in idx_par.items():
        res_value = jnp.real(innprod(dh[param], Dh, psd, freqs))
        res_value = jax.lax.cond(
            log_flag[param],
            lambda x: x * par[param],
            lambda x: x,
            res_value
        )
        bias = bias.at[index].set(res_value)
    
    return bias

# ------------ wavefrom derivs --------


def get_FI(freqs, red_param, idx_par, psd, log_flag):
    dh_H1  = get_dh_H1(red_param, freqs)
    dh_L1  = get_dh_L1(red_param, freqs)
    dh_V1  = get_dh_V1(red_param, freqs)
    
    fi_H1 = fish(freqs, dh_H1, red_param, idx_par, psd, log_flag)
    fi_L1 = fish(freqs, dh_L1, red_param, idx_par, psd, log_flag)
    fi_V1 = fish(freqs, dh_V1, red_param, idx_par, psd, log_flag)
    return fi_H1, fi_L1, fi_V1

def get_snrs(freqs, red_param, psd):
    h_H1   = get_h_H1(red_param, freqs)
    h_L1   = get_h_L1(red_param, freqs)
    h_V1   = get_h_V1(red_param, freqs)

    snr_H1 = jnp.real(innprod(h_H1, h_H1, psd, freqs)**(1/2))
    snr_L1 = jnp.real(innprod(h_L1, h_L1, psd, freqs)**(1/2))
    snr_V1 = jnp.real(innprod(h_V1, h_V1, psd, freqs)**(1/2))

    return snr_H1, snr_L1, snr_V1, (snr_H1**2 + snr_L1**2 + snr_V1**2)**(1/2)



def get_FI_ppe(freqs, red_param, idx_par, psd, log_flag, k):
    dh_H1  = get_dh_H1(red_param, freqs)
    dh_L1  = get_dh_L1(red_param, freqs)
    dh_V1  = get_dh_V1(red_param, freqs)
    
    h_H1   = get_h_H1(red_param, freqs)
    h_L1   = get_h_L1(red_param, freqs)
    h_V1   = get_h_V1(red_param, freqs)

    dpsi_ppe = get_dpsi_ppe(freqs, red_param, k)

    ########## alot more work needs to be done. I think I need to calculate the value for ppe first then recompute the FI matrix at that point 
    dh_H1["phi_k"] = 1j*dpsi_ppe*h_H1
    dh_L1["phi_k"] = 1j*dpsi_ppe*h_L1
    dh_V1["phi_k"] = 1j*dpsi_ppe*h_V1

    fi_H1 = fish(freqs, dh_H1, red_param, idx_par, psd, log_flag)
    fi_L1 = fish(freqs, dh_L1, red_param, idx_par, psd, log_flag)
    fi_V1 = fish(freqs, dh_V1, red_param, idx_par, psd, log_flag)


    return fi_H1, fi_L1, fi_V1

# ------------ dephasing terms --------

def get_dpsi_ppe_inner(freqs, par, k):

    Mc = par["M_c"]
    η = par["eta"]
    
    M = pycbc.conversions.mtotal_from_mchirp_eta(Mc,η)*Ms
    phi0 = 1
    phi1 = 0
    phi2 = 3715/756 + 55/9*η
    phi3 = -16*np.pi
    phi4 =  15293365/508032+27145 *η /503+ 3085 *η**2 / 72
    phi5 = ((38645 * np.pi / 756) - (65 * np.pi * η / 9))
    pi = np.pi
    gamma_e = np.euler_gamma
    eta = η
    phi6 =  (11583231236531 / 4694215680) - (6848 * gamma_e / 21) - (640 * pi**2 / 3) + (-15737765635 / 3048192 + 2255 * pi**2 / 12) * eta + 76055 * eta**2 / 1728 -     127825 * eta**3 / 1296 
    phi7 = (77096675 * pi / 254016) + (378515 * pi * eta / 1512) - (74045 * pi * eta**2 / 756) 
    # k = kargs["k"]
    # δφ_k = par['dphi_k']
    δφ_k = 1
    
    # these come from eq B3 of 1508.07253
    if k == -2:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == -1:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 0:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi0 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 1:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 2:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi2 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 3:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi3 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 4:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi4 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 5:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi5 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 6:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi6 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 7:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi7 * δφ_k * (np.pi * freqs * M)**(k/3)
    else:
        print(k)
        print("power error defn")
    
    return dpsi

def get_dpsi_ppe(freqs, par, k):
    fend = 0.04257918562317578 # to match eob file 
    fstart = 0.004432985313285457
    Mc = par["M_c"]
    eta = par["eta"]
    fend = fend/pycbc.conversions.mtotal_from_mchirp_eta(Mc,eta)/Ms
    fstart = fstart/pycbc.conversions.mtotal_from_mchirp_eta(Mc,eta)/Ms

    dpsi = get_dpsi_ppe_inner(freqs, par, k)
    # dpsi[freqs>fend] = get_dpsi_ppe_inner(fend, par) # numpy version
    dpsiend = get_dpsi_ppe_inner(fend, par, k)
    dpsi = jnp.where(freqs>fend, get_dpsi_ppe_inner(fend, par, k), dpsi)- dpsiend # jax version
    # dpsi = dpsi.at[freqs>fend].set(get_dpsi_ppe_inner(fend, par, k)) 
    return dpsi


# -------- class -----------

class Fisher(object):
    def __init__(self, fmin = 20, fmax = 1000, n_freq = 2000., waveform="IMRPhenomPv2", f_ref=20, fisher_parameters=None, psdid="O3"):
        self.fmin = fmin
        self.fmax = fmax
        self.freqs = jnp.logspace(jnp.log10(fmin), jnp.log10(fmax), num = int(n_freq))
        if waveform == "IMRPhenomD":
            self.waveform = RippleIMRPhenomD(f_ref=f_ref)
            self.paramdiffgr = ["M_c", "eta", "d_L", "ra", "dec", "iota", "psi", "t_c", "phase_c"]
            self.paramdiffgr_latex = [r"$M_c$", r"$\eta$", r"$d_L$", r"$\text{ra}$", r"$\text{dec}$", r"$\iota$", r"$\psi$", r"$t_c$", r"$\phi_c$"]
        elif waveform == "IMRPhenomPv2":
            self.waveform = RippleIMRPhenomPv2(f_ref=f_ref)
            self.paramdiffgr = ["M_c", "eta", "d_L", "ra", "dec", "iota", "psi", "t_c", "phase_c", 's1_z', 's1_x']
            self.paramdiffgr_latex = [r"$M_c$", r"$\eta$", r"$d_L$", r"$\text{ra}$", r"$\text{dec}$", r"$\iota$", r"$\psi$", r"$t_c$", r"$\phi_c$", r"$s_{1x}$", r"$s_{1z}$"]
        self.paramgr = ["M_c", "eta", "d_L", "ra", "dec", "iota", "psi", "t_c", "phase_c", 's1_x', 's1_y', 's1_z', 's2_x', 's2_y', 's2_z','gmst', 'epoch']
        self.k2str = {k: f"phi_{k}" for k in range(-2, 8)}
        self.str2k = {v: k for k, v in self.k2str.items()}

        xvals = [ 3.00000000e+01,  2.46559096e-01,  3.90000000e+02,
        1.69254929e+00,  9.39189162e-01,  2.35481238e+00,
       -1.20559143e+00,  0.00000000e+00,  0.00000000e+00,
        1.00000000e-06,  1.00000000e-06,  1.00000000e-06,
        1.00000000e-06,  1.00000000e-06,  1.00000000e-06,
        0.00000000e+00,  0.00000000e+00]
        xkeys = ['M_c','eta','d_L','ra','dec','iota','psi','t_c','phase_c','s1_x','s1_y','s1_z','s2_x','s2_y','s2_z','gmst','epoch']
        xtest = dict(zip(xkeys, xvals))
        self.jitted_h1 = jax.jit(lambda x: self.get_h_slow(x, H1))
        self.jitted_h2 = jax.jit(lambda x: self.get_h_slow(x, L1))
        self.jitted_h3 = jax.jit(lambda x: self.get_h_slow(x, V1))

        idx_diff = tuple(i for i, key in enumerate(self.paramgr) if key in self.paramdiffgr)
        self.jitted_dh1 = jax.jit(jax.jacfwd(lambda *args: self._get_h_args(*args, det=H1), argnums = idx_diff))
        self.jitted_dh2 = jax.jit(jax.jacfwd(lambda *args: self._get_h_args(*args, det=L1), argnums = idx_diff))
        self.jitted_dh3 = jax.jit(jax.jacfwd(lambda *args: self._get_h_args(*args, det=V1), argnums = idx_diff))

        self.det1 = H1
        self.det2 = L1
        self.det3 = V1



        self.psdid = psdid
        self.psdO3 = read_mag(self.freqs, "curves/o3_l1.txt")**2
        self.psdCE = read_mag(self.freqs, "curves/ce1.txt")**2

        def reset_matplotlib():
            mpl.rcdefaults()
            default_handler_map = {
                mpl.lines.Line2D: HandlerLine2D(numpoints=1),
                mpl.patches.Patch: HandlerPatch()
            }
            mpl.legend.Legend.update_default_handler_map(default_handler_map)
        reset_matplotlib()

    def get_h_slow(self, x, det):
        ff = self.freqs
        h_sky = self.waveform(ff, x)
        align_time = jnp.exp(-1j * 2 * jnp.pi * ff * (x['epoch'] + x['t_c']))
        signal = det.fd_response(ff, h_sky, x) * align_time
        return signal

    def _get_h_args(self, *args, det):
        keys = self.paramgr
        y = dict(zip(keys, args)) 
        return self.get_h_slow(y, det)

    def get_h_gr(self, x):
        return {'H1': self.jitted_h1(x), 'L1': self.jitted_h2(x), 'V1': self.jitted_h3(x)}

    def get_dh_gr(self, x):
        xvalues = list(x.values())
        dh = {
            'H1': dict(zip(self.paramdiffgr, self.jitted_dh1(*xvalues))),
            'L1': dict(zip(self.paramdiffgr, self.jitted_dh2(*xvalues))),
            'V1': dict(zip(self.paramdiffgr, self.jitted_dh3(*xvalues)))
        }
        return dh

    def get_snrs_gr(self, x):
        if self.psdid == "O3":
            psd = self.psdO3
        elif self.psdid == "CE":
            psd = self.psdCE
        freqs = self.freqs

        h = self.get_h_gr(x)
        snrs = {d: jnp.real(innprod(h[d], h[d], psd, freqs)**(1/2)) for d in ["H1", "L1", "V1"]}
        
        snrs['total'] = (snrs['H1']**2 + snrs['L1']**2 + snrs['V1']**2)**(1/2)
        self.snr1 = snrs['H1']
        self.snr2 = snrs['L1']
        self.snr3 = snrs['V1']
        self.snrt = snrs['total']
        return snrs

    def compute_joint_fish(self, x, paramx, k = None):
        if self.psdid == "O3":
            psd = self.psdO3
        elif self.psdid == "CE":
            psd = self.psdCE
        dh = self.get_dh_gr(x)
        idx_par = {paramx[i] : i for i in range(len(paramx))}
        log_flag =  {paramx[i] : 0 for i in range(len(paramx))}; log_flag["M_c"] = 1; log_flag["d_L"] = 1
        freqs = self.freqs
        self.idx_par = idx_par
        self.log_flag = log_flag

        # saving the value of fend for ppe attachment
        fend = 0.04257918562317578 
        Mc = x["M_c"]
        eta = x["eta"]
        self.fend = fend/pycbc.conversions.mtotal_from_mchirp_eta(Mc,eta)/Ms

        if k is not None:
            h = self.get_h_gr(x)

            dpsi_ppe = get_dpsi_ppe(freqs, x, k)
            for d in ["H1", "L1", "V1"]:
                dh[d]["phi_k"] = 1j * dpsi_ppe * h[d]
        
        self.fi1 = fish(freqs, dh["H1"], x, idx_par, psd, log_flag)
        self.fi2 = fish(freqs, dh["L1"], x, idx_par, psd, log_flag)
        self.fi3 = fish(freqs, dh["V1"], x, idx_par, psd, log_flag)
        self.fi = self.fi1 + self.fi2 + self.fi3
        return self.fi
    
    def compute_biasip(self, x, Dh, paramx, k = None):
        if self.psdid == "O3":
            psd = self.psdO3
        elif self.psdid == "CE":
            psd = self.psdCE
        dh = self.get_dh_gr(x)
        idx_par = {paramx[i] : i for i in range(len(paramx))}
        log_flag =  {paramx[i] : 0 for i in range(len(paramx))}; log_flag["M_c"] = 1; log_flag["d_L"] = 1
        freqs = self.freqs

        if k is not None:
            h = self.get_h_gr(x)

            dpsi_ppe = get_dpsi_ppe(freqs, x, k)
            for d in ["H1", "L1", "V1"]:
                dh[d]["phi_k"] = 1j * dpsi_ppe * h[d]
        

        self.biasip1 = bias_innerprod(freqs, dh["H1"], x, Dh["H1"], idx_par, psd, log_flag)
        self.biasip2 = bias_innerprod(freqs, dh["L1"], x, Dh["L1"], idx_par, psd, log_flag)
        self.biasip3 = bias_innerprod(freqs, dh["V1"], x, Dh["V1"], idx_par, psd, log_flag)
        # bias = sum(bias_innerprod(freqs, dh[d], x, dh[d], idx_par, psd, log_flag) for d in ["H1", "L1", "V1"])
        self.biasip = self.biasip1 + self.biasip2 + self.biasip3
        
        return self.biasip

    def compute_fisher_raw(self, dh, x, param):
        if self.psdid == "O3":
            psd = self.psdO3
        elif self.psdid == "CE":
            psd = self.psdCE
            
        idx_par = {param[i] : i for i in range(len(param))}
        log_flag =  {param[i] : 0 for i in range(len(param))}; log_flag["M_c"] = 1; log_flag["d_L"] = 1
        freqs = self.freqs
        self.idx_par = idx_par
        self.log_flag = log_flag

                # saving the value of fend for ppe attachment
        fend = 0.04257918562317578 
        Mc = x["M_c"]
        eta = x["eta"]
        self.fend = fend/pycbc.conversions.mtotal_from_mchirp_eta(Mc,eta)/Ms
        
        fi1 = fish(freqs, dh["H1"], x, idx_par, psd, log_flag)
        fi2 = fish(freqs, dh["L1"], x, idx_par, psd, log_flag)
        fi3 = fish(freqs, dh["V1"], x, idx_par, psd, log_flag)
        fi = fi1 + fi2 + fi3
        return fi
    
    def compute_biasip_raw(self, dh, Dh, x, paramx):
        if self.psdid == "O3":
            psd = self.psdO3
        elif self.psdid == "CE":
            psd = self.psdCE
        idx_par = {paramx[i] : i for i in range(len(paramx))}
        log_flag =  {paramx[i] : 0 for i in range(len(paramx))}; log_flag["M_c"] = 1; log_flag["d_L"] = 1
        freqs = self.freqs

        biasip1 = bias_innerprod(freqs, dh["H1"], x, Dh["H1"], idx_par, psd, log_flag)
        biasip2 = bias_innerprod(freqs, dh["L1"], x, Dh["L1"], idx_par, psd, log_flag)
        biasip3 = bias_innerprod(freqs, dh["V1"], x, Dh["V1"], idx_par, psd, log_flag)
        # bias = sum(bias_innerprod(freqs, dh[d], x, dh[d], idx_par, psd, log_flag) for d in ["H1", "L1", "V1"])
        biasip = biasip1 + biasip2 + biasip3
        
        return biasip