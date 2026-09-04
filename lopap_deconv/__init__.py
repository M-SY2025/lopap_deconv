"""
lopap_deconv — LOPAP signal deconvolution using a dual inverse-Gaussian IRF.

Public API
----------
IO:
    read_lopap_raw, read_lopap_manifest, read_lopap_test_list
Preprocessing:
    cfg_default, pack_lopap_only, estimate_tau_from_lopap
Training:
    fit_irf_sd_train
Deconvolution:
    rtd_ig, irf_from_theta, estimate_c_given_kernel, run_fixed_irf_recon
Plotting:
    plot_irf_diagnostics, plot_recon_diagnostics
"""

from .io import read_lopap_raw, read_lopap_manifest, read_lopap_test_list
from .preprocess import cfg_default, pack_lopap_only, estimate_tau_from_lopap
from .train import fit_irf_sd_train
from .deconvolve import rtd_ig, irf_from_theta, estimate_c_given_kernel, run_fixed_irf_recon
from .plotting import plot_irf_diagnostics, plot_recon_diagnostics

__version__ = "0.1.0"
