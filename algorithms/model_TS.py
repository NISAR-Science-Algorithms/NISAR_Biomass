'''
Estimate the Gamma (T) and S terms in forward radar model (Saatchi 1997)
'''

from numpy.lib.scimath import sqrt as csqrt
import numpy as np


def get_Gamma(inc_angle, ks_frqcxrghnss, e_epsilon):
    sin0 = np.sin(inc_angle * np.pi / 180. )
    cos0 = np.cos(inc_angle * np.pi / 180. )
    R_H = (cos0 - csqrt(e_epsilon - sin0**2)) / \
            (cos0 + csqrt(e_epsilon - sin0**2))
    R_V = (e_epsilon * cos0 - csqrt(e_epsilon - sin0**2)) / \
            (e_epsilon * cos0 + csqrt(e_epsilon - sin0**2))
    e0 = np.exp(-4 * ks_frqcxrghnss**2 * cos0**2)
    # print(e0)

    Gamma_HH = np.absolute(R_H * R_H.conj()) * e0
    Gamma_HV = np.absolute(R_H * R_V.conj()) * e0
    Gamma_VV = np.absolute(R_V * R_V.conj()) * e0
    return Gamma_HH, Gamma_HV, Gamma_VV


def get_S(inc_angle, ks_frqcxrghnss, e_epsilon):
    sin0 = np.sin(inc_angle * np.pi / 180. )
    cos0 = np.cos(inc_angle * np.pi / 180. )
    R_H = (cos0 - csqrt(e_epsilon - sin0**2)) / \
            (cos0 + csqrt(e_epsilon - sin0**2))
    R_V = (e_epsilon * cos0 - csqrt(e_epsilon - sin0**2)) / \
            (e_epsilon * cos0 + csqrt(e_epsilon - sin0**2))
    R0 = np.absolute((1-csqrt(e_epsilon)) / (1+csqrt(e_epsilon)))**2
    g = 0.7 * (1 - np.exp(-0.65 * ks_frqcxrghnss**1.8))
    q = 0.23 * (1 - np.exp(-ks_frqcxrghnss)) * np.sqrt(R0)
    p = 1 - (2*inc_angle*np.pi/180/np.pi) ** (1/3/R0) * np.exp(-ks_frqcxrghnss)

    S_HH = g*np.sqrt(p) * cos0**3 * (np.absolute(R_H)**2 + np.absolute(R_V)**2)
    S_VV = g/np.sqrt(p) * cos0**3 * (np.absolute(R_H)**2 + np.absolute(R_V)**2)
    S_HV = q * S_VV
    return S_HH, S_HV, S_VV


def get_S_sm(inc_angle, ks_frqcxrghnss, sm):
    M = np.real(0.11 * sm ** 0.7)
    A = np.real(np.cos(inc_angle * np.pi / 180) ** 2.2)
    K = np.real(1 - np.exp(-0.32 * ks_frqcxrghnss ** 1.8))
    S_HV = M * A * K
    return S_HV