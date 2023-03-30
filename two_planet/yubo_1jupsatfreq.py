'''
calculate the Jupiter saturn precession frequency using the formalism of
Appendix A of Su & Lai 2022
'''

import numpy as np

AJUP = 5.202        # AU
ASAT = 9.537        # AU
MJUP = 0.000954     # Msun
MSAT = 0.000286     # Msun
MSUN = 1            # Msun
G = 4 * np.pi**2    # in units of (Msun, AU, yr)

def get_laplace_exact(a, j=1):
    '''
    b_{3/2}^j(alpha) / 3 * alpha, determined numerically
    '''
    psi = np.linspace(0, 2 * np.pi, 10000)
    integrand = np.cos(j * psi) / (
        1 - 2 * a * np.cos(psi) + a**2)**(3/2)
    return np.mean(integrand) * 2 / (3 * a)

def get_laplace_lazy(a):
    ''' compare to basic formula '''
    return 3 * a * (1 + 15 * a**2 / 8 + 175 * a**4 / 65)

def get_laplace(a):
    return get_laplace_exact(a)

if __name__ == '__main__':
    a_in = AJUP
    a_out = ASAT
    alpha = a_in / a_out

    # Eq (A2), j = Jupiter, k = Saturn
    omega_JS = (
        MSAT / (4 * MSUN)
        * AJUP * a_in / a_out**2
        * np.sqrt(G * MSUN / AJUP**3)
        * get_laplace(alpha) * 3 * alpha
    )
    # Eq (A2), j = Saturn, k = Jupiter
    omega_SJ = (
        MJUP / (4 * MSUN)
        * ASAT * a_in / a_out**2
        * np.sqrt(G * MSUN / ASAT**3)
        * get_laplace(alpha) * 3 * alpha
    )
    mat = np.array([
        [-omega_JS, omega_JS],
        [omega_SJ, -omega_SJ]
    ])
    eigs, eigv = np.linalg.eig(mat)
    prec_freq = eigs[np.where(np.abs(eigs) > 1e-10)[0]]
    print('Jupiter & Saturn Precession Frequency is:', 2 * np.pi / prec_freq)
    print('Mode amplitudes:', eigv)
