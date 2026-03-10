#Standard python imports
from __future__ import division
from pathlib import Path
from optparse          import OptionParser
from scipy.interpolate import interp1d, LinearNDInterpolator
try:
    from scipy.signal import butter, filtfilt, tukey
except ImportError:
    from scipy.signal import butter, filtfilt
    from scipy.signal.windows import tukey
import numpy as np, os, scipy.linalg as sl, traceback, warnings

#LVC imports
import lal

#Package internal imports
try:
    import surfinBH

    def final_state_surfinBH(Mtot, q, chi1, chi2, f_ref):

        """
        
        This function computes the final mass and spin of a binary black hole system using the surfinBH package.

        Parameters
        ----------

        Mtot : float
            Total mass of the binary system in solar masses.
        q : float
            Mass ratio of the binary system.
        chi1 : float
            Dimensionless spin of the more massive black hole.
        chi2 : float
            Dimensionless spin of the less massive black hole.
        f_ref : float
            Reference frequency in Hz.

        Returns
        -------

        Mf : float
            Final mass of the binary system in solar masses.
        af : float
            Final dimensionless spin of the binary system.
        
        """

        fit = surfinBH.LoadFits('NRSur7dq4Remnant')
        #Adapt q to surrogate conventions.
        if(q < 1.): q = 1./q
        #This is the orbital frequency (hence the missing factor of 2, since f_ref is the GW frequency) in units of rad/M (seehttps://github.com/vijayvarma392/surfinBH/blob/master/examples/example_7dq4.ipynb for more info).
        omega_ref = np.pi*f_ref/Mtot

        Mf_sBH, _ = fit.mf(q, chi1, chi2, omega0=omega_ref)
        af_sBH, _ = fit.chif(q, chi1, chi2, omega0=omega_ref)
        Mf        = Mtot*Mf_sBH
        af        = np.sqrt(af_sBH[0]**2+af_sBH[1]**2+af_sBH[2]**2)

        return Mf, af

except:
    warnings.warn("* The `surfinBH` package is not automatically installed due to possible conflicts. If you wish to use its functionalities, it needs to be installed separately.")

def print_section(name):

    """
    
    This function prints a section header.

    Parameters
    ----------

    name : str
        Name of the section.

    Returns
    -------

    Nothing, but prints a section header.

    """

    pad = "#" * len(name)

    print('\n\n\n##{}##'.format(pad))
    print('# \u001b[\u001b[38;5;39m{}\u001b[0m #'.format(name))
    print('##{}##\n'.format(pad))

    return

def print_subsection(name):

    """

    This function prints a subsection header.

    Parameters
    ----------

    name : str
        Name of the subsection. 

    Returns
    -------

    Nothing, but prints a subsection header.

    """

    pad = "-" * len(name)
    
    print('\n--{}--'.format(pad))
    print('- \u001b[\u001b[38;5;39m{}\u001b[0m -'.format(name))
    print('--{}--\n'.format(pad))

    return

def print_out_of_bounds_warning(name):

    """
    
    This function prints a warning message when the injected values are outside the prior bounds.

    Parameters
    ----------

    name : str
        Name of the parameter.
    
    Returns
    -------

    Nothing, but prints a warning message.
    
    """
    
    print('\n\n######################### WARNING ############################')
    print('# The {} injected values are outside the prior bounds. #'.format(name))
    print('##############################################################\n\n')

    return

def print_fixed_parameters(fixed_params):

    """

    This function prints the fixed parameters.

    Parameters
    ----------

    fixed_params : dict
        Dictionary containing the fixed parameters.

    Returns
    -------

    Nothing, but prints the fixed parameters.

    """

    if not fixed_params:
        print('\n* No parameter was fixed.')
    else:
        for name in fixed_params.keys():
            print('{} : {}'.format(name.ljust(len('cos_altitude')), fixed_params[name]))

    return 

def set_prefix(warning_message=True):
    
    """
        Set the prefix path for the data files.

        Parameters
        ----------

        warning_message : bool
            If True, a warning message is printed if the environment variable is not set.

        Returns
        -------

        prefix : str
            Path to the data files.

    """
    
    # Check environment
    try:
        prefix = os.path.join(os.environ['PYRING_PREFIX'], 'pyRing')
    except KeyError:
        prefix = ''
        if(warning_message):
            warnings.warn("The requested functionality requires data not included in the package. Please set a $PYRING_PREFIX variable which contains the path to such data. This can be done by setting 'export PYRING_PREFIX= yourpath' in your ~/.bashrc file. Typically, PYRING_PREFIX contains the path to the clone of the repository containing the source code.")
    return prefix

def import_datafile_path(filename):

    """

    This function returns the path to a data file, including the pyRing relative path.

    Parameters
    ----------

    filename : str
        Name of the data file.

    Returns
    ------- 

    package_path : str
        Path to the data file.
    
    """

    package_path = str((Path(__file__).resolve().parent / filename).resolve())

    return package_path

def check_NR_dir():

    """

    This function checks if the directory LVC-NR waveforms is present.

    Parameters
    ----------

    None.

    Returns
    -------

    Nothing, but raises an exception if the directory is not present.

    """
    
    PYRING_PREFIX = set_prefix()
    if not(os.path.isdir(os.path.join(PYRING_PREFIX, 'data/NR_data/lvcnr-lfs'))):
        raise Exception("pyRing supports NR injections using the LVC-NR injection infrastructure. If you wish to inject NR simulations, please clone the LVC-NR injection infrastructure repository, located here: https://git.ligo.org/waveforms/lvcnr-lfs \nFor tutorials and info on how to use the LVC NR injection infrastructure see:\n - https://git.ligo.org/sebastian-khan/waveform-f2f-berlin/blob/master/notebooks/2017WaveformsF2FTutorial_NRDemo.ipynb \n - https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Waveforms/NR/InjectionInfrastructure \n - https://arxiv.org/pdf/1703.01076.pdf")
    return

def review_warning():

    """

    This function prints a warning message if the code block is not reviewed.

    Parameters
    ----------

    None.

    Returns
    -------

    Nothing, but prints a warning message.

    """

    print("* Warning: You are using a code block which is not reviewed. Non-reviewed code cannot be used for producing LVC results.")

def qnm_interpolate(s,l,m,n):

    """

    This function interpolates the complex frequencies of the QNM modes.

    Parameters
    ----------

    s : int
        Spin-weight of the mode.
    l : int
        Orbital angular momentum of the mode.
    m : int 
        Azimuthal angular momentum of the mode.
    n : int
        Radial quantum number of the mode.
    
    Returns
    -------

    w_r : interp1d object
        Interpolant function of the real part of the complex frequency.
    w_i : interp1d object
        Interpolant function of the imaginary part of the complex frequency.

    """

    assert not(np.abs(m) > l), "QNM interpolation: m cannot be greater than l in modulus."
    assert (s==0 or s==1 or s==2), "QNM interpolation: supported s values are [0,1,2] ({} was passed)."
    try:
        PYRING_PREFIX = set_prefix()
        # Adapt to Berti conventions (start counting from 1): n -> n+1
        if (m<0):
            af, w_r, w_i, _, _ = np.loadtxt(os.path.join(PYRING_PREFIX,'data/NR_data/Kerr_BH/s{}l{}'.format(s, l),'n{}l{}mm{}.dat'.format(n+1, l, np.abs(m))), unpack=True)
        else:
            af, w_r, w_i, _, _ = np.loadtxt(os.path.join(PYRING_PREFIX,'data/NR_data/Kerr_BH/s{}l{}'.format(s, l),'n{}l{}m{}.dat'.format(n+1, l, m)), unpack=True)
    except:
        raise Exception("If you wish to use perturbation theory NR data not stored on the repository, please download the corresponding files from `https://pages.jh.edu/~eberti2/ringdown` and place them within directories with the following structure: `pyring_installation_directory/pyring/pyRing/data/NR_data/Kerr_BH/sXlY`, where `X` is the value of the spin perturbation considered and `Y` the value of the `l` QNM index. This feature requires the installation of the source code and is not currently supported by pip.\nQNM interpolation failed with error: {}.".format(traceback.print_exc()))
    return interp1d(af, w_r, kind='cubic'), interp1d(af, w_i, kind='cubic')

def qnm_interpolate_KN(s,l,m,n):

    """

    This function interpolates the complex frequencies of the QNM modes for the Kerr-Newman spacetime.

    Parameters
    ----------

    s : int
        Spin-weight of the mode.
    l : int
        Orbital angular momentum of the mode.
    m : int
        Azimuthal angular momentum of the mode.
    n : int
        Radial quantum number of the mode.
    
    Returns
    -------

    w_r : interp1d object
        Interpolant function of the real part of the complex frequency.
    w_i : interp1d object
        Interpolant function of the imaginary part of the complex frequency.
    
    """

    assert not(np.abs(m) > l), "QNM interpolation: m cannot be greater than l in modulus."
    assert (s==0 or s==1 or s==2), "QNM interpolation: supported s values are [0,1,2] ({} was passed)."
    try:
        PYRING_PREFIX = set_prefix()
        if (m<0):
            Q, af, w_r, w_i = np.loadtxt(os.path.join(PYRING_PREFIX,'data/NR_data/KN_BH/s{}l{}'.format(s, l),'n{}l{}mm{}.dat'.format(n, l, np.abs(m))), unpack=True)
        else:
            Q, af, w_r, w_i = np.loadtxt(os.path.join(PYRING_PREFIX,'data/NR_data/KN_BH/s{}l{}'.format(s, l),'n{}l{}m{}.dat'.format(n, l, m)), unpack=True)
    except:
        raise Exception("Loading KN data failed. Exiting.".format(traceback.print_exc()))

    coords   = np.column_stack((af,Q))
    interp_r = LinearNDInterpolator(coords, w_r)
    interp_i = LinearNDInterpolator(coords, w_i)

    return interp_r, interp_i

def qnm_interpolate_braneworld(s,l,m,n):

    """

    This function interpolates the complex frequencies of the QNM modes for the Braneworld spacetime.

    Parameters
    ----------

    s : int
        Spin-weight of the mode.
    l : int
        Orbital angular momentum of the mode.
    m : int
        Azimuthal angular momentum of the mode.
    n : int
        Radial quantum number of the mode.
    
    Returns
    -------

    w_r : interp1d object
        Interpolant function of the real part of the complex frequency.
    w_i : interp1d object
        Interpolant function of the imaginary part of the complex frequency.
    
    """

    print('\nPerforming interpolation of ringdown Braneworld complex frequencies of {}{}{}{} mode.'.format(s,l,m,n))
    assert not(np.abs(m) > l), "QNM interpolation: m cannot be greater than l in modulus."
    assert (s==0 or s==1 or s==2), "QNM interpolation: supported s values are [0,1,2] ({} was passed)."
    try:
        PYRING_PREFIX = set_prefix()
        if (m<0):
            af, qf, w_r, w_i = np.loadtxt(os.path.join(PYRING_PREFIX,'data/NR_data/Braneworld/s{}l{}'.format(s, l),'n{}l{}mm{}.dat'.format(n, l, np.abs(m))), unpack=True)
        else:
            af, qf, w_r, w_i = np.loadtxt(os.path.join(PYRING_PREFIX,'data/NR_data/Braneworld/s{}l{}'.format(s, l),'n{}l{}m{}.dat'.format(n, l, m)), unpack=True)
    except:
        raise Exception("Loading Braneworld data failed. Exiting.".format(traceback.print_exc()))

    coords   = np.column_stack((af, qf))
    interp_r = LinearNDInterpolator(coords, w_r)
    interp_i = LinearNDInterpolator(coords, w_i)

    return interp_r, interp_i


def check_modes_naming_scheme(modes, quad_modes):

    for (s,l,m,n) in modes:
        if(s>9 or l>9 or m>9 or n>9): raise ValueError("The naming scheme for Kerr amplitudes is currently only compatible with s,l,m,n <= 9. Aborting.")
    if quad_modes is not None:
        for quad_term in quad_modes.keys():
            for ((s,l,m,n),(s1,l1,m1,n1),(s2,l2,m2,n2)) in quad_modes[quad_term]: 
                if(s>9 or l>9 or m>9 or n>9 or s1>9 or l1>9 or m1>9 or n1>9 or s2>9 or l2>9 or m2>9 or n2>9): raise ValueError("The naming scheme for Kerr amplitudes is currently only compatible with s,l,m,n <= 9. Aborting.")

    return

def construct_full_modes(modes, quad_modes):

    """

    This function constructs the full list of modes (combining linear and quadratic modes) to be used in the ringdown fitting procedure.

    Parameters
    ----------

    modes : list
        List of linear modes to be used in the fitting procedure.
    quad_modes : dict
        Dictionary of quadratic modes to be used in the fitting procedure.
    
    Returns
    -------

    modes_full : list
        Combined list of linear and quadratic modes to be used in the fitting procedure.
    
    """

    modes_full = []
    for mode in modes: modes_full.append(mode)
    if quad_modes is not None:
        for quad_term in quad_modes.keys():
            for mode in quad_modes[quad_term]: 
                modes_full.append(mode[0])
                modes_full.append(mode[1])
                modes_full.append(mode[2])

    # Remove duplicates.
    modes_full = list(dict.fromkeys(modes_full))

    return modes_full 

def bandpass_around_ringdown(strain, dt, f_min, mf, alpha_window=0.1):

    """

    This function bandpasses the strain around the ringdown frequency of the BH.

    Parameters
    ----------

    strain : array
        Strain time series.
    dt : float
        Time step of the strain time series.
    f_min : float
        Minimum frequency of the bandpass.
    mf : float
        Final mass of the BH.
    alpha_window : float
        Tukey window parameter.

    Returns
    -------

    strain : array
        Bandpassed strain time series.
    
    """

    srate_dt = 1./dt
    Nt       = len(strain)

    if not(mf==0.0):
        # Typical ringdown frequency (220 mode) for a BH with af=0.7 (Berti+ fit), only for plotting purposes.
        central_f_ringdown = ((lal.C_SI*lal.C_SI*lal.C_SI)/(2.*np.pi*lal.G_SI*mf*lal.MSUN_SI)) * (1.5251-1.1568*(1-0.7)**0.1292)

        window = tukey(Nt, alpha=alpha_window)
        strain = strain*window
        bb, ab = butter(4, [f_min/(0.5*srate_dt), (central_f_ringdown*2.)/(0.5*srate_dt)], btype='band')
        strain = filtfilt(bb, ab, strain)

    return strain

def whiten_TD(x, cholesky_L, method='solve-triangular'):

    """

    Whiten in the time domain the time series x using the Cholesky decomposition of the covariance matrix.

    Parameters
    ----------

    x : array
        Time series to be whitened.
    cholesky_L : array
        Cholesky decomposition of the covariance matrix.
    method : str
        Method to be used to solve the linear system.
    
    Returns
    -------

    x_whitened : array
        Whitened time series.

    """

    # If x is a multivariate gaussian variable with covariance C (p(x) ~ x^T * C^{-1} * x), one can use the Cholesky decomposition to obtain C in terms of a lower triangular matrix L: `C = L * L^T`. This implies that `z~N(0,1)` with `z = L^{-1} * x` (below called `x_whitened`), i.e. we need to solve the linear system `L * z = x` for the unknown `z`.

    if(  method=='solve'):            x_whitened = sl.solve(           cholesky_L, x, lower=True, check_finite=False)
    elif(method=='solve-triangular'): x_whitened = sl.solve_triangular(cholesky_L, x, lower=True, check_finite=False)
    elif(method=='solve-numpy'):      x_whitened = np.linalg.solve(    cholesky_L, x)

    else:                             raise ValueError('Unknown whitening method requested')
    
    return x_whitened

def whiten_FD(strain, interp_psd, dt, f_min, f_max):

    """

    This function whitens the strain time series in the frequency domain.

    Parameters
    ----------

    strain : array
        Strain time series.
    interp_psd : array
        Interpolated PSD.
    dt : float
        Time step of the strain time series.
    f_min : float
        Minimum frequency of the bandpass.
    f_max : float
        Maximum frequency of the bandpass.
    
    Returns
    -------

    white_ht : array
        Whitened strain time series.

    """

    #########################################################################
    # Function to whiten the data. Transform to freq domain, divide by asd, #
    # then transform back, taking care to get normalization right.          #
    #########################################################################

    # Initialise auxiliary quantities
    Nt       = len(strain)
    freqs    = np.fft.rfftfreq(Nt, dt)
    srate_dt = 1./dt

    # Clean PSD. Required because when we inject a given PSD, the extrapolation sets it to 0 in the region outside the interpolation range. Such a 0 would cause the whitening to crash.
    psd_cleaned = interp_psd(freqs)
    for i in range(0, len(psd_cleaned)):
        if((freqs[i]<f_min) or (freqs[i]>f_max)): psd_cleaned[i] = np.inf

    hf       = np.fft.rfft(strain)
    white_hf = hf / (np.sqrt(psd_cleaned/dt/2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)

    return white_ht

def inner_product_TD(h1, h2, InvCov):

    """
    
    This function computes the inner product between two time series h1 and h2 using the inverse covariance matrix InvCov.

    Parameters
    ----------

    h1 : array
        First time series.
    h2 : array
        Second time series.
    InvCov : array
        Inverse covariance matrix.
    
    Returns
    -------

    inner_product : float
        Inner product between h1 and h2.
    
    """
    
    return np.dot(h1,np.dot(InvCov,h2))

def inner_product_FD(h1, h2, psd, df):

    """

    This function computes the inner product between two frequency series h1 and h2 using the PSD psd.

    Parameters
    ----------

    h1 : array
        First frequency series.
    h2 : array
        Second frequency series.
    psd : array
        PSD.
    df : float
        Frequency step of the frequency series.
    
    Returns
    -------

    inner_product : float
        Inner product between h1 and h2.
    
    """
    
    return 4.0*df*np.sum(np.conj(h1)*h2/psd).real

def compute_SNR_TD(data, template, weights, method='direct-inversion'):

    """

    This function computes the SNR between a data time series and a template time series in the time domain, using various methods.

    Parameters
    ----------

    data : array
        Data time series.
    template : array    
        Template time series.
    weights : array
        Weights to be used in the inner product. In the case of the time domain, this can be the inverse covariance matrix (for the 'direct-inversion' method) or the Cholesky decomposition of the covariance matrix (for the 'cholesky-solve-triangular' method) or the ACF (for the 'toeplitz-inversion' method).
    method : str
        Method to be used to compute the inner product. Options are: 'direct-inversion', 'toeplitz-inversion', 'cholesky-solve-triangular'.

    Returns
    -------

    SNR : float
        SNR of the template in the given data.
    
    """

    # These methods have been found to give all identical results (up to the 11th decimal digit of GW150914 SNR at 10M and seglen=0.1).
    # The computational time hierarchy is: 'direct-inversion'~0.2ms, 'toeplitz-inversion'~0.7ms, 'cholesky-solve-triangular'~4ms.
    if(method=='direct-inversion'):
        # In this case weights is C^{-1}, the inverse covariance matrix
        hh = inner_product_TD(template, template, weights)
        dh = inner_product_TD(data,     template, weights)
    elif(method=='cholesky-solve-triangular'):
        # In this case weights is L, the Cholesky decomposition of the covariance matrix C
        whiten_h = whiten_TD(template, weights, method='solve-triangular')
        whiten_d = whiten_TD(data,     weights, method='solve-triangular')
        hh       = np.dot(whiten_h, whiten_h)
        dh       = np.dot(whiten_d, whiten_h)
    elif(method=='toeplitz-inversion'):
        # In this case weights is ACF, the autocorrelation function from which the covariance matrix C is computed
        whiten_whiten_h = sl.solve_toeplitz(weights, template, check_finite=False)
        hh              = np.dot(template, whiten_whiten_h)
        dh              = np.dot(data,     whiten_whiten_h)
    else:
        raise ValueError('Unknown method requested to compute the TD SNR.')

    return dh/np.sqrt(hh)

def compute_SNR_FD(data, template, psd, df):

    """

    This function computes the SNR between a data frequency series and a template frequency series in the frequency domain.

    Parameters
    ----------

    data : array
        Data frequency series.
    template : array
        Template frequency series.
    psd : array
        PSD.
    df : float
        Frequency step of the frequency series.

    Returns
    -------

    SNR : float
        SNR of the template in the given data.
    
    """

    return inner_product_FD(data, template, psd, df)/np.sqrt(inner_product_FD(template, template, psd, df))

def railing_check(samples, prior_bins, tolerance):

    """

    This function checks whether the density of samples is within the tolerance percentage for the bins at the lower and higher boundaries of the prior, i.e. if there is railing of the posterior against the prior.

    Parameters
    ----------

    samples : array
        Samples from the posterior.
    prior_bins : array
        Bins of the prior.
    tolerance : float
        Tolerance percentage.
    
    Returns
    -------

    low_end_railing : bool
        True if the density of samples is within the tolerance percentage for the bin at the lower boundary of the prior.
    high_end_railing : bool
        True if the density of samples is within the tolerance percentage for the bin at the higher boundary of the prior.
    
    """


    hist, bin_edges = np.histogram(samples, bins=prior_bins, density=True)

    highest_hist     = np.amax(hist)
    lower_support    = hist[0]  / highest_hist * 100
    higher_support   = hist[-1] / highest_hist * 100

    low_end_railing  = lower_support  > tolerance
    high_end_railing = higher_support > tolerance

    return low_end_railing, high_end_railing

##########################################
# From here downwards, NO REVIEW NEEDED. #
##########################################

def construct_interpolant(param_name, N_bins=32, par_file = 'ringdown/random_parameters_interpolation.txt'):

    """

    This function constructs a spline interpolant for the prior of a given parameter.

    Parameters
    ----------

    param_name : str
        Name of the parameter.
    N_bins : int
        Number of bins to be used to construct the spline interpolant.
    par_file : str
        File containing the simulated events.
    
    Returns
    -------

    spline_interpolant : scipy.interpolate.UnivariateSpline
        Spline interpolant for the prior of the given parameter.
    
    """
    
    # Read in the simulated events and create spline interpolants.
    review_warning()
    from scipy.interpolate import UnivariateSpline
    print('\n* Reweighting the priors for a random population of injections.')
    try:
        print("\nReweighting the prior of {}".format(param_name))
        PYRING_PREFIX      = set_prefix()
        values_interp      = np.genfromtxt(os.path.join(PYRING_PREFIX, par_file), names=True)[param_name]
        m                  = np.histogram(values_interp, bins=N_bins, density=True)
        bins               = 0.5 * (m[1][1:] + m[1][:-1])
        spline_interpolant = UnivariateSpline(bins, m[0], k=1, ext=1, s=0)
    except:
        raise Exception("\n* Prior railing file generation failed with error: {}.".format(traceback.print_exc()))

    return spline_interpolant

def EsGB_corrections(name, dim):

    """

    This function returns the coefficients of the Parspec expansion for the gravitational polar-led modes in the EdGB model.

    Parameters
    ----------

    name : str
        Name of the mode deviation.
    dim : int
        Number of terms to be used in the Parspec expansion.
    
    Returns
    -------

    corr : array
        Coefficients of the Parspec expansion for the gravitational polar-led modes in the EdGB model.
    
    """
    
    # Coefficients of Parspec expansion for gravitational polar-led modes in EdGB, from arXiv:2103.09870, arXiv:2207.11267
    all_corr = {
                'domega_220': [-0.03773, -0.1668 , -0.278],
                'dtau_220'  : [-0.0528 , -0.08   ,  3.914],
                }
    if not name in all_corr.keys(): raise ValueError("Currently EdsGB corrections only support the (l,m,n)=(2,2,0) mode deviation.")

    corr = []
    for i in range(dim+1): corr.append(all_corr[name][i])

    return corr

def EsGB_corrections_Carson_Yagi(name):

    """

    This function returns the coefficients of the Parspec expansion for the gravitational polar-led modes in the EdGB model according to the Carson-Yagi prescription.

    Parameters
    ----------

    name : str
        Name of the mode deviation.
    
    Returns
    -------

    corr : array
        Coefficients of the Parspec expansion for the gravitational polar-led modes in the EdGB model according to the Carson-Yagi prescription.
    
    """
    
    if(name=='domega_220'):
        a0_GR =  0.373672
        a1_GR =  0.2438
        a2_GR = -1.2722
        a0_GB = -0.1874
        a1_GB = -0.6552
        a2_GB = -0.6385
        corr  = [a0_GB/a0_GR, (a0_GB*a1_GB)/a1_GR, (a0_GB*a2_GB)/a2_GR]
    elif(name=='dtau_220'):
        b0_GR =  11.240715
        b1_GR =  2.3569
        b2_GR = -5.0014
        b0_GB = 1.0/(-0.0622)
        b1_GB = 0.0
        b2_GB = 0.0
        corr  = [b0_GB/b0_GR, (b0_GB*b1_GB)/b1_GR, (b0_GB*b2_GB)/b2_GR]
        raise ValueError("Tau corrections to be filled yet.")
    else:
        raise ValueError("Currently EdsGB corrections only support the (l,m,n)=(2,2,0) mode deviation.")

    return corr

def mtot(m1,m2): 
    
    """
    
    This function returns the total mass of a binary system as a function of the component masses.

    Parameters
    ----------

    m1 : float
        Mass of the first component.
    m2 : float
        Mass of the second component.

    Returns
    -------

    mtot : float
        Total mass of the binary system.
    
    """
    
    return m1+m2

def mc(m1,m2): 
    
    """

    This function returns the chirp mass of a binary system as a function of the component masses.

    Parameters
    ----------

    m1 : float
        Mass of the first component.
    m2 : float
        Mass of the second component.

    Returns
    -------

    mc : float
        Chirp mass of the binary system.

    """
    
    return (m1*m2)**(3./5.)/(m1+m2)**(1./5.)

def eta(m1,m2): 
    
    """

    This function returns the symmetric mass ratio of a binary system as a function of the component masses.

    Parameters
    ----------

    m1 : float
        Mass of the first component.
    m2 : float
        Mass of the second component.

    Returns
    -------

    eta : float
        Symmetric mass ratio of the binary system.

    """
    
    return (m1*m2)/(m1+m2)**2

def q(m1,m2):

    """

    This function returns the mass ratio of a binary system as a function of the component masses.

    Parameters
    ----------

    m1 : float
        Mass of the first component.
    m2 : float
        Mass of the second component.

    Returns
    -------

    q : float
        Mass ratio of the binary system.

    """

    return [(np.minimum(m1,m2))/(np.maximum(m1,m2)), (np.maximum(m1,m2))/(np.minimum(m1,m2))]  #Return both of the q conventions

def m1_from_m_q(m, q):

    """

    This function returns the mass of the first component of a binary system as a function of the total mass and the mass ratio.

    Parameters
    ----------

    m : float
        Total mass of the binary system.
    q : float
        Mass ratio of the binary system.
    
    Returns
    -------

    m1 : float
        Mass of the first component of the binary system.

    """

    return m*q/(1+q)

def m2_from_m_q(m, q):

    """

    This function returns the mass of the second component of a binary system as a function of the total mass and the mass ratio.

    Parameters
    ----------

    m : float
        Total mass of the binary system.
    q : float
        Mass ratio of the binary system.
    
    Returns
    -------

    m2 : float
        Mass of the second component of the binary system.

    """

    return m/(1+q)

def m1_from_mc_q(mc, q): 
    
    """

    This function returns the mass of the first component of a binary system as a function of the chirp mass and the mass ratio.

    Parameters
    ----------

    mc : float
        Chirp mass of the binary system.
    q : float
        Mass ratio of the binary system.

    Returns
    -------

    m1 : float
        Mass of the first component of the binary system.

    """
    
    return mc*((1+q)**(1./5.))*q**(-3./5.)

def m2_from_mc_q(mc, q): 
    
    """

    This function returns the mass of the second component of a binary system as a function of the chirp mass and the mass ratio.

    Parameters
    ----------

    mc : float
        Chirp mass of the binary system.
    q : float
        Mass ratio of the binary system.

    Returns
    -------

    m2 : float
        Mass of the second component of the binary system.  

    """

    return mc*((1+q)**(1./5.))*q**(2./5.)

def chi_eff(q, a1, a2):

    """

    This function returns the effective spin of a binary system as a function of the mass ratio and the spins of the components.
    Note: in the equal mass case, eta=0.25, implying chi_1=chi_2=0.5. Hence, in this case, chi_eff is the aritmetic mean of (a1,a2).

    
    Parameters
    ----------

    q : float
        Mass ratio of the binary system.
    a1 : float
        Spin of the first component of the binary system.
    a2 : float
        Spin of the second component of the binary system.

    Returns
    -------

    chi_eff : float
        Effective spin of the binary system.

    """

    eta = q/(1+q)**2
    chi_1 = 0.5*(1.0+np.sqrt(1.0-4.0*eta))
    chi_2 = 1.0-chi_1

    return chi_1*a1 + chi_2*a2

def mchirp_from_mtot_eta(mtot, eta):

    """

    This function returns the chirp mass of a binary system as a function of the total mass and the symmetric mass ratio.

    Parameters
    ----------

    mtot : float
        Total mass of the binary system.
    eta : float
        Symmetric mass ratio of the binary system.

    Returns
    -------

    mc : float
        Chirp mass of the binary system.

    """

    return mtot*eta**(3./5.)

def mchirp_from_mtot_q(mtot, q):

    """

    This function returns the chirp mass of a binary system as a function of the total mass and the mass ratio.

    Parameters
    ----------

    mtot : float
        Total mass of the binary system.

    q : float
        Mass ratio of the binary system.

    Returns
    -------

    mc : float
        Chirp mass of the binary system.

    """

    return mtot*(q/(1+q)**2)**(3./5.)

def eta_from_q(q):

    """

    This function returns the symmetric mass ratio of a binary system as a function of the mass ratio.

    Parameters
    ----------

    q : float
        Mass ratio of the binary system.

    Returns
    -------

    eta : float
        Symmetric mass ratio of the binary system.

    """

    return q/(1+q)**2

def project_python_wrapper(hs,hvx, hvy, hp, hc, detector, ra, dec, psi, tgps):

    """
    
        Compute the complex time series projected onto the given detector.

        Parameters
        ----------

            hs : array of shape (n,)
                The complex time series of the breathing mode.

            hvx : array of shape (n,)
                The complex time series of the longitudinal mode in the x direction.

            hvy : array of shape (n,)
                The complex time series of the longitudinal mode in the y direction.

            hp : array of shape (n,)
                The complex time series of the plus polarization.

            hc : array of shape (n,)
                The complex time series of the cross polarization.

            detector : laldetector structure
                The detector.

            ra : double
                The right ascension.

            dec : double
                The declination.

            psi : double
                The polarisation angle.

            tgps : double
                The time (GPS seconds).

    """

    #==============================================================================#
    # Project complex time series onto the given detector (laldetector structure). #
    # Signal is shifted in time relative to the geocenter.                         #
    # ra   - right ascension                                                       #
    # dec  - declination                                                           #
    # psi  - polarisation angle                                                    #
    # tgps - time (GPS seconds)                                                    #
    #==============================================================================#

    gmst = lal.GreenwichMeanSiderealTime(tgps)
    #The breathing and longitudinal modes act on a L-shaped detector in the same way up to a constant amplitude, thus we just use one. See arXiv:1710.03794.
    fp, fc, fb, fs, fvx, fvy = lal.ComputeDetAMResponseExtraModes(detector.response, ra, dec, psi, gmst)

    waveform = fs*hs + fvx*hvx + fvy*hvy + fp*hp + fc*hc

    return waveform

def cholesky_logdet_C(covariance):

    """

    Compute the log determinant of a covariance matrix using Cholesky decomposition.

    Parameters
    ----------

    covariance : array
        Covariance matrix.

    Returns
    -------

    logdet : float
        Log determinant of the covariance matrix.
    
    """

    R = sl.cholesky(covariance)

    return 2.*np.sum([np.log(R[i,i]) for i in range(R.shape[0])])

def resize_time_series(inarr, N, dt, starttime, desiredtc):

    """
    Zero pad inarr and align its peak to the desired tc in the segment.

    Parameters
    ----------

    inarr : array
        Input time series.
    N : int
        Length of the output time series.
    dt : float
        Sampling time of the output time series.
    starttime : float
        Start time of the output time series.
    desiredtc : float
        Desired time of coalescence of the output time series.
    
    Returns
    -------

    outarr : array
        Output time series.
    
    """
    
    review_warning()

    waveLength = inarr.shape[0]

    # Find the sample at which we wish tc to be, and the actual tc.
    tcSample = int(np.floor((desiredtc-starttime)/dt))
    injTc    = starttime + tcSample*dt

    # find the sample in waveform space at which tc happens, using the square amplitude as reference
    waveTcSample = np.argmax(inarr[:,0]**2+inarr[:,1]**2)

    wavePostTc = waveLength - waveTcSample

    if (tcSample >= waveTcSample)  : bufstartindex =  tcSample - waveTcSample
    else                           : bufstartindex = 0
    if (wavePostTc + tcSample <= N): bufendindex   = wavePostTc + tcSample
    else                           : bufendindex   = N

    bufWaveLength = bufendindex - bufstartindex
    if (tcSample >= waveTcSample): waveStartIndex = 0
    else                         : waveStartIndex = waveTcSample - tcSample

    # Allocate the arrays of zeros which work as a buffer.
    hp = np.zeros(N,dtype = np.float64)
    hc = np.zeros(N,dtype = np.float64)

    # Copy the waveform over.
    waveEndIndex = waveStartIndex + bufWaveLength

    hp[bufstartindex:bufstartindex+bufWaveLength] = inarr[waveStartIndex:waveEndIndex,0]
    hc[bufstartindex:bufstartindex+bufWaveLength] = inarr[waveStartIndex:waveEndIndex,1]

    return hp,hc

class UNUSED_NR_waveform(object):

    """
    
    Class to generate NR waveforms.

    Parameters
    ----------

    object : object
        Object to be used.
    
    """

    # ===============================================================================================#
    # NR injection setup adapted from the inject_NR.py script used in IMR consistency test studies.  #
    # Credits to: Abhirup Ghosh, Archisman Ghosh, Ashok Choudhary, KaWa Tsang, Laura Van Der Schaaf, #
    # Nathan K Johnson-McDaniel, Peter Pang.                                                         #
    # ===============================================================================================#

    def __init__(self, **kwargs):
        review_warning()
        self.incl_inj  = kwargs['injection-parameters']['incl']
        self.phi_inj   = kwargs['injection-parameters']['phi']
        self.SXS_ID    = kwargs['injection-parameters']['SXS-ID']
        PYRING_PREFIX = set_prefix()
        self.data_file = os.path.join(PYRING_PREFIX, 'data/NR_data/SXS_data/BBH0{0}/rhOverM_Asymptotic_GeometricUnits_CoM.h5'.format(self.SXS_ID))
        self.N         = kwargs['injection-parameters']['N']
        # Load the data
        sys.stdout.write('\n\n----NR injection section----')
        sys.stdout.write('\nLoading SXS NR data from %s\n'%(os.path.realpath(self.data_file)))
        sys.stdout.write('Extrapolation order N = %d\n'%(self.N))
        ff               = h5py.File(self.data_file, 'r')
        available_modes  = ff.get('Extrapolated_N%d.dir'%(self.N)).keys()
        self.lmax        = int(kwargs['injection-parameters']['lmax'])
        self.fix_NR_mode = kwargs['injection-parameters']['fix-NR-mode']
        try:
            (l_fix, m_fix) = self.fix_NR_mode[0]
        except:
            (l_fix, m_fix) = (None, None)
        self.absmmin     = 0
        sys.stdout.write('lmax = %d\n'%(self.lmax))
        (self.t22_geom, hr_geom22, hi_geom22) = (ff.get('Extrapolated_N%d.dir/Y_l%d_m%d.dat'%(self.N, 2, 2)).value).T
        self.h_plus   = np.array([0.]*len(self.t22_geom))
        self.h_cross  = np.array([0.]*len(self.t22_geom))

        if not((l_fix, m_fix)==(None, None)):
            l = l_fix
            m = m_fix
            assert not(m==0), "m=0 modes not yet supported."
            sys.stdout.write('Including (%d,%d) mode.\n'%(l, m))
            (t_geom, hr_geom, hi_geom) = (ff.get('Extrapolated_N%d.dir/Y_l%d_m%d.dat'%(self.N, l, m)).value).T
            if abs(m) >= self.absmmin:
                Amp = np.sqrt(hr_geom**2+hi_geom**2)
                #FIXME: check the convention which enforces this minus sign (it stops the people being upside down)
                Phi = np.unwrap(np.angle(hr_geom - 1j*hi_geom))
                Y_p = wf.SWSH(2,l,m)(self.incl_inj,self.phi_inj)
                self.h_plus  =  Amp*(np.cos(Phi)*np.real(Y_p) + np.sin(Phi)*np.imag(Y_p))
                self.h_cross = -Amp*(np.cos(Phi)*np.imag(Y_p) - np.sin(Phi)*np.real(Y_p))
                self.h_dressed = self.h_plus-1j*self.h_cross

        else:
            for l in range(2, self.lmax+1):
                for m in range(-l, l+1):
                    #FIXME: This expansion is not true for m=0. m=0 expansion needs to be implemented
                    if not(m==0):
                        sys.stdout.write('Including (%d,%d) mode.\n'%(l, m))
                        (t_geom, hr_geom, hi_geom) = (ff.get('Extrapolated_N%d.dir/Y_l%d_m%d.dat'%(self.N, l, m)).value).T
                        if abs(m) >= self.absmmin:
                            Y_p = wf.SWSH(2,l,m)(self.incl_inj,self.phi_inj)
                            Y_m = wf.SWSH(2,l,-m)(self.incl_inj,self.phi_inj)
                            Amp = np.sqrt(hr_geom**2+hi_geom**2)
                            Phi = np.unwrap(np.angle(hr_geom - 1j*hi_geom))

                            self.h_plus  += Amp*(np.cos(Phi)*(np.real(Y_p)+np.real(Y_m)) - np.sin(Phi)*(-np.imag(Y_p)+np.imag(Y_m)))
                            self.h_cross -= Amp*(np.cos(Phi)*(np.imag(Y_p)+np.imag(Y_m)) - np.sin(Phi)*(np.real(Y_p)-np.real(Y_m)))
            self.h_dressed = self.h_plus-1j*self.h_cross
        sys.stdout.write('\n')

def UNUSED_inject_NR_signal(lenstrain, tstart, length, ifo, triggertime, **kwargs):

    """
    
    Inject a NR signal into the data. The NR signal is generated using the FIXME

    Parameters
    ----------

    lenstrain : int
        Length of the strain data vector
    
    tstart : float
        Start time of the data

    length : float
        Length of the data
    
    ifo : str
        Detector name
    
    triggertime : float
        GPS time of the trigger

    **kwargs : dict
        Dictionary of keyword arguments
    
    Returns
    -------

    hplus : numpy.ndarray
        Plus polarization of the NR signal
    
    hcross : numpy.ndarray
        Cross polarization of the NR signal
    
    """

    mass      = kwargs['injection-parameters']['M']
    dist      = kwargs['injection-parameters']['dist']
    psi       = kwargs['injection-parameters']['psi']
    M_inj_sec = mass*lal.MTSUN_SI
    tM_gps    = lal.LIGOTimeGPS(float(triggertime))
    detector  = lal.cached_detector_by_prefix[ifo]
    ref_det   = lal.cached_detector_by_prefix[kwargs['ref-det']]

    if (kwargs['sky-frame']=='detector'):
        tg, ra, dec = DetFrameToEquatorial(lal.cached_detector_by_prefix[kwargs['ref-det']],
                                           lal.cached_detector_by_prefix[kwargs['nonref-det']],
                                           triggertime,
                                           np.arccos(kwargs['injection-parameters']['cos_altitude']),
                                           kwargs['injection-parameters']['azimuth'])
    elif (kwargs['sky-frame']=='equatorial'):
        ra  = kwargs['injection-parameters']['ra']
        dec = kwargs['injection-parameters']['dec']
    else:
        raise ValueError("Invalid option for sky position sampling.")

    time_delay = lal.ArrivalTimeDiff(detector.location, ref_det.location, ra, dec, tM_gps)

    # Build NR waveform.
    NR_wf_obj = NR_waveform(**kwargs)
    t_phys    = NR_wf_obj.t22_geom*M_inj_sec
    hp        = NR_wf_obj.h_plus
    hc        = NR_wf_obj.h_cross

    time = tstart+np.linspace(0, length, lenstrain)

    # Interpolate the waveform over a uniform grid, NR sampling (t_phys) is NOT uniform.
    h_p_int = np.interp(time, tstart+t_phys, hp)
    h_c_int = np.interp(time, tstart+t_phys, hc)

    # Shift the waveform to the desidered tc.
    hp,hc = resize_time_series(np.column_stack((h_p_int,h_c_int)),
                               lenstrain,
                               time[1]-time[0],
                               tstart,
                               triggertime+time_delay)


    # Project the waveform onto a given detector, switching from geometrical to physical units.
    hs, hvx, hvy = np.zeros(len(hp)), np.zeros(len(hp)), np.zeros(len(hp))
    h = project(hs, hvx, hvy, hp, hc, detector, ra, dec, psi, tM_gps)

    # timeshift the waveform to the desired merger time in the given detector tM = THanford+time_delay.
    h *= mass * lal.MSUN_SI * lal.G_SI / (dist * lal.PC_SI*10**6 * lal.C_SI**2)

    return h

def F_mrg_Nagar_v0(m1, m2, a1, a2):

    """
    
    This function returns the merger frequency of a binary black hole system. Old version of the merger frequency, defined with respect to the 22 mode of the inspiral.
    
    Parameters
    ----------

    m1 : float
        Mass of the first black hole
    
    m2 : float
        Mass of the second black hole
    
    a1 : float
        Dimensionless spin of the first black hole
    
    a2 : float
        Dimensionless spin of the second black hole
    
    Returns
    -------

    res : float
        Merger frequency of the binary black hole system

    """

    review_warning()

    q       = m1/m2 #Husa conventions, m1>m2 [https://arxiv.org/abs/1611.00332]
    eta     = q/(1+q)**2
    M_tot   = m1+m2
    chi_1   = 0.5*(1.0+np.sqrt(1.0-4.0*eta))
    chi_2   = 1.0-chi_1
    chi_eff = chi_1*a1 + chi_2*a2

    A = -0.28562363*eta + 0.090355762
    B = -0.18527394*eta + 0.12596953
    C =  0.40527397*eta + 0.25864318

    res = (A*chi_eff**2 + B*chi_eff + C)*((2*np.pi*M_tot)*lal.G_SI*lal.C_SI**(-3))**(-1)
    return res

#NO-REVIEW-NEEDED
def F_mrg_Nagar(m1, m2, a1, a2, geom=0):

    """

    This function returns the merger frequency of a binary black hole system. New version of the merger frequency, defined with respect to the 22 mode of the inspiral.

    Parameters
    ----------

    m1 : float
        Mass of the first black hole
    
    m2 : float
        Mass of the second black hole

    a1 : float
        Dimensionless spin of the first black hole

    a2 : float
        Dimensionless spin of the second black hole

    geom : int
        Flag to activate geometrical units. Default is 0, SI units.

    Returns
    -------

    res : float
        Merger frequency of the binary black hole system

    """

    q           = m1/m2
    nu          = q/(1+q)**2
    M           = m1+m2
    X12         = (m1-m2)/M
    Shat        = (m1**2*a1 + m2**2*a2)/M**2
    # Orbital fits calibrated to the non-spinning SXS data
    omg_tp         = 0.273356     # for this one I won't give an error the TP
    # waveform was generated with the TEUKCode
    # and I think it is pretty much acurate
    omg1         = 0.84074
    omg1_err     = 0.014341
    omg2         = 1.6976
    omg2_err     = 0.075488
    orb          = omg_tp*(1+omg1*nu+omg2*nu**2)

    # Equal Mass fit calibrated to the q=1 SXS data
    b1          = -0.42311
    b1_err      = 0.088583
    b2          = -0.066699
    b2_err      = 0.042978
    b3          = -0.83053
    b3_err      = 0.084516

    # Unequal Mass corrections to the q=1 fit based on SXS, BAM and TP data
    c1          = 0.066045
    c1_err      = 0.13227
    c2          = -0.23876
    c2_err      = 0.29338
    c3          = 0.76819
    c3_err      = 0.01949
    c4          = -0.9201
    c4_err      = 0.025167
    num         = 1.+((b1+c1*X12)/(1.+c2*X12))*Shat+b2*Shat**2
    denom       = 1.+((b3+c3*X12)/(1.+c4*X12))*Shat

    if(geom): res         = (orb*num/denom)*((2*np.pi*M))**(-1)
    else    : res         = (orb*num/denom)*((2*np.pi*M)*lal.G_SI*lal.C_SI**(-3))**(-1)  

    return res

def F_mrg_Bohe(m1,m2,a1,a2):

    """

        Computees the merger frequency fit as described in Bohe et al. arXiv:1611.03703   

        Parameters
        ----------

        m1 : float
            Mass of the first black hole

        m2 : float
            Mass of the second black hole
        
        a1 : float
            Dimensionless spin of the first black hole
        
        a2 : float
            Dimensionless spin of the second black hole
        
        Returns
        -------

        res : float
            Merger frequency of the binary black hole system
        
        """

    review_warning()

    q       = m1/m2
    M_tot   = m1+m2
    nu      = q/(1+q)**2
    delta   = np.sqrt(1.-4.*nu)
    chi_S   = 0.5*(a1+a2)
    chi_A   = 0.5*(a1-a2)
    chi     = chi_S + chi_A*delta*((1.-2.*nu)**(-1))

    p0_TPL  = + 0.562679
    p1_TPL  = - 0.087062
    p2_TPL  = + 0.001743
    p3_TPL  = + 25.850378
    p4_TPL  = + 25.819795

    p3_EQ   = 10.262073
    p4_EQ   = 7.629922

    A3      = p3_EQ + 4.*(p3_EQ - p3_TPL)*(nu-1./4.)
    A4      = p4_EQ + 4.*(p4_EQ - p4_TPL)*(nu-1./4.)

    res     = (p0_TPL + (p1_TPL + p2_TPL*chi)*np.log(A3 - A4*chi))*((2.*np.pi*M_tot)*lal.G_SI*lal.C_SI**(-3))**(-1)
    return res

def F_mrg_Healy(m1,m2,a1,a2):

    """

    Computes the merger frequency fit as described in Healy et al. Phys. Rev. D 97, 084002(2018)
    This implementation uses the convention m1>m2 and thus has a sign differences in the definitions of dm and Delta comparing to the original paper.

    Parameters
    ----------

    m1 : float
        Mass of the first black hole
    
    m2 : float
        Mass of the second black hole
    
    a1 : float
        Dimensionless spin of the first black hole
    
    a2 : float
        Dimensionless spin of the second black hole
    
    Returns
    -------

    res : float
        Merger frequency of the binary black hole system
    
    """

    M_tot = m1+m2
    dm  = (m2-m1)/(m1+m2)
    Delta=(a1*m1-a2*m2)/(m1+m2)
    St  = (m1**2*a1+m2**2*a2)/(m1+m2)**2


    W0  = 0.3587
    W1  = 0.14189
    W2a = -0.01461
    W2b = 0.05505
    W2c = 0.00878
    W2d = -0.1211
    W3a = -0.16841
    W3b = 0.04874
    W3c = 0.09181
    W3d = -0.08607
    W4a = -0.02185
    W4b = 0.11183
    W4c = -0.01704
    W4d = 0.21595
    W4e = -0.12378
    W4f = 0.0432
    W4g = 0.00167
    W4h = -0.13224
    W4i = -0.09933
    o1     = W0 + W1*St + W2a*Delta*dm + W2b*St**2 + W2c*Delta**2 + W2d*dm**2
    o2     = W3a*Delta*St*dm + W3b*St*Delta**2 + W3c*St**3 + W3d*St*dm**2
    o3     = W4a*Delta*St**2*dm + W4b*Delta**3*dm + W4c*Delta**4 + W4d*St**4
    o4     = W4e*Delta**2*St**2 + W4f*dm**4 + W4g*Delta*dm**3 + W4h*Delta**2*dm**2 + W4i*St**2*dm**2
    res = (o1 + o2 + o3 + o4)*((2.*np.pi*M_tot)*lal.G_SI*lal.C_SI**(-3))**(-1)
    
    return res

def A_mrg_Bohe(m1,m2,a1,a2):

    review_warning()

    # Frequency fit from Bohe et al. arXiv:1611.03703
    q               = m1/m2
    nu      = q/(1+q)**2
    delta   = np.sqrt(1.-4.*nu)
    chi_S   = 0.5*(a1+a2)
    chi_A   = 0.5*(a1-a2)
    chi     = chi_S + chi_A*delta*((1.-2.*nu)**(-1))

    e00     = +1.452857
    e01     = +0.166134
    e02     = +0.027356
    e03     = -0.020073

    e10     = -0.034424
    e11     = -1.218066
    e12     = -0.568373
    e13     = +0.401114


    Amp_tp  = e00 + e01*chi + e02*chi**2 + e03*chi**3
    Amp_lin = e10 + e11*chi + e12*chi**2 + e13*chi**3

    e20     = + 16*1.577458 - 16*e00 - 4*e10
    e21     = - 16*0.007695 - 16*e01 - 4*e11
    e22     = + 16*0.021887 - 16*e02 - 4*e12
    e23     = + 16*0.023268 - 16*e03 - 4*e13

    Amp_quad= e20 + e21*chi + e22*chi**2 + e23*chi**3
    res     = nu*(Amp_tp + Amp_lin*nu + Amp_quad*nu**2)
    return res

def A_mrg_Healy(m1,m2,a1,a2):
    
        # Phys. Rev. D 97, 084002(2018)
        # Important: this Fit uses the convention m1>m2 and thus has
        # Sign differences in the definitions of dm and Delta comparing to
        # the orignial paper!
        # When comparing with the fit presented in the paper in eq (20) then
        # it is important to note that the coefficients are referred to as
        # H.. instead of A

    review_warning()

    dm  = (m2-m1)/(m1+m2)
    Delta=(a1*m1-a2*m2)/(m1+m2)
    St  = (m1**2*a1+m2**2*a2)/(m1+m2)**2
    nu      = m1*m2*(m1+m2)**(-2)
    A0  = 0.3937
    A1  = -0.00252
    A2a = 0.00385
    A2b = 0.00495
    A2c = -0.00145
    A2d = -0.0526
    A3a = 0.00331
    A3b = 0.01775
    A3c = 0.03202
    A3d = 0.05267
    A4a = 0.11029
    A4b = -0.00552
    A4c = 0.00558
    A4d = 0.04593
    A4e = -0.04754
    A4f = 0.0179
    A4g = -0.00516
    A4h = 0.00163
    A4i = -0.02098

    h1      = A0 + A1*St + A2a*Delta*dm + A2b*St**2 + A2c*Delta**2
    h2              = A2d*dm**2 +A3a*Delta*St*dm + A3b*St*Delta**2 + A3c*St**3
    h3              = A3d*St*dm**2 + A4a*Delta*St**2*dm + A4b*Delta**3*dm
    h4              = A4c*Delta**4 + A4d*St**4 + A4e*Delta**2*St**2 + A4f*dm**4
    h5              = A4g*Delta*dm**3 + A4h*Delta**2*dm**2 + A4i*St**2*dm**2
    res     = 4*nu*(h1 + h2 + h3 + h4 + h5)
    return res

#NO-REVIEW-NEEDED
def A_mrg_Nagar(m1, m2, a1, a2):

    q    = m1/m2
    nu   = q/(1+q)**2
    M    = m1+m2
    X12  = (m1-m2)/M
    Shat = (m1**2*a1 + m2**2*a2)/M**2

    # Orbital fits calibrated to the non-spinning SXS data
    omg_tp       = 0.273356     # for this one I won't give an error the TP waveform was generated with the TEUKCode and I think it is pretty much acurate
    omg1         = 0.84074
    omg1_err     = 0.014341
    omg2         = 1.6976
    omg2_err     = 0.075488
    orb          = omg_tp*(1.+omg1*nu+omg2*nu**2)

    # Equal Mass fit calibrated to the q=1 SXS data
    b1             = -0.42311
    b1_err         = 0.088583
    b2             = -0.066699
    b2_err         = 0.042978
    b3             = -0.83053
    b3_err         = 0.084516

    # Unequal Mass corrections to the q=1 fit based on SXS, BAM and TP data
    c1            = 0.066045
    c1_err        = 0.13227
    c2            = -0.23876
    c2_err        = 0.29338
    c3            = 0.76819
    c3_err        = 0.01949
    c4            = -0.9201
    c4_err        = 0.025167
    num           = 1.+((b1+c1*X12)/(1.+c2*X12))*Shat+b2*Shat**2
    denom         = 1.+((b3+c3*X12)/(1.+c4*X12))*Shat
    omgmx         = (orb*num/denom)

    scale       = 1. - Shat*omgmx
    # Orbital fits calibrated to the non-spinning SXS data
    Amax_tp     = 0.295897  # for this one I won't give an error the TP
    # waveform was generated with the TEUKCode
    # and I think it is pretty much acurate
    Amax1       = -0.041285
    Amax1_err   = 0.0078878
    Amax2       = 1.5971
    Amax2_err   = 0.041521

    orb_A       = Amax_tp*(1+Amax1*nu+Amax2*nu**2)

    # Equal Mass fit calibrated to the q=1 SXS data
    b1Amax         = -0.74124
    b1Amax_err     = 0.016178
    b2Amax         = -0.088705
    b2Amax_err     = 0.0081611
    b3Amax         = -1.0939
    b3Amax_err     = 0.015318

    # Unequal Mass corrections to the q=1 fit based on SXS, BAM and TP data
    c1Amax        = 0.44467
    c1Amax_err    = 0.037352
    c2Amax        = -0.32543
    c2Amax_err    = 0.081211
    c3Amax        = 0.45828
    c3Amax_err    = 0.066062
    c4Amax        = -0.21245
    c4Amax_err    = 0.080254

    num_A       = 1+((b1Amax+c1Amax*X12)/(1+c2Amax*X12))*Shat+b2Amax*Shat**2
    denom_A     = 1+((b3Amax+c3Amax*X12)/(1+c4Amax*X12))*Shat
    res         = nu*orb_A*scale*num_A*(denom_A**(-1))*np.sqrt(24)

    return res

#UNUSED code to interpolate NR wfs.
## Interpolate the waveform over a uniform grid, NR sampling (t_phys) is NOT uniform. Then pad it to the total strain length.
#    endtime            = tstart+length
#    dt_uniform         = length/lenstrain
#    nr_wf_len          = t_phys[-1]-t_phys[0]
#    npoints            = nr_wf_len/dt_uniform
#    t_phys_uniform     = np.linspace(t_phys[0], t_phys[-1], npoints)
#    h_p_int            = np.interp(t_phys_uniform, t_phys, hp)
#    h_c_int            = np.interp(t_phys_uniform, t_phys, hc)
#    t_peak             = t_phys_uniform[np.argmax(h_p_int**2 + h_c_int**2)]
#    dt_peak_start      = t_peak - t_phys_uniform[0]
#    dt_peak_end        = t_phys_uniform[-1] - t_peak
#    dt_buffer_start    = (triggertime+time_delay-dt_peak_start) - tstart
#    dt_buffer_end      = endtime - (triggertime+time_delay+dt_peak_end)
#    buffer_start_len   = int(dt_buffer_start/dt_uniform)
#    buffer_end_len     = int(dt_buffer_end/dt_uniform)
#    zeros_start        = np.zeros(buffer_start_len)
#    zeros_end          = np.zeros(buffer_end_len)
#    h_p_buffered_start = np.concatenate((zeros_start, np.array(h_p_int)), axis=None)
#    h_c_buffered_start = np.concatenate((zeros_start, np.array(h_c_int)), axis=None)
#    hp                 = np.concatenate((h_p_buffered_start, zeros_end), axis=None)
#    hc                 = np.concatenate((h_c_buffered_start, zeros_end), axis=None)
