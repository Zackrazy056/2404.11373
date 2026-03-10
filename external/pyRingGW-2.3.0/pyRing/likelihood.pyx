#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: language_level=3
#cython: embedsignature=True

#Standard python imports
from __future__   import division
from scipy.linalg import inv, solve_toeplitz, solve_triangular, toeplitz
from libc.math   cimport log, M_PI

import  numpy as np, scipy
cimport numpy as np
cimport cython

#LVC imports
import lal

#Package internal imports
from pyRing import noise

cdef double log2pi = log(2.0*M_PI)

cpdef tuple toeplitz_slogdet(np.ndarray[double, ndim=1] r):

    """
        Method from Marano et al. "Fitting Earthquake Spectra: Colored Noise and Incomplete Data", Bulletin of the Seismological Society of America, Vol. 107, No. 1, pp. –, February 2017, doi: 10.1785/0120160030
        Code available here: http://mercalli.ethz.ch/~marra/publications/2017_fitting_earthquake_spectra_colored_noise_and_incomplete_data/
        All credits go to the original authors.

        Compute the log determinant of a positive-definite symmetric toeplitz matrix.
        The determinant is computed recursively. The intermediate solutions of the
        Levinson recursion are exploited.

        Parameters
        ----------
        
            r : array of shape (n,)
                The first row of the Toeplitz matrix. 

        Returns
        -------

            sign : 
                Sign of the determinant

            logdet :
                Natural log of the determinant
    """

    cdef int k, n
    cdef double r_0, logdet, sign, alpha, beta, d, mu
    cdef np.ndarray[double, ndim=1] x, y, b

    n      = len(r)
    r_0    = r[0]
    r      = np.concatenate((r, np.array([r_0])))
    r     /= r_0 # normalize the system so that the T matrix has diagonal of ones
    logdet = n*np.log(np.abs(r_0))
    sign   = np.sign(r_0)**n

    if(n == 1): return (sign, logdet)

    # From this point onwards, is a modification of Levinson algorithm.
    y       = np.zeros((n,))
    x       = np.zeros((n,))
    b       = -r[1:n+1]
    r       = r[:n]
    y[0]    = -r[1]
    x[0]    = b[0]
    beta    = 1
    alpha   = -r[1]
    d       = 1 + np.dot(-b[0], x[0])
    sign   *= np.sign(d)
    logdet += np.log(np.abs(d))

    for k in range(0, n-2):

        beta     = (1 - alpha*alpha)*beta
        mu       = (b[k+1] - np.dot(r[1:k+2], x[k::-1])) /beta
        x[0:k+1] = x[0:k+1] + mu*y[k::-1]
        x[k+1]   = mu

        d        = 1 + np.dot(-b[0:k+2], x[0:k+2])
        sign    *= np.sign(d)
        logdet  += np.log(np.abs(d))

        if(k < n-2):
            alpha    = -(r[k+2] + np.dot(r[1:k+2], y[k::-1]))/beta
            y[0:k+1] = y[0:k+1] + alpha * y[k::-1]
            y[k+1]   = alpha

    return (sign, logdet)

cpdef double vector_module(np.ndarray[double, ndim=1] X):

    """
        Compute the module of a vector.

        Parameters
        ----------

            X : array of shape (n,)
                The vector.
        
        Returns
        -------

            module : double
                The module of the vector.

    """

    return np.dot(X,X)

cpdef double inner_product_direct_inversion(np.ndarray[double, ndim=1] vector1           ,
                                            np.ndarray[double, ndim=1] vector2           ,
                                            np.ndarray[double, ndim=2] inverse_covariance):

    """

        Compute the inner product between two vectors using the inverse of the covariance matrix.

        Parameters
        ----------

            vector1 : array of shape (n,)
                The first vector.

            vector2 : array of shape (n,)
                The second vector.

            inverse_covariance : array of shape (n,n)
                The inverse of the covariance matrix.

        Returns
        -------

            inner_product : double
                The inner product between the two vectors.

    """

    return np.dot(vector1, np.dot(inverse_covariance, vector2))

cpdef double residuals_inner_product_direct_inversion(np.ndarray[double, ndim=1] residuals         ,
                                                      np.ndarray[double, ndim=2] inverse_covariance):

    """

        Compute the inner product between the residuals using the direct inverse of the covariance matrix.

        Parameters
        ----------

            residuals : array of shape (n,)
                The residuals.

            inverse_covariance : array of shape (n,n)
                The inverse of the covariance matrix.
        
        Returns
        -------

            inner_product : double
                The inner product between the residuals.

    """

    return np.dot(residuals, np.dot(inverse_covariance, residuals))

cpdef double residuals_inner_product_toeplitz_inversion(np.ndarray[double, ndim=1] residuals,
                                                        np.ndarray[double, ndim=1] acf      ):

    """

        Compute the inner product between the residuals using the inverse of the covariance matrix through the scipy `solve_toeplitz` method.

        Parameters
        ----------

            residuals : array of shape (n,)
                The residuals.

            acf : array of shape (n,)
                The autocorrelation function of the residuals.
        
        Returns
        -------

            inner_product : double
                The inner product between the residuals.

    """

    # `solve_toeplitz` returns `np.dot(inv(toeplitz(acf)), residuals)`.
    return np.dot(residuals, solve_toeplitz(acf, residuals))

cpdef double residuals_inner_product_cholesky_solve_triangular(np.ndarray[double, ndim=1] residuals,
                                                               np.ndarray[double, ndim=2] cholesky ):

    """

        Compute the inner product between the residuals using the inverse of the covariance matrix through the scipy `solve_triangular` method, exploiting the Cholesky decomposition.

        Parameters
        ----------

            residuals : array of shape (n,)
                The residuals.

            cholesky : array of shape (n,n)
                The Cholesky decomposition of the covariance matrix.
        
        Returns
        -------

            inner_product : double
                The inner product between the residuals.

    """

    # `solve_triangular` returns the whitened residuals.
    return vector_module(solve_triangular(cholesky, residuals, lower=True))

cpdef np.ndarray[double,ndim=1] project(np.ndarray[double,ndim=1] hs,
                                        np.ndarray[double,ndim=1] hvx,
                                        np.ndarray[double,ndim=1] hvy,
                                        np.ndarray[double,ndim=1] hp,
                                        np.ndarray[double,ndim=1] hc,
                                        object detector,
                                        double ra,
                                        double dec,
                                        double psi,
                                        object tgps):

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

    cdef double gmst, fs, fvx, fvy, fp, fc
    gmst = lal.GreenwichMeanSiderealTime(tgps)
    #The breathing and longitudinal modes act on a L-shaped detector in the same way up to a constant amplitude, thus we just use one. See arXiv:1710.03794.
    fp, fc, fb, fs, fvx, fvy = lal.ComputeDetAMResponseExtraModes(detector.response, ra, dec, psi, gmst)

    cdef np.ndarray[double, ndim=1] waveform = fs*hs + fvx*hvx + fvy*hvy + fp*hp + fc*hc

    return waveform

def loglikelihood(object       model                                 ,
                  object       x                                     ,
                  object       waveform_model                        ,
                  double       ra                                    ,
                  double       dec                                   ,
                  double       psi                                   ,
                  double       t_start                               ,
                  dict         time_delay                            ,
                  str          ref_det                               ,
                  int          truncate                              ,
                  int          duration_n                            ,
                  unsigned int OnsourceACF       = 0                 ,
                  unsigned int MaxEntPSD         = 0                 ,
                  unsigned int Dirac_comb        = 0                 ,
                  unsigned int Zeroing_data      = 0                 ,
                  str          likelihood_method = 'direct-inversion',
                  unsigned int split_inner_prod  = 0                 ):

    return _loglikelihood(model                                ,
                          x                                    ,
                          waveform_model                       ,
                          ra                                   ,
                          dec                                  ,
                          psi                                  ,
                          t_start                              ,
                          time_delay                           ,
                          ref_det                              ,
                          truncate                             ,
                          duration_n                           ,
                          OnsourceACF       = OnsourceACF      ,
                          MaxEntPSD         = MaxEntPSD        ,
                          Dirac_comb        = Dirac_comb       ,
                          Zeroing_data      = Zeroing_data     ,
                          likelihood_method = likelihood_method,
                          split_inner_prod  = split_inner_prod )

cdef double _loglikelihood(object       model                                 ,
                           object       x                                     ,
                           object       waveform_model                        ,
                           double       ra                                    ,
                           double       dec                                   ,
                           double       psi                                   ,
                           double       t_start                               ,
                           dict         time_delay                            ,
                           str          ref_det                               ,
                           int          truncate                              ,
                           int          duration_n                            ,
                           unsigned int OnsourceACF       = 0                 ,
                           unsigned int MaxEntPSD         = 0                 ,
                           unsigned int Dirac_comb        = 0                 ,
                           unsigned int Zeroing_data      = 0                 ,
                           str          likelihood_method = 'direct-inversion',
                           unsigned int split_inner_prod  = 0                 ):

    """

        Compute the log-likelihood of the data given the model.

        Parameters
        ----------

            model : object
                The model.

            x : array of shape (n,)
                The parameters of the model.

            waveform_model : object
                The waveform model.

            ra : double
                The right ascension.

            dec : double
                The declination.

            psi : double
                The polarisation angle.

            t_start : double
                The start time of the waveform.

            time_delay : dict
                The time delay between the detectors.

            ref_det : str
                The reference detector.

            truncate : int
                The number of samples to truncate the waveform.

            duration_n : int
                The number of samples in the waveform.

            OnsourceACF : unsigned int
                Whether to use the on-source ACF. Optional, default is 0.

            MaxEntPSD : unsigned int
                Whether to use the maximum entropy PSD. Optional, default is 0.

            Dirac_comb : unsigned int
                Whether to use the Dirac comb. Optional, default is 0.

            Zeroing_data : unsigned int
                Whether to zero the data. Optional, default is 0.

            likelihood_method : str
                The method to use to compute the log-likelihood. Optional, default is 'direct-inversion'.

            split_inner_prod : unsigned int
                Whether to split the inner product computation. Optional, default is 0.

        Returns
        -------

            logL : double
                The log-likelihood.

    """

    # Initialise the required structures.
    cdef double dt, log_normalisation, dd, dh, hh, residuals_inner_product
    cdef double logL = 0.0
    cdef np.ndarray[double, ndim=1] residuals, time_array, time_array_raw, data, prediction, zeros_array, zerod_data, hs, hvx, hvy, hp, hc, ACF
    cdef np.ndarray[double, ndim=2] inverse_Covariance, cholesky
    cdef object tref

    for d in model.detectors.keys():

        # Set the origin of the time axis. The waveform starts at `t=0`, so we need the `t=0` to correspond to `model.tevent+dt`. Sample times for each detector are `d.time-(model.tevent + dt)`.
        dt   = time_delay['{0}_'.format(ref_det)+d]
        tref = lal.LIGOTimeGPS(t_start+dt+model.tevent)

        # Select the data segment and the corresponding time axis. The `>=` implies a maximum discretisation data loss of dt.
        if not truncate:
            time_array     = model.detectors[d].time - (model.tevent+dt)
            data           = model.detectors[d].time_series
        else:
            # Crop data.
            time_array_raw = model.detectors[d].time - (model.tevent+dt)
            time_array     = time_array_raw[time_array_raw >= t_start][:duration_n]
            data           = model.detectors[d].time_series[time_array_raw >= t_start][:duration_n]

        # Generate the model prediction.
        if waveform_model is not None:

            wf_model             = waveform_model.waveform(time_array)
            hs, hvx, hvy, hp, hc = wf_model[0], wf_model[1], wf_model[2], wf_model[3], wf_model[4]
            prediction           = project(hs, hvx, hvy, hp, hc, model.detectors[d].lal_detector, ra, dec, psi, tref)

            #NO-REVIEW-NEEDED
            if not truncate:
                if(Dirac_comb):     prediction = np.concatenate((data[time_array < t_start], prediction[time_array >= t_start]), axis=None)
                elif(Zeroing_data): data       = np.concatenate((np.zeros(time_array.shape[0], dtype='double')[time_array < t_start], data[time_array >= t_start]), axis=None)

            residuals = data - prediction
        else:
            prediction = np.zeros(data.shape[0], dtype='double')
            residuals  = data

        # When marginalising over the noise properties, update the noise estimate at each sample.
        # WARNING: this section is still experimental
        if(OnsourceACF):
            if(MaxEntPSD):
                freqs_maxent, psd_maxent = noise.mem_psd(residuals, model.detectors[d].sampling_rate)
                df                       = model.detectors[d].sampling_rate/residuals.shape[0]
                ACF_onsource             = 0.5*np.real(np.fft.irfft(psd_maxent*df))*residuals.shape[0]
            else:
                ACF_onsource = noise.acf(residuals)
            # We are using the one-sided PSD, thus it is twice the Fourier transform of the autocorrelation function (see eq. 7.15 of Maggiore Vol.1). We take the real part just to convert the complex output of fft to a real numpy float. The imaginary part if already 0 when coming out of the fft.
            log_normalisation = -0.5*toeplitz_slogdet(ACF_onsource)[1] - 0.5*(len(ACF_onsource))*log2pi
        else:
            log_normalisation = model.detectors[d].log_normalisation

        if not(split_inner_prod):
            # Compute the residuals inner product.
            # The different types of inner products have separate functions since to optimise the computation in cython we need to declare the dimension of the numpy ndarray; given that in some cases we need to pass a matrix to scipy routines, while in some others we need to pass a vector, the dimension is not the same and so the function needs to be specialised.
            if(likelihood_method=='direct-inversion'):
                if(OnsourceACF): inverse_Covariance = inv(toeplitz(ACF_onsource))
                else:            inverse_Covariance = model.detectors[d].inverse_covariance
                residuals_inner_product = residuals_inner_product_direct_inversion(residuals, inverse_Covariance)
            elif(likelihood_method=='cholesky-solve-triangular'):
                if(OnsourceACF): cholesky = np.linalg.cholesky(toeplitz(ACF_onsource))
                else:            cholesky = model.detectors[d].cholesky
                residuals_inner_product = residuals_inner_product_cholesky_solve_triangular(residuals, cholesky)
            elif(likelihood_method=='toeplitz-inversion'):
                if(OnsourceACF): ACF = ACF_onsource
                else:            ACF = model.detectors[d].acf
                residuals_inner_product = residuals_inner_product_toeplitz_inversion(residuals, ACF)
            else: raise ValueError('Unknown likelihood method requested.')

        # This formulation deals only with numbers of O(10^few). Used to further check for numerical stability in the subtraction of two small quantities, hence is implemented only for a single inversion method.
        # IMPROVEME: If the data segment is fixed, the dd term should be computed only once outside the likelihood.
        else:

            if(OnsourceACF): inverse_Covariance = inv(toeplitz(ACF_onsource))
            else:            inverse_Covariance = model.detectors[d].inverse_covariance

            dd = inner_product_direct_inversion(data,       data,       inverse_Covariance)
            dh = inner_product_direct_inversion(data,       prediction, inverse_Covariance)
            hh = inner_product_direct_inversion(prediction, prediction, inverse_Covariance)
            residuals_inner_product = dd - 2. * dh + hh

        # Finally, compute the likelihood.
        logL += -0.5*residuals_inner_product + log_normalisation

    return logL
