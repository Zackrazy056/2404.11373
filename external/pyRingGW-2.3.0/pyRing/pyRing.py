#!/usr/bin/env python
#Standard python imports
import ast, itertools as it, json, h5py, matplotlib, numpy as np, os, shutil, sys, time, traceback, warnings
try:                import configparser
except ImportError: import ConfigParser as configparser
from optparse       import OptionParser

#LVC imports
from lalinference                import DetFrameToEquatorial
from lalinference.imrtgr.nrutils import bbh_final_mass_projected_spins, bbh_final_spin_projected_spins, bbh_Kerr_trunc_opts
import lal, lalsimulation as lalsim

#Package internal imports
import cpnest.model

import pyRing
from pyRing            import initialise, plots, utils, waveform as wf
from pyRing.likelihood import loglikelihood, residuals_inner_product_direct_inversion
from pyRing.noise      import detector



ell_units_factor = (lal.C_SI**2 * 10**3)/(lal.MSUN_SI*lal.G_SI)
matplotlib.rc('text', usetex=False)

def get_param_override(fixed_params, x, name):
    """
        Function returning either a sample or the fixed value for the parameter considered.
        ---------------
        
        Returns x[name], unless it is over-ridden by
        value in the fixed_params dictionary.
        
    """
    if name in fixed_params: return fixed_params[name]
    else:                    return x[name]

def read_parameter_bounds(Config, configparser, kwargs, name, default_bounds, approx_name, extra_polarisation=None):
    
    single_bounds = [0.0,0.0]
    
    if(extra_polarisation==None):
        composite_name = name
    else:
        pol, i = extra_polarisation
        composite_name = '{}_{}_{}'.format(name, pol,i)
        
    try:
        single_bounds[0] = Config.getfloat("Priors", composite_name+'-min')
    except (KeyError, configparser.NoOptionError, configparser.NoSectionError, configparser.NoSectionError):
        single_bounds[0] = default_bounds[name][0]
    try:
        single_bounds[1] = Config.getfloat("Priors", composite_name+'-max')
    except (KeyError, configparser.NoOptionError, configparser.NoSectionError, configparser.NoSectionError):
        single_bounds[1] = default_bounds[name][1]

    # This try-except is needed because we might inject and recover a different set of parameters
    try:
        if (kwargs['injection-approximant']==approx_name):
            if(extra_polarisation==None):
                if (not(single_bounds[0] <= kwargs['injection-parameters']['{}'.format(name)] <= single_bounds[1])): utils.print_out_of_bounds_warning(name)
            else:
                if (name=='logA'):
                    if (not(10**(single_bounds[0]) <= kwargs['injection-parameters']['A'][pol][i] <= 10**(single_bounds[1]))): utils.print_out_of_bounds_warning(name)
                else:
                    if (not(single_bounds[0] <= kwargs['injection-parameters'][name][pol][i] <= single_bounds[1])): utils.print_out_of_bounds_warning(name)
    except(IndexError, KeyError):
        pass

    print(('{} : [{}, {}]'.format(composite_name.ljust(len('cos_altitude')), single_bounds[0], single_bounds[1])))

    return single_bounds



class LIGOVirgoModel(cpnest.model.Model):

    """
    Parent class for all the models used in the package.
    Reads the data and sets sky position parameters common to all waveform models.
    """
        
    def __init__(self, **kwargs):

        super(LIGOVirgoModel,self).__init__()
        self.gr_time_prior     = kwargs['gr-time-prior']
        self.srate             = kwargs['sampling-rate']
        self.dt                = 1./self.srate
        self.noise_chunksize   = kwargs['noise-chunksize']
        self.signal_chunksize  = kwargs['signal-chunksize']
        self.OnsourceACF       = kwargs['onsource-ACF']
        self.likelihood_method = kwargs['likelihood-method']
        self.split_inner_prod  = kwargs['split-inner-products']
        self.Dirac_comb        = kwargs['Dirac-comb']
        self.Zeroing_data      = kwargs['Zeroing-data']
        self.truncate          = kwargs['truncate']
        self.duration_n        = kwargs['analysis-duration-n']

        self.Config = configparser.ConfigParser()
        self.Config.read(kwargs['config-file'])

        utils.print_section('Setting the start time')

        # Estimate of peak time in reference detector.
        self.tevent = kwargs['trigtime']
        self.tgps   = lal.LIGOTimeGPS(float(self.tevent))
        print('Trigtime in {} {}: {:f}\n'.format(kwargs['ref-det'], ''.ljust(12), self.tevent))

        utils.print_section('Data reading')

        # Set up the datafiles and the detectors.
        # Analyze a stretch T of data centered around the t_merger (default trigtime in H1).
        # T must be bigger than light travel time between detectors + lenght of ringdown
        # (light travel time across Earth diameter: 0.042s)
        self.datafiles = {det:kwargs['data-{}'.format(det)] if(not(kwargs['data-{}'.format(det)]=='')) else None for det in kwargs['detectors']}
        self.detectors = {'{}'.format(ifo): detector(ifo, fname, **kwargs) for ifo,fname in list(self.datafiles.items())}
        
        if(kwargs['run-type']=='noise-estimation-only'): exit()
        
        self.len_det       = len(kwargs['detectors'])
        self.ref_det       = kwargs['ref-det']
        self.sky_frame     = kwargs['sky-frame']
        self.signal_seglen = int(self.srate*self.signal_chunksize)
        if(self.len_det > 1): self.non_ref_det = kwargs['nonref-det']

        # Set common parameters structures.
        self.names         = []
        self.bounds        = []
        self.fixed_params  = {}

        utils.print_section('Model initialisation')

        # Set sky position parameters.
        if (self.sky_frame == 'detector'):
            print('* Using detectors network-based sky coordinates.\n')
            self.default_angles_bounds = {'azimuth'      : [0.0,2.0*np.pi],
                                          'cos_altitude' : [-1.0,1.0]     ,
                                          'psi'          : [0.0,np.pi]    }
            print('\n* Using azimuth and cos_altitude to sample sky position.\n')
        elif(self.sky_frame == 'equatorial'):
            print('* Using equatorial sky coordinates.\n')
            self.default_angles_bounds = {'ra'  : [0.0,2.0*np.pi]     ,
                                          'dec' : [-np.pi/2.,np.pi/2.],
                                          'psi' : [0.0,np.pi]         }
        else:
            raise ValueError("Invalid option for sky position coordinates.")

        utils.print_section('Priors')

        print('I\'ll be running with the following prior bounds:\n')

        utils.print_subsection('Extrinsic')
                 
        for name in self.default_angles_bounds.keys():
            try:
                self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
                assert not((name=='dec') and ((self.fixed_params['dec'] > (self.default_angles_bounds['dec'][1])) or (self.fixed_params['dec'] < (self.default_angles_bounds['dec'][0])))), "Injected value of declination out of definition interval."
                assert not((name=='ra')  and ((self.fixed_params['ra']  > (self.default_angles_bounds['ra'][1]))  or (self.fixed_params['ra']  < (self.default_angles_bounds['ra'][0])))),  "Injected value of right ascension out of definition interval."
                assert not((name=='psi') and ((self.fixed_params['psi'] > (self.default_angles_bounds['psi'][1])) or (self.fixed_params['psi'] < (self.default_angles_bounds['psi'][0])))), "Injected value of polarization out of definition interval."
            except(configparser.NoOptionError, configparser.NoSectionError):
                try:
                    single_bounds = [self.Config.getfloat("Priors",name+'-min'),self.Config.getfloat("Priors",name+'-max')]
                except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
                    single_bounds = self.default_angles_bounds[name]
                print(('{} : [{}, {}]'.format(name.ljust(len('cos_altitude')), single_bounds[0], single_bounds[1])))
                if ( not(kwargs['injection-approximant']=='') and (self.len_det == 1) ):
                    if (not(single_bounds[0] <= kwargs['injection-parameters']['{}'.format(name)] <= single_bounds[1])): utils.print_out_of_bounds_warning(name)
                self.names.append('{}'.format(name))
                self.bounds.append(single_bounds)

        if not self.truncate:
            # Compute the probability that the whole chunk is explained by gaussian noise.
            # FIXME: this should use the same call as below, so it can consistently switch between different likelihood methods.
            self.logZnoise = np.sum([-0.5*residuals_inner_product_direct_inversion(d.time_series, d.inverse_covariance)+d.log_normalisation for d in list(self.detectors.values())])
        else:
            try:
            
                if (self.sky_frame == 'detector'):
        
                    cos_altitude_noise = self.fixed_params['cos_altitude']
                    azimuth_noise      = self.fixed_params['azimuth']

                    _, ra_noise, dec_noise = DetFrameToEquatorial(self.detectors[self.ref_det].lal_detector,
                                                       self.detectors[self.non_ref_det].lal_detector,
                                                       self.tevent,
                                                       np.arccos(cos_altitude_noise),
                                                       azimuth_noise)
                elif (self.sky_frame == 'equatorial'):
                    ra_noise  = self.fixed_params['ra']
                    dec_noise = self.fixed_params['dec']

                t_start_noise    = self.Config.getfloat("Priors",'fix-t')
                time_delay_noise = {'{}_'.format(self.ref_det)+d2: lal.ArrivalTimeDiff(self.detectors[d2].location,lal.cached_detector_by_prefix[self.ref_det].location, ra_noise, dec_noise, self.tgps) for d2 in list(self.detectors.keys())}
            
            except (configparser.NoOptionError, configparser.NoSectionError):
                t_start_noise    = 0.0
                time_delay_noise = {'{}_'.format(self.ref_det)+d2: 0.0 for d2 in list(self.detectors.keys())}
                print("In the case where truncation is active and the data segment analyzed varies with t0 and sky location (not currently used), we need to compute a distribution of logZs for each (t0, skypos) sample.\nNOT YET IMPLEMENTED (to be done in post-processing), so computing logZ_noise with t=0.0 in all detectors.")

            if not(kwargs['maxent-psd']==''): self.MaxEntPSD = 1
            else:                             self.MaxEntPSD = 0
                
            self.logZnoise = loglikelihood(self,
                                           None,
                                           None,
                                           0.0,
                                           0.0,
                                           0.0,
                                           t_start_noise         ,
                                           time_delay_noise      ,
                                           self.ref_det          ,
                                           self.truncate         ,
                                           self.duration_n       ,
                                           self.OnsourceACF      ,
                                           self.MaxEntPSD        ,
                                           self.Dirac_comb       ,
                                           self.Zeroing_data     ,
                                           self.likelihood_method,
                                           self.split_inner_prod )

class RingDownModel(LIGOVirgoModel):

    """
    Parent class for all ringdown models.
    Sets common time prior and likelihood.
    """
    
    def __init__(self,**kwargs):

        super(RingDownModel,self).__init__(**kwargs)
        self.Config = configparser.ConfigParser()
        self.Config.read(kwargs['config-file'])
        self.qnm_interpolants = {}
        try:
            self.fixed_params['t'] = self.Config.getfloat("Priors",'fix-t')
        except (configparser.NoOptionError, configparser.NoSectionError):
            if (self.gr_time_prior):
                try:
                    if not(kwargs['injection-approximant']==''): mf_time_prior = kwargs['injection-parameters']['Mf']
                    else:                                        mf_time_prior = kwargs['mf-time-prior']
                    assert (-100 <= self.Config.getfloat("Priors",'t-min') <= 100), "GR prior is currently only compatible with start times of [-100,100]M after the merger. If you know what you are doing, tweak the code to change these hardcoded values."
                    Tmerger = np.array([self.Config.getfloat("Priors",'t-min'),self.Config.getfloat("Priors",'t-max')])*mf_time_prior*lal.MTSUN_SI
                    self.time_prior = [Tmerger[0], Tmerger[1]]
                    print('An estimate of the final mass is being used to select a GR prior on time ([{:.6f}, {:.6f}] Mf after peak): {:.2f}'.format(Tmerger[0], Tmerger[1], mf_time_prior))
                except(KeyError, configparser.NoOptionError, configparser.NoSectionError):
                    raise Exception("\nIf a GR prior on time is selected, an estimate of the final mass, together with start and end times must be passed.")
            else:
                try:
                    self.time_prior = [self.Config.getfloat("Priors",'t-min'),self.Config.getfloat("Priors",'t-max')]
                except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
                    self.time_prior = [0.0,0.01]
                    print("\nWarning: No time prior given, setting prior to default: [{},{}]".format(self.time_prior[0], self.time_prior[1]))
                if((self.time_prior[0] > 1) or (self.time_prior[1] > 1)):
                    print("\nWarning: If no time prior option is passed the time prior is set in units of seconds. You passed a time prior with a value greater than 1s, are you sure you didn't mean to pass it in units of mass?")

        if not('t' in self.fixed_params.keys()):
            assert not((self.time_prior[1]+kwargs['analysis-duration']) > (kwargs['signal-chunksize']/2.) ), "The selected analysis duration plus the upper bound of the time prior, exceeds half of the signal seglen. This will raise an 'index out of bound' error. Please decrease the analysis duration or increase the signal seglen."
            assert not((self.time_prior[1]-self.time_prior[0]) > (kwargs['signal-chunksize'])), "The selected time prior range exceeds the signal seglen. This will raise an 'index out of bound' error. Please decrease the time prior ranges or increase the signal seglen."
            if (self.gr_time_prior):
                if(((self.time_prior[0])/(mf_time_prior*lal.MTSUN_SI) < 20) and (kwargs['template']=='MMRDNP')):
                    print("\nWarning: You are using the MMRDNP model outside its calibration domain.")
                elif(((self.time_prior[0])/(mf_time_prior*lal.MTSUN_SI) < 10) and (kwargs['template']=='MMRDNS')):
                    print("\nWarning: You are using the MMRDNS model outside its calibration domain.")
        else:
            assert not((self.fixed_params['t']+kwargs['analysis-duration']) > (kwargs['signal-chunksize']/2.)), "The selected analysis duration plus the fixed time, exceeds half of the signal seglen. This will raise an 'index out of bound' error. Please decrease the analysis duration or increase the signal seglen."

    def log_likelihood(self,x):

        if (self.sky_frame == 'detector'):
            
            # ===================================================================================================================#
            # DetFrameToEquatorial makes the sampling on sky position parameters much easier                                     #
            # by parametrizing in a convenient way the ring in the sky in the case of two detectors.                             #
            # In the one detector case there is no advantage in using it, thus we will rely on ra and dec directly,              #
            # while in the three detector case we will use as detectors H1 and L1 because they break the degeneracy more easily. #
            # We need to pass the trigtime of the first detector as input.                                                       #
            # The returned tg is the time at geocenter.                                                                          #
            # ===================================================================================================================#
            
            cos_altitude = get_param_override(self.fixed_params,x,'cos_altitude')
            azimuth      = get_param_override(self.fixed_params,x,'azimuth')

            tg, ra, dec = DetFrameToEquatorial(self.detectors[self.ref_det].lal_detector,
                                               self.detectors[self.non_ref_det].lal_detector,
                                               self.tevent,
                                               np.arccos(cos_altitude),
                                               azimuth)
        elif (self.sky_frame == 'equatorial'):
            ra  = get_param_override(self.fixed_params,x,'ra')
            dec = get_param_override(self.fixed_params,x,'dec')
        else:
            raise ValueError("Invalid option for sky position sampling.")

        psi = get_param_override(self.fixed_params,x,'psi')

        if ('t' in self.fixed_params): t_start = self.fixed_params['t']
        else:                          t_start = x['t0']

        self.time_delay = {'{}_'.format(self.ref_det)+d2: lal.ArrivalTimeDiff(self.detectors[d2].location,lal.cached_detector_by_prefix[self.ref_det].location, ra, dec, self.tgps) for d2 in list(self.detectors.keys())}
        wf_model        = self.get_waveform(x)
        
        logL            = loglikelihood(self                  ,
                                        x                     ,
                                        wf_model              ,
                                        ra                    ,
                                        dec                   ,
                                        psi                   ,
                                        t_start               ,
                                        self.time_delay       ,
                                        self.ref_det          ,
                                        self.truncate         ,
                                        self.duration_n       ,
                                        self.OnsourceACF      ,
                                        self.MaxEntPSD        ,
                                        self.Dirac_comb       ,
                                        self.Zeroing_data     ,
                                        self.likelihood_method,
                                        self.split_inner_prod )
                                                
        return logL

class IMR_Ringdown_Model(RingDownModel):
    
    """
    Class implementing a non-precessing ringdown model extracted from LAL IMR waveforms.
    """

    def __init__(self, **kwargs):
        super(IMR_Ringdown_Model,self).__init__(**kwargs)

        utils.review_warning()

        self.Config = configparser.ConfigParser()
        self.Config.read(kwargs['config-file'])

        #FIXME: read it from config
        IMR_approx_name = 'LAL-X'

        # ==================================#
        # Initialise model names and bounds #
        # for mandatory parameters.         #
        # ==================================#
        
        utils.print_subsection('Inspiral model')

        self.default_bounds = {'m1'     : [20.,100.]     ,
                               'm2'     : [20.,100.]     ,
                               's1z'    : [-0.99,0.99]   ,
                               's2z'    : [-0.99,0.99]   ,
                               'dist'   : [10.,1000.]    ,
                               'cosiota': [-1.0,1.0]     ,
                               'phi'    : [0.0, 2*np.pi] }

        if not('t' in self.fixed_params):
            self.names.append('t0')
            self.bounds.append(self.time_prior)
            print(('{}: [{},{}]'.format('t0'.ljust(len('cosiota')), self.time_prior[0], self.time_prior[1])))

        for name in self.default_bounds.keys():
            try:
                self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
            except(configparser.NoOptionError):
                single_bounds = read_parameter_bounds(self.Config, configparser, kwargs, name, self.default_bounds, IMR_approx_name)
                self.names.append('{}'.format(name))
                self.bounds.append(single_bounds)

    def log_prior(self,x):
        
        #Order the masses: m1>m2, same convention as in LAL.
        try:
            if x['m1']<x['m2']: return -np.inf
        except(KeyError):
            pass
        return super(RingDownModel,self).log_prior(x)

    def get_waveform(self,x):

        insp_params = {}
        for name in self.names:
            insp_params[name] = get_param_override(self.fixed_params,x,name)

        return wf.IMR_WF(insp_params['m1'],
                         insp_params['m2'],
                         insp_params['s1z'],
                         insp_params['s2z'],
                         insp_params['dist'],
                         insp_params['cosiota'],
                         insp_params['phi'],
                         insp_params['t0'],
                         self.dt,
                         float(0.0),#FIXME needs to read the start-time
                         float(self.signal_seglen))

class DampedSinusoids(RingDownModel):

    """
    Class implementing a superposition of damped sinusoids as ringdown model.
    Frequency, damping time, amplitude and phase are sampled over for each mode.
    """

    def __init__(self,**kwargs):
        super(DampedSinusoids,self).__init__(**kwargs)

        self.Config = configparser.ConfigParser()
        self.Config.read(kwargs['config-file'])

        # ==================================#
        # Initialise model names and bounds #
        # for mandatory parameters.         #
        # ==================================#

        utils.print_subsection('Damped Sinusoids model')

        self.default_bounds = {'logA': [-22.,-20.]     ,
                               'f'   : [20., 1000.]    ,
                               'tau' : [0.0005,0.0500] ,
                               'phi' : [0.0,2.0*np.pi] }

        self.n_modes         = kwargs['n-ds-modes']
        self.mode_ordering   = kwargs['ds-ordering']
        self.amp_flat_prior  = kwargs['ds-amp-flat-prior']

        self.multiple_modes = False
        for pol in self.n_modes.keys():
            if(self.n_modes[pol] > 1): self.multiple_modes = True
        
        #To avoid incompatibilities, all polarisations start at the same time, since t0 is used in likelihood. If needed, change this to allow for different start times.
        if not('t' in self.fixed_params):
            for pol in self.n_modes.keys():
                for i in (list(range(self.n_modes[pol]))):
                    self.names.append('t{}'.format(i))
                    self.bounds.append(self.time_prior)
                    print(('{}{}: [{}, {}]'.format('t', str(i), self.time_prior[0], self.time_prior[1])))
                break

        # Since we are agnostic, all the modes are given the same prior bounds.
        for pol in self.n_modes.keys():
            for i,name in it.product(list(range(self.n_modes[pol])),self.default_bounds.keys()):
                try:
                    self.fixed_params[name+'_{}_{}'.format(pol,i)] = self.Config.getfloat("Priors",'fix-'+name+'_{}_{}'.format(pol,i))
                except(configparser.NoOptionError):
                    single_bounds = read_parameter_bounds(self.Config, configparser, kwargs, name, self.default_bounds, 'Damped-sinusoids', extra_polarisation=[pol,i])
                    if((name=='f') and (single_bounds[0]<kwargs['f-min-bp'])):
                        warnings.warn("The prior upper bound on the frequency is smaller than the smallest frequency allowed by bandpassing. Setting the lower bound of the frequency to the smallest bandpassing frequency={} for consistency.".format(kwargs['f-min-bp']))
                        single_bounds[0] = kwargs['f-min-bp']
                    if((name=='f') and (single_bounds[1]>kwargs['f-max-bp'])):
                        warnings.warn("The prior upper bound on the frequency is higher than the highest frequency allowed by bandpassing. Setting the upper bound of the frequency to the highest bandpassing frequency={} for consistency.".format(kwargs['f-max-bp']))
                        single_bounds[1] = kwargs['f-max-bp']
                    self.names.append('{}_{}_{}'.format(name, pol, i))
                    self.bounds.append(single_bounds)

        utils.print_subsection('Fixed')

        if not self.fixed_params:
            print('\n* No parameter was fixed.')
        else:
            for name in self.fixed_params.keys():
                print('{} : {}'.format(name.ljust(len('cos_altitude')), self.fixed_params[name]))

    def log_prior(self,x):
        
        if(self.amp_flat_prior):
            for pol in self.n_modes.keys():
                if(self.n_modes[pol]>1):
                    for i in range(1,self.n_modes[pol]):
                        logPrior += x['logA_{}_{}'.format(pol,i)]

        if(self.multiple_modes):
            # Order the modes per given polarisation (same as m1>m2 in LAL).
            for pol in self.n_modes.keys():
                if(self.n_modes[pol]>1):
                    for i in range(1,self.n_modes[pol]):
                        # Skip when fixing parameters
                        try:
                            if((self.mode_ordering=='freq')   and (x['f_{}_{}'.format(   pol,i)] < x['f_{}_{}'.format(   pol,i-1)])): return -np.inf
                            elif((self.mode_ordering=='tau')  and (x['tau_{}_{}'.format( pol,i)] < x['tau_{}_{}'.format( pol,i-1)])): return -np.inf
                            elif((self.mode_ordering=='amp')  and (x['logA_{}_{}'.format(pol,i)] < x['logA_{}_{}'.format(pol,i-1)])): return -np.inf
                            elif((self.mode_ordering=='time') and (x['t_{}_{}'.format(   pol,i)] < x['t_{}_{}'.format(   pol,i-1)])): return -np.inf
                        except(KeyError): pass
                        
        return super(RingDownModel,self).log_prior(x)

    def get_waveform(self,x):

        params = {}
        for name in self.default_bounds.keys():
            if ('logA' not in name):
                params[name] = {}
                for pol in self.n_modes.keys():
                    params[name][pol] = []
                    for i in range(self.n_modes[pol]):
                        try:    params[name][pol].append(self.fixed_params[name+'_{}_{}'.format(pol,i)])
                        except: params[name][pol].append(x[name+'_{}_{}'.format(pol,i)])
            else:
                params['A'] = {}
                for pol in self.n_modes.keys():
                    params['A'][pol] = []
                    for i in range(self.n_modes[pol]):
                        try:    params['A'][pol].append(10**(self.fixed_params[name+'_{}_{}'.format(pol,i)]))
                        except: params['A'][pol].append(10**(x[name+'_{}_{}'.format(pol,i)]))
        params['t'] = {}
        for pol in self.n_modes.keys():
            params['t'][pol] = []
            for i in range(self.n_modes[pol]):
                try:    params['t'][pol].append(self.fixed_params['t'])
                except: params['t'][pol].append(x['t{}'.format(i)])

        return wf.Damped_sinusoids(params['A']  ,
                                   params['f']  ,
                                   params['tau'],
                                   params['phi'],
                                   params['t']  )


class MorletGaborWavelets(RingDownModel):

    """
    Class implementing a superposition of Morlet-Gabor-wavelets as waveform model.
    Frequency, damping time, amplitude and phase are sampled over for each mode.
    """

    def __init__(self,**kwargs):
        super(MorletGaborWavelets,self).__init__(**kwargs)

        self.Config = configparser.ConfigParser()
        self.Config.read(kwargs['config-file'])

        # ==================================#
        # Initialise model names and bounds #
        # for mandatory parameters.         #
        # ==================================#

        utils.print_subsection('Intrinsic: Morlet-Gabor model')

        self.default_bounds = {'logA': [-22., -20.]    ,
                               'f'   : [20., 1000.]    ,
                               'tau' : [0.0005, 0.0500],
                               'phi' : [0.0, 2.0*np.pi]}

        self.n_modes = kwargs['n-ds-modes']

        #To avoid incompatibilities, all polarisations start at the same time, since t0 is used in likelihood. If needed, change this to allow for different start times.

        if not('t' in self.fixed_params):
            for pol in self.n_modes.keys():
                for i in (list(range(self.n_modes[pol]))):
                    self.names.append('t{}'.format(i))
                    self.bounds.append(self.time_prior)
                    print(('{}{}: [{}, {}]'.format('t', str(i), self.time_prior[0], self.time_prior[1])))
                break

        # Since we are agnostic, all the modes are given the same prior bounds.
        for pol in self.n_modes.keys():
            for i,name in it.product(list(range(self.n_modes[pol])),self.default_bounds.keys()):
                try:
                    self.fixed_params[name+'_{}_{}'.format(pol,i)] = self.Config.getfloat("Priors",'fix-'+name+'_{}_{}'.format(pol,i))
                except(configparser.NoOptionError):
                    single_bounds = read_parameter_bounds(self.Config, configparser, kwargs, name, self.default_bounds, 'Morlet-Gabor-wavelets', extra_polarisation=[pol,i])
                    if((name=='f') and (single_bounds[0]<kwargs['f-min-bp'])):
                        warnings.warn("The prior upper bound on the frequency is smaller than the smallest frequency allowed by bandpassing. Setting the lower bound of the frequency to the smallest bandpassing frequency={} for consistency.".format(kwargs['f-min-bp']))
                        single_bounds[0] = kwargs['f-min-bp']
                    if((name=='f') and (single_bounds[1]>kwargs['f-max-bp'])):
                        warnings.warn("The prior upper bound on the frequency is higher than the highest frequency allowed by bandpassing. Setting the upper bound of the frequency to the highest bandpassing frequency={} for consistency.".format(kwargs['f-max-bp']))
                        single_bounds[1] = kwargs['f-max-bp']
                    self.names.append('{}_{}_{}'.format(name, pol, i))
                    self.bounds.append(single_bounds)

        utils.print_subsection('Fixed')

        if not self.fixed_params:
            print('\n* No parameter was fixed.')
        else:
            for name in self.fixed_params.keys():
                print('{} : {}'.format(name.ljust(len('cos_altitude')), self.fixed_params[name]))

    def log_prior(self,x):
        # Order the frequencies per given polarisation (same as m1>m2 in LAL).
        for pol in self.n_modes.keys():
            for i in range(1,self.n_modes[pol]):
                try:
                    if (x['f_{}_{}'.format(pol,i)] < x['f_{}_{}'.format(pol,i-1)]):
                        return -np.inf
                except(KeyError):
                    pass
        return super(RingDownModel,self).log_prior(x)

    def get_waveform(self,x):

        params = {}
        for name in self.default_bounds.keys():
            if ('logA' not in name):
                params[name] = {}
                for pol in self.n_modes.keys():
                    params[name][pol] = []
                    for i in range(self.n_modes[pol]):
                        try:    params[name][pol].append(self.fixed_params[name+'_{}_{}'.format(pol,i)])
                        except: params[name][pol].append(x[name+'_{}_{}'.format(pol,i)])
            else:
                params['A'] = {}
                for pol in self.n_modes.keys():
                    params['A'][pol] = []
                    for i in range(self.n_modes[pol]):
                        try:    params['A'][pol].append(10**(self.fixed_params[name+'_{}_{}'.format(pol,i)]))
                        except: params['A'][pol].append(10**(x[name+'_{}_{}'.format(pol,i)]))
        params['t'] = {}
        for pol in self.n_modes.keys():
            params['t'][pol] = []
            for i in range(self.n_modes[pol]):
                try:    params['t'][pol].append(self.fixed_params['t'])
                except: params['t'][pol].append(x['t{}'.format(i)])

        return wf.Morlet_Gabor_wavelets(params['A']  ,
                                        params['f']  ,
                                        params['tau'],
                                        params['phi'],
                                        params['t']  )

class KerrModel(RingDownModel):

    """
    Class implementing a Kerr waveform as ringdown model.
    Frequencies and damping times of the specified (s,l,m,n) modes are fixed as a function of the mass and spin of the remnant BH.
    Relative amplitudes and phases of the modes are instead free to vary.
    """

    def __init__(self, **kwargs):
        super(KerrModel,self).__init__(**kwargs)
        
        self.Config     = configparser.ConfigParser()
        self.Config.read(kwargs['config-file'])

        self.modes      = kwargs['kerr-modes']
        self.quad_modes = kwargs['quadratic-modes']
        self.tail_modes = kwargs['tail-modes']

        # ==================================#
        # Initialise model names and bounds #
        # for mode-independent parameters.  #
        # ==================================#

        self.default_bounds = {'Mf'         : [10.0,500.0]                  ,
                               'af'         : [0.0,0.99]                    ,
                               'logdistance': [np.log(10.0),np.log(10000.0)],
                               'cosiota'    : [-1.0,1.0]                    ,
                               'phi'        : [0.0, 2*np.pi]                }

        # Testing GR options
        self.TGR_params             = []
        self.AreaQuantization       = kwargs['area-quantization']
        self.braneworld             = kwargs['braneworld']
        self.charge                 = kwargs['charge']
        self.ParSpec                = kwargs['ParSpec']
        self.ParSpec_Dmax_TGR       = kwargs['ParSpec_Dmax_TGR']
        self.ParSpec_Dmax_charge    = kwargs['ParSpec_Dmax_charge']
        self.EsGB                   = kwargs['EsGB']
        self.default_AQ_names       = ['alpha']
        self.default_AQ_bounds      = {'alpha': [0.0,50.0]}
        self.tau_AQ                 = kwargs['tau-AQ']
        self.domega_tgr_modes       = kwargs['domega-tgr-modes']
        self.dtau_tgr_modes         = kwargs['dtau-tgr-modes']
        self.TGR_overtones_ordering = kwargs['TGR-overtones-ordering']

        if(self.EsGB):
            
            if(self.ParSpec_Dmax_TGR>2):
                warnings.warn("Einstein-scalar-Gauss-Bonnet option is currently only compatible with M_max <=2. Forcing `ParSpec_Dmax_TGR` = 2.")
                self.ParSpec_Dmax_TGR    = 2
            if not(self.ParSpec_Dmax_charge==4):
                warnings.warn("Einstein-scalar-Gauss-Bonnet option is currently only compatible with p=4. Forcing `ParSpec_Dmax_charge` = 4.")
                self.ParSpec_Dmax_charge = 4
            
            self.EsGB_corrections_dict = {}
            self.EsGB_corrections_dict['domega_220'] = utils.EsGB_corrections('domega_220', self.ParSpec_Dmax_TGR)
            self.EsGB_corrections_dict['dtau_220']   = utils.EsGB_corrections('dtau_220'  , self.ParSpec_Dmax_TGR)

        # GR options
        self.dist_flat_prior        = kwargs['dist-flat-prior']
        self.coherent_n             = kwargs['coherent-n']
        self.Spheroidal             = kwargs['spheroidal']
        self.qnm_fit                = kwargs['qnm-fit']
        self.amp_non_prec_sym       = kwargs['amp-non-prec-sym']
        self.prior_reweight         = kwargs['prior-reweight']
        self.reference_amplitude    = kwargs['reference-amplitude']
        self.max_amp_ratio          = kwargs['max-Kerr-amp-ratio']
        self.quad_lin_prop          = kwargs['quadratic-linear-prop']

        # Consistency checks
        if(self.max_amp_ratio < 0.0): raise ValueError("The maximum Kerr amplitudes ratio is positive defined. Aborting.")
   
        if(self.EsGB or (self.ParSpec and (self.ParSpec_Dmax_charge > 0))):
            if not(self.reference_amplitude==0.0):
                raise ValueError('The `reference-amplitude` option was passed, but sampling on the distance is required when considering corrections determined by dimensionful couplings (since they are rescaled by the source frame mass, hence we need to remove the redshift factor).')
   
        utils.print_subsection('Intrinsic: Kerr model')

        utils.check_modes_naming_scheme(self.modes, self.quad_modes)
        print('* Running the Kerr model with modes: `{}`.\n'.format(self.modes))
   
        # Read mode-independent parameters from config file
        if not('t' in self.fixed_params):
            self.names.append('t0')
            self.bounds.append(self.time_prior)
            print(('{} : [{},{}]'.format('t0'.ljust(len('cos_altitude')), self.time_prior[0], self.time_prior[1])))

        for name in self.default_bounds.keys():
            if(name=='logdistance' and self.reference_amplitude): continue
            try:
                self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
            except(configparser.NoOptionError):
                single_bounds = read_parameter_bounds(self.Config, configparser, kwargs, name, self.default_bounds, 'Kerr')
                self.names.append('{}'.format(name))
                self.bounds.append(single_bounds)

        utils.print_subsection('Intrinsic: Beyond-Kerr')

        # Read beyond-Kerr parameters
        if ((self.domega_tgr_modes is not None) or (self.dtau_tgr_modes is not None)):
            if (self.domega_tgr_modes is not None):
                for mode in self.domega_tgr_modes:
                    (l,m,n) = mode
                    if not(self.ParSpec):
                        name = 'domega_{}{}{}'.format(l, m, n)
                        bound = [-1.0,1.0]
                        self.names.append(name)
                        self.bounds.append(bound)
                        self.TGR_params.append(name)
                        print(('{} : [{},{}]'.format(name.ljust(len('cos_altitude')), bound[0], bound[1])))
                    else:
                        basename = 'domega_{}{}{}'.format(l, m, n)
                        self.TGR_params.append(basename)
                        if not(self.EsGB):
                            for i in range(0,self.ParSpec_Dmax_TGR+1):
                                name  = basename+'_{}'.format(i)
                                bound = [-0.5,0.5]
                                self.names.append(name)
                                self.bounds.append(bound)
                                print(('{} : [{},{}]'.format(name.ljust(len('cos_altitude')), bound[0], bound[1])))

            if (self.dtau_tgr_modes is not None):
                for mode in self.dtau_tgr_modes:
                    (l,m,n) = mode
                    if not(self.ParSpec):
                        name = 'dtau_{}{}{}'.format(l, m, n)
                        bound = [-0.9,1.0]
                        self.names.append(name)
                        self.bounds.append(bound)
                        self.TGR_params.append(name)
                        print(('{} : [{},{}]'.format(name.ljust(len('cos_altitude')), bound[0], bound[1])))
                    else:
                        basename = 'dtau_{}{}{}'.format(l, m, n)
                        self.TGR_params.append(basename)
                        if not(self.EsGB):
                            for i in range(0,self.ParSpec_Dmax_TGR+1):
                                name  = basename+'_{}'.format(i)
                                bound = [-0.5,0.5]
                                self.names.append(name)
                                self.bounds.append(bound)
                                print(('{} : [{},{}]'.format(name.ljust(len('cos_altitude')), bound[0], bound[1])))

            if((self.ParSpec) and (self.ParSpec_Dmax_charge > 0)):
                self.names.append('ell')
                self.bounds.append([0,75])#km units
                print(('{} : [{},{}]'.format('ell'.ljust(len('cos_altitude')), 0, 75)))

        elif (self.AreaQuantization):
            utils.review_warning()
            if (self.tau_AQ):
                self.default_AQ_names.append('tau_AQ')
                self.default_AQ_bounds['tau_AQ'] = [0.0005,0.0500]
            for name in self.default_AQ_names:
                try:
                    self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
                except(configparser.NoOptionError):
                    try:
                        single_bounds = [self.Config.getfloat("Priors",name+'-min'),self.Config.getfloat("Priors",name+'-max')]
                    except(KeyError, configparser.NoOptionError, configparser.NoSectionError):
                        single_bounds = self.default_AQ_bounds[name]
                    print(('{} : [{},{}]'.format(name.ljust(len('cos_altitude')), single_bounds[0], single_bounds[1])))
                    if (kwargs['inject-area-quantization']):
                        if (not(single_bounds[0] <= kwargs['injection-parameters']['{}'.format(name)] <= single_bounds[1])):  utils.print_out_of_bounds_warning(name)
                    self.names.append('{}'.format(name))
                    self.bounds.append(single_bounds)
        elif (self.charge):
            utils.review_warning()
            charge_single_bounds = [0,1]
            charge_name = 'Q'
            try:
                self.fixed_params[charge_name] = self.Config.getfloat("Priors",'fix-'+charge_name)
            except(configparser.NoOptionError):
                try:
                    single_bounds = [self.Config.getfloat("Priors",charge_name+'-min'),self.Config.getfloat("Priors",charge_name+'-max')]
                except(KeyError, configparser.NoOptionError, configparser.NoSectionError):
                    single_bounds = charge_single_bounds
                print(('{} : [{},{}]'.format(charge_name.ljust(len('cos_altitude')), single_bounds[0], single_bounds[1])))
                if (kwargs['inject-charge']):
                    if (not(single_bounds[0] <= kwargs['injection-parameters'][charge_name] <= single_bounds[1])): utils.print_out_of_bounds_warning(charge_name)
                self.names.append(charge_name)
                self.bounds.append(single_bounds)
        elif (self.braneworld):
            utils.review_warning()
            braneworld_default_bounds = [-0.495,0]
            braneworld_name = 'beta'
            try:
                self.fixed_params[braneworld_name] = self.Config.getfloat("Priors",'fix-'+braneworld_name)
            except(configparser.NoOptionError):
                single_bounds = [0.0,0.0]

                try                                                                      : single_bounds[0] = self.Config.getfloat("Priors",braneworld_name+'-min')
                except(KeyError, configparser.NoOptionError, configparser.NoSectionError): single_bounds[0] = braneworld_default_bounds[0]
                try                                                                      : single_bounds[1] = self.Config.getfloat("Priors",braneworld_name+'-max')
                except(KeyError, configparser.NoOptionError, configparser.NoSectionError): single_bounds[1] = braneworld_default_bounds[1]

                print(('{} : [{},{}]'.format(braneworld_name.ljust(len('cos_altitude')), single_bounds[0], single_bounds[1])))
                if (kwargs['inject-braneworld']):
                    if (not(single_bounds[0] <= kwargs['injection-parameters'][braneworld_name] <= single_bounds[1])): utils.print_out_of_bounds_warning(braneworld_name)
                self.names.append(braneworld_name)
                self.bounds.append(single_bounds)

        else:
            print('* No TGR parameter was considered.')

        # Read amplitudes and phases
        utils.print_subsection('Modes amplitudes')

        amps,phis = self.amp_phi_names()
        for name in np.concatenate((amps, phis), axis=0):
            name = str(name)
            try:
                self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
            except(configparser.NoOptionError):
                try:
                    single_bounds = [self.Config.getfloat("Priors",name+'-min'),self.Config.getfloat("Priors",name+'-max')]
                except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
                    if ('phi' in name): single_bounds = [0.0, 2*np.pi]
                    elif ('A' in name): single_bounds = [0.0, 50.0]
                    else              : raise ValueError("* Kerr amplitudes section: you tried to fix an unknown parameter.")
                self.names.append(name)
                self.bounds.append(single_bounds)
                print(('{} : [{}, {}]'.format(name.ljust(len('cos_altitude')), single_bounds[0], single_bounds[1])))

        # Read tail parameters
        if(self.tail_modes is not None):
            utils.print_subsection('Tails')

            tails_amps, tails_phis, tails_ps = self.tail_names()
            for name in np.concatenate((tails_amps, tails_phis, tails_ps), axis=0):
                name = str(name)
                try:
                    self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
                except(configparser.NoOptionError):
                    try:
                        single_bounds = [self.Config.getfloat("Priors",name+'-min'),self.Config.getfloat("Priors",name+'-max')]
                    except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
                        if ('phi-tail-' in name): single_bounds = [0.0, 2*np.pi]
                        elif ('A-tail-' in name): single_bounds = [0.0, 50.0]
                        elif ('p-tail-' in name): single_bounds = [0.0, 10.0]
                        else                    : raise ValueError("* Kerr tails section: you tried to fix an unknown parameter.")
                    self.names.append(name)
                    self.bounds.append(single_bounds)
                    print(('{} : [{}, {}]'.format(name.ljust(len('cos_altitude')), single_bounds[0], single_bounds[1])))

        # Print fixed parameters
        utils.print_subsection('Fixed')
        utils.print_fixed_parameters(self.fixed_params)

        # Build prior reweighting interpolants
        if not(self.prior_reweight[0]==''):
            self.spline_interpolants = {}
            for param_name in self.prior_reweight: self.spline_interpolants[param_name] = utils.construct_interpolant(param_name)

        # Build QNM interpolants
        if(kwargs['qnm-fit'] == 0):
            full_modes = utils.construct_full_modes(self.modes, self.quad_modes)

            for (s,l,m,n) in full_modes:

                if(self.charge):       interpolate_freq, interpolate_tau = utils.qnm_interpolate_KN(s,l,m,n)
                elif(self.braneworld): interpolate_freq, interpolate_tau = utils.qnm_interpolate_braneworld(s,l,m,n)
                else:                  interpolate_freq, interpolate_tau = utils.qnm_interpolate(s,l,m,n)
                self.qnm_interpolants[(s,l,m,n)]  = {'freq': interpolate_freq, 'tau': interpolate_tau}

    def log_prior(self,x):
        logPrior = 0
        if('logdistance' in self.names):
            if(self.dist_flat_prior): logPrior +=     x['logdistance']
            else:                     logPrior += 3.0*x['logdistance'] #Assume uniform comoving density, hence p(D|I)~D^2 --> p(log(D)|I)~D^3
        
        #If requested by the user, impose a hierarchy on the amplitudes. Higher modes cannot have an amplitude bigger than the one of the l=m=2, n=0. Does not apply to overtones of the l=m=2 mode.
        if(not(self.max_amp_ratio==0.0) and (('A2220' in self.names) or ('A2220_1' in self.names))):
            for (s,l,m,n) in self.modes:
                if(l==2 and m==2): continue
                else:
                    if(self.amp_non_prec_sym):
                        if(x['A{}{}{}{}'.format(s,l,m,n)]   > self.max_amp_ratio * x['A2220']  ): return -np.inf
                    else:
                        if(x['A{}{}{}{}_1'.format(s,l,m,n)] > self.max_amp_ratio * x['A2220_1']): return -np.inf
                        if(x['A{}{}{}{}_2'.format(s,l,m,n)] > self.max_amp_ratio * x['A2220_2']): return -np.inf
    
        if not(self.prior_reweight[0]==''):
            utils.review_warning()
            for param_name in self.prior_reweight: logPrior += np.log(self.spline_interpolants[param_name](x[param_name]))

        # You shall not break the Cosmic Censorship Conjecture.
        if(self.charge):
            af_tmp = get_param_override(self.fixed_params,x,'af')
            Q_tmp  = get_param_override(self.fixed_params,x,'Q')

            if not(af_tmp**2 + Q_tmp**2 < 0.99): return -np.inf

        # Implement an ordering of the modes when testing GR using overtones. This is not imposing the GR modes structure to the template, since all possible template values can still be reached by shifting (Mf,af), it is just preventing that a given mode in the data can be latched upon by different modes in the template.
        if((self.domega_tgr_modes is not None) and (self.dtau_tgr_modes is not None) and not(self.TGR_overtones_ordering=='Unordered')):
            # The fundamental mode is defined to be the one with the largest damping time OR largest frequency.
            if(self.TGR_overtones_ordering=='freq'):
                for mode in self.domega_tgr_modes:
                    (l,m,n)   = mode
                    next_mode = (2,l,m,n+1)
                    # If higher overtones are included, check that the mode ordering is not broken by these higher overtones.
                    if(next_mode in self.modes):
                        freq_eff              = wf.QNM_fit(l,m,  n).f(x['Mf'], x['af'])*(1.0+x['domega_{}{}{}'.format(l, m,   n)])
                        try:    freq_eff_next = wf.QNM_fit(l,m,n+1).f(x['Mf'], x['af'])*(1.0+x['domega_{}{}{}'.format(l, m, n+1)])
                        except: freq_eff_next = wf.QNM_fit(l,m,n+1).f(x['Mf'], x['af'])
                        if(freq_eff < freq_eff_next): return -np.inf
                    # Here we are supposing that all the lower overtones wrt a given n are included (except for the n=0 of course) and check for mode ordering compared to lower overtones.
                    if(n>0):
                        freq_eff              = wf.QNM_fit(l,m,  n).f(x['Mf'], x['af'])*(1.0+x['domega_{}{}{}'.format(l, m,   n)])
                        try:    freq_eff_prev = wf.QNM_fit(l,m,n-1).f(x['Mf'], x['af'])*(1.0+x['domega_{}{}{}'.format(l, m, n-1)])
                        except: freq_eff_prev = wf.QNM_fit(l,m,n-1).f(x['Mf'], x['af'])
                        if(freq_eff_prev < freq_eff): return -np.inf
            elif(self.TGR_overtones_ordering=='tau'):
                for mode in self.dtau_tgr_modes:
                    (l,m,n)   = mode
                    next_mode = (2,l,m,n+1)
                    # If higher overtones are included, check that the mode ordering is not broken by these higher overtones.
                    if(next_mode in self.modes):
                        tau_eff              = wf.QNM_fit(l,m,  n).tau(x['Mf'], x['af'])*(1.0+x['dtau_{}{}{}'.format(l, m,   n)])
                        try:    tau_eff_next = wf.QNM_fit(l,m,n+1).tau(x['Mf'], x['af'])*(1.0+x['dtau_{}{}{}'.format(l, m, n+1)])
                        except: tau_eff_next = wf.QNM_fit(l,m,n+1).tau(x['Mf'], x['af'])
                        if(tau_eff_next > tau_eff): return -np.inf
                    # Here we are supposing that all the lower overtones wrt a given n are included (except for the n=0 of course) and check for mode ordering compared to lower overtones.
                    if(n>0):
                        tau_eff              = wf.QNM_fit(l,m,  n).tau(x['Mf'], x['af'])*(1.0+x['dtau_{}{}{}'.format(l, m,   n)])
                        try:    tau_eff_prev = wf.QNM_fit(l,m,n-1).tau(x['Mf'], x['af'])*(1.0+x['dtau_{}{}{}'.format(l, m, n-1)])
                        except: tau_eff_prev = wf.QNM_fit(l,m,n-1).tau(x['Mf'], x['af'])
                        if(tau_eff_prev < tau_eff): return -np.inf

        return super(KerrModel,self).log_prior(x)+logPrior

    def tail_names(self):

        # Initialise structures
        tails_amps, tails_phis, tails_ps = [], [], []
        for (l,m) in self.tail_modes:
        
            tail_string = '{}{}'.format(l,m)

            if(self.amp_non_prec_sym):
                tails_amps.append( 'A-tail-' +tail_string)
                tails_phis.append('phi-tail-'+tail_string)
                tails_ps.append(   'p-tail-' +tail_string)

            else:
                tails_amps.append( 'A-tail-' +tail_string+'_1')
                tails_amps.append( 'A-tail-' +tail_string+'_2')
                tails_phis.append('phi-tail-'+tail_string+'_1')
                tails_phis.append('phi-tail-'+tail_string+'_2')
                tails_ps.append(   'p-tail-' +tail_string+'_1')
                tails_ps.append(   'p-tail-' +tail_string+'_2')

        return tails_amps, tails_phis, tails_ps

    def amp_phi_names(self):

        # Initialise structures
        amps, phis = [], []

        # Fill-in linear amplitudes names
        for (s,l,m,n) in self.modes:
            if(self.amp_non_prec_sym):
                amps.append('A'+self.modename(s,l,m,n))
            else:
                amps.append('A'+self.modename(s,l,m,n)+'_1')
                amps.append('A'+self.modename(s,l,m,n)+'_2')

        # Fill-in linear phases names
        if not self.coherent_n:
            for (s,l,m,n) in self.modes:
                if(self.amp_non_prec_sym):
                    phis.append('phi'+self.modename(s,l,m,n))
                else:
                    phis.append('phi'+self.modename(s,l,m,n)+'_1')
                    phis.append('phi'+self.modename(s,l,m,n)+'_2')
        else:
            # For coherent n modes, only set for l,m
            for (s,l,m,n) in self.modes:
                name='phi{}{}{}'.format(s,l,m)
                if name not in phis:
                    phis.append(name)

        # Fill in quadratic amplitudes names
        if self.quad_modes is not None:
            for quad_term in self.quad_modes.keys():
                for ((s,l,m,n),(s1,l1,m1,n1),(s2,l2,m2,n2)) in self.quad_modes[quad_term]: 
                    
                    quad_string = '{}-{}{}{}{}-{}{}{}{}-{}{}{}{}'.format(quad_term, s,l,m,n, s1,l1,m1,n1, s2,l2,m2,n2)
                    if(self.amp_non_prec_sym):
                        amps.append('A-'  +quad_string)
                        phis.append('phi-'+quad_string)
                    else:
                        amps.append('A-'+quad_string+'_1')
                        amps.append('A-'+quad_string+'_2')
                        phis.append('phi-'+quad_string+'_1')
                        phis.append('phi-'+quad_string+'_2')

        return amps, phis
    
    @staticmethod
    def modename(s,l,m,n):
        """
        Returns "slmn"
        """
        return "{}{}{}{}".format(s,l,m,n)

    def get_cplx_amp(self,x,mode):
        
        s,l,m,n = mode

        name = self.modename(s,l,m,n)
        
        if(self.amp_non_prec_sym):
            
            if not self.coherent_n: phi = get_param_override(self.fixed_params,x,'phi'+name)
            else                  : phi = get_param_override(self.fixed_params,x,'phi{}{}{}'.format(s,l,m))
            amp  = get_param_override(self.fixed_params,x,'A'+name)
            amp *= np.exp(1j*phi)
            
            return amp
        
        else:
            
            phi_1  = get_param_override(self.fixed_params,x,'phi'+name+'_1')
            phi_2  = get_param_override(self.fixed_params,x,'phi'+name+'_2')
            amp_1  = get_param_override(self.fixed_params,x,'A'  +name+'_1')
            amp_2  = get_param_override(self.fixed_params,x,'A'  +name+'_2')
            amp_1 *= np.exp(1j*phi_1)
            amp_2 *= np.exp(1j*phi_2)
            
            return (amp_1, amp_2)

    def get_cplx_amp_quad(self,x,quad_term, s,l,m,n, s1,l1,m1,n1, s2,l2,m2,n2):
        
        quad_string = '{}-{}{}{}{}-{}{}{}{}-{}{}{}{}'.format(quad_term, s,l,m,n, s1,l1,m1,n1, s2,l2,m2,n2)

        if(self.amp_non_prec_sym):

            quad_amp    = get_param_override(self.fixed_params,x,'A-'  +quad_string)
            quad_phi    = get_param_override(self.fixed_params,x,'phi-'+quad_string)
            amp         = quad_amp*np.exp(1j*quad_phi)

            return amp 

        else:

            quad_amp_1  = get_param_override(self.fixed_params,x,'A-'  +quad_string+'_1')
            quad_amp_2  = get_param_override(self.fixed_params,x,'A-'  +quad_string+'_2')
            quad_phi_1  = get_param_override(self.fixed_params,x,'phi-'+quad_string+'_1')
            quad_phi_2  = get_param_override(self.fixed_params,x,'phi-'+quad_string+'_2')
            amp_1       = quad_amp_1*np.exp(1j*quad_phi_1)
            amp_2       = quad_amp_2*np.exp(1j*quad_phi_2)

            return (amp_1, amp_2)

    def get_tail_parameters(self,x,tail_string):

        tail_dict = {}

        if(self.amp_non_prec_sym):

            tail_dict['A']   = get_param_override(self.fixed_params,x,  'A-tail-'+tail_string)
            tail_dict['phi'] = get_param_override(self.fixed_params,x,'phi-tail-'+tail_string)
            tail_dict['p']   = get_param_override(self.fixed_params,x,  'p-tail-'+tail_string)

        else:

            tail_dict['A_1']  = get_param_override(self.fixed_params,x,  'A-tail-'+tail_string+'_1')
            tail_dict['A_2']  = get_param_override(self.fixed_params,x,  'A-tail-'+tail_string+'_2')
            tail_dict['phi_1']= get_param_override(self.fixed_params,x,'phi-tail-'+tail_string+'_1')
            tail_dict['phi_2']= get_param_override(self.fixed_params,x,'phi-tail-'+tail_string+'_2')
            tail_dict['p_1']  = get_param_override(self.fixed_params,x,  'p-tail-'+tail_string+'_1')
            tail_dict['p_2']  = get_param_override(self.fixed_params,x,  'p-tail-'+tail_string+'_2')

        return tail_dict

    def get_waveform(self,x):

        # Initialise empty structures.
        amps, quad_amps, tail_parameters, TGR_parameters= {}, {}, {}, {}

        # Build the linear amplitudes.
        for mode in self.modes: amps[mode] = self.get_cplx_amp(x,mode)
        
        # Build the quadratic amplitudes.
        if self.quad_modes is not None:
            for quad_term in self.quad_modes.keys():
                quad_amps[quad_term] = {}
                for ((s,l,m,n),(s1,l1,m1,n1),(s2,l2,m2,n2)) in self.quad_modes[quad_term]: 
                    quad_amps[quad_term][((s,l,m,n),(s1,l1,m1,n1),(s2,l2,m2,n2))] = self.get_cplx_amp_quad(x,quad_term, s,l,m,n, s1,l1,m1,n1, s2,l2,m2,n2)

        # Build the tail parameters.
        if(self.tail_modes is not None):
            for (l,m) in self.tail_modes: 
                tail_parameters[(l,m)] = self.get_tail_parameters(x,'{}{}'.format(l,m))

        # Build the TGR parameters.
        if(self.AreaQuantization):
            TGR_parameters['alpha']                   = get_param_override(self.fixed_params,x,'alpha')
            if(self.tau_AQ):
                TGR_parameters['tau_AQ']              = get_param_override(self.fixed_params,x,'tau_AQ')
        elif(self.charge):     TGR_parameters['Q']    = get_param_override(self.fixed_params,x,'Q')
        elif(self.braneworld): TGR_parameters['beta'] = get_param_override(self.fixed_params,x,'beta')
        else:
            if not(self.ParSpec):
                for name in self.TGR_params:
                    TGR_parameters[name] = x[name]
            else:
                if (self.ParSpec_Dmax_charge > 0):
                    z = ((np.exp(x['logdistance'])*lal.PC_SI*1e6)/lal.C_SI)*lal.H0_SI
                    # ell is measured in km
                    TGR_parameters['gamma'] = ((ell_units_factor*x['ell']*(1+z))/x['Mf'])**(self.ParSpec_Dmax_charge)
                else:
                    TGR_parameters['gamma'] = 1.0
                # In this case, TGR_parameters[name] will be an array with all the spin expansion coefficients for a given mode.
                if not(self.EsGB):
                    for name in self.TGR_params:
                        TGR_parameters[name] = []
                        for i in range(self.ParSpec_Dmax_TGR+1):
                            TGR_parameters[name].append(x[name+'_{}'.format(i)])
                        TGR_parameters[name] = np.array(TGR_parameters[name])
                else:
                    for name in self.TGR_params:
                        TGR_parameters[name] = np.array(self.EsGB_corrections_dict[name])

        # Build the default Kerr parameters.
        Kerr_params = {}
        for name in (self.default_bounds.keys()):
            if(name=='logdistance' and self.reference_amplitude): Kerr_params[name] = 0.0
            else:                                                 Kerr_params[name] = get_param_override(self.fixed_params,x,name)
        try:    Kerr_params['t0'] = self.fixed_params['t']
        except: Kerr_params['t0'] = x['t0']

        return wf.KerrBH(Kerr_params['t0'],
                         Kerr_params['Mf'],
                         Kerr_params['af'],
                         amps,
                         np.exp(Kerr_params['logdistance']),
                         np.arccos(Kerr_params['cosiota']),
                         Kerr_params['phi'],

                         reference_amplitude = self.reference_amplitude,
                         geom                = 0,
                         qnm_fit             = self.qnm_fit,
                         interpolants        = self.qnm_interpolants,

                         Spheroidal          = self.Spheroidal,
                         amp_non_prec_sym    = self.amp_non_prec_sym,
                         tail_parameters     = tail_parameters,
                         quadratic_modes     = quad_amps,
                         quad_lin_prop       = self.quad_lin_prop,

                         TGR_params          = TGR_parameters,
                         AreaQuantization    = self.AreaQuantization,
                         ParSpec             = self.ParSpec,
                         charge              = self.charge,
                         braneworld          = self.braneworld)


class MMRDNSModel(RingDownModel):

        """
        Class implementing the MMRDNS waveform as ringdown model (valid for non-spinning progenitors).
        Frequencies and damping times of the specified (l,m,n) modes are fixed as a function of the mass and spin of the remnant BH.
        Relative amplitudes and phases are fixed as a function of the symmetric mass ratio.
        Reference: https://arxiv.org/pdf/1404.3197.pdf
        """

        def __init__(self, **kwargs):
            super(MMRDNSModel,self).__init__(**kwargs)

            utils.review_warning()

            self.Config = configparser.ConfigParser()
            self.Config.read(kwargs['config-file'])

            self.single_mode     = kwargs['single-mode']
            self.dist_flat_prior = kwargs['dist-flat-prior']
            self.Spheroidal      = kwargs['spheroidal']
            self.qnm_fit         = kwargs['qnm-fit']
            self.tensor_index    = 2
            self.TGR_params      = []
            #Syntax: [(s,l,m,n)]
            self.modes =  [(2,2,0), (2,2,1), (2,1,0), (3,3,0), (3,3,1), (3,2,0), (4,4,0), (4,3,0), (5,5,0)]
            
            utils.print_subsection('Intrinsic: MMRDNS model')

            # ==================================#
            # Initialise model names and bounds #
            # for mandatory parameters.         #
            # ==================================#

            self.default_bounds = {'Mf'         : [10,500]                      ,
                                   'af'         : [0.0,0.99]                    ,
                                   'eta'        : [0.0, 0.25]                   ,
                                   'logdistance': [np.log(10.0),np.log(10000.0)],
                                   'cosiota'    : [-1,1]                        ,
                                   'phi'        : [0.0,2*np.pi]                 }

            if not('t' in self.fixed_params):
                self.names.append('t0')
                self.bounds.append(self.time_prior)
                print(('{}: [{}, {}]'.format('t0'.ljust(len('logdistance')), self.time_prior[0], self.time_prior[1])))

            for name in self.default_bounds.keys():
                try:
                    self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
                except(configparser.NoOptionError):
                    single_bounds = read_parameter_bounds(self.Config, configparser, kwargs, name, self.default_bounds, 'MMRDNS')
                    self.names.append('{}'.format(name))
                    self.bounds.append(single_bounds)

            utils.print_subsection('Fixed')

            if not self.fixed_params:
                print('\n* No parameter was fixed.')
            else:
                for name in self.fixed_params.keys():
                    print('{} : {}'.format(name.ljust(len('cos_altitude')), self.fixed_params[name]))

            utils.print_subsection('Intrinsic: Testing-GR')

            if ((kwargs['domega-tgr-modes'] is not None) or (kwargs['dtau-tgr-modes'] is not None)):
                if (kwargs['domega-tgr-modes'] is not None):
                    for mode in kwargs['domega-tgr-modes']:
                        (l,m,n) = mode
                        name  = 'domega_{}{}{}'.format(l, m, n)
                        bound = [-1.0,1.0]
                        self.names.append(name)
                        self.bounds.append(bound)
                        self.TGR_params.append(name)
                        print(('{} : [{},{}]'.format(name.ljust(len('domega_000')), bound[0], bound[1])))
                if (kwargs['dtau-tgr-modes'] is not None):
                    for mode in kwargs['dtau-tgr-modes']:
                        (l,m,n) = mode
                        name  = 'dtau_{}{}{}'.format(l, m, n)
                        bound = [-1.0,1.0]
                        self.names.append(name)
                        self.bounds.append(bound)
                        self.TGR_params.append(name)
                        print(('{} : [{},{}]'.format(name.ljust(len('domega_000')), bound[0], bound[1])))
            else:
                print("No TGR parameter was considered.")

            if(self.qnm_fit == 0):
                for (l,m,n) in self.modes:
                    interpolate_freq, interpolate_tau = utils.qnm_interpolate(2,l,m,n)
                    self.qnm_interpolants[(2,l,m,n)]  = {'freq': interpolate_freq, 'tau': interpolate_tau}
            else:
                print("\nWarning: No interpolant selected, running MMRDNS without the (5,5,0) mode. If you wish to turn on this mode, please pass the option: 'qnm-fit=0'.")

        def log_prior(self,x):
            logPrior = 0
            if ('logdistance' in self.names):
                if(self.dist_flat_prior): logPrior +=     x['logdistance']
                else:                     logPrior += 3.0*x['logdistance']
            return super(MMRDNSModel,self).log_prior(x)+logPrior

        def get_waveform(self,x):

            MMRDNS_params = {}
            for name in (self.default_bounds.keys()):
                MMRDNS_params[name] = get_param_override(self.fixed_params,x,name)

            try:    MMRDNS_params['t0'] = self.fixed_params['t']
            except: MMRDNS_params['t0'] = x['t0']

            TGR_parameters = {}
            for name in self.TGR_params:
                TGR_parameters[name] = x[name]

            if (self.single_mode is not None):
                single_mode_flag = 1
                single_mode_l, single_mode_m = self.single_mode[0]

            else:
                single_mode_flag = 0
                single_mode_l, single_mode_m = (0,0)

            return wf.MMRDNS(MMRDNS_params['t0']                  ,
                             MMRDNS_params['Mf']                  ,
                             MMRDNS_params['af']                  ,
                             MMRDNS_params['eta']                 ,
                             np.exp(MMRDNS_params['logdistance']) ,
                             np.arccos(MMRDNS_params['cosiota'])  ,
                             MMRDNS_params['phi']                 ,
                             TGR_parameters                       ,
                             single_l     = single_mode_l         ,
                             single_m     = single_mode_m         ,
                             single_n     = 0                     ,
                             single_mode  = single_mode_flag      ,
                             Spheroidal   = self.Spheroidal       ,
                             interpolants = self.qnm_interpolants ,
                             qnm_fit      = self.qnm_fit          )

class MMRDNPModel(RingDownModel):

        """
        Class implementing the MMRDNP waveform as ringdown model (valid for spinning, non-precessing progenitors).
        Reference: https://arxiv.org/pdf/1801.08208.pdf
        Technical notes: https://github.com/llondon6/kerr_public/blob/master/notes/ns/mmrd.pdf
        """

        def __init__(self, **kwargs):
            super(MMRDNPModel,self).__init__(**kwargs)

            self.Config          = configparser.ConfigParser()
            self.Config.read(kwargs['config-file'])

            self.single_mode     = kwargs['single-mode']
            self.dist_flat_prior = kwargs['dist-flat-prior']
            self.qnm_fit         = kwargs['qnm-fit']
            self.TGR_params      = []
            #Syntax: [(l,m,n)]
            self.modes = [(2,2,0),(2,1,0),(3,3,0),(3,2,0),(4,4,0),(4,3,0)] + [(2,-2,0),(2,-1,0),(3,-3,0),(3,-2,0),(4,-4,0),(4,-3,0)]

            utils.print_subsection('Intrinsic: MMRDNP model')

            # ==================================#
            # Initialise model names and bounds #
            # for mandatory parameters.         #
            # ==================================#

            self.default_bounds = {'m1'          : [10,250]                      ,
                                   'm2'          : [10,250]                      ,
                                   'chi1'        : [-0.99,0.99]                  ,
                                   'chi2'        : [-0.99,0.99]                  ,
                                   'logdistance' : [np.log(10.0),np.log(10000.0)],
                                   'cosiota'     : [-1,1]                        ,
                                   'phi'         : [0.0,2*np.pi]                 }

            if not('t' in self.fixed_params):
                self.names.append('t0')
                self.bounds.append(self.time_prior)
                print(('{}: [{}, {}]'.format('t0'.ljust(len('logdistance')), self.time_prior[0], self.time_prior[1])))

            for name in self.default_bounds.keys():
                try:
                    self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
                except(configparser.NoOptionError):
                    single_bounds = read_parameter_bounds(self.Config, configparser, kwargs, name, self.default_bounds, 'MMRDNP')
                    self.names.append('{}'.format(name))
                    self.bounds.append(single_bounds)

            utils.print_subsection('Fixed')

            if not self.fixed_params:
                print('\n* No parameter was fixed.')
            else:
                for name in self.fixed_params.keys():
                    print('{} : {}'.format(name.ljust(len('cos_altitude')), self.fixed_params[name]))

            utils.print_subsection('Intrinsic: Testing-GR')

            if ((kwargs['domega-tgr-modes'] is not None) or (kwargs['dtau-tgr-modes'] is not None)):
                if (kwargs['domega-tgr-modes'] is not None):
                    for mode in kwargs['domega-tgr-modes']:
                        (l,m,n) = mode
                        name = 'domega_{}{}{}'.format(l, m, n)
                        bound = [-1.0,1.0]
                        self.names.append(name)
                        self.bounds.append(bound)
                        self.TGR_params.append(name)
                        print(('{} : [{},{}]'.format(name.ljust(len('domega_000')), bound[0], bound[1])))
                if (kwargs['dtau-tgr-modes'] is not None):
                    for mode in kwargs['dtau-tgr-modes']:
                        (l,m,n) = mode
                        name = 'dtau_{}{}{}'.format(l, m, n)
                        bound = [-1.0,1.0]
                        self.names.append(name)
                        self.bounds.append(bound)
                        self.TGR_params.append(name)
                        print(('{} : [{},{}]'.format(name.ljust(len('domega_000')), bound[0], bound[1])))
            else:
                print("No TGR parameter was considered.")

            if(self.qnm_fit == 0):
                for (l,m,n) in self.modes:
                    interpolate_freq, interpolate_tau = utils.qnm_interpolate(2,l,m,n)
                    self.qnm_interpolants[(2,l,m,n)]  = {'freq': interpolate_freq, 'tau': interpolate_tau}
                    interpolate_freq_negm, interpolate_tau_negm = utils.qnm_interpolate(2,l,-m,n)
                    self.qnm_interpolants[(2,l,-m,n)]  = {'freq': interpolate_freq_negm, 'tau': interpolate_tau_negm}

        def log_prior(self,x):
            logPrior = 0
            if ('logdistance' in self.names):
                if(self.dist_flat_prior): logPrior +=     x['logdistance']
                else:                     logPrior += 3.0*x['logdistance']
            return super(MMRDNPModel,self).log_prior(x)+logPrior

        def get_waveform(self,x):

            MMRDNP_params = {}
            for name in (self.default_bounds.keys()):
                MMRDNP_params[name] = get_param_override(self.fixed_params,x,name)

            try:    MMRDNP_params['t0'] = self.fixed_params['t']
            except: MMRDNP_params['t0'] = x['t0']

            TGR_parameters = {}
            for name in self.TGR_params:
                TGR_parameters[name] = x[name]

            if (self.single_mode is not None):
                single_mode_flag = 1
                single_mode_l, single_mode_m = self.single_mode[0]

            else:
                single_mode_flag = 0
                single_mode_l, single_mode_m = (0,0)

            # Adapt to final state fits conventions
            if(MMRDNP_params['chi1'] < 0): tilt1_fit = np.pi
            else: tilt1_fit = 0.0
            if(MMRDNP_params['chi2'] < 0): tilt2_fit = np.pi
            else: tilt2_fit = 0.0
            chi1_fit  = np.abs(MMRDNP_params['chi1'])
            chi2_fit  = np.abs(MMRDNP_params['chi2'])
            MMRDNP_params['Mf']   = bbh_final_mass_projected_spins(MMRDNP_params['m1'], MMRDNP_params['m2'], chi1_fit, chi2_fit, tilt1_fit, tilt2_fit, 'UIB2016')
            MMRDNP_params['af']   = bbh_final_spin_projected_spins(MMRDNP_params['m1'], MMRDNP_params['m2'], chi1_fit, chi2_fit, tilt1_fit, tilt2_fit, 'UIB2016', truncate = bbh_Kerr_trunc_opts.trunc)

            MMRDNP_params['Mi']   = MMRDNP_params['m1'] + MMRDNP_params['m2']
            MMRDNP_params['eta']  = (MMRDNP_params['m1']*MMRDNP_params['m2'])/(MMRDNP_params['Mi'])**2
            MMRDNP_params['chis'] = (MMRDNP_params['m1']*MMRDNP_params['chi1'] + MMRDNP_params['m2']*MMRDNP_params['chi2'])/(MMRDNP_params['Mi'])
            MMRDNP_params['chia'] = (MMRDNP_params['m1']*MMRDNP_params['chi1'] - MMRDNP_params['m2']*MMRDNP_params['chi2'])/(MMRDNP_params['Mi'])

            return wf.MMRDNP(MMRDNP_params['t0']                  ,
                             MMRDNP_params['Mf']                  ,
                             MMRDNP_params['af']                  ,
                             MMRDNP_params['Mi']                  ,
                             MMRDNP_params['eta']                 ,
                             MMRDNP_params['chis']                ,
                             MMRDNP_params['chia']                ,
                             np.exp(MMRDNP_params['logdistance']) ,
                             np.arccos(MMRDNP_params['cosiota'])  ,
                             MMRDNP_params['phi']                 ,
                             TGR_parameters                       ,
                             single_l     = single_mode_l         ,
                             single_m     = single_mode_m         ,
                             single_mode  = single_mode_flag      ,
                             interpolants = self.qnm_interpolants ,
                             qnm_fit      = self.qnm_fit          )

#OPTIMISEME: parametrise KHS in terms of initial parameters
class KHS_2012Model(RingDownModel):

        """
        Class implementing the KHS_2012 waveform as ringdown model (valid for spinning, non-precessing progenitors).
        Frequencies and damping times of the specified (l,m,n) modes are fixed as a function of the mass and spin of the remnant BH.
        Relative amplitudes and phases are fixed as a function of the symmetric mass ratio and effective spin.
        References: https://arxiv.org/abs/1207.0399, https://arxiv.org/abs/1406.3201
        """

        def __init__(self, **kwargs):
            super(KHS_2012Model,self).__init__(**kwargs)

            utils.review_warning()

            self.Config = configparser.ConfigParser()
            self.Config.read(kwargs['config-file'])

            self.single_mode     = kwargs['single-mode']
            self.dist_flat_prior = kwargs['dist-flat-prior']
            self.TGR_params      = []
            
            utils.print_subsection('Intrinsic: KHS_2012 model')

            # ==================================#
            # Initialise model names and bounds #
            # for mandatory parameters.         #
            # ==================================#

            self.default_bounds = {'Mf'          : [10,500]                      ,
                                   'af'          : [0.0,0.99]                    ,
                                   'eta'         : [0.0, 0.25]                   ,
                                   'chi_eff'     : [-1.0, 1.0]                   ,
                                   'logdistance' : [np.log(10.0),np.log(10000.0)],
                                   'cosiota'     : [-1,1]                        ,
                                   'phi'         : [0.0,2*np.pi]                 }

            if not('t' in self.fixed_params):
                self.names.append('t0')
                self.bounds.append(self.time_prior)
                print(('{}: [{}, {}]'.format('t0'.ljust(len('logdistance')), self.time_prior[0], self.time_prior[1])))

            for name in self.default_bounds.keys():
                try:
                    self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
                except(configparser.NoOptionError):
                    single_bounds = read_parameter_bounds(self.Config, configparser, kwargs, name, self.default_bounds, 'KHS_2012')
                    self.names.append('{}'.format(name))
                    self.bounds.append(single_bounds)

            utils.print_subsection('Fixed')

            if not self.fixed_params:
                print('\n* No parameter was fixed.')
            else:
                for name in self.fixed_params.keys():
                    print('{} : {}'.format(name.ljust(len('cos_altitude')), self.fixed_params[name]))

            utils.print_subsection('Intrinsic: Testing-GR')

            if ((kwargs['domega-tgr-modes'] is not None) or (kwargs['dtau-tgr-modes'] is not None)):
                if (kwargs['domega-tgr-modes'] is not None):
                    for mode in kwargs['domega-tgr-modes']:
                        (l,m,n) = mode
                        name = 'domega_{}{}{}'.format(l, m, n)
                        bound = [-1.0,1.0]
                        self.names.append(name)
                        self.bounds.append(bound)
                        self.TGR_params.append(name)
                        print(('{} : [{},{}]'.format(name.ljust(len('domega_000')), bound[0], bound[1])))
                if (kwargs['dtau-tgr-modes'] is not None):
                    for mode in kwargs['dtau-tgr-modes']:
                        (l,m,n) = mode
                        name = 'dtau_{}{}{}'.format(l, m, n)
                        bound = [-1.0,1.0]
                        self.names.append(name)
                        self.bounds.append(bound)
                        self.TGR_params.append(name)
                        print(('{} : [{},{}]'.format(name.ljust(len('domega_000')), bound[0], bound[1])))
            else:
                print("No TGR parameter was considered.")


        def log_prior(self,x):
            logPrior = 0
            if ('logdistance' in self.names):
                if(self.dist_flat_prior): logPrior +=     x['logdistance']
                else:                     logPrior += 3.0*x['logdistance']
            return super(KHS_2012Model,self).log_prior(x)+logPrior

        def get_waveform(self,x):

            KHS_2012_params = {}
            for name in (self.default_bounds.keys()):
                KHS_2012_params[name] = get_param_override(self.fixed_params,x,name)

            try:    KHS_2012_params['t0'] = self.fixed_params['t']
            except: KHS_2012_params['t0'] = x['t0']

            TGR_parameters = {}
            for name in self.TGR_params:
                TGR_parameters[name] = x[name]

            if (self.single_mode is not None):
                single_mode_flag = 1
                single_mode_l, single_mode_m = self.single_mode[0]

            else:
                single_mode_flag = 0
                single_mode_l, single_mode_m = (0,0)

            return wf.KHS_2012(KHS_2012_params['t0']                 ,
                               KHS_2012_params['Mf']                 ,
                               KHS_2012_params['af']                 ,
                               KHS_2012_params['eta']                ,
                               KHS_2012_params['chi_eff']            ,
                               np.exp(KHS_2012_params['logdistance']),
                               np.arccos(KHS_2012_params['cosiota']) ,
                               KHS_2012_params['phi']                ,
                               TGR_parameters                        ,
                               single_l    = single_mode_l           ,
                               single_m    = single_mode_m           ,
                               single_mode = single_mode_flag        )

class TEOBPMModel(RingDownModel):

    """
    Class implementing the EOBPM waveform as post-merger model (valid for non-precessing progenitors).
    References: arXiv: 1406.0401, 1606.03952, 2001.09082.
    """

    def __init__(self, **kwargs):
        super(TEOBPMModel,self).__init__(**kwargs)

        self.Config = configparser.ConfigParser()
        self.Config.read(kwargs['config-file'])
        
        self.multipoles            = kwargs['teob-modes']
        self.dist_flat_prior       = kwargs['dist-flat-prior']
        self.mass_ratio_flat_prior = kwargs['mass-ratio-flat-prior']
        self.TGR_params            = []
        
        utils.print_subsection('Intrinsic: TEOBPM model')
        utils.review_warning()

        # ==================================#
        # Initialise model names and bounds #
        # for mandatory parameters.         #
        # ==================================#
        
        # Spin and q ranges from https://arxiv.org/abs/2011.01958
        self.default_bounds = {'m1'          : [10,200]                      ,
                               'm2'          : [10,200]                      ,
                               'chi1'        : [-0.8,0.95]                   ,
                               'chi2'        : [-0.8,0.95]                   ,
                               'phase'       : [0,2*np.pi]                   ,
                               'logdistance' : [np.log(10.0),np.log(10000.0)],
                               'cosiota'     : [-1,1]                        ,
                               'phi'         : [0,2*np.pi]                   }

        if not('t' in self.fixed_params):
            self.names.append('t0')
            self.bounds.append(self.time_prior)
            print('{}: [{}, {}]'.format('t0'.ljust(len('logdistance')), self.time_prior[0], self.time_prior[1]))

        for name in self.default_bounds.keys():
            
            if(name=='phase'):
                # FIXME: Temporary solution setting the prior equal for all merger phases, phases should be fit.
                for multipole in self.multipoles:
                    (l,m) = multipole
                    phase_name = 'phase_{}{}'.format(l,m)
                    try:
                        self.fixed_params[phase_name] = self.Config.getfloat("Priors",'fix-{}'.format(phase_name))
                    except(configparser.NoOptionError):
                        single_bounds = self.default_bounds[name]
                        print('{} : [{}, {}]'.format(phase_name.ljust(len('logdistance')), single_bounds[0], single_bounds[1]))
                        self.names.append(phase_name)
                        self.bounds.append(single_bounds)
            else:
                try:
                    self.fixed_params[name] = self.Config.getfloat("Priors",'fix-'+name)
                except(configparser.NoOptionError):
                    single_bounds = read_parameter_bounds(self.Config, configparser, kwargs, name, self.default_bounds, 'TEOBResumSPM')
                    self.names.append('{}'.format(name))
                    self.bounds.append(single_bounds)
            
        utils.print_subsection('Fixed')

        if not self.fixed_params:
            print('\n* No parameter was fixed.')
        else:
            for name in self.fixed_params.keys(): print('{}: {}'.format(name.ljust(len('cos_altitude')), self.fixed_params[name]))

        utils.print_subsection('Intrinsic: Testing-GR')

        if ((kwargs['domega-tgr-modes'] is not None) or (kwargs['dtau-tgr-modes'] is not None)):
            if (kwargs['domega-tgr-modes'] is not None):
                for mode in kwargs['domega-tgr-modes']:
                    (l,m,n) = mode
                    if not(n==0): raise ValueError('With the TEOBResumSPM model currently only n=0 TGR modes deviations are supported.')
                    name = 'domega_{}{}{}'.format(l, m, n)
                    bound = [-1.0,1.0]
                    self.names.append(name)
                    self.bounds.append(bound)
                    self.TGR_params.append(name)
                    print(('{} : [{},{}]'.format(name.ljust(len('domega_000')), bound[0], bound[1])))
            if (kwargs['dtau-tgr-modes'] is not None):
                for mode in kwargs['dtau-tgr-modes']:
                    (l,m,n) = mode
                    if not(n==0): raise ValueError('With the TEOBResumSPM model currently only n=0 TGR modes deviations are supported.')
                    name = 'dtau_{}{}{}'.format(l, m, n)
                    bound = [-1.0,1.0]
                    self.names.append(name)
                    self.bounds.append(bound)
                    self.TGR_params.append(name)
                    print(('{} : [{},{}]'.format(name.ljust(len('domega_000')), bound[0], bound[1])))
        else:
            print("No TGR parameter was considered.")

    def log_prior(self,x):
        
        logPrior = 0
        if ('logdistance' in self.names):
            if(self.dist_flat_prior): logPrior +=     x['logdistance']
            else:                     logPrior += 3.0*x['logdistance']
        # Swap masses and spins to respect model conventions (m1 is the heaviest BH).
        try:
            if (x['m1']<x['m2']):
                x['m1'],   x['m2']   = x['m2'],   x['m1']
                x['chi1'], x['chi2'] = x['chi2'], x['chi1']
        except(KeyError):
            pass

        if ('m1' in self.names) and ('m2' in self.names):
            # If selected, set a flat prior in mass ratio.
            if(self.mass_ratio_flat_prior):
                    M_chirp   = (x['m1']*x['m2'])**(3/5) / (x['m1']+x['m2'])**(1/5)
                    logPrior += np.log(M_chirp) - 2 * np.log(x['m2'])
        
        return super(TEOBPMModel,self).log_prior(x)+logPrior


    def get_waveform(self,x):

        TEOBPM_params, merger_phases = {}, {}
        for name in (self.default_bounds.keys()):
            if(name=='phase'):
                for multipole in self.multipoles:
                    (l,m) = multipole
                    merger_phases[(l,m)] = get_param_override(self.fixed_params,x,'phase_{}{}'.format(l,m))
            else:
                TEOBPM_params[name] = get_param_override(self.fixed_params,x,name)
        try:    TEOBPM_params['t0'] = self.fixed_params['t']
        except: TEOBPM_params['t0'] = x['t0']

        TGR_parameters = {}
        for name in self.TGR_params:
            TGR_parameters[name] = x[name]
                      
        return wf.TEOBPM(TEOBPM_params['t0']                 ,
                         TEOBPM_params['m1']                 ,
                         TEOBPM_params['m2']                 ,
                         TEOBPM_params['chi1']               ,
                         TEOBPM_params['chi2']               ,
                         merger_phases                       ,
                         np.exp(TEOBPM_params['logdistance']),
                         np.arccos(TEOBPM_params['cosiota']) ,
                         TEOBPM_params['phi']                ,
                         self.multipoles                     ,
                         TGR_parameters                      )




def main():
    
    # Print ascii art.
    try   : print("\u001b[\u001b[38;5;39m{}\u001b[0m".format(initialise.my_art))
    except: pass
    print(initialise.__ascii_art__)
    
    # Initialise and read config.
    start_execution_time = time.time()
    parser               = OptionParser(initialise.usage)
    parser.add_option('--config-file', type='string', metavar = 'config_file', default = None)
    (opts,args)          = parser.parse_args()
    config_file          = opts.config_file

    if not config_file:
        parser.print_help()
        parser.error('Please specify a config file.')
    if not os.path.exists(config_file): parser.error('Config file {} not found.'.format(config_file))
    Config = configparser.ConfigParser()
    Config.read(config_file)



    # ===================================#
    # Deviate stdout and stderr to file. #
    # ===================================#

    try:                                          directory = str(Config.get("input",'output'))
    except(KeyError, configparser.NoOptionError): directory = 'pyRing_default_output_directory'
    if not os.path.exists(directory): os.makedirs(directory)
    try:
        if not(Config.getint("input",'screen-output')):
            sys.stdout = open(os.path.join(directory,'stdout_pyRing.txt'), 'w')
            sys.stderr = open(os.path.join(directory,'stderr_pyRing.txt'), 'w')
        else: pass
    except(configparser.NoOptionError):
        sys.stdout = open(os.path.join(directory,'stdout_pyRing.txt'), 'w')
        sys.stderr = open(os.path.join(directory,'stderr_pyRing.txt'), 'w')

    print('* Using pyRing version: `{}`.\n'.format(pyRing.__version__))

    utils.print_section('Input parameters')
    print(('* Reading config file : `{}`.'.format(config_file)))
    print( '* With sections       : {}.\n'.format(str(Config.sections())))
    print( '* I\'ll be running with the following values:\n')

    # ==================================================#
    # Initialize and read from config input parameters. #
    # ==================================================#

    input_par = initialise.read_config(Config, config_file)

    #=================================================================#
    # Print git info on a file (if running from inside a repository). #
    #=================================================================#

    try:
        if (input_par['run-type']=='full' or input_par['run-type']=='noise-estimation-only'):
            initialise.create_directories(input_par['output'])
            initialise.store_git_info(input_par['output'])
            shutil.copy2(opts.config_file, os.path.join(input_par['output'], os.path.basename(opts.config_file)))
    except: pass

    #=======================#
    # Select prior options. #
    #=======================#

    try: input_par['gr-time-prior'] = ast.literal_eval(Config.get("Priors",'gr-time-prior'))
    except(KeyError, configparser.NoOptionError, configparser.NoSectionError): pass
    print("{name} : {value}".format(name='gr-time-prior'.ljust(initialise.max_len_keyword), value=input_par['gr-time-prior']))
           
    try: input_par['dist-flat-prior'] = ast.literal_eval(Config.get("Priors",'dist-flat-prior'))
    except(KeyError, configparser.NoOptionError, configparser.NoSectionError): pass
    print("{name} : {value}".format(name='dist-flat-prior'.ljust(initialise.max_len_keyword), value=input_par['dist-flat-prior']))

    try: input_par['mass-ratio-flat-prior'] = ast.literal_eval(Config.get("Priors",'mass-ratio-flat-prior'))
    except(KeyError, configparser.NoOptionError, configparser.NoSectionError): pass
    print("{name} : {value}".format(name='mass-ratio-flat-prior'.ljust(initialise.max_len_keyword), value=input_par['mass-ratio-flat-prior']))

    try: input_par['ds-amp-flat-prior'] = ast.literal_eval(Config.get("Priors",'ds-amp-flat-prior'))
    except(KeyError, configparser.NoOptionError, configparser.NoSectionError): pass

    print("{name} : {value}".format(name='ds-amp-flat-prior'.ljust(initialise.max_len_keyword), value=input_par['ds-amp-flat-prior']))

    #Compatibility checks
    try: assert not(Config.getfloat("Plot",'mf')), "You are running with an old config file. Now the final mass used in the time prior computation must be passed through 'mf-time-prior' option in the 'Priors' section."
    except(configparser.NoOptionError, configparser.NoSectionError): pass
    try: input_par['mf-time-prior'] = Config.getfloat("Priors",'mf-time-prior')
    except(configparser.NoOptionError, configparser.NoSectionError): pass
    print(("{name} : {value}".format(name='mf-time-prior'.ljust(initialise.max_len_keyword), value=input_par['mf-time-prior'])))

    #=======================================#
    # Read injection parameters, if needed. #
    #=======================================#

    if not(input_par['injection-approximant']==''):
        
        injection_parameters = initialise.read_injection_parameters(input_par, Config)

        for key in injection_parameters:
            print("{name} : {value}".format(name=key.ljust(initialise.max_len_keyword), value=injection_parameters[key]))
            input_par['injection-parameters'] = injection_parameters

    #===========================#
    # Select waveform template. #
    #===========================#

    if   (input_par['template']=='Kerr'):                  signal_model = KerrModel(          **input_par)
    elif (input_par['template']=='Damped-sinusoids'):      signal_model = DampedSinusoids(    **input_par)
    elif (input_par['template']=='Morlet-Gabor-wavelets'): signal_model = MorletGaborWavelets(**input_par)
    elif (input_par['template']=='MMRDNS'):                signal_model = MMRDNSModel(        **input_par)
    elif (input_par['template']=='MMRDNP'):                signal_model = MMRDNPModel(        **input_par)
    elif (input_par['template']=='KHS_2012'):              signal_model = KHS_2012Model(      **input_par)
    elif (input_par['template']=='IMR'):                   signal_model = IMR_Ringdown_Model( **input_par)
    elif (input_par['template']=='TEOBResumSPM'):          signal_model = TEOBPMModel(        **input_par)
    else:                                                  raise Exception("Unknown template selected")

    if (input_par['run-type']=='noise-estimation-only'):
        print('* Noise estimation done, `noise-estimation-only` option was selected. Exiting.')
        exit()

    #================================#
    # Initialize sampler parameters. #
    #================================#

    utils.print_section('Sampler initialisation')

    try:                                                                       input_par['sampler'] = Config.get("Sampler settings",'sampler')
    except(KeyError, configparser.NoOptionError, configparser.NoSectionError): input_par['sampler'] = 'cpnest'
    if(input_par['sampler']=='cpnest'):
        sampler_params={'verbose'                     : 2,
                        'nlive'                       : 1024,
                        'maxmcmc'                     : 1024,
                        'poolsize'                    : 128,
                        'nthreads'                    : 1,
                        'seed'                        : 1234,
                        'resume'                      : 1,
                        'periodic_checkpoint_interval': 900,
                        }
    elif(input_par['sampler']=='raynest'):
        sampler_params={'verbose'                     : 2,
                        'nlive'                       : 1024,
                        'maxmcmc'                     : 1024,
                        'nnest'                       : 1,
                        'nensemble'                   : 1,
                        'seed'                        : 1234,
                        'resume'                      : 1,
                        'periodic_checkpoint_interval': 900,
                        }
    else:
        raise ValueError('Unsupported sampler passed under the option `sampler` of the `Sampler settings` section.')

    print("* I'll be running with the following sampler parameters:\n")
    for key in sampler_params:
        if isinstance(sampler_params[key], int):
            try:                                                                       sampler_params[key] = Config.getint("Sampler settings", key)
            except(KeyError, configparser.NoOptionError, configparser.NoSectionError): pass
            print("{name} : {value}".format(name=key.ljust(len('periodic_checkpoint_interval')), value=sampler_params[key]))
    # Adapt to cpnest convention.
    if(sampler_params['periodic_checkpoint_interval']==0): sampler_params['periodic_checkpoint_interval'] = None

    logZ_noise = signal_model.logZnoise
    print('\nNoise evidence: {:.3f}'.format(logZ_noise))

    #================#
    # Output strain. #
    #================#

    # TD whitening requires a start time, which is not provided in `strain_only=1` mode. Hence, produce FD strain-only plots.
    print('\n\n* Whitened plots using the FD PSD are illustrative, and do not correspond to the full TD treatment applied in the analysis.')
    plots.plot_strain(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, whiten_flag=False, whiten_method = 'FD', strain_only = 1, spectrum = 0, **input_par)
    plots.plot_strain(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, whiten_flag=True,  whiten_method = 'FD', strain_only = 1, spectrum = 0, **input_par)
    plots.plot_strain(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, whiten_flag=False, whiten_method = 'FD', strain_only = 1, spectrum = 1, **input_par)
    plots.plot_strain(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, whiten_flag=True,  whiten_method = 'FD', strain_only = 1, spectrum = 1, **input_par)

    if (input_par['run-type']=='full'):

        #=============================================#
        # Start the CPNest Nested Sampling algorithm. #
        #=============================================#

        pre_sampling_execution_time = (time.time() - start_execution_time)/60.0
        print('\n* Pre-sampling execution time (min): {:.2f}'.format(pre_sampling_execution_time))

        utils.print_section('Start sampling')

        if(input_par['sampler']=='cpnest'):

            print('* Using CPNest version: `{}`.\n'.format(cpnest.__version__))
            print('* The sampling output appears in the `{}/Nested_sampler/cpnest.log` file.\n'.format(input_par['output']))

            work=cpnest.CPNest(signal_model,
                            verbose  = sampler_params['verbose'],
                            nlive    = sampler_params['nlive'],
                            maxmcmc  = sampler_params['maxmcmc'],
                            poolsize = sampler_params['poolsize'],
                            nthreads = sampler_params['nthreads'],
                            output   = input_par['output']+'/Nested_sampler',
                            seed     = sampler_params['seed'],
                            resume   = sampler_params['resume'])
            work.run()
            x           = work.get_posterior_samples(filename='posterior.dat')
            logZ_signal = work.NS.logZ

        elif(input_par['sampler']=='raynest'):

            try   : import raynest
            except: raise Exception('The raynest sampler requires the ray package. Please install it through `pip install ray`.')
            print('* Using raynest version: `{}`.\n'.format(raynest.__version__))
            print('* The sampling output appears in the `{}/Nested_sampler/raynest.log` file.\n'.format(input_par['output']))

            work=raynest.raynest(signal_model,
                                 verbose   = sampler_params['verbose'],
                                 nlive     = sampler_params['nlive'],
                                 maxmcmc   = sampler_params['maxmcmc'],
                                 nnest     = sampler_params['nnest'],
                                 nensemble = sampler_params['nensemble'],
                                 output    = input_par['output']+'/Nested_sampler',
                                 seed      = sampler_params['seed'],
                                 resume    = sampler_params['resume'])
            work.run()
            x           = work.posterior_samples.ravel()
            logZ_signal = work.logZ
        else:
            raise ValueError('Unsupported sampler passed under the option `sampler` of the `Sampler settings` section.')

        logB     = logZ_signal-logZ_noise
        logL_max = x['logL'][-1]
        SNR      = np.sqrt(2*(logL_max - logZ_noise))
        
        print('\n{}: {:.3f}'.format('Signal evidence'.ljust(len('Estimated SNR from logB ')), logZ_signal))
        print('{}: {:.3f}'.format('ln B'.ljust(len('Estimated SNR from logB ')), logB))
        print('{}: {:.3f}'.format('Estimated SNR from logB '.ljust(len('Estimated SNR from logB ')), SNR))

        if(input_par['truncate']):
            # FIXME: Compute the logB distribution, since in this case the data chunk is changing at each step.
            pass

        sampling_execution_time = (time.time() - pre_sampling_execution_time)/60.0
        print('\n* Sampling execution time (min): {:.2f}'.format(sampling_execution_time))

    elif (input_par['run-type']=='post-processing'):

        #===============================================================#
        # If a NS run has already been performed, read out the results. #
        #===============================================================#

        utils.print_section('Sampling post-processing')

        # All the logs are natural logarithms
        if(input_par['sampler']=='cpnest'):
            x = np.genfromtxt(os.path.join(input_par['output'],'Nested_sampler/posterior.dat'),names=True,deletechars="")
            # genfromtxt removes - signs from column names by default, so set deletechars="" to disable validation
            logZ_signal = np.loadtxt(os.path.join(input_par['output'],'Nested_sampler/chain_{}_{}.txt_evidence.txt'.format(sampler_params['nlive'], sampler_params['seed'])))[0]
        elif(input_par['sampler']=='raynest'):
            filename    = os.path.join(input_par['output'],'Nested_sampler','cpnest.h5')
            h5_file     = h5py.File(filename,'r')
            x           = h5_file['combined'].get('posterior_samples')
            logZ_signal = h5_file['combined']['logZ'][()]
        else:
            raise ValueError('Unsupported sampler passed under the option `sampler` of the `Sampler settings` section.')
        print('{}: {:.3f}'.format('Signal evidence'.ljust(len('Estimated SNR from logB ')), logZ_signal))
        logB = logZ_signal-logZ_noise
        print('{}: {:.3f}'.format('ln B'.ljust(len('Estimated SNR from logB ')), logB))
        logL_max = x['logL']
        SNR = np.sqrt(2*(logL_max[-1] - logZ_noise))
        print('{}: {:.3f}'.format('Estimated SNR from logB '.ljust(len('Estimated SNR from logB ')), SNR))
    else:
        raise Exception('Unknown run type was selected.')

    outFile_evidence = open(os.path.join(input_par['output'],'Nested_sampler/Evidence.txt'), 'w')
    outFile_evidence.write('lnZ_noise \t lnZ_signal \t lnB \t Estimated_SNR \n')
    outFile_evidence.write('{} \t {} \t {} \t {} \n'.format(logZ_noise, logZ_signal, logB, SNR))
    outFile_evidence.close()

    #============================#
    # Posterior railing section. #
    #============================#

    utils.print_section('Posterior railing tests')

    try:
        railing_parameters  = []
        header = ''
        for (i,param) in enumerate(signal_model.names):
            Prior_bins = np.linspace(signal_model.bounds[i][0], signal_model.bounds[i][-1], 100)
            low_rail, high_rail = utils.railing_check(samples=x[param], prior_bins= Prior_bins, tolerance=2.0)
            header +='{par}_low\t{par}_up\t'.format(par=param)
            if(low_rail):
                railing_parameters.append(1)
                print('{}'.format(param.ljust(len('cos_altitude'))), 'is railing against the lower prior bound.')
            else:
                railing_parameters.append(0)
            if(high_rail):
                railing_parameters.append(1)
                print('{}'.format(param.ljust(len('cos_altitude'))), 'is railing against the upper prior bound.')
            else:
                railing_parameters.append(0)
        np.savetxt(os.path.join(input_par['output'], 'Nested_sampler/Parameters_prior_railing.txt'), np.column_stack(railing_parameters), fmt= "%d", header=header)
    except:
        print("\n* Warning: Prior railing file generation failed with error: {}.".format(traceback.print_exc()))

    #====================#
    # pesummary section. #
    #====================#

    if(input_par['pesummary']):
        try:
            print('\nCreating pesummary metafile...\n')
            os.system('summarypages --webdir {outdir}/Plots/pesummary_postproc \
                                    --samples {outdir}/Nested_sampler/posterior.dat \
                                    --config {conf} \
                                    --no_ligo_skymap \
                                    --verbose \
                                    --labels {runlabel}'.format(outdir=input_par['output'], conf=config_file, runlabel=input_par['run-tag']))
            #These data can be extracted as (e.g.): f.extra_kwargs[0]['other']['lnZ_noise']
            os.system('summarymodify --samples {outdir}/Plots/pesummary_postproc/samples/posterior_samples.h5 --overwrite --delimiter / --kwargs {runlabel}/lnZ_signal:{logZ_signal_value}'.format(outdir=input_par['output'], runlabel=input_par['run-tag'], logZ_signal_value=logZ_signal))
            os.system('summarymodify --samples {outdir}/Plots/pesummary_postproc/samples/posterior_samples.h5 --overwrite --delimiter / --kwargs {runlabel}/lnZ_noise:{logZ_noise_value}'.format(  outdir=input_par['output'], runlabel=input_par['run-tag'], logZ_noise_value =logZ_noise ))
            # If available add the information as an extra parameter (useful to estimate logB error). Since this is only printed, but not stored by cpnest, we extract it from the stderr when available (a bit ugly, but does the job).
            if not(input_par['screen-output'] or (input_par['run-type']=='post-processing')):
                try:
                    # The path depends on the sampler version used.
                    if(input_par['sampler']=='cpnest'):
                        if os.path.exists('{outdir}/Nested_sampler/cpnest.log'.format( outdir=input_par['output'])): H_file = '{outdir}/Nested_sampler/cpnest.log'.format( outdir=input_par['output'])
                        else                                                                                       : H_file = '{outdir}/stderr_pyRing.txt'.format(         outdir=input_par['output'])
                    elif(input_par['sampler']=='raynest'):
                        if os.path.exists('{outdir}/Nested_sampler/raynest.log'.format(outdir=input_par['output'])): H_file = '{outdir}/Nested_sampler/raynest.log'.format(outdir=input_par['output'])
                        else                                                                                       : H_file = '{outdir}/stderr_pyRing.txt'.format(         outdir=input_par['output'])              
                    with open(H_file, 'r+') as fd:
                        contents = fd.readlines()
                        for line in contents:
                            if('Information' in line):
                                H = line.split('Information:')[-1]
                    fd.close()
                    os.system('summarymodify --samples {outdir}/Plots/pesummary_postproc/samples/posterior_samples.h5 --overwrite --delimiter / --kwargs {runlabel}/H:{H_value}'.format(outdir=input_par['output'], runlabel=input_par['run-tag'], H_value=H))
                except:
                    print("\n* Warning: Information storage in pesumarry failed with error: {}.".format(traceback.print_exc()))
            os.system('mv {outdir}/Plots/pesummary_postproc/samples/posterior_samples.h5 {outdir}/Plots/pesummary_postproc/samples/{runlabel}_pesummary_metafile.h5'.format(outdir=input_par['output'], runlabel=input_par['run-tag']))
        except:
            print("\n* Warning: PESummary metafile generation failed with error: {}.".format(traceback.print_exc()))

    #=============================#
    # Plots common to all models. #
    #=============================#

    utils.print_section('SNR computation')

    # Here, we also check the amount of SNR lost by truncation
    try:
        plots.SNR_plots(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, params = x, **input_par)
    except(KeyError, ValueError, configparser.NoOptionError, configparser.NoSectionError):
        print("\n* Warning: Waveform generation failed with error: {}.".format(traceback.print_exc()))

    utils.print_section('Plots generation')

    try   : injection_values = [injection_parameters[key] for key in injection_parameters.keys()]
    except: injection_values = None
        
    try:
        plots.global_corner(x, signal_model.names, input_par['output'], truths=injection_values)
    except Exception:
        print("\n* Warning: Global corner plot failed with error: {}.".format(traceback.print_exc()))
                                
    if(('ra' in signal_model.fixed_params.keys()) or ('dec' in signal_model.fixed_params.keys()) or ('cos_altitude' in signal_model.fixed_params.keys()) or ('azimuth' in signal_model.fixed_params.keys()) or ('logdistance' in signal_model.fixed_params.keys())):
        print('* Skipping sky position plot, since at least one of the sky position parameters was fixed.\n')
    else:
        try:                                                                                   plots.sky_location_plots(x, Config, **input_par)
        except(KeyError, ValueError, configparser.NoOptionError, configparser.NoSectionError): print("\nWarning: Plot of sky position failed with error: {}.".format(traceback.print_exc()))
    if(('t0' in signal_model.fixed_params.keys()) or (input_par['truncate'])):
        print('* Skipping time plot, since time was fixed.\n')
    else:
        try:                                                                                   plots.t_start_plot(x['t0'], input_par['mf-time-prior'], **input_par)
        except(KeyError, ValueError, configparser.NoOptionError, configparser.NoSectionError): print("\nWarning: Plot of start time failed with error: {}.".format(traceback.print_exc()))

    try:
        plots.plot_strain(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, params=x, whiten_flag=True,  whiten_method = 'FD', spectrum = 0, **input_par)
        plots.plot_strain(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, params=x, whiten_flag=True,  whiten_method = 'TD', spectrum = 0, **input_par)
        plots.plot_strain(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, params=x, whiten_flag=False, whiten_method = 'TD', spectrum = 0, **input_par)
        plots.plot_strain(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, params=x, whiten_flag=True,  whiten_method = 'FD', spectrum = 1, **input_par)
        plots.plot_strain(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, params=x, whiten_flag=True,  whiten_method = 'TD', spectrum = 1, **input_par)
        plots.plot_strain(signal_model.get_waveform, signal_model.detectors, signal_model.fixed_params, signal_model.tgps, params=x, whiten_flag=False, whiten_method = 'TD', spectrum = 1, **input_par)

    except(KeyError, ValueError, configparser.NoOptionError, configparser.NoSectionError):
        print("* Warning: Waveform reconstruction failed with error: {}.\n".format(traceback.print_exc()))

    Mf_dist, af_dist = plots.read_Mf_af_IMR_posterior(Config, **input_par)

    #=========================================================#
    # Specific plots for different waveform model parameters. #
    #=========================================================#

    # IMPROVEME: skip plotting when we fix some of the intrinsic params, write a generic parameters handling structure to take care of this

    try:
        if((input_par['template']=='Damped-sinusoids') or (input_par['template']=='Morlet-Gabor-wavelets')):
            try:                          plots.f_tau_amp_corner(x, **input_par)
            except(KeyError, ValueError): print("\n* Warning: Agnostic template intrinsic parameters plot failed.")
            try:
                if((Mf_dist is not None) and (af_dist is not None)):
                    for pol in input_par['n-ds-modes'].keys():
                        for i in range(input_par['n-ds-modes'][pol]):
                            plots.f0_t0_gr(x, Mf_dist, af_dist, i, **input_par)
            except(KeyError, ValueError): print("\n* Warning: Agnostic template spectral parameters plot failed.")

        else:
            
            # Plot common to all non-agnostic families of templates.
            if (('logdistance' in signal_model.names) and ('cosiota' in signal_model.names)):
                try:                          plots.orientation_corner(x, Config, **input_par)
                except(KeyError, ValueError): print("\n* Warning: Plot of orientation_corner failed.")

            # Non-agnostic template-specific plots.
            if (input_par['template']=='Kerr'):
                if not(input_par['truncate']):
                    try:                                                                                   plots.Kerr_intrinsic_corner(x, **input_par)
                    except(KeyError, ValueError, configparser.NoOptionError, configparser.NoSectionError): print("\n* Warning: Plot of intrinsic Kerr parameters failed with error: {}.".format(traceback.print_exc()))
                if (('logdistance' in signal_model.names) and ('cosiota' in signal_model.names)):
                    try:                                                                                   plots.amplitudes_corner(x, **input_par)
                    except(AttributeError, KeyError, ValueError, configparser.NoOptionError, configparser.NoSectionError): print("\n* Warning: Plot of amplitudes failed with error: {}.".format(traceback.print_exc()))
                if (input_par['inject-area-quantization']==1 and input_par['area-quantization']==1):
                    try:                                                                                   plots.Kerr_intrinsic_alpha_corner(x, **input_par)
                    except(KeyError, ValueError, configparser.NoOptionError, configparser.NoSectionError): print("\n* Warning: Plot of alpha failed with error: {}.".format(traceback.print_exc()))
                if (input_par['inject-braneworld']==1 and input_par['braneworld']==1):
                    try:                                                                                   plots.Kerr_intrinsic_braneworld_corner(x, **input_par)
                    except(KeyError, ValueError, configparser.NoOptionError, configparser.NoSectionError): print("\n* Warning: Plot of braneworld failed with error: {}.".format(traceback.print_exc()))
                if(input_par['charge']):
                    try:                                                                                   plots.Kerr_Newman_intrinsic_corner(x, **input_par)
                    except(KeyError, ValueError, configparser.NoOptionError, configparser.NoSectionError): print("\n* Warning: Plot of intrinsic Kerr-Newman parameters failed with error: {}.".format(traceback.print_exc()))
            elif (input_par['template']=='MMRDNS'):
                try:                          plots.MMRDNS_intrinsic_corner(x, **input_par)
                except(KeyError, ValueError): print("\n* Warning: MMRDNS plots failed.")
            elif (input_par['template']=='MMRDNP'):
                try:                          plots.MMRDNP_intrinsic_corner(x, **input_par)
                except(KeyError, ValueError): print("\n* Warning: MMRDNP intrinsic parameters plot failed.")
                try:                          plots.MMRDNP_amplitude_parameters_corner(x, **input_par)
                except(KeyError, ValueError): print("\n* Warning: MMRDNP amplitude parameters plot failed.")
                try:                          plots.MMRDNP_Mf_af_plot(x, **input_par)
                except(KeyError, ValueError): print("\n* Warning: MMRDNP final parameters plot failed.")
            elif (input_par['template']=='TEOBResumSPM'):
                try:                          plots.TEOBPM_intrinsic_corner(x, **input_par)
                except(KeyError, ValueError): print("\n* Warning: TEOB intrinsic parameters plot failed.")

            if((input_par['template']=='Kerr') or (input_par['template']=='MMRDNS')):
                try:                          plots.Mf_af_plot(x, Mf_dist, af_dist, **input_par)
                except(KeyError, ValueError): print("\n* Warning: Plot of Mf_af failed.")

            # TGR plots
            if((input_par['template']=='Kerr') or (input_par['template']=='MMRDNP') or (input_par['template']=='TEOBResumSPM')):
                try:
                    if((input_par['domega-tgr-modes'] is not None) or (input_par['dtau-tgr-modes'] is not None)):
                        plots.omega_tau_eff_plot(x, **input_par)
                except:
                    print("\n* Warning: QNM deviation parameters plots failed.")

    except(KeyError, ValueError, configparser.NoOptionError, configparser.NoSectionError):
        print("\n* Warning: Plotting failed with error: {}.".format(traceback.print_exc()))


    #===============================================================#
    # Print execution time and move plots to appropriate directory. #
    #===============================================================#

    if (input_par['run-type']=='full'):
        execution_time = (time.time() - start_execution_time)/60.0
        print('* Total execution time (min): {:.2f}'.format(execution_time))
    if (input_par['run-type']=='full' or input_par['run-type']=='post-processing'):
        if any(plot.endswith(".png") for plot in os.listdir("{}/Nested_sampler/".format(input_par['output']))): os.system('mv {0}/Nested_sampler/*.png {0}/Plots/Parameters/. '.format(input_par['output']))
        if any(plot.endswith(".pdf") for plot in os.listdir("{}/Nested_sampler/".format(input_par['output']))): os.system('mv {0}/Nested_sampler/*.pdf {0}/Plots/Parameters/. '.format(input_par['output']))
        else: pass


if __name__=='__main__':
    main()
