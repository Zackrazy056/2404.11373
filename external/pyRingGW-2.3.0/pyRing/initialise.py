# -*- coding: utf-8 -*-
#Standard python imports
import ast, json, h5py, numpy as np, os, subprocess
try:                import configparser
except ImportError: import ConfigParser as configparser

#LVC imports
from lalinference.imrtgr.nrutils import bbh_final_mass_projected_spins, bbh_final_spin_projected_spins, bbh_final_spin_precessing, bbh_Kerr_trunc_opts
import lal, lalsimulation as lalsim
#Package internal imports
from pyRing            import waveform as wf
from pyRing.utils      import check_NR_dir, F_mrg_Nagar, import_datafile_path, print_section, review_warning, set_prefix

def store_git_info(output):

    """
    
    Store git info in a file.

    Parameters
    ----------

    output : str
        Output directory.

    Returns
    -------

    Nothing, but creates a file with the git info.
    
    """

    git_info = open(os.path.join(output, 'git_info.txt'), 'w')
    pipe1 = str(subprocess.Popen("git branch | grep \* | cut -d ' ' -f2", shell=True, stdout=subprocess.PIPE).stdout.read())[2:-1]
    pipe2 = str(subprocess.Popen("git log --pretty=format:'%H' -n 1 ",    shell=True, stdout=subprocess.PIPE).stdout.read())[2:-1]
    git_info.write('pyRing\nbranch: {}\t commit: {}\n'.format(pipe1, pipe2))
    pipe1 = str(subprocess.Popen('git config user.name',                  shell=True, stdout=subprocess.PIPE).stdout.read())[2:-1]
    pipe2 = str(subprocess.Popen('git config user.email',                 shell=True, stdout=subprocess.PIPE).stdout.read())[2:-1]
    git_info.write('Author: {} {}'.format(pipe1, pipe2))
    pipe = str(subprocess.Popen('git diff',                               shell=True, stdout=subprocess.PIPE).stdout.read())[2:-1]
    git_info.write('\n\nGit diff:\n{}'.format(pipe))
    git_info.close()

    return

def create_directories(output):

    """

    Create the directories where the output will be stored.

    Parameters
    ----------

    output : str
        Output directory.
    
    Returns
    -------

    Nothing, but creates the directories.

    """

    dirs_list = ['Noise'           ,
                 'Plots'           ,
                 'Plots/Parameters',
                 'Plots/SNR'       ,
                 'Plots/Strains'   ,
                 'Nested_sampler'  ]
    
    for dir_to_create in dirs_list:
        if not os.path.exists(os.path.join(output, dir_to_create)): os.makedirs(os.path.join(output, dir_to_create))

    return

def read_config(Config, config_file):

    """

    Read the configuration file.

    Parameters
    ----------

    Config : configparser.ConfigParser
        ConfigParser object.
    config_file : str
        Configuration file.
    
    Returns
    -------

    input_par : dict
        Dictionary with the input parameters.

    """

    input_par={
               #==============#
               # Data inputs. #
               #==============#

               'data-H1'                  : '',
               'data-L1'                  : '',
               'data-V1'                  : '',
               'ignore-data-filename'     : 0,
               'download-data'            : 0,
               'datalen-download'         : 4096.,
               'gw-data-find'             : 0,
               'gw-data-type-H1'          : '',
               'gw-data-type-L1'          : '',
               'gw-data-type-V1'          : '',
               'channel-H1'               : 'GWOSC',
               'channel-L1'               : 'GWOSC',
               'channel-V1'               : 'GWOSC',

               #===============================#
               # Run configuration parameters. #
               #===============================#

               'config-file'              : config_file,
               'run-type'                 : 'full',
               'output'                   : 'pyRing_default_output_directory',
               'run-tag'                  : '',
               'screen-output'            : 0,
               'debug'                    : 0,
               'pesummary'                : 1,

               #==============================================#
               # Detectors, time and sky position parameters. #
               #==============================================#
               
               'trigtime'                 : float(1126259462.423), #Default GW150914 H1 trigtime, consistent with H1 being the ref detector (GW150914 L1 trigtime is 1126259462.416)
               'detectors'                : 'H1,L1,V1',
               'ref-det'                  : 'H1',
               'sky-frame'                : 'default',

               #=========================================#
               # Noise and signal processing parameters. #
               #=========================================#

               'acf-H1'                     : '',
               'acf-L1'                     : '',
               'acf-V1'                     : '',
               'psd-H1'                     : '',
               'psd-L1'                     : '',
               'psd-V1'                     : '',

               'signal-chunksize'           : 2.0,
               'noise-chunksize'            : 2.0,
               'window-onsource'            : 0,
               'window'                     : 1,
               'alpha-window'               : 0.1,
               'sampling-rate'              : 4096.0,
               'f-min-bp'                   : 20.,
               'f-max-bp'                   : 2038.,
               'bandpassing'                : 1,

               'noise-averaging-method'     : 'mean',
               'fft-acf'                    : 1,
               'acf-simple-norm'            : 1,
               'no-lognorm'                 : 0,
               'truncate'                   : 1,
               'analysis-duration'          : 0.1,
               'analysis-duration-n'        : 410,

               'zero-noise'                 : 0,
               'gaussian-noise'             : '',
               'gaussian-noise-seed'        : -1,
               'gaussian-noise-white-sigma' : 1e-21,

               'chisquare-computation'      : 0,
               'non-stationarity-check'     : 0,

               'maxent-psd'                 : '',
               'PSD-investigation'          : 0,

               #========================#
               # Likelihood parameters. #
               #========================#

               'onsource-ACF'               : 0,
               'Dirac-comb'                 : 0,
               'Zeroing-data'               : 0,
               'likelihood-method'          : 'direct-inversion',
               'split-inner-products'       : 0,

               #=======================#
               # Injection parameters. #
               #=======================#

               'injection-parameters'       : None,
               'injection-approximant'      : '',
               'inject-n-ds-modes'          : {"s": 0, "v": 0, "t": 1},
               'inject-area-quantization'   : 0,
               'inject-braneworld'          : 0,
               'inject-charge'              : 0,
               'injection-scaling'          : 1.0,
               'injection-T'                : 64.0,

               #======================#
               # Waveform parameters. #
               #======================#

               'template'                   : 'Kerr',
               'single-mode'                : None,
               'teob-modes'                 : [(2,2)],

               #==============================#
               # Damped sinusoids parameters. #
               #==============================#

               'n-ds-modes'                 : {"s": 0, "v": 0, "t": 1},
               'ds-ordering'                : 'freq',

               #=====================#
               # Kerr GR parameters. #
               #=====================#

               'kerr-modes'                 : [(2,2,2,0)],
               'reference-amplitude'        : 0.0,
               'spheroidal'                 : 0,
               'amp-non-prec-sym'           : 0,
               'tail-modes'                 : None,
               'quadratic-modes'            : None,
               'quadratic-linear-prop'      : 0,
               'qnm-fit'                    : 1,
               'coherent-n'                 : 0,
               'max-Kerr-amp-ratio'         : 0.0,

               #============================#
               # Kerr beyond GR parameters. #
               #============================#

               'TGR-overtones-ordering'     : 'Unordered',
               'domega-tgr-modes'           : None,
               'dtau-tgr-modes'             : None,
               'area-quantization'          : 0,
               'braneworld'                 : 0,
               'tau-AQ'                     : 0,
               'prior-reweight'             : '',
               'ParSpec'                    : 0,
               'ParSpec_Dmax_TGR'           : 2,
               'ParSpec_Dmax_charge'        : 0,
               'EsGB'                       : 0,
               'charge'                     : 0,

               #===========================#
               # Default prior parameters. #
               #===========================#

               'gr-time-prior'              : 1,
               'dist-flat-prior'            : 0,
               'mass-ratio-flat-prior'      : 0,
               'ds-amp-flat-prior'          : 0,
               'free-time-prior'            : 0,
               'mf-time-prior'              : 0.0

    }

    # FIXME: the [input] section should be split in smaller subsections with more appropriate names.
    #General input read.
    for key in input_par:

        # These parameters will be read in the [Priors] section.
        if('-prior' in key): continue

        keytype = type(input_par[key])
        try:
            if   ('ds-modes'        in key): input_par[key] = json.loads(      Config.get("input",'{}'.format(key)))
            elif ('quadratic-modes' in key): input_par[key] = eval(            Config.get("input",'{}'.format(key)))
            elif ('tail-modes'      in key): input_par[key] = eval(            Config.get("input",'{}'.format(key)))
            elif ('mode'            in key): input_par[key] = ast.literal_eval(Config.get("input",key))
            else                           : input_par[key] = keytype(         Config.get("input",key))
        except (KeyError, configparser.NoOptionError, TypeError): pass
        print("{name} : {value}".format(name=key.ljust(max_len_keyword), value=input_par[key]))

    if(input_par['run-tag']==''): input_par['run-tag'] = input_par['output'].split('/')[-1]

    #Set up likelihood specific inputs.
    if(input_par['truncate']):
        assert not(input_par['Dirac-comb'] or input_par['Zeroing-data']), "Running with truncation is incompatible with Dirac comb or zeroing out the data."
        if not(input_par['analysis-duration']):
            raise Exception("Running with the 'truncate' option requires an 'analysis-duration' option to be passed (in seconds).")
        elif((input_par['signal-chunksize']/2.) < input_par['analysis-duration']):
            raise Exception("Analysis duration ({})  must be shorter than half of the signal chunksize ({})".format(input_par['analysis-duration'], input_par['signal-chunksize']/2.))
        else:
            input_par['analysis-duration-n'] = int(input_par['analysis-duration'] * input_par['sampling-rate'])
    print("{name} : {value}".format(name='analysis-duration-n'.ljust(max_len_keyword), value=input_par['analysis-duration-n']))

    #Set detector input and check for incompatibilites.
    try:                                assert not(Config.get("input",'trigtime-H1') is not None), "You are running with an old config file. trigtime-H1 is now deprecated in favour of trigtime (of the correspondent ref-det)."
    except(configparser.NoOptionError): pass

    input_par['detectors']      = [det     for det     in (input_par['detectors']     ).split(',')]
    input_par['prior-reweight'] = [par_rew for par_rew in (input_par['prior-reweight']).split(',')]
    len_det                     = len(input_par['detectors'])

    if(input_par['sky-frame'] == 'default'):
        if(len_det > 1): input_par['sky-frame'] = 'detector'
        else:            input_par['sky-frame'] = 'equatorial'
    elif(input_par['sky-frame'] == 'detector'):
        check = None
        try: check = Config.getfloat("Priors","fix-ra")
        except: pass
        try: check = Config.getfloat("Priors","fix-dec")
        except: pass
        assert (check == None), "Fixed sky position parameters are inconsistent with respect to the 'sky-frame = detector' option selected. If you want to fix ra and dec, please pass the 'sky-frame = equatorial' option."
    elif(input_par['sky-frame'] == 'equatorial'):
        check = None
        try: check = Config.getfloat("Priors","fix-azimuth")
        except: pass
        try: check = Config.getfloat("Priors","fix-cos_altitude")
        except: pass
        assert (check == None), "Fixed sky position parameters are inconsistent with respect to the 'sky-frame = equatorial' option selected. If you want to fix azimuth and cos_altitude, please pass the 'sky-frame = detector' option."
    else:
        raise ValueError("Invalid option for sky position sampling.")

    assert not((len_det == 1) and (input_par['sky-frame'] == 'detector')), "You cannot select detector frame using only a single detector. It needs at least two of them. Please select the skyframe 'default' or 'equatorial' option."

    # Try to fetch the required data from the files included in the package or from the PYRING_PREFIX path.
    for data_string in ['data', 'acf', 'psd']:
        for det in input_par['detectors']:
            if(not(input_par['{}-{}'.format(data_string, det)]=='')):

                package_datapath = import_datafile_path(input_par['{}-{}'.format(data_string, det)])
                PYRING_PREFIX    = set_prefix(warning_message=False)
                custom_datapath  = os.path.join(PYRING_PREFIX, input_par['{}-{}'.format(data_string, det)])
                
                if(os.path.exists(package_datapath)):  input_par['{}-{}'.format(data_string, det)] = package_datapath
                elif(os.path.exists(custom_datapath)): input_par['{}-{}'.format(data_string, det)] = custom_datapath
                else:                                  pass

    # Do some cleaning of the json format for dictionary reading.

    try:
        n_ds_modes = {}
        for key in list(input_par['inject-n-ds-modes'].keys()):
            if (input_par['inject-n-ds-modes'][key] > 0):
                key = key[0]
                n_ds_modes[key] = int(input_par['inject-n-ds-modes'][key])
                assert ((key=='s') or (key=='v') or (key=='t')), "Supported polarisations are: s (scalar perturbations), v (vector perturbations), t (tensor perturbations). The requested polarization {} is not supported.".format(key)
        input_par['inject-n-ds-modes'] = n_ds_modes
        print("{name} : {value}".format(name='inject-n-ds-modes'.ljust(max_len_keyword), value=input_par['inject-n-ds-modes']))
    except (AttributeError, KeyError):
        pass

    try:
        n_ds_modes = {}
        for key in list(input_par['n-ds-modes'].keys()):
            if (input_par['n-ds-modes'][key] > 0):
                key = key[0]
                n_ds_modes[key] = int(input_par['n-ds-modes'][key])
                if not((key=='s') or (key=='v') or (key=='t')):
                    raise ValueError("Supported polarisations are: s (scalar perturbations), v (vector perturbations), t (tensor perturbations). The requested polarization {} is not supported.".format(key))
        input_par['n-ds-modes'] = n_ds_modes
        print("{name} : {value}".format(name='n-ds-modes'.ljust(max_len_keyword), value=input_par['n-ds-modes']))
    except (AttributeError, KeyError):
        pass

    # A few sanity checks
    if (len_det > 1):
        assert not(not (('H1' in input_par['detectors']) or ('L1' in input_par['detectors']))), "When running with more than one detector, H1 and L1 are currently required for sky parameters."
        if ('H1'==input_par['ref-det']): input_par['nonref-det'] = 'L1'
        else:                            input_par['nonref-det'] = 'H1'
    else:
        input_par['nonref-det'] = ''
    if not((input_par['ds-ordering']=='freq') or (input_par['ds-ordering']=='tau') or (input_par['ds-ordering']=='amp')):
        raise ValueError('Unknown ds-ordering method selected.')
    if not((input_par['noise-averaging-method']=='mean') or (input_par['noise-averaging-method']=='median')):
        raise ValueError('Unknown noise-averaging method selected.')

    if(input_par['spheroidal']):
        try:    import pykerr
        except: raise ValueError("The `spheroidal` option relies on the `pykerr` package; importing the latter however failed with error: {}.\nThe probable cause is that the `pykerr` package is not installed: you can solve this by pip-installing locally.\nThe reason why this is not done automatically by the package is that `pykerr` has not been deployed on conda, hence the conda-build fails and `pykerr` cannot be registered as a requirement in the `pyRing` package.".format(traceback.print_exc()))

    return input_par

def read_injection_parameters(input_par, Config):

    """
    
    Read injection parameters from the configuration file.

    Parameters
    ----------

    input_par : dict
        Dictionary containing the input parameters.
    Config : configparser.ConfigParser
        Configuration file parser.

    Returns
    -------

    injection_parameters : dict
        Dictionary containing the injection parameters.
    
    """

    #==============================================================================#
    # Injections section: initialize injection parameters for each waveform model. #
    #==============================================================================#

    injection_parameters = {}

    injection_angles_default = {'ra'           : 0.0,
                                'dec'          : 0.0,
                                'psi'          : 0.0,
                                'cos_altitude' : 0.0,
                                'azimuth'      : 0.0
                               }
    for j in list(injection_angles_default.keys()):
        try:
            injection_parameters[j] = Config.getfloat("Injection",j)
        except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
            injection_parameters[j] = injection_angles_default[j]

    print_section('{} injection'.format(input_par['injection-approximant']))
    print("* I\'ll be injecting the following parameters:\n")

    if ((input_par['injection-approximant']=='Damped-sinusoids') or (input_par['injection-approximant']=='Morlet-Gabor-wavelets')):
        
        injection_parameters_default = {'A'  : 1e-21,
                                        'f'  : 100.0,
                                        'tau': 0.007,
                                        'phi': 0.0,
                                        't'  : 0.0
                                        }

        for par in injection_parameters_default.keys():
            injection_parameters[par] = {}
            for pol in input_par['inject-n-ds-modes'].keys():
                injection_parameters[par][pol] = []
                for i in range(0, input_par['inject-n-ds-modes'][pol]):
                    k = '{}_{}_{}'.format(par,pol,i)
                    try:
                        injection_parameters[par][pol].append(Config.getfloat("Injection",k))
                    except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
                        injection_parameters[par][pol].append(injection_parameters_default[par])

    if (input_par['injection-approximant']=='Kerr'):
        injection_parameters_default = {'t0'                  : 0.004,
                                        'Mf'                  : 67.0,
                                        'af'                  : 0.67,
                                        'logdistance'         : 6.0857,
                                        'cosiota'             : 1.0,
                                        'phi'                 : 0.0,
                                        'alpha'               : 8.*np.pi, # Bekenstein value for area quantization
                                        'beta'                : -0.4, # Typical value for braneworld parameter
                                        'Q'                   : 0.0, # Absolute value of the adimensional electric charge,
                                        'kerr-amplitudes'     : {(2,2,2,0): 0.0},
                                        'kerr-phases'         : {(2,2,2,0): 0.0},

                                        'kerr-tail-amplitudes': None, # syntax: {(l,m): value}, i.e. {(2,2): 0.0}
                                        'kerr-tail-phases'    : None, #{(2,2): 0.0}
                                        'kerr-tail-exponents' : None, #{(2,2): 0.0}

                                        # Otherwise split sum and diff in separate dicts 
                                        'kerr-quad-amplitudes': None, # syntax: {"sum": {((2,4,4,0),(2,2,2,0),(2,2,2,0)): 0.0}, "diff": {((2,4,4,0),(2,2,2,0),(2,2,2,0)): 0.0}}
                                        'kerr-quad-phases'    : None, # syntax: {"sum": {((2,4,4,0),(2,2,2,0),(2,2,2,0)): 0.0}, "diff": {((2,4,4,0),(2,2,2,0),(2,2,2,0)): 0.0}}

                                        'kerr-domegas'        : {(2,2,2,0): 0.0},
                                        'kerr-dtaus'          : {(2,2,2,0): 0.0}
                                        }

        #Temporary constraint. To be relaxed.
        print("* Warning: Imposing non-precessing Kerr amplitude symmetry for both injection and recovery (to be relaxed in future versions).\n")
        input_par['amp-non-prec-sym'] = 1
        
        for p in list(injection_parameters_default.keys()):
            try:
                if ('kerr-' in p):
                    injection_parameters[p] = json.loads(Config.get("Injection",p))
                else:
                    partype = type(injection_parameters_default[p])
                    injection_parameters[p] = partype(Config.get("Injection",p))
            except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
                injection_parameters[p] = injection_parameters_default[p]

        # Do some cleaning of the json format.
        try:
            Kerr_amps_complex = {}
            for key in list(injection_parameters['kerr-amplitudes'].keys()):
                # Syntax: (s,l,m,n)
                if ('-' in key): new_key = (int(key[0]), int(key[1]), -int(key[3]), int(key[4]))
                else           : new_key = (int(key[0]), int(key[1]),  int(key[2]), int(key[3]))
                Kerr_amps_complex[new_key] = injection_parameters['kerr-amplitudes'][key]*np.exp(1j*injection_parameters['kerr-phases'][key])
            injection_parameters['kerr-amplitudes'] = Kerr_amps_complex
            del injection_parameters['kerr-phases']
        except (AttributeError, KeyError):
            pass
        injection_parameters['kerr-tail-parameters'] = None
        try:
            Kerr_tail_params = {}
            for key in list(injection_parameters['kerr-tail-amplitudes'].keys()):
                # Syntax: (l,m)
                if ('-' in key): new_key = (int(key[0]), -int(key[1]))
                else           : new_key = (int(key[0]),  int(key[1]))
                Kerr_tail_params[new_key]        = {}
                Kerr_tail_params[new_key]['A'  ] = injection_parameters['kerr-tail-amplitudes'][key]
                Kerr_tail_params[new_key]['phi'] = injection_parameters['kerr-tail-phases'    ][key]
                Kerr_tail_params[new_key]['p'  ] = injection_parameters['kerr-tail-exponents' ][key]
            injection_parameters['kerr-tail-parameters'] = Kerr_tail_params
            del injection_parameters['kerr-tail-amplitudes']
            del injection_parameters['kerr-tail-phases']
            del injection_parameters['kerr-tail-exponents']
        except (AttributeError, KeyError):
            pass
        try:
            Kerr_quad_amps_complex = {}
            for quad_term in list(injection_parameters['kerr-quad-amplitudes'].keys()):
                Kerr_quad_amps_complex[quad_term] = {}
                for key in list(injection_parameters['kerr-quad-amplitudes'][quad_term].keys()):
                    # Syntax: ((s,l,m,n),(s1,l1,m1,n1),(s2,l2,m2,n2)), with {1,2} the parent modes
                    child_mode, parent_1_mode, parent_2_mode = key.split(',')
                    if ('-' in child_mode)   : new_child_mode    = (int(child_mode[0]   ), int(child_mode[1]   ), -int(child_mode[3]   ), int(child_mode[4]   ))
                    else                     : new_child_mode    = (int(child_mode[0]   ), int(child_mode[1]   ),  int(child_mode[2]   ), int(child_mode[3]   ))
                    if ('-' in parent_1_mode): new_parent_1_mode = (int(parent_1_mode[0]), int(parent_1_mode[1]), -int(parent_1_mode[3]), int(parent_1_mode[4]))
                    else                     : new_parent_1_mode = (int(parent_1_mode[0]), int(parent_1_mode[1]),  int(parent_1_mode[2]), int(parent_1_mode[3]))
                    if ('-' in parent_2_mode): new_parent_2_mode = (int(parent_2_mode[0]), int(parent_2_mode[1]), -int(parent_2_mode[3]), int(parent_2_mode[4]))
                    else                     : new_parent_2_mode = (int(parent_2_mode[0]), int(parent_2_mode[1]),  int(parent_2_mode[2]), int(parent_2_mode[3]))
                    new_key = (new_child_mode, new_parent_1_mode, new_parent_2_mode)
                    Kerr_quad_amps_complex[quad_term][new_key] = injection_parameters['kerr-quad-amplitudes'][quad_term][key]*np.exp(1j*injection_parameters['kerr-quad-phases'][quad_term][key])
            injection_parameters['kerr-quad-amplitudes'] = Kerr_quad_amps_complex
            del injection_parameters['kerr-quad-phases']
        except (AttributeError, KeyError):
            pass

        try:
            Kerr_domegas_dict = {}
            for key in list(injection_parameters['kerr-domegas'].keys()):
                # Syntax: (s,l,m,n)
                if ('-' in key): new_key = (int(key[0]), int(key[1]), -int(key[3]), int(key[4]))
                else:            new_key = (int(key[0]), int(key[1]),  int(key[2]), int(key[3]))
                Kerr_domegas_dict[new_key] = injection_parameters['kerr-domegas'][key]
            injection_parameters['kerr-domegas'] = Kerr_domegas_dict
        except (AttributeError, KeyError):
            pass
        try:
            Kerr_dtaus_dict = {}
            for key in list(injection_parameters['kerr-dtaus'].keys()):
                # Syntax: (s,l,m,n)
                if ('-' in key): new_key = (int(key[0]), int(key[1]), -int(key[3]), int(key[4]))
                else:            new_key = (int(key[0]), int(key[1]),  int(key[2]), int(key[3]))
                Kerr_dtaus_dict[new_key] = injection_parameters['kerr-dtaus'][key]
            injection_parameters['kerr-dtaus'] = Kerr_dtaus_dict
        except (AttributeError, KeyError):
            pass

    if (input_par['injection-approximant']=='MMRDNS'):
        review_warning()
        injection_parameters_default = {'t0'          : 0.004,
                                        'Mf'          : 67.0,
                                        'af'          : 0.67,
                                        'eta'         : 0.25,
                                        'logdistance' : 6.0857,
                                        'cosiota'     : 1.0,
                                        'phi'         : 0.0
                                        }

        for p in list(injection_parameters_default.keys()):
            try:
                partype = type(injection_parameters_default[p])
                injection_parameters[p] = partype(Config.get("Injection",p))
            except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
                injection_parameters[p] = injection_parameters_default[p]

    if (input_par['injection-approximant']=='MMRDNP'):
        injection_parameters_default = {'t0'          : 0.004,
                                        'm1'          : 35.0,
                                        'm2'          : 32.0,
                                        'chi1'        : 0.0,
                                        'chi2'        : 0.0,
                                        'logdistance' : 6.0857,
                                        'cosiota'     : 1.0,
                                        'phi'         : 0.0
                                        }

        for p in list(injection_parameters_default.keys()):
            try:
                partype = type(injection_parameters_default[p])
                injection_parameters[p] = partype(Config.get("Injection",p))
            except(KeyError, configparser.NoOptionError, configparser.NoSectionError):
                injection_parameters[p] = injection_parameters_default[p]

        injection_parameters['Mi'] = injection_parameters['m1'] + injection_parameters['m2']

        if(injection_parameters['chi1'] < 0): tilt1_fit = np.pi
        else: tilt1_fit = 0.0
        if(injection_parameters['chi2'] < 0): tilt2_fit = np.pi
        else: tilt2_fit = 0.0
        chi1_fit  = np.abs(injection_parameters['chi1'])
        chi2_fit  = np.abs(injection_parameters['chi2'])
        injection_parameters['Mf']   = bbh_final_mass_projected_spins(injection_parameters['m1'], injection_parameters['m2'], chi1_fit, chi2_fit, tilt1_fit, tilt2_fit, 'UIB2016')
        injection_parameters['af']   = bbh_final_spin_projected_spins(injection_parameters['m1'], injection_parameters['m2'], chi1_fit, chi2_fit, tilt1_fit, tilt2_fit, 'UIB2016', truncate = bbh_Kerr_trunc_opts.trunc)

        injection_parameters['eta']  = (injection_parameters['m1']*injection_parameters['m2'])/(injection_parameters['Mi'])**2
        injection_parameters['chis'] = (injection_parameters['m1']*injection_parameters['chi1'] + injection_parameters['m2']*injection_parameters['chi2'])/(injection_parameters['Mi'])
        injection_parameters['chia'] = (injection_parameters['m1']*injection_parameters['chi1'] - injection_parameters['m2']*injection_parameters['chi2'])/(injection_parameters['Mi'])

    if (input_par['injection-approximant']=='TEOBResumSPM'):
        injection_parameters_default = {'t0'             : 0.004,
                                        'm1'             : 50.0,
                                        'm2'             : 50.0,
                                        'chi1'           : 0.0,
                                        'chi2'           : 0.0,
                                        'phi22'          : 0.0,
                                        'phi21'          : 0.0,
                                        'phi33'          : 0.0,
                                        'phi32'          : 0.0,
                                        'phi31'          : 0.0,
                                        'phi44'          : 0.0,
                                        'phi43'          : 0.0,
                                        'phi42'          : 0.0,
                                        'phi41'          : 0.0,
                                        'phi55'          : 0.0,
                                        'cosiota'        : 1.0,
                                        'logdistance'    : 6.0857,
                                        'phi'            : 0.0,
                                        'inject-modes'   : [(2,2)],
                                        }

        for p in list(injection_parameters_default.keys()):
            try:
                if(p=='inject-modes'):
                   injection_parameters[p] = ast.literal_eval(Config.get("Injection",'inject-modes'))
                else:
                   partype = type(injection_parameters_default[p])
                   injection_parameters[p] = partype(Config.get("Injection",p))
            except(KeyError, configparser.NoOptionError, configparser.NoSectionError):
                injection_parameters[p] = injection_parameters_default[p]

        injection_parameters['M'] = injection_parameters['m1'] + injection_parameters['m2']
        injection_parameters['q'] = injection_parameters['m1']/injection_parameters['m2']

        #For the documentation on final state fits, see https://lscsoft.docs.ligo.org/lalsuite/lalinference/namespacelalinference_1_1imrtgr_1_1nrutils.html#a351987f8201d32f50af2fd38a7e4190e
        
        if(injection_parameters['chi1'] < 0): tilt1_fit = np.pi
        else: tilt1_fit = 0.0
        if(injection_parameters['chi2'] < 0): tilt2_fit = np.pi
        else: tilt2_fit = 0.0
        chi1_fit  = np.abs(injection_parameters['chi1'])
        chi2_fit  = np.abs(injection_parameters['chi2'])
        injection_parameters['Mf'] = bbh_final_mass_projected_spins(injection_parameters['m1'], injection_parameters['m2'], chi1_fit, chi2_fit, tilt1_fit, tilt2_fit, 'UIB2016')
        injection_parameters['af'] = bbh_final_spin_projected_spins(injection_parameters['m1'], injection_parameters['m2'], chi1_fit, chi2_fit, tilt1_fit, tilt2_fit, 'UIB2016', truncate = bbh_Kerr_trunc_opts.trunc)

    if (input_par['injection-approximant']=='KHS_2012'):
        review_warning()
        injection_parameters_default = {'t0'          : 0.004,
                                        'Mf'          : 67.0,
                                        'af'          : 0.67,
                                        'eta'         : 0.25,
                                        'chi_eff'     : 0.0,
                                        'logdistance' : 6.0857,
                                        'cosiota'     : 1.0,
                                        'phi'         : 0.0
                                        }

        for p in list(injection_parameters_default.keys()):
            try:
                partype = type(injection_parameters_default[p])
                injection_parameters[p] = partype(Config.get("Injection",p))
            except(KeyError, configparser.NoOptionError, configparser.NoSectionError):
                injection_parameters[p] = injection_parameters_default[p]

    if ((input_par['injection-approximant']=='NR') or ('LAL' in input_par['injection-approximant'])):

        #FIXME: Section C of IMRPhenomX model paper https://dcc.ligo.org/DocDB/0165/P2000011/003/IMRPhenomXHM.pdf
        # says that the angles that enters the spherical harmonics 'phi' = pi/2 - phiref
        # Currently we pass phi = phiref, so there should be a disagreement of pi/2 wrt the injected value.
        # If this convention is valid for all wfs, included NR, then we should implement it in the injections as well.
        if (input_par['injection-approximant']=='NR'):
            injection_parameters_default = {'M'                  : 70.0,
                                            'dist'               : 400.0,
                                            'incl'               : 0.0,
                                            'phi'                : 0.0,
                                            'NR-ID'              : '0180',
                                            'NR-catalog'         : 'SXS',
                                            'resolution-level'   : -1
                                            }

            for j in list(injection_parameters_default.keys()):
                keytype = type(injection_parameters_default[j])
                try:
                    injection_parameters[j] = keytype(Config.get("Injection",j))
                except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
                    injection_parameters[j] = injection_parameters_default[j]

            assert not(injection_parameters['resolution-level'] == -1), "A positive resolution level must be passed from the user."

            # For more details on the LVC NR injection infrastructure see this paper: arxiv-1703.01076 or this wiki page https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Waveforms/NR/InjectionInfrastructure

            # NR metadata
            PYRING_PREFIX = set_prefix()
            check_NR_dir()

            NR_catalog                          = injection_parameters['NR-catalog']
            NR_ID                               = injection_parameters['NR-ID']
            resolution_level                    = injection_parameters['resolution-level']
            injection_parameters['NR-datafile'] = os.path.join(PYRING_PREFIX, 'data/NR_data/lvcnr-lfs/{0}/{0}_BBH_{1}_Res{2}.h5'.format(NR_catalog, NR_ID, resolution_level))
            NR_injection                        = h5py.File(injection_parameters['NR-datafile'], 'r')

            # Binary parameters
            mass  = injection_parameters['M']
            m1_NR = NR_injection.attrs['mass1']
            m2_NR = NR_injection.attrs['mass2']
            
            # NR has unitary total mass, so we need to rescale the masses.
            injection_parameters['m1']       = m1_NR * mass / (m1_NR + m2_NR)
            injection_parameters['m2']       = m2_NR * mass / (m1_NR + m2_NR)
            injection_parameters['q']        = injection_parameters['m1']/injection_parameters['m2']
            injection_parameters['theta_LN'] = injection_parameters['incl']

            injection_parameters['f-lower-NR'] = NR_injection.attrs['f_lower_at_1MSUN']/mass  # this generates the whole NR waveform
            injection_parameters['f-ref']      = injection_parameters['f-lower-NR']
            injection_parameters['f-start']    = injection_parameters['f-lower-NR']
#                input_par['f-min-bp']            = injection_parameters['f-lower-NR']
            spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(injection_parameters['f-ref'], mass, injection_parameters['NR-datafile'])
            injection_parameters['s1x_LALSim'], injection_parameters['s1y_LALSim'], injection_parameters['s1z_LALSim'] = spins[0], spins[1], spins[2]
            injection_parameters['s2x_LALSim'], injection_parameters['s2y_LALSim'], injection_parameters['s2z_LALSim'] = spins[3], spins[4], spins[5]
            NR_injection.close()

            spin_tolerance = 1e-3
            if(np.abs(injection_parameters['s1x_LALSim'])>=spin_tolerance or np.abs(injection_parameters['s1y_LALSim'])>=spin_tolerance or np.abs(injection_parameters['s2x_LALSim'])>=spin_tolerance or np.abs(injection_parameters['s2y_LALSim'])>=spin_tolerance):
                raise ValueError("Precessing NR injections are not yet supported, need to edit the code to correctly define spin angles.")
            else:
                if(injection_parameters['s1z_LALSim'] < 0): injection_parameters['tilt1']  = np.pi
                else: injection_parameters['tilt1']  = 0.0
                if(injection_parameters['s2z_LALSim'] < 0): injection_parameters['tilt2'] = np.pi
                else: injection_parameters['tilt2']  = 0.0

                injection_parameters['phi_12'] = 0.0

        else:
            # If not explicitely specified (e.g. with the suffix _LALSim), all parameters are referred to the reference frame where the z-axis is fixed and aligned with the total angular momentum direction J
            injection_parameters_default = {'m1'                 : 35.0,
                                            'm2'                 : 35.0,
                                            's1'                 : 0.0,
                                            's2'                 : 0.0,
                                            'tilt1'              : 0.0,
                                            'tilt2'              : 0.0,
                                            'dist'               : 400.0,
                                            'incl'               : 0.0, #'incl' is *defined* as theta_JN
                                            'phi'                : 0.0,
                                            'phi_JL'             : 0.0,
                                            'phi_12'             : 0.0,
                                            'f-ref'              : 20.,
                                            'amp-order'          : -1,
                                            'phase-order'        : -1,
                                            }

            for j in list(injection_parameters_default.keys()):
                keytype = type(injection_parameters_default[j])
                try:
                    injection_parameters[j] = keytype(Config.get("Injection",j))
                except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
                    injection_parameters[j] = injection_parameters_default[j]

            # Set appropriate starting frequency in case where HMs are used and change bandpassing frequency accordingly.
            injection_parameters['f-start']  = lalsim.SimInspiralfLow2fStart(input_par['f-min-bp'], injection_parameters['amp-order'], lalsim.SimInspiralGetApproximantFromString(input_par['injection-approximant'].strip('LAL-')))

            # This step is needed because for precessing systems the reference frame used by LALSimulation to define the spins is different wrt the one used in NR or PE (both LALInference and pyRing). Full documentation and parameters definitions can be found here: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html
            injection_parameters['theta_LN'], injection_parameters['s1x_LALSim'], injection_parameters['s1y_LALSim'], injection_parameters['s1z_LALSim'], injection_parameters['s2x_LALSim'], injection_parameters['s2y_LALSim'], injection_parameters['s2z_LALSim'] = lalsim.SimInspiralTransformPrecessingNewInitialConditions(injection_parameters['incl'],
                                                                          injection_parameters['phi_JL'],
                                                                          injection_parameters['tilt1'],
                                                                          injection_parameters['tilt2'],
                                                                          injection_parameters['phi_12'],
                                                                          injection_parameters['s1'],
                                                                          injection_parameters['s2'],
                                                                          injection_parameters['m1'],
                                                                          injection_parameters['m2'],
                                                                          injection_parameters['f-ref'],
                                                                          injection_parameters['phi'])

        #IMPROVEME: write document with details on our reference frame
       # This section is temporarily commented out, since gwsurrogate was giving installation problems.
       # try:
       #     q     = injection_parameters['m1']/injection_parameters['m2']
       #     M_tot = injection_parameters['m1']+injection_parameters['m2']
       #     chi1  = [injection_parameters['s1x_LALSim'], injection_parameters['s1y_LALSim'], injection_parameters['s1z_LALSim']]
       #     chi2  = [injection_parameters['s2x_LALSim'], injection_parameters['s2y_LALSim'], injection_parameters['s2z_LALSim']]
       #     injection_parameters['Mf'], injection_parameters['af'] = utils.final_state_surfinBH(M_tot, q, chi1, chi2, injection_parameters['f-ref'])
       # except:
        # Case where we are outside the validity region of the surrogate.
       # print("\nWarning: Surrogate remnant fits failed with error {}.\nUsing analytical fits from UIB instead.".format(traceback.print_exc()))
        #FIXME: check the reference frame in which spin z-components should be specified
        chi1  = np.sqrt(injection_parameters['s1x_LALSim']**2 + injection_parameters['s1y_LALSim']**2 + injection_parameters['s1z_LALSim']**2)
        chi2  = np.sqrt(injection_parameters['s2x_LALSim']**2 + injection_parameters['s2y_LALSim']**2 + injection_parameters['s2z_LALSim']**2)
        tilt1 = injection_parameters['tilt1']
        tilt2 = injection_parameters['tilt2']

        #For the documentation on final state fits, see https://lscsoft.docs.ligo.org/lalsuite/lalinference/namespacelalinference_1_1imrtgr_1_1nrutils.html#a351987f8201d32f50af2fd38a7e4190e
        injection_parameters['af'] = bbh_final_spin_precessing(injection_parameters['m1'], injection_parameters['m2'], chi1, chi2, tilt1, tilt2, injection_parameters['phi_12'], 'UIB2016', truncate=bbh_Kerr_trunc_opts.trunc)
        injection_parameters['Mf'] = bbh_final_mass_projected_spins(injection_parameters['m1'], injection_parameters['m2'], chi1, chi2, tilt1, tilt2, 'UIB2016')

        injection_parameters['f_220']      = wf.QNM_fit(2,2,0).f(injection_parameters['Mf'], injection_parameters['af'])
        injection_parameters['tau_220']    = wf.QNM_fit(2,2,0).tau(injection_parameters['Mf'], injection_parameters['af'])
        injection_parameters['f_220_peak'] = F_mrg_Nagar(injection_parameters['m1']*lal.MSUN_SI, injection_parameters['m2']*lal.MSUN_SI, injection_parameters['s1z_LALSim'], injection_parameters['s2z_LALSim'])

        try:
            injection_parameters['inject-modes'] = ast.literal_eval(Config.get("Injection",'inject-modes'))
        except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
            injection_parameters['inject-modes'] = None
        try:
            injection_parameters['inject-l-modes'] = ast.literal_eval(Config.get("Injection",'inject-l-modes'))
        except (KeyError, configparser.NoOptionError, configparser.NoSectionError):
            injection_parameters['inject-l-modes'] = None

    if (input_par['sky-frame'] == 'detector'):
        check = None
        try: check = Config.getfloat("Injection","ra")
        except: pass
        try: check = Config.getfloat("Injection","dec")
        except: pass
        assert (check == None), "Injected sky position parameters are inconsistent with respect to the 'sky-frame = detector' option selected. If you want to inject ra and dec, please pass the 'sky-frame = equatorial' option."
    elif (input_par['sky-frame'] == 'equatorial'):
        check = None
        try: check = Config.getfloat("Injection","azimuth")
        except: pass
        try: check = Config.getfloat("Injection","cos_altitude")
        except: pass
        assert (check == None), "Injected sky position parameters are inconsistent with respect to the 'sky-frame = equatorial' option selected. If you want to inject azimuth and cos_altitude, please pass the 'sky-frame = detector' option."
    else:
        raise ValueError("Invalid option for sky position sampling.")

    return injection_parameters

#Description of the package. Printed on stdout if --help option is give.
usage="""\n\n %prog --config-file config.ini\n
Parameter estimation package targeting time-domain ringdown analyses.

Options syntax: type, default values and sections of the configuration
file where each parameter should be passed are declared below.
By convention, booleans are represented by the integers [0,1].
To use default values, do not include the parameter in the configuration
file: empty fields are interpreted as empty strings.
A dot is present at the end of each description line and is not
to be intended as part of the default value.



       ***************************************************
       * Parameters to be passed to the [input] section. *
       ***************************************************

               #==========================================================#
               # Data inputs.                                             #
               # Data reading from file assumes the following convention: #
               # "IFO-frame_type-segment_start_time-segment_duration"     #
               # e.g.: H-H1_GWOSC_4_V1-1126259446-32.txt                  #
               #==========================================================#

               data-H1                    Name of H1 data file (either path relative to the package data, or to a PYRING_PREFIX or absolute path). Default: ''.
               data-L1                    Name of L1 data file (either path relative to the package data, or to a PYRING_PREFIX or absolute path). Default: ''.
               data-V1                    Name of LV data file (either path relative to the package data, or to a PYRING_PREFIX or absolute path). Default: ''.
               ignore-data-filename       Flag to ignore naming of data file if loading from disk. Default: False.
               download-data              Flag to allow for data fetching through gwpy. If true, a channel name is required for each detector. Default: 0.
               datalen-download           Length (in s) of the data to be downloaded through gwpy. Default: 4096.
               gw-data-find               Flag to fetch LVC proprietary data not available through GWPY. Default: 0.
               gw-data-type-H1            Type of the H1.gwf file fetched through gw_data_find. Default: ''.
               gw-data-type-L1            Type of the L1.gwf file fetched through gw_data_find. Default: ''.
               gw-data-type-V1            Type of the V1.gwf file fetched through gw_data_find. Default: ''.
               channel-H1                 Name of H1 interferometric channel. Defult: 'GWOSC'.
               channel-L1                 Name of L1 interferometric channel. Defult: 'GWOSC'.
               channel-V1                 Name of V1 interferometric channel. Defult: 'GWOSC'.

               #===============================#
               # Run configuration parameters. #
               #===============================#

               run-type                   Type of run to be launched. Available options: ['full', 'noise-estimation-only', 'post-processing']. Default: 'full'.
               output                     Output directory of the run. Default: 'pyRing_default_output_directory'.
               run-tag                    Label of the run in the PESummary metafile. Default: outdir.
               screen-output              Flag to allow for the stdout to appear on screen. Default: 0.
               debug                      Flag to activate debugging mode, printing additional info. Default: 0.
               pesummary                  Flag to create a PESummary metafile at the end of the run containing all the needed inputs to completely reproduce the run and the run output. Default: 1.
               
               #==============================================#
               # Detectors, time and sky position parameters. #
               #==============================================#

               trigtime                   Reference time in the reference detector 'ref-det' from which the analysis will start. For ringdown-only injections will be the start of the signal, while IMR injections will be aligned in such a way that the trigtime corresponds to the peak of hp^2 + hc^2. Default: 1126259462.423.
               detectors                  Detectors to be used in the run. Default: 'H1,L1'.
               ref-det                    Detector corresponding to the given trigtime. Default: 'H1'.
               sky-frame                  Sky-frame reference systems. Valid options: ['equatorial', 'detector', 'default']. Default: 'default'.

               #=========================================#
               # Noise and signal processing parameters. #
               #=========================================#

               acf-H1                     Name of H1 ACF from file (either path relative to the package data, or to a PYRING_PREFIX or absolute path). Default: ''.
               acf-L1                     Name of L1 ACF from file (either path relative to the package data, or to a PYRING_PREFIX or absolute path). Default: ''.
               acf-V1                     Name of V1 ACF from file (either path relative to the package data, or to a PYRING_PREFIX or absolute path). Default: ''.
               psd-H1                     Name of H1 PSD from file (either path relative to the package data, or to a PYRING_PREFIX or absolute path). Default: ''.
               psd-L1                     Name of L1 PSD from file (either path relative to the package data, or to a PYRING_PREFIX or absolute path). Default: ''.
               psd-V1                     Name of V1 PSD from file (either path relative to the package data, or to a PYRING_PREFIX or absolute path). Default: ''.

               signal-chunksize           Length (in s) of the chunk that contains the signal. Default: 2.0.
               noise-chunksize            Length (in s) of the chunks used in ACF estimation. Should be equal to signal-chunksize if our standard method is employed (but truncation method does not require it). Default: 2.0.
               window-onsource            Flag to add a window on the signal chunk. Default: 0.
               window                     Flag to add a window on the noise chunk. Default: 1.
               alpha-window               Value setting the rise and decay time of the noise window. Default: 0.1 (same value as LALInference).
               sampling-rate              Inverse of the data time step (assumed uniform). If bandpassing is on, requires a consistent bandpassing maximum frequency. Default: 4096.0.
               f-min-bp                   Minimum frequency of the bandpass filter. Default: 20.0.
               f-max-bp                   Maximum frequency of the bandpass filter. Default: 2038.0.
               bandpassing                Flag to bandpassing the data. Default: 1.

               noise-averaging-method     Option to perform the noise averaging using different methods. Available options: ['mean', 'median']. Deafult: 'mean'.
               fft-acf                    Flag to compute the ACF through an FFT. Default: 1.
               acf-simple-norm            Option to activate the simple 1/N norm of the ACF estimator. With this norm, the ACF estimator is biased, but reduces the influence of the terms with large lag, where the ACF variance is larger (see arXiv:2107.05609 for references). Instead by default, at lag k, we use the unbiased estimator with norm 1/(N-k). Default: False.
               no-lognorm                 Flag to omit the likelihood normalisation. Default: 0.
               truncate                   Flag to computed the likelihood only on data starting at t0. Default: 1.
               analysis-duration          Option used to set seglen (in seconds) if truncating. Default: 0.1.

               zero-noise                 Flag to inject a waveform without any detector noise. Default: 0.
               gaussian-noise             Option to generate a strain from a gaussian stochastic process. Available options: ['white', 'coloured', 'None']. Default: None.
               gaussian-noise-seed        Option to fix a random seed for the noise generation. If -1, noise will be different at each iteration. Default: -1.
               gaussian-noise-white-sigma Standard deviation of white gaussian noise. Default: 1e-21.
               
               chisquare-computation      Flag to output a chisquare computation on the data. Default: 0.
               non-stationarity-check     Flag to perform a simple non-stationarity check on the data. Default: 0.
               
               maxent-psd                 Option to use the maximum entropy estimation method to compute the PSD. Available options: ['average', 'onsource-chunk', 'pre-onsource-chunk', 'post-onsource-chunk']. Default:''.
               PSD-investigation          Flag to plot the PSDs from all chunks. Useful to identify localized non-stationarities. Default: 0.

               #========================#
               # Likelihood parameters. #
               #========================#

               onsource-ACF               Flag to use the ACF computed on-source in the likelihood. Technically the correct thing to do, but still in testing phase. Default: 0.
               Dirac-comb                 Flag to use as inspiral model the data itself. Sets to 0 all the contributions except for ringdown inside the likelihood. Default: 0.
               Zeroing-data               Flag to zero-out the data before the start of the analysis. Sets to 0 all the contributions except for ringdown inside the likelihood, but inserts spurious frequencies in the data (FIXME: implement window to cure this). Default: 0.
               likelihood-method          Option to change the numerical method with which the likelihood is computed. Use to check for numerical stability of the results. Ignored when `split-inner-products=1`. Available options: ['direct-inversion', 'cholesky-solve-triangular', 'toeplitz-inversion']. Default: 'direct-inversion'.
               split-inner-products       Flag to change the numerical implementation with which the likelihood is computed, avoiding computing the subtraction of two small quantities. Use to check for numerical stability of the results. Default: 0.

               #=======================#
               # Injection parameters. #
               #=======================#

               injection-approximant      Option to select the waveform approximant to be injected.  Current available options: ['Damped-sinusoids', 'Morlet-Gabor-wavelets', 'Kerr', 'MMRDNS', 'MMRDNP', 'KHS_2012', 'TEOBResumSPM', 'NR', 'LAL-X'where X is any LAL approximant]. If 'NR', requires a PYRING_PREFIX to be set. Default: ''.
               inject-n-ds-modes        Number of scalar, vector and tensor damped sinuoids to be injected for 'Damped-sinusoids' template. Default: '{"s": 0, "v": 0, "t": 1}'.
               inject-area-quantization   Flag to inject a Kerr waveform with frequencies set by the area quantization proposal in the data. Default: 0.
               inject-charge              Flag to inject a Kerr-Newman waveform. Default: 0.
               injection-scaling          Parameter used to scale the distance of the injection. Default: 1.0.
               injection-T                Duration of data segment when performing an injection. Default: 64.0.

               #======================#
               # Waveform parameters. #
               #======================#

               template                   Template to be used in the run. Current available options: ['Damped-sinusoids', 'Kerr', 'MMRDNS', 'MMRDNP', 'KHS_2012', 'TEOBResumSPM']. Default: 'Kerr'.
               single-mode                Option to use a single mode in multi-modal NR waveforms ['MMRDNS', 'MMRDNP', 'KHS_2012']. Example of expected syntax: [(l,m)]. For MMRDNS it fixes the (l,m,0) mode. Default: None.
               teob-modes                 Modes to be used in 'TEOBResumSPM' template. Available modes: [(2,2), (2,1), (3,3), (3,2), (3,1), (4,4), (4,3), (4,2), (4,1), (5,5)]. For the accuracy of specific modes, see Chapter 6.2.2 and Appendix C.3 of `https://etd.adm.unipi.it/t/etd-10022022-163345`. Default: [(2,2)] (syntax: [(l,m)]).
               
               #==============================#
               # Damped sinusoids parameters. #
               #==============================#

               n-ds-modes                 Number of scalar, vector and tensor damped sinusoids to be used for 'Damped-sinusoids' template. Default: '{"s": 0, "v": 0, "t": 1}'.
               ds-ordering                Sort the modes according to increasing values of this parameter, relevant for damped-sinusoid waveforms. Available options: ['freq', 'tau', 'amp']. Default: 'freq'.
               ds-amp-flat-prior          Flag to select a flat prior on the damped-sinusoids amplitude. Default: 0.

               #=====================#
               # Kerr GR parameters. #
               #=====================#

               kerr-modes                 Modes to be used in 'Kerr' template. Default: [(2,2,2,0)] (syntax: [(s,l,m,n)]). Note: s \=2 modes correspond to non-tensorial QNMs coupling to the tensorial gravitational degrees of freedom.
               reference-amplitude        Option to set a reference amplitude (e.g. 1E-21) in place of the M_f/D_L waveform prefactor (in geometrised units) since this factor is completely degenerate with modes free amplitudes. Overwrites sampling in distance. Default: 0.0 (not active).
               spheroidal                 Flag to allow for spheroidal decomposition of the waveform. Default: 0.
               amp-non-prec-sym           Flag to enforce non-precessing symmetry on Kerr amplitudes. Default: 0.
               tail-modes                 Modes to be used in the tail terms of the 'Kerr' template (s=2 only). Default: None (syntax: [(l,m)]).
               quadratic-modes            Quadratic modes to be used in the 'Kerr' template (s=2 only). Default: None. Syntax: {"sum:": [((s,l,m,n),(s1,l1,m1,n1),(s2,l2,m2,n2))], "diff": [((s',l',m',n'),(s1',l1',m1',n1'),(s2',l2',m2',n2'))]} i.e. ('child_term': child-parent1-parent2), child mode has the sum and difference of the frequencies of parent modes. E.g. {"sum": [((2,4,4,0),(2,2,2,0),(2,2,2,0)), ((2,4,4,0), (2,2,2,0), (2,3,2,0))], "diff": [((2,4,4,0),(2,2,2,0),(2,2,2,0))]}.
               quadratic-linear-prop      Flag to fix the child quadratic amplitudes as product of parents amplitudes, as alpha A_1 * A_2, with alpha a proportionality constant to be sampled and (A_1, A_2) the corresponding linear amplitudes. If active, the parent amplitudes have to be present in the `kerr-modes` list and the value stored in the quadratic amplitudes structures will correspond alpha. Default: 0.
               qnm-fit                    Flag to use fits for QNM complex frequencies, instead of interpolating numerical relativity data directly. Available only up to n=2 and l=4, for s=2. If 0, requires a PYRING_PREFIX to be set. Default: 1.
               coherent-n                 Option to allow for different overtones to be summed up coherently in phase. Default: False.
               max-Kerr-amp-ratio         Option to force amplitudes of modes different than l=m=2 to be smaller than max-Kerr-amp-ratio*A_220. Does not apply to overtones of the l=m=2 mode. Default: 0.0

               #============================#
               # Kerr beyond GR parameters. #
               #============================#

               TGR-overtones-ordering     Sort the Kerr modes according to increasing values of this parameter, relevant for Kerr waveforms when including overtones and parametrised deviations. Available options: ['Unordered', 'freq', 'tau']. Default: 'Unordered'.
               domega-tgr-modes           Modes on which to sample on deviations from GR predictions in the QNM frequencies. Available for models: ['Kerr', 'MMRDNP', 'TEOBResumSPM']. For 'TEOBResumSPM' only n=0 modes can currently be selected. Example of expected syntax: [(2,2,0)]. Default: None.
               dtau-tgr-modes             Modes on which to sample on deviations from GR predictions in the QNM damping times. Available for models: ['Kerr', 'MMRDNP', 'TEOBResumSPM']. For 'TEOBResumSPM' only n=0 modes can currently be selected.  Example of expected syntax: [(2,2,0)]. Default: None.
               area-quantization          Flag to allow for area-quantization test. Default: 0.
               braneworld                 Flag to allow for braneworld corrections to the Kerr spectrum. Default: 0.
               tau-AQ                     Flag to sample on tau in area-quantization test. Default: 0.
               prior-reweight             List of parameters for which the prior has to be reweighted according to custom file. Example of expected syntax: 'Mf,af,A2220'. Default: ''.
               ParSpec                    Flag to use the ParSpec description of the QNM spectrum. Required to sample GR deviations using the parametrised spinning formalism. Default: 0.
               ParSpec_Dmax_TGR           Option to specify the maximum order of the testing GR deviation coefficients in the parametrised spinning expansion. Default: 5.
               ParSpec_Dmax_charge        Option (integer) to specify the dimension of the charge in the parametrised spinning expansion. Default: 0.
               EsGB                       Flag to specify the Einstein-scalar-Gauss-Bonnet corrections within the ParSpec formalism. Default: 0.
               charge                     Flag to allow for the inclusion of charge. Default: 0.

       ****************************************************
       * Parameters to be passed to the [Priors] section. *
       ****************************************************

               #========================#
               # Time prior parameters. #
               #========================#

               gr-time-prior              Flag to select a time prior between [X,Y] in units of mf-time-prior. Default: 1.
               dist-flat-prior            Flag to select a flat prior on the luminosity distance. Default: 0.
               mass-ratio-flat-prior      Flag to select a flat prior on the mass ratio. Default: 0.
               mf-time-prior              Value of the mass used to set the time prior. Default: 0.0.

               #==============================================#
               # Waveform parameters priors and fixed values. #
               #==============================================#

               Prior default bounds can be changed by adding 'param-min=value' or 'param-max=value' to this section.
               Currently for sky position parameters and Kerr modes specific parameters (amplitudes, phases and QNM deviations) it is not possible to change only the lower or upper bound separately, thus the above commands should both be passed (FIXME).
               All sampled parameters can be fixed adding the 'fix-name=value' option.
               The list of available waveforms, waveform parameters and default parameters priors can be accessed through FIXME MISSING.

       **************************************************************
       * Parameters to be passed to the [Sampler settings] section. *
       **************************************************************

               #=====================#
               # Sampler parameters. #
               #=====================#

               sampler                      Option to switch between different samplers. Available options: [`cpnest`, `raynest`]. Default: 'cpnest'.
               
               #===============================#
               # Branch-independent parameters #
               #===============================#
               
               verbose                      Option to set the output verbosity of the sampler. Available options: FIXME MISSING. Default: 2.
               nlive                        Option to set the number of live points (main feature controlling the accuracy of the sampling) to be used by the sampler. Default: 1024.
               maxmcmc                      Option to set the number of maximum Markov Chain Monte Carlo steps at each sample (main feature limiting the autocorrelation legth of the samples) to be used by the sampler. Default: 1024.

               seed                         Option to set the random seed to be used by the sampler. Default: 1234.
               resume                       Flag to allow the sampler to resume the run after it was killed. Default: 1.
               periodic_checkpoint_interval Option to checkpoint the sampler every `periodic_checkpoint_interval` seconds.  Default: 900.
               
               #=============================#
               # Branch-dependent parameters #
               #=============================#
               
               poolsize                     Option to set the number of pools (`cpnest` sampler only). Default: 128.
               nthreads                     Option to set the number of parallel threads to be used (`cpnest` sampler only). Default: 1.
               nnest                        Number of nested samplers to run in parallel (`raynest` sampler only). Default: 1.
               nensemble                    Total number of ensemble processes running (`raynest` sampler only). nensemble = nnest * N_ev, where N_ev is the number of live points being  \
    substituted at each NS step. Requires N_ev << nlive. Also n_cpu = nnest+nensemble. Default: 1.

       *******************************************************
       * Parameters to be passed to the [Injection] section. *
       *******************************************************

               inject-modes               Option to select a list of modes to be injected. Example of expected syntax: [(l1,m1), (l2,m2), ...]. Default: None.
               inject-l-modes             Option to select a list of l-modes to be injected. All |m|<l modes will be injected by default. Example of expected syntax: [l1, l2, ...]. Default: None.

               FIXME: MISSING description of injection parameters for the different waveforms.
               
       **************************************************
       * Parameters to be passed to the [Plot] section. *
       **************************************************
       
                imr-samples               File containing the posterior samples from an IMR run to compare against ringdown values in plots. Needs to be either a full posterior in the GWTC-1 format or the marginalised posterior on Mf-af, with two columns and the ['Mf', 'af'] header.

"""
                                                     
try:
    import art
    my_art = art.text2art("            Launching     pyRing") # Return ASCII text (default font)
except: print("* Warning: The `art` package could not be imported. Please consider installing it locally for best visual renditions. The cause of this not being taken care of automatically by the `pyRing` package is that the `art` package is not deployed on conda, hence the conda-build fails and `art` cannot be listed as a requirement for the `pyRing` package.")

__ascii_art__ = """\n\n \u001b[\u001b[38;5;39m
                                         @.
                                        &  @
                                        @  ,
                                        (
                                                       *
                                            &            @
                                       #    @        @
                                       @             .    ,
                                       *    .             @
                                                     @
                                                     ,    &
                                      (     #             @           @
                                      *     @                       @   @
                                      *     &       /
                                            .       @      #       @     @          *
*   @  %       *       @       &     @                     %                      @    &          *    @     &    @     @
                                                    *      *              @      @      @     @
                                             &                    @                        %
                                                                 .&        @   @
                                                   .        @                &
                                             @                   @
                                                   @
                                             *               @  @
                                                   .            &
                                                              %&
                                              *
                                              .
                                              @    @
                                              
                                               @  .
                                               /
                                                 @
\u001b[0m"""

max_len_keyword = len('gaussian-noise-white-sigma')