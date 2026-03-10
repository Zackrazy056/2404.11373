#! /usr/bin/env python
#Standard python imports
import matplotlib
matplotlib.use("pdf")
import corner, h5py, glob, matplotlib.patches as mpatches, matplotlib.pyplot as plt, numpy as np, os, sys, traceback, warnings
from matplotlib.ticker import FormatStrFormatter
from scipy.stats       import kstest, gaussian_kde
try:
    from scipy.signal import tukey
except ImportError:
    from scipy.signal.windows import tukey

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

#LVC imports
import lal
from lalinference                import DetFrameToEquatorial
from lalinference.imrtgr.nrutils import *

#Package internal imports
from pyRing.waveform         import *
from pyRing.utils            import bandpass_around_ringdown, compute_SNR_TD, compute_SNR_FD, import_datafile_path, qnm_interpolate, set_prefix, whiten_TD, whiten_FD, project_python_wrapper as project


def init_plotting():

    """
    
    Function to set the default plotting parameters.

    Parameters
    ----------
    None

    Returns
    -------
    Nothing, but sets the default plotting parameters.
    
    """
    
    plt.rcParams['figure.max_open_warning'] = 0
    
    plt.rcParams['figure.figsize']    = (5, 5)
    plt.rcParams['font.size']         = 10
    plt.rcParams['mathtext.fontset']  = 'stix'
    plt.rcParams['font.family']       = 'STIXGeneral'
    
    plt.rcParams['axes.linewidth']    = 1
    plt.rcParams['axes.labelsize']    = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize']    = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.major.size']  = 3
    plt.rcParams['xtick.minor.size']  = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size']  = 3
    plt.rcParams['ytick.minor.size']  = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    
    plt.rcParams['legend.frameon']             = False
    plt.rcParams['legend.loc']                 = 'center left'
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    
    return

def get_param_override(fixed_params,x,name):

    """
        Function returning either a sample or the fixed value for the parameter considered.
        ---------------
        
        Returns x[name], unless it is over-ridden by value in the fixed_params dictionary.

        Parameters
        ----------
        fixed_params : dict
            Dictionary of fixed parameters.
        x : dict
            Dictionary of samples.
        name : str
            Name of the parameter.

        Returns
        -------
        Either x[name] or fixed_params[name].
        
    """

    if name in fixed_params: return fixed_params[name]
    else:                    return x[name]

def FindHeightForLevel(inArr, adLevels):

    """
    
    Function to find the height of a contour line in a 2D array.

    Parameters
    ----------

    inArr : array
        2D array of values.
    adLevels : array
        Array of levels.

    Returns
    -------
    
    adHeights : array
        Array of heights.
    
    """

    # flatten the array
    oldshape = np.shape(inArr)
    adInput= np.reshape(inArr,oldshape[0]*oldshape[1])
    # GET ARRAY SPECIFICS
    nLength = np.size(adInput)
    # CREATE REVERSED SORTED LIST
    adTemp = -1.0 * adInput
    adSorted = np.sort(adTemp)
    adSorted = -1.0 * adSorted
    # CREATE NORMALISED CUMULATIVE DISTRIBUTION
    adCum = np.zeros(nLength)
    adCum[0] = adSorted[0]
    for i in range(1,nLength):
        adCum[i] = np.logaddexp(adCum[i-1], adSorted[i])
    adCum = adCum - adCum[-1]
    # FIND VALUE CLOSEST TO LEVELS
    adHeights = []
    for item in adLevels:
        idx=(np.abs(adCum-np.log(item))).argmin()
        adHeights.append(adSorted[idx])
    adHeights = np.array(adHeights)

    return np.sort(adHeights)

def plot_contour(samples_stacked, level=[0.9], linest = 'dotted', label= None, color='k', line_w=1.2, plot_legend=1, zorder=None):

    """

    Function to plot a contour line.

    Parameters
    ----------

    samples_stacked : array
        Array of samples.
    level : array
        Array of levels.
    linest : str
        Linestyle.
    label : str
        Label.
    color : str
        Color.
    line_w : float
        Line width.
    plot_legend : int
        Plot legend.
    zorder : int
        Zorder.
    
    Returns
    -------

    Nothing, but plots a contour line.    

    """

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    kde         = gaussian_kde(samples_stacked.T)
    x_flat      = np.r_[samples_stacked[:,0].min():samples_stacked[:,0].max():128j]
    y_flat      = np.r_[samples_stacked[:,1].min():samples_stacked[:,1].max():128j]
    X,Y         = np.meshgrid(x_flat,y_flat)
    grid_coords = np.append(X.reshape(-1,1),Y.reshape(-1,1),axis=1)
    pdf         = kde(grid_coords.T)
    pdf         = pdf.reshape(128,128)
    pdf[np.where(pdf==0.)] = 1.e-100

    hs  = []
    lgs = []
    for l in level:
        if zorder is not None: cntr = plt.contour(X,Y,np.log(pdf),levels = np.sort(FindHeightForLevel(np.log(pdf),[l])), colors=color, linewidths=line_w, linestyles=linest)
        else:                  cntr = plt.contour(X,Y,np.log(pdf),levels = np.sort(FindHeightForLevel(np.log(pdf),[l])), colors=color, linewidths=line_w, linestyles=linest, zorder=zorder)
        if(plot_legend):
            h,_ = cntr.legend_elements()
            hs.append(h[0])
            if not(label==None):  lgs.append(r'${0} - {1} \% \, CI$'.format(label,int(l*100.)))
            else:                 lgs.append(r'${0} \% \, CI$'.format(int(l*100.)))
    if(plot_legend): plt.legend([h_x for h_x in hs], [lg for lg in lgs], loc='upper left')

    warnings.filterwarnings('default', category=RuntimeWarning)
    
    return

def initialise_plot_strain(strain_only, whiten_method, dets, **kwargs):

    if((strain_only==1) and (whiten_method == 'TD')): raise Exception('TD whitening requires a start time, which is not provided in `strain_only=1` mode.')

    init_plotting()
    model_waveforms        = {d: [] for d in list(dets.keys())}
    dt_dict                = {d: [] for d in list(dets.keys())}
    timeseries_whitened_TD = {d: [] for d in list(dets.keys())}

    ref_det         = kwargs['ref-det']
    tevent          = kwargs['trigtime']
    srate           = kwargs['sampling-rate']
    sky_frame       = kwargs['sky-frame']
    dt              = 1./srate
    duration_n      = kwargs['analysis-duration-n']

    #If there is no injected Mf, get an estimate of final mass from the tau of a chif=0.7 BH, for plotting purposes only.
    if(not(kwargs['injection-approximant']=='Damped-sinusoids') and not(kwargs['injection-approximant']=='')):
        mf_time_prior = kwargs['injection-parameters']['Mf']
    elif(kwargs['injection-approximant']=='Damped-sinusoids'):
        mf_time_prior = ((kwargs['injection-parameters']['tau']['t'][0]*lal.C_SI**3)/lal.G_SI)/(lal.MSUN_SI*20.)
    else:
        mf_time_prior = kwargs['mf-time-prior']

    return model_waveforms, dt_dict, timeseries_whitened_TD, ref_det, tevent, sky_frame, dt, duration_n, mf_time_prior

def read_skypos(fixed_params, p, dets, sky_frame, ref_det, tevent, **kwargs):
            
    if (sky_frame == 'detector'):
        non_ref_det  = kwargs['nonref-det']
        cos_altitude = get_param_override(fixed_params,p,'cos_altitude')
        azimuth      = get_param_override(fixed_params,p,'azimuth')
        tg, ra, dec  = DetFrameToEquatorial(dets[ref_det].lal_detector, dets[non_ref_det].lal_detector, tevent, np.arccos(cos_altitude), azimuth)
    elif (sky_frame == 'equatorial'):
        ra  = get_param_override(fixed_params,p,'ra')
        dec = get_param_override(fixed_params,p,'dec')
    else:
        if (len(dets) > 1):
            non_ref_det  = kwargs['nonref-det']
            cos_altitude = get_param_override(fixed_params,p,'cos_altitude')
            azimuth      = get_param_override(fixed_params,p,'azimuth')
            tg, ra, dec  = DetFrameToEquatorial(dets[ref_det].lal_detector, dets[non_ref_det].lal_detector, tevent, np.arccos(cos_altitude), azimuth)
        else:
            ra  = get_param_override(fixed_params,p,'ra')
            dec = get_param_override(fixed_params,p,'dec')
    psi = get_param_override(fixed_params,p,'psi')

    return ra, dec, psi

def read_waveforms(params, fixed_params, get_waveform, dets, ref_det, tevent, sky_frame, whiten_method, whiten_flag, mf_time_prior, duration_n, tgps, dt, model_waveforms, dt_dict, timeseries_whitened_TD, spectrum, **kwargs):

    for p in params:
        if p is not None:
            
            if ('t' in fixed_params): t_start = fixed_params['t']
            else:                     t_start = p['t0']

            # Select sky position parameters
            ra, dec, psi = read_skypos(fixed_params, p, dets, sky_frame, ref_det, tevent, **kwargs)

            # Generate waveform
            waveform_polarisations = get_waveform(p)
            for d in list(dets.keys()):
                detector             = dets[d]
                time_delay           = lal.ArrivalTimeDiff(detector.location, lal.cached_detector_by_prefix[ref_det].location, ra, dec, tgps)
                time_array           = detector.time - (tevent+time_delay)
                hs, hvx, hvy, hp, hc = waveform_polarisations.waveform(time_array)
                waveform_proj        = project(hs, hvx, hvy, hp, hc, detector.lal_detector, ra, dec, psi, tgps)

                if whiten_flag:
                    waveform_proj = bandpass_around_ringdown(waveform_proj, dt, kwargs['f-min-bp'], mf_time_prior, alpha_window=0.1)
                    if(  whiten_method=='FD'):
                        waveform = whiten_FD(waveform_proj, detector.psd, dt, kwargs['f-min-bp'], kwargs['f-max-bp'])
                    elif(whiten_method=='TD'):

                        # In the case where we truncate and use the TD domain, the whitening depends on when the analysis starts, i.e. the samples. Hence, the whitened data will be different depending on the sample.
                        if(kwargs['truncate']==1):
                            waveform_proj = waveform_proj[time_array >= t_start][:duration_n]
                            
                            # This block seems out of place, but it's needed here since in the case of a variable start time we need to loop over tstart.
                            timeseries_tmp = bandpass_around_ringdown(detector.time_series, dt, kwargs['f-min-bp'], mf_time_prior, alpha_window=0.1)
                            timeseries_tmp = timeseries_tmp[time_array >= t_start][:duration_n]
                            timeseries_tmp = whiten_TD(timeseries_tmp, detector.cholesky)
                            timeseries_whitened_TD[d].append(timeseries_tmp)
                        
                        waveform = whiten_TD(waveform_proj, detector.cholesky)
                
                    else:
                        raise ValueError('Unknown whitening method requested.')
                else:
                    waveform = waveform_proj
                
                if(spectrum): waveform = np.absolute(np.fft.rfft(waveform*dt))

                model_waveforms[d].append(waveform)
                dt_dict[d].append(time_delay+t_start)
        else:
            waveform_polarisations = None

    return model_waveforms, dt_dict, timeseries_whitened_TD

def read_dataseries(d, whiten_flag, detector, tevent, dt, whiten_method, mf_time_prior, timeseries_whitened_TD, duration_n, dt_dict, spectrum, **kwargs):

    if whiten_flag:

        if(whiten_method=='FD'):
            timeseries_tmp = bandpass_around_ringdown(detector.time_series, dt, kwargs['f-min-bp'], mf_time_prior, alpha_window=0.1)
            timeseries     = whiten_FD(timeseries_tmp, detector.psd, dt, kwargs['f-min-bp'], kwargs['f-max-bp'])
            time_axis      = detector.time-tevent
        elif(whiten_method=='TD'):

            if(kwargs['truncate']==1):
                # In the case where the tstart is fixed, the median of the timeseries is simply the timeseries. 
                timeseries_regions = np.percentile(np.array(timeseries_whitened_TD[d]),[5,50,95], axis=0)
                timeseries         = timeseries_regions[1]
                dt_regions         = np.percentile(np.array(dt_dict[d]),[5,50,95], axis=0)
                time_axis          = (detector.time-tevent-dt_regions[1])
                time_axis          = time_axis[time_axis >= 0][:duration_n]
            else:
                timeseries = bandpass_around_ringdown(detector.time_series, dt, kwargs['f-min-bp'], mf_time_prior, alpha_window=0.1)
                timeseries = whiten_TD(timeseries, detector.cholesky)
                time_axis  = detector.time-tevent
        else:
            raise ValueError('Unknown whitening method requested.')

        wtleg      = 'whitened'
        label_y    = r'$\mathrm{s_{%s}(t)}$'%(d)
    
    else:
        timeseries = detector.time_series
        time_axis  = detector.time-tevent
        wtleg      = ''
        label_y    = r'$\mathrm{Strain}$'

    if(spectrum):
        data_axis  = np.fft.rfftfreq(len(timeseries), d=dt)
        dataseries = np.absolute(np.fft.rfft(timeseries*dt))
    else:
        data_axis  = time_axis
        dataseries = timeseries

    return data_axis, dataseries, wtleg, label_y

def plot_strain(get_waveform, dets, fixed_params, tgps, params = None, whiten_flag = False, whiten_method = 'TD', strain_only = 0, spectrum = 0, **kwargs):

    """

    Function to plot the strain.

    Parameters
    ----------

    get_waveform : function
        Function to get the waveform.
    dets : dict
        Dictionary of detectors.
    fixed_params : dict
        Dictionary of fixed parameters.
    tgps : float
        GPS time.
    params : dict
        Dictionary of parameters.
    whiten_flag : bool
        Whitening flag.
    whiten_method : str
        Whitening method.
    strain_only : int
        Strain only.
    kwargs : dict
        Dictionary of keyword arguments.

    Returns
    -------

    Nothing, but plots the strain.

    """


    ##############
    # Initialise #
    ##############

    model_waveforms, dt_dict, timeseries_whitened_TD, ref_det, tevent, sky_frame, dt, duration_n, mf_time_prior = initialise_plot_strain(strain_only, whiten_method, dets, **kwargs)


    ##################
    # Read waveforms #
    ##################

    if not(strain_only): model_waveforms, dt_dict, timeseries_whitened_TD = read_waveforms(params, fixed_params, get_waveform, dets, ref_det, tevent, sky_frame, whiten_method, whiten_flag, mf_time_prior, duration_n, tgps, dt, model_waveforms, dt_dict, timeseries_whitened_TD, spectrum, **kwargs)


    #########################
    # Data vs waveform plot #
    #########################
    
    if(spectrum):
        if(strain_only): filename = 'strain_fft'
        else:            filename = 'reconstructed_waveform_fft'
    else:
        if(strain_only): filename = 'strain'
        else:            filename = 'reconstructed_waveform'

    fig         = plt.figure()
    nsub_panels = len(dets)

    # Plot the data
    for i,d in enumerate(dets.keys()):
        detector = dets[d]
        ax       = fig.add_subplot(nsub_panels,1,i+1)
        plt.grid(False)

        # Read the timeseries
        data_axis, dataseries, wtleg, label_y = read_dataseries(d, whiten_flag, detector, tevent, dt, whiten_method, mf_time_prior, timeseries_whitened_TD, duration_n, dt_dict, spectrum, **kwargs)

        # Plot the frequency/time series
        if(spectrum): ax.loglog(data_axis, dataseries, label='{}'.format(d)+' ffted-strain '+wtleg, c='black', linestyle='-', lw=1.0)
        else:         ax.plot(  data_axis, dataseries, label='{}'.format(d)+' strain '+wtleg, c='black', linestyle='-', lw=1.0)

        # Axes
        if not(spectrum):
            if(whiten_flag):
                if (np.max(np.abs(dataseries)) < 5.): ax.set_ylim([-5,5])
                else:                                 ax.set_ylim([-10,10])
        ax.set_ylabel(label_y)
        if (d=='H1' and (nsub_panels > 1)):
            ax.get_xaxis().set_visible(False)
            ax.grid(False)
        
        # Plot waveform median and credible regions
        if not(strain_only):
            
            waveform_regions = np.percentile(np.array(model_waveforms[d]), [5,50,95], axis=0)
            dt_regions       = np.percentile(np.array(dt_dict[d])        , [5,50,95], axis=0)

            # Plot the waveform fft or timeseries
            if(spectrum): ax.loglog(data_axis, waveform_regions[1], label='model '+wtleg, color='gold', lw=0.8)
            else        : ax.plot(  data_axis, waveform_regions[1], label='model '+wtleg, color='gold', lw=0.8)
            ax.fill_between(data_axis, waveform_regions[0], waveform_regions[2], facecolor='gold', lw=0.5, alpha=0.4)

            # Axes
            if not(spectrum):
                try:
                    if(  whiten_method=='FD'):
                        ax.set_xlim([dt-150*mf_time_prior*lal.MTSUN_SI, dt+150*mf_time_prior*lal.MTSUN_SI])
                        ax.axvline(0.0, c='deeppink', linestyle='dashed', lw=1.0, label=r'$\mathrm{t_{start}}$')
                    elif(whiten_method=='TD'):
                        if not whiten_flag: ax.set_xlim([dt-150*mf_time_prior*lal.MTSUN_SI, dt+150*mf_time_prior*lal.MTSUN_SI])
                        else              : ax.set_xlim([dt-  0*mf_time_prior*lal.MTSUN_SI, dt+150*mf_time_prior*lal.MTSUN_SI])
                except (KeyError,configparser.NoOptionError, configparser.NoSectionError):
                    print("\nWarning: Failed to complete strain plot due to error: {}.".format(traceback.print_exc()))

        if not(spectrum):
            ax.set_xlabel('Time [s]')
        else:
            ax.set_xlim([kwargs['f-min-bp'], kwargs['f-max-bp']])
            ax.set_xlabel('Freq [Hz]')

        # Legend
        ax.legend(loc='upper left', prop={'size': 6})

    # Save the plot
    plt.subplots_adjust(wspace=0, hspace=0)
    if not(wtleg==''): filename = filename + '_' + wtleg + '_{}'.format(whiten_method)
    plt.savefig(os.path.join(kwargs['output'],'Plots/Strains', filename+'.pdf'), bbox_inches='tight')


    ##################
    # Residuals plot #
    ##################

    if(whiten_flag and not(strain_only)):

        # Initialise
        fig = plt.figure()
        for i,d in enumerate(dets.keys()):

            detector = dets[d]
            ax       = fig.add_subplot(nsub_panels,1,i+1)
            
            # Read the timeseries
            data_axis, dataseries, wtleg, label_y = read_dataseries(d, whiten_flag, detector, tevent, dt, whiten_method, mf_time_prior, timeseries_whitened_TD, duration_n, dt_dict, spectrum, **kwargs)

            # Axes
            ax.set_ylabel(label_y)
            if (d=='H1' and (nsub_panels > 1)):
                ax.get_xaxis().set_visible(False)
                ax.grid(False)
            
            # Compute residuals against waveform median and 90% region
            waveform_regions = np.percentile(np.array(model_waveforms[d]),[5,50,95], axis=0)
            residuals_lower  = dataseries-waveform_regions[0]
            residuals_median = dataseries-waveform_regions[1]
            residuals_upper  = dataseries-waveform_regions[2]

            # Plot residuals
            if(spectrum): ax.loglog(data_axis, residuals_median, label='whitened residuals', color='gold', lw=0.8)
            else        : ax.plot(  data_axis, residuals_median, label='whitened residuals', color='gold', lw=0.8)
            ax.fill_between(data_axis, residuals_lower, residuals_upper, facecolor='gold', lw=0.5, alpha=0.4)

            # Axes
            if not(spectrum):

                # Confidence bands
                ax.axhline( 1.0, c='black',   linestyle='dotted', lw=0.6, label=r'$\pm 1 \sigma$')
                ax.axhline(-1.0, c='black',   linestyle='dotted', lw=0.6)
                ax.axhline( 2.0, c='darkred', linestyle='dotted', lw=0.6, label=r'$\pm 2 \sigma$')
                ax.axhline(-2.0, c='darkred', linestyle='dotted', lw=0.6)

                try:
                    if(  whiten_method=='FD'):
                        ax.set_ylim([-5,5])
                        ax.set_xlim([dt-150*mf_time_prior*lal.MTSUN_SI, dt+80*mf_time_prior*lal.MTSUN_SI])
                        ax.axvline(0.0, c='deeppink', linestyle='dashed', lw=1.0, label=r'$\mathrm{t_{start}}$')
                    elif(whiten_method=='TD'):
                        ax.set_ylim([-3,3])
                        ax.set_xlim([0.0, dt+150*mf_time_prior*lal.MTSUN_SI])
                except (KeyError,configparser.NoOptionError, configparser.NoSectionError):
                    print("\nWarning: Failed to set axes for residuals plot due to error: {}.".format(traceback.print_exc()))
                ax.set_xlabel('Time [s]')
            else: 
                ax.set_xlim([kwargs['f-min-bp'], kwargs['f-max-bp']])
                ax.set_xlabel('Freq [Hz]')

            # Legend
            ax.legend(loc='upper left', prop={'size': 6})
        
        # Finalise and save plot
        plt.grid(False)
        plt.subplots_adjust(wspace=0, hspace=0)
        if(spectrum): filename = 'whitened_residuals' +'_{}'.format(whiten_method) + '_fft'
        else        : filename = 'whitened_residuals' +'_{}'.format(whiten_method)
        plt.savefig(os.path.join(kwargs['output'],'Plots/Strains', filename+'.pdf'), bbox_inches='tight')


        ######################
        # Gaussianity checks #
        ######################

        if not(spectrum):
            if(whiten_method=='TD'):

                # Plot whitened residuals against a normal distribution, to visually check gaussianity.
                
                # FIXME: this number should be experimented with.
                nbins        = 50
                normal_draws = np.random.normal(size=1000000)

                # Plot the data
                for i,d in enumerate(dets.keys()):
                    detector = dets[d]
                    fig      = plt.figure()

                    label_y = r'$\mathrm{s_{%s}(t)}$'%(d)
                    plt.ylabel(label_y)

                    for i in range(len(model_waveforms[d])):
                        plt.hist(timeseries_whitened_TD[d][i]-model_waveforms[d][i], histtype='step', bins=nbins, stacked=True, fill=False, density=True, color='gold', lw=0.5, alpha=0.4)

                    gaussian_x, bins_gauss, _ = plt.hist(normal_draws, label='Expected distribution', histtype='step', bins=nbins, stacked=True, fill=False, density=True, color='black', linewidth=2.0)
    # Work in progress
    #                sigma                     = [gaussian_x[i]*(1-gaussian_x[i]) for i in range(len(gaussian_x))]
    #                lower, upper              = gaussian_x - sigma, gaussian_x + sigma
    #                plt.plot(bins_gauss[:-1], lower, color='black', linewidth=1.7, linestyle='dashed')
    #                plt.plot(bins_gauss[:-1], upper, color='black', linewidth=1.7, linestyle='dashed')
                    
                    plt.legend(loc='best')
                    plt.xlabel('Whitened residuals')
                    plt.savefig(os.path.join(kwargs['output'],'Plots/Strains','Histrogram_whitened_residuals_{}_{}.pdf'.format(d, whiten_method)), bbox_inches='tight')

                    # Now let's compute a quantitative measure of gaussianity with zero mean and unit variance.
                    KS_p_values = []
                    p_value_threshold = 0.01
                    for i in range(len(model_waveforms[d])):
                        res_x = timeseries_whitened_TD[d][i]-model_waveforms[d][i]
                        KS_statistic, p_value = kstest(res_x, "norm")
                        KS_p_values.append(p_value)

                    KS_p_values   = np.array(KS_p_values)
                    mask_outliers = KS_p_values < p_value_threshold
                    N_outliers    = len(KS_p_values[mask_outliers])
                    len_tot       = len(KS_p_values)
                    n, bins, _    = plt.hist(KS_p_values)
                    bin_width     = bins[1] - bins[0]
                    integral      = bin_width * sum(n)

                    plt.figure()
                    plt.hist(KS_p_values, histtype='step', bins=nbins, stacked=True, fill=False, color='black', linewidth=2.0)
                    plt.axvline(p_value_threshold, label = 'Significance level: {}\nN outliers: {}/{}'.format(p_value_threshold, N_outliers, len_tot), color='darkred', linestyle='dashed', linewidth=1.5)
                    plt.xlabel('Kolmogorov–Smirnov p-values')
                    plt.title('Integral: {:.6f}'.format(integral))
                    plt.legend(loc='upper right')
                    plt.savefig(os.path.join(kwargs['output'],'Plots/Strains', 'Histogram_Kolmogorov_Smirnov_test_whitened_residuals_{}_{}.pdf'.format(d, whiten_method)), bbox_inches='tight')

                    plt.figure()
                    plt.scatter(np.arange(0,len(KS_p_values)), KS_p_values, color='black', marker='x')
                    plt.axhline(p_value_threshold, label = 'Significance level: {}\nN outliers: {}/{}'.format(p_value_threshold, N_outliers, len_tot), color='darkred', linestyle='dashed', linewidth=1.5)
                    plt.xlabel('Samples')
                    plt.ylabel('Kolmogorov–Smirnov p-values')
                    plt.ylim([0,2])
                    plt.legend(loc='upper right')
                    plt.savefig(os.path.join(kwargs['output'],'Plots/Strains', 'Scatter_Kolmogorov_Smirnov_test_whitened_residuals_{}_{}.pdf'.format(d, whiten_method)), bbox_inches='tight')

                    print('* A Kolmogorov–Smirnov test of whitened {} residuals gave {}/{} normality outliers (at {} % significance).\n'.format(d, N_outliers, len_tot, p_value_threshold*100))

    return

def SNR_plots(get_waveform, dets, fixed_params, tgps, params = None, **kwargs):

    """
    
    Plot the SNR of the network and of each detector.

    Parameters
    ----------

    get_waveform : function
        Function that returns the waveform model.
    dets : dict
        Dictionary of detectors.
    fixed_params : dict
        Dictionary of fixed parameters.
    tgps : float
        GPS time of the event.
    params : dict
        Dictionary of parameters sampled on.
    kwargs : dict
        Dictionary of keyword arguments.

    Returns
    -------

    Nothing, but creates the SNR plots.
    
    """

    # Initialise plotting.
    init_plotting()
    
    # Initialise structures.
    SNRs, network_SNRs, SNRs_loss = {}, {}, {}
    
    # Read auxiliary quantities.
    ref_det           = kwargs['ref-det']
    tevent            = kwargs['trigtime']
    srate             = kwargs['sampling-rate']
    sky_frame         = kwargs['sky-frame']
    seglen            = int(srate*kwargs['signal-chunksize'])
    dt                = 1./srate
    freqs             = np.fft.rfftfreq(seglen, d=dt)
    df                = freqs[1] - freqs[0]
    duration_n        = kwargs['analysis-duration-n']
    alpha_window      = kwargs['alpha-window']
    likelihood_method = kwargs['likelihood-method']
    
    # Do it only in TD, FD needs windowing etc.
    domain            = 'TD'
    
    # First, find maxL index to compute an estimate of the SNR loss due to truncation
    maxL = np.max(params['logL'])

    SNRs[domain]         = {}
    network_SNRs[domain] = {}
    
    # Loop over the type of SNR.
    print('Medians:\n')
    for SNR_type in ['matched_filter', 'optimal']:
        
        SNRs[domain][SNR_type]         = {d: []  for d in list(dets.keys())}
        network_SNRs[domain][SNR_type] = []
        SNRs_loss[SNR_type]            = {d: 0.0 for d in list(dets.keys())}

        # Loop over posterior samples.
        for p in params:

            # Read time and sky position parameters, required separately to compute truncated time axis.
            if ('t' in fixed_params): t_start = fixed_params['t']
            else:                     t_start = p['t0']
            if (sky_frame == 'detector'):
                non_ref_det  = kwargs['nonref-det']
                cos_altitude = get_param_override(fixed_params,p,'cos_altitude')
                azimuth      = get_param_override(fixed_params,p,'azimuth')
                tg, ra, dec  = DetFrameToEquatorial(dets[ref_det].lal_detector, dets[non_ref_det].lal_detector, tevent, np.arccos(cos_altitude), azimuth)
            elif (sky_frame == 'equatorial'):
                ra  = get_param_override(fixed_params,p,'ra')
                dec = get_param_override(fixed_params,p,'dec')
            else:
                if (len(dets) > 1):
                    non_ref_det = kwargs['nonref-det']
                    cos_altitude = get_param_override(fixed_params,p,'cos_altitude')
                    azimuth      = get_param_override(fixed_params,p,'azimuth')
                    tg, ra, dec = DetFrameToEquatorial(dets[ref_det].lal_detector, dets[non_ref_det].lal_detector, tevent, np.arccos(cos_altitude), azimuth)
                else:
                    ra  = get_param_override(fixed_params,p,'ra')
                    dec = get_param_override(fixed_params,p,'dec')
            psi = get_param_override(fixed_params,p,'psi')

            # Generate waveform polarisations.
            waveform_polarisations = get_waveform(p)

            # Loop onto detectors.
            for d in list(dets.keys()):
                
                # Compute detector time axis.
                detector      = dets[d]
                time_delay    = lal.ArrivalTimeDiff(detector.location, lal.cached_detector_by_prefix[ref_det].location, ra, dec, tgps)
                time_array    = detector.time - (tevent+time_delay)

                # Compute polarisations.
                data_TD = detector.time_series

                if(domain=='FD'):
                    window               = tukey(seglen,alpha_window)
                    windowNorm           = seglen/np.sum(window**2)
                    data                 = np.real(np.fft.rfft(data_TD*window*dt))*windowNorm
                
                    hs, hvx, hvy, hp, hc = waveform_polarisations.waveform(time_array)
                    waveform_TD_tmp      = project(hs, hvx, hvy, hp, hc, detector.lal_detector, ra, dec, psi, tgps)
                    waveform             = np.real(np.fft.rfft(waveform_TD_tmp*dt))
                
                elif(domain=='TD'):
                    if(kwargs['truncate']==1):
                        data                = data_TD[time_array >= t_start][:duration_n]
                        time_array_cropped  = time_array[time_array >= t_start][:duration_n]
                    else:
                        data             = data_TD
                    hs, hvx, hvy, hp, hc = waveform_polarisations.waveform(time_array_cropped)
                    waveform             = project(hs, hvx, hvy, hp, hc, detector.lal_detector, ra, dec, psi, tgps)

                # Compute inner product weights.
                if(  domain=='FD'):
                    weights_FD = detector.psd(freqs)
                    if(SNR_type=='optimal'):          SNR_sample = compute_SNR_FD(waveform, waveform, weights_FD, df)
                    elif(SNR_type=='matched_filter'): SNR_sample = compute_SNR_FD(data,     waveform, weights_FD, df)
                elif(domain=='TD'):
                    TD_method = likelihood_method
                    if(  TD_method=='direct-inversion'         ): weights_TD = detector.inverse_covariance
                    elif(TD_method=='cholesky-solve-triangular'): weights_TD = detector.cholesky
                    elif(TD_method=='toeplitz-inversion'       ): weights_TD = detector.acf
                    
                    if(SNR_type=='optimal'):          SNR_sample = compute_SNR_TD(waveform, waveform, weights_TD, method=TD_method)
                    elif(SNR_type=='matched_filter'): SNR_sample = compute_SNR_TD(data,     waveform, weights_TD, method=TD_method)

                # Store SNR sample.
                SNRs[domain][SNR_type][d].append(SNR_sample)

                # Compute the SNR loss due to truncation
                if(p['logL']==maxL):

                    # Extract the uncropped ACF
                    acf_file     = glob.glob(os.path.join(kwargs['output'],'Noise',f'ACF_{d}_*.txt'))[0]
                    _, ACF_full  = np.loadtxt(acf_file, unpack=True)

                    data_double_length     =    data_TD[time_array >= t_start][:int(2*duration_n)]
                    time_arr_double_length = time_array[time_array >= t_start][:int(2*duration_n)]
                    hs_dl, hvx_dl, hvy_dl, hp_dl, hc_dl = waveform_polarisations.waveform(time_arr_double_length)
                    waveform_dl            = project(hs_dl, hvx_dl, hvy_dl, hp_dl, hc_dl, detector.lal_detector, ra, dec, psi, tgps)
                    ACF_double_length      = ACF_full[:int(2*duration_n)]
                    SNR_double_length      = compute_SNR_TD(data_double_length, waveform_dl, ACF_double_length, method='toeplitz-inversion')
                    SNR_loss               = ((SNR_double_length - SNR_sample)/SNR_sample) * 100
                    SNRs_loss[SNR_type][d] = SNR_loss

            # Compute network SNR.
            network_SNRs[domain][SNR_type].append(np.sqrt(np.sum([SNRs[domain][SNR_type][d][-1]**2 for d in list(dets.keys())])))
        # Plot the results.
        SNR_header    = 'Network\t'
        SNR_filestack = (network_SNRs[domain][SNR_type],)

        for d in list(dets.keys()):
            SNR_header += '{}\t'.format(d)
            SNR_filestack  = SNR_filestack  + (SNRs[domain][SNR_type][d],)
            print('{} {} SNR ({}): {:.3f}'.format(d.ljust(len('Network')), SNR_type.ljust(len('matched_filter')), domain, np.median(SNRs[domain][SNR_type][d])))
            plt.figure()
            plt.hist(SNRs[domain][SNR_type][d], histtype='step', bins=70, stacked=True, fill=False, density=True, label = '{} SNR (TD)'.format(d))
            plt.savefig(os.path.join(kwargs['output'],'Plots', 'SNR/{}_{}_SNR_{}.pdf'.format(d, SNR_type, domain)), bbox_inches='tight')
        plt.figure()
        plt.hist(network_SNRs[domain][SNR_type], histtype='step', bins=70, stacked=True, fill=False, density=True, label = 'Network SNR ({})'.format(domain))
        plt.savefig(os.path.join(kwargs['output'],'Plots', 'SNR/{}_network_SNR_{}.pdf'.format(SNR_type, domain)), bbox_inches='tight')
        np.savetxt(os.path.join(kwargs['output'],'Nested_sampler/{}_SNR_{}.dat'.format(SNR_type, domain)), np.column_stack(SNR_filestack), header=SNR_header)

        print('Network {} SNR ({}): {:.3f}'.format(SNR_type.ljust(len('matched_filter')), domain, np.median(network_SNRs[domain][SNR_type])))

    print(f'\nPercentage SNR loss due to truncation (in maxL sample):\n')
    for SNR_type in ['matched_filter', 'optimal']:
        SNR_header    = ''
        SNR_filestack = ()
        for d in list(dets.keys()):
            SNR_header += '{}\t'.format(d)
            SNR_filestack  = SNR_filestack  + (SNRs_loss[SNR_type][d],)
            print('{} {}: {:.1f}'.format(d.ljust(len('Network')), SNR_type.ljust(len('matched_filter')), SNRs_loss[SNR_type][d])+'%')
        np.savetxt(os.path.join(kwargs['output'],'Nested_sampler/{}_trunc_loss_perc_SNR_maxL_{}.txt'.format(SNR_type, domain)), np.column_stack(SNR_filestack), header=SNR_header)

    return

def global_corner(x, names, output, truths=None):

    """
    
    Create a corner plot of all parameters.
    
    Parameters
    ----------

    x       : dictionary    
        Dictionary of parameters.
    names   : list
        List of parameter names.
    output  : string
        Output directory.

    Returns
    -------

    Nothing, but saves a corner plot to the output directory.

    """

    samples = []
    for xy in names: samples.append(np.array(x[xy]))
    samples = np.transpose(samples)
    mask    = [i for i in range(samples.shape[-1]) if not all(samples[:,i]==samples[0,i]) ]

    fig = plt.figure(figsize=(10,10))
    C   = corner.corner(samples[:,mask],
                        quantiles     = [0.05, 0.5, 0.95],
                        labels        = names,
                        color         = 'darkred',
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths = truths
                        )
    plt.savefig(os.path.join(output,'Plots','Parameters', 'corner.png'), bbox_inches='tight')

    return

def mode_corner(samples,filename=None,**kwargs):

    """

    Create a corner plot of the QNM parameters.

    Parameters
    ----------

    samples : array
        Array of samples.
    filename : string
        Name of the file to save the plot to.   
    kwargs : dictionary
        Dictionary of keyword arguments.

    Returns
    -------

    fig : figure
        Figure object.
    
    """
    
    fig = plt.figure(figsize=(10,10))
    C   = corner.corner(samples,**kwargs)
    if filename is not None: plt.savefig(filename,bbox_inches='tight')
    
    return fig

def read_QNM(fit_flag, l, m, n):

    qnm_interpolants              = {}

    if(fit_flag == 1):
        qnm                               = QNM_fit(l,m,n)
    else:
        interpolate_freq, interpolate_tau = qnm_interpolate(2,l,m,n)
        qnm_interpolants[(2,l,m,n)]       = {'freq': interpolate_freq, 'tau': interpolate_tau}
        qnm                               = QNM(2,l,m,n,qnm_interpolants)

    return qnm

def f0_t0_gr(samples_x, Mf_s, af_s, Num_mode, probs_flag=1, scatter_samples_flag=1, delta_QNM_plots=0, predictions_contour_flag=0, **kwargs):
    
    """

    Plot the posterior distributions of the QNM frequencies and damping times, and compare them to the GR prediction coming from an IMR analysis (if IMR posterior is available).

    Parameters
    ----------

    samples_x : dictionary
        Dictionary of samples.
    Mf_s : array
        Array of frequencies.
    af_s : array
        Array of damping times.
    Num_mode : int
        Number of modes.
    probs_flag : int
        Flag to plot the posterior probabilities of the QNM parameters.
    scatter_samples_flag : int
        Flag to plot the scatter of the samples.
    delta_QNM_plots : int
        Flag to plot the difference between the QNM parameters and the GR prediction.
    predictions_contour_flag : int
        Flag to plot the contours of the GR prediction.
    kwargs : dictionary
        Dictionary of keyword arguments.

    Returns
    -------

    Nothing, but saves a plot to the output directory.

    """

    plt.rcParams['mathtext.fontset']  = 'stix'
    plt.rcParams['font.family']       = 'STIXGeneral'
    output_dir = os.path.join(kwargs['output'], 'Plots')
    
    # Initialise structures.
    theoretical_values, labels    = [], []
    i, Num_modes                  = 0, 0
    n_values                      = [0,1]
    l_values                      = [2,3]
    for l in l_values: Num_modes += (2*l+1)

    fig = plt.figure()
    ax  = plt.axes()

    # Plot the posteriors from the tensorial agnostic reconstruction.
    pol = 't'
    for i in range(kwargs['n-ds-modes'][pol]):
        
        samples = np.column_stack((samples_x['f_{}_{}'.format(pol,i)],1e3*samples_x['tau_{}_{}'.format(pol,i)]))
        plot_contour(samples, level=[0.5, 0.9], linest = 'solid', color='k', line_w=0.8, plot_legend=0, zorder=1)
        if(scatter_samples_flag): plt.scatter(samples[:,0], samples[:,1], cmap='seismic',  marker='.', alpha=0.3, s=2)

    # Plot the GR prediction for frequency and damping time coming from an IMR run.
    for n in n_values:
        
        #IMPROVEME: Palettes for generic N modes, now it's set for 2 sets of overtones modes.
        if(n==0): palette = iter(matplotlib.cm.Spectral(np.linspace( 0, 1, Num_modes)))
        if(n==1): palette = iter(matplotlib.cm.rainbow_r(np.linspace(0, 1, Num_modes)))
        
        qnm_ref    = read_QNM(kwargs['qnm-fit'], 2, 2, 0)
        f0_ref_vec = [qnm_ref.f(M,a)       for M,a in zip(Mf_s,af_s)]
        t0_ref_vec = [1e3*qnm_ref.tau(M,a) for M,a in zip(Mf_s,af_s)]
        freq_ref   = np.median(f0_ref_vec)
        tau_ref    = np.median(t0_ref_vec)

        for l in l_values:
            for m in range(-l,l+1,1):
                
                col = next(palette)
                
                qnm = read_QNM(kwargs['qnm-fit'], l, m, n)
                
                f0 = [qnm.f(M,a)       for M,a in zip(Mf_s,af_s)]
                t0 = [1e3*qnm.tau(M,a) for M,a in zip(Mf_s,af_s)]
                theoretical_values.append([f0,t0])
                if(m<0): label_mode = r'$%d\bar{%d}%d$'%(l,-m,n)
                else   : label_mode = r'$%d%d%d$'%(      l, m,n)
                labels.append(label_mode)
                
                if(predictions_contour_flag): plot_contour(samples, level=[0.9], linest = 'solid', color=col, line_w=1.2, plot_legend=0)

                # Fine tune text positioning.
                shift_f   = 0.05
                shift_tau = 0.08
                if(l==3                   ): shift_tau += 0.08
                if(l==2 and m==-2 and n==0): shift_f   += 0.02

                # apply shift as multiplicative factor of the median value.
                shift_f   = shift_f   * np.median(freq_ref)
                shift_tau = shift_tau * np.median(tau_ref)

                # Invert label positioning between l=2 and l=3.
                if(l==2): shift_tau = -shift_tau
                font_text = {'color':  'black', 'weight': 'normal', 'size': 10}
                plt.plot(np.median(f0),         np.median(t0), color=col, marker='*', markersize=8,      zorder=2)
                plt.text(np.median(f0)-shift_f, np.median(t0)-shift_tau, label_mode, fontdict=font_text, zorder=3)

                i = i+1
                if(delta_QNM_plots):
                    difference_freq = np.array([(samples[j,0]-f0[k])/f0[k] for k in range(len(f0)) for j in range(0,samples.shape[0])])
                    difference_tau  = np.array([(samples[j,1]-t0[k])/t0[k] for k in range(len(t0)) for j in range(0,samples.shape[0])])
                    plt.figure()
                    plt.hist(difference_freq, histtype='step', bins=70, stacked=True, fill=False, normed=1)
                    plt.xlabel(r'$\mathrm{\delta f_{%d%d%d}}$'%(l,m,n))
                    plt.ylabel(r'$\mathrm{P(\delta f_{%d%d%d}|D)}$'%(l,m,n))
                    plt.savefig(os.path.join(output_dir,'dfreq_{}{}{}_{}.png'.format(l,m,n,Num_mode)) ,bbox_inches='tight')
                    plt.figure()
                    plt.hist(difference_tau, histtype='step', bins=70, stacked=True, fill=False, normed=1)
                    plt.xlabel(r'$\mathrm{\delta \tau_{%d%d%d}}$'%(l,m,n))
                    plt.ylabel(r'$\mathrm{P(\delta \tau_{%d%d%d}|D)}$'%(l,m,n))
                    plt.savefig(os.path.join(output_dir,'Parameters/dtau_{}{}{}_{}.png'.format(l,m,n,Num_mode)) ,bbox_inches='tight')

    # Compute the probability that each of the modes is the one corresponding to the samples.
    # To understand why the kde evaluation on the samples, combined with the mean of the probability, is the correct procedure to compute this number, see TeX/Likelihood.tex
    if(probs_flag):

        pol = 't'
        for i in range(kwargs['n-ds-modes'][pol]):
            samples = np.column_stack((samples_x['f_{}_{}'.format(pol,i)],1e3*samples_x['tau_{}_{}'.format(pol,i)]))
            theoretical_values = np.array(theoretical_values)
            kde                = gaussian_kde(samples.T)
            probs              = np.array([kde(t) for t in theoretical_values])

            mprob = np.array([np.mean(probs[i,:]) for i in range(len(labels))])
            mprob /= mprob.sum()
            patches_vec = []

            counter = 0
            n_values = [0]
            for n in n_values:

                if(n==0): palette = iter(matplotlib.cm.Spectral(np.linspace( 0, 1, Num_modes)))
                if(n==1): palette = iter(matplotlib.cm.rainbow_r(np.linspace(0, 1, Num_modes)))

                for l in l_values:
                    for m in range(-l,l+1,1):

                        col = next(palette)

                        if(m<0): label_mode = r'$p( %d\bar{%d}%d ) = $'%(l,-m,n)
                        else   : label_mode = r'$p( %d%d%d ) = $'%(      l, m,n)
                        patches_vec.append(mpatches.Patch(color=col, label = label_mode+'{:.2f}'.format(mprob[counter])))

                        legend_x = ax.legend(handles     = patches_vec,
                                    fancybox      = True,
                                    loc           = "upper right",
                                    borderaxespad = 0.,
                                    fontsize      = 8.5)
                       
                        counter += 1

    ax.set_xlabel(r'$\mathrm{f\,(Hz)}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{\tau\, (ms)}$', fontsize=16)
    plt.grid(False)
    plt.savefig(os.path.join(output_dir,'Parameters/frequency_tau_{}.pdf'.format(Num_mode)), bbox_inches='tight')

    return fig

def Kerr_intrinsic_corner(x, **input_par):

    """
    
    Create the corner plot of the intrinsic parameters of the final Kerr black hole.

    Parameters
    ----------

    x : dictionary
        Dictionary containing the samples of the intrinsic parameters of the final Kerr black hole.
    input_par : dictionary
        Dictionary containing the input parameters of the run.
    
    Returns
    -------

    Nothing, but saves the corner plot in the output directory.
    
    """

    # Corner plot of final mass and final spin
    pos             = np.column_stack((x['Mf'], x['af'], x['t0']))
    injected_values = None
    if (input_par['injection-approximant']=='Kerr'):
        injected_values = [input_par['injection-parameters']['Mf'], input_par['injection-parameters']['af'], input_par['injection-parameters']['t0']]
    mode_corner(  pos,
                  labels        = [r'$M_f (M_{\odot})$',
                                   r'$a_f$'            ,
                                   r'$t_{start}$'      ],
                  quantiles     = [0.05, 0.5, 0.95],
                  show_titles   = True,
                  title_kwargs  = {"fontsize": 12},
                  use_math_text = True,
                  truths        = injected_values,
                  filename      = os.path.join(input_par['output'],'Plots/Parameters/Kerr_intrinsic_corner.png'))
    
    return

def Kerr_intrinsic_alpha_corner(x, **input_par):

    """

    Create the corner plot of the intrinsic parameters of the final Kerr black hole when including the alpha area quantisatio parameter.

    Parameters
    ----------

    x : dictionary
        Dictionary containing the samples of the intrinsic parameters of the final Kerr black hole.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the corner plot in the output directory.

    """

    # Corner plot of final mass, final spin, and alpha.
    pos             = np.column_stack((x['Mf'], x['af'], x['alpha']))
    injected_values = None
    if (input_par['inject-area-quantization']):
        injected_values = [input_par['injection-parameters']['Mf'], input_par['injection-parameters']['af'], input_par['injection-parameters']['alpha']]

    mode_corner(  pos,
                labels        = [r'$M_f (M_{\odot})$',
                                 r'$a_f$'            ,
                                 r'$\alpha$'      ],
                quantiles     = [0.05, 0.5, 0.95],
                show_titles   = True,
                title_kwargs  = {"fontsize": 12},
                use_math_text = True,
                truths        = injected_values,
                filename      = os.path.join(input_par['output'],'Plots/Parameters/Kerr_intrinsic_alpha_corner.png'))
    
    return

def Kerr_intrinsic_braneworld_corner(x, **input_par):

    """

    Create the corner plot of the intrinsic parameters of the final Kerr black hole when including the braneworld parameters.

    Parameters
    ----------

    x : dictionary
        Dictionary containing the samples of the intrinsic parameters of the final Kerr black hole.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the corner plot in the output directory.

    """
    
    # Corner plot of final mass, final spin, and beta.
    pos             = np.column_stack((x['Mf'], x['af'], x['beta']))
    injected_values = None
    if (input_par['inject-braneworld']):
        injected_values = [input_par['injection-parameters']['Mf'], input_par['injection-parameters']['af'], input_par['injection-parameters']['beta']]
    
    mode_corner(  pos,
                labels        = [r'$M_f (M_{\odot})$',
                                 r'$a_f$'            ,
                                 r'$\beta$'      ],
                quantiles     = [0.05, 0.5, 0.95],
                show_titles   = True,
                title_kwargs  = {"fontsize": 12},
                use_math_text = True,
                truths        = injected_values,
                filename      = os.path.join(input_par['output'],'Plots/Parameters/Kerr_intrinsic_braneworld_corner.png'))
    
    return

def MMRDNS_intrinsic_corner(x, **input_par):

    """

    Create the corner plot of the intrinsic parameters for the MMRDNS model.

    Parameters
    ----------

    x : dictionary
        Dictionary containing the samples of the intrinsic parameters of the final Kerr black hole.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the corner plot in the output directory.

    """

    pos             = np.column_stack( (x['Mf'], x['af'], x['eta'], x['t0']) )
    injected_values = None

    if (input_par['injection-approximant']=='NR'):
        M    = input_par['injection-parameters']['M']
        q    = input_par['injection-parameters']['q']
        eta = q/((1+q)*(1+q))
        injected_values = [input_par['injection-parameters']['Mf'], input_par['injection-parameters']['af'], eta, 12*input_par['injection-parameters']['M']*lal.MTSUN_SI]
    elif (input_par['injection-approximant']=='Kerr'):
        injected_values = [input_par['injection-parameters']['Mf'], input_par['injection-parameters']['af'], None, 12*input_par['injection-parameters']['Mf']*lal.MTSUN_SI]
    elif(input_par['injection-approximant']=='MMRDNS'):
        injected_values = [input_par['injection-parameters']['Mf'], input_par['injection-parameters']['af'], input_par['injection-parameters']['eta'], input_par['injection-parameters']['t0']]
    mode_corner(pos,
                labels        = [r'$M_f (M_{\odot})$',
                                 r'$a_f$'            ,
                                 r'$\eta$'           ,
                                 r'$t_{start}$'      ],
                quantiles     = [0.05, 0.5, 0.95],
                show_titles   = True,
                title_kwargs  = {"fontsize": 12},
                use_math_text = True,
                truths        = injected_values,
                filename      = os.path.join(input_par['output'],'Plots/Parameters/MMRDNS_intrinsic_corner.png'))
    
    return

def MMRDNP_intrinsic_corner(x, **input_par):

    """

    Create the corner plot of the intrinsic parameters for the MMRDNP model.

    Parameters
    ----------

    x : dictionary
        Dictionary containing the samples of the amplitude parameters for the MMNRDNS model.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the corner plot in the output directory.

    """

    pos             = np.column_stack((x['m1'], x['m2'], x['chi1'], x['chi2']))
    injected_values = None

    if((input_par['injection-approximant']=='NR') or (input_par['injection-approximant']=='MMRDNP') or (input_par['injection-approximant']=='TEOBResumSPM') or ('LAL' in input_par['injection-approximant'])):

        injected_values = [input_par['injection-parameters']['m1'],
                           input_par['injection-parameters']['m2'],
                           input_par['injection-parameters']['chi1'],
                           input_par['injection-parameters']['chi2']]

    mode_corner(pos,
                labels        = [r'$m1 (M_{\odot})$',
                                 r'$m2 (M_{\odot})$',
                                 r'$\chi_1$'        ,
                                 r'$\chi_2$'        ],
                quantiles     = [0.05, 0.5, 0.95],
                show_titles   = True,
                title_kwargs  = {"fontsize": 12},
                use_math_text = True,
                truths        = injected_values,
                filename      = os.path.join(input_par['output'],'Plots/Parameters/MMRDNP_intrinsic_corner.png'))
    
    return

def MMRDNP_amplitude_parameters_corner(x, **input_par):

    """

    Create the corner plot of the amplitude parameters for the MMRDNP model.

    Parameters
    ----------

    x : dictionary
        Dictionary containing the samples of the amplitude parameters for the MMNRDNP model.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the corner plot in the output directory.

    """

    m1   = x['m1']
    m2   = x['m2']
    chi1 = x['chi1']
    chi2 = x['chi2']
    M    = m1 + m2
    q    = m1/m2
    chis = (m1*chi1 + m2*chi2)/M
    chia = (m1*chi1 - m2*chi2)/M
    eta  = q/((1+q)*(1+q))

    pos = np.column_stack((eta, chis, chia))
    injected_values = None

    if((input_par['injection-approximant']=='NR') or (input_par['injection-approximant']=='TEOBResumSPM') or ('LAL' in input_par['injection-approximant'])):

        M_inj    = input_par['injection-parameters']['M']
        q_inj    = input_par['injection-parameters']['q']
        m1_inj   = input_par['injection-parameters']['m1']
        m2_inj   = input_par['injection-parameters']['m2']
        if not(input_par['injection-approximant']=='TEOBResumSPM'):
            chi1_inj = input_par['injection-parameters']['s1z_LALSim']
            chi2_inj = input_par['injection-parameters']['s2z_LALSim']
        else:
            chi1_inj = input_par['injection-parameters']['chi1']
            chi2_inj = input_par['injection-parameters']['chi2']
        chis_inj = (m1_inj*chi1_inj + m2_inj*chi2_inj)/M_inj
        chia_inj = (m1_inj*chi1_inj - m2_inj*chi2_inj)/M_inj
        eta_inj  = q_inj/((1+q_inj)*(1+q_inj))

        injected_values = [eta_inj, chis_inj, chia_inj]

    elif(input_par['injection-approximant']=='MMRDNP'):

        injected_values = [input_par['injection-parameters']['eta'],
                           input_par['injection-parameters']['chis'],
                           input_par['injection-parameters']['chia']]

    mode_corner(pos,
                labels        = [r'$\eta$' ,
                                r'$\chi_s$',
                                r'$\chi_a$'],
                quantiles     = [0.05, 0.5, 0.95],
                show_titles   = True,
                title_kwargs  = {"fontsize": 12},
                use_math_text = True,
                truths        = injected_values,
                filename      = os.path.join(input_par['output'],'Plots/Parameters/MMRDNP_amplitude_parameters_corner.png'))
    
    return

def MMRDNP_Mf_af_plot(x, **input_par):

    """

    Create the plot of the final mass and final spin for the MMRDNP model.

    Parameters
    ----------

    x : dictionary
        Dictionary containing the samples of the amplitude parameters for the MMNRDNP model.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the plot in the output directory.

    """

    Mf = []
    af = []
    for par in x:
        if(par['chi1'] < 0): tilt1_fit = np.pi
        else: tilt1_fit = 0.0
        if(par['chi2'] < 0): tilt2_fit = np.pi
        else: tilt2_fit = 0.0
        chi1_fit  = np.abs(par['chi1'])
        chi2_fit  = np.abs(par['chi2'])
        Mf_x = bbh_final_mass_projected_spins(par['m1'], par['m2'], chi1_fit, chi2_fit, tilt1_fit, tilt2_fit, 'UIB2016')
        af_x = bbh_final_spin_projected_spins(par['m1'], par['m2'], chi1_fit, chi2_fit, tilt1_fit, tilt2_fit, 'UIB2016', truncate = bbh_Kerr_trunc_opts.trunc)
        Mf.append(Mf_x)
        af.append(af_x)

    pos             = np.column_stack((Mf, af))
    injected_values = None
    if (not((input_par['injection-approximant']=='Damped-sinusoids') or (input_par['injection-approximant']=='Morlet-Gabor-wavelets')) and not(input_par['injection-approximant']=='')):
        injected_values = [input_par['injection-parameters']['Mf'], input_par['injection-parameters']['af']]

    mode_corner(pos,
                labels        = [r'$M_f (M_{\odot})$',
                                 r'$a_f$'            ],
                quantiles     = [0.05, 0.5, 0.95],
                show_titles   = True,
                title_kwargs  = {"fontsize": 12},
                use_math_text = True,
                truths        = injected_values,
                filename      = os.path.join(input_par['output'],'Plots/Parameters/MMRDNP_Mf_af_plot.png'))

    return

def orientation_corner(x, Config, **input_par):

    """

    Create the corner plot of the orientation parameters.

    Parameters
    ----------

    x : dictionary
        Dictionary containing the samples of the orientation parameters.
    Config : configparser object
        Configparser object containing the configuration of the run.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the plot in the output directory.

    """
    
    pos             = np.column_stack((np.arccos(x['cosiota']), np.exp(x['logdistance'])))
    if not(input_par['injection-approximant']=='Damped-sinusoids') and not(input_par['injection-approximant']==''):
        expected_values_orientation = [np.arccos(input_par['injection-parameters']['cosiota']), np.exp(input_par['injection-parameters']['logdistance'])]
    else:
        try:
            file = str(Config.get("Plot",'imr-samples'))
            expected_values_orientation = None
            if('GWTC' in file):
                BBH             = h5py.File(file, 'r')['IMRPhenomPv2_posterior']
                IMR_distance    = np.median(BBH['luminosity_distance_Mpc'])
                IMR_inclination = np.median(np.arccos(BBH['costheta_jn']))
                expected_values_orientation = [IMR_inclination, IMR_distance]
                sys.stdout.write(('Using %s to get orientation IMR samples.\n'%(file)))
        except(KeyError,configparser.NoOptionError, configparser.NoSectionError):
            expected_values_orientation = None
    mode_corner(pos,
                labels        = [r'$\iota (rad)$',
                                r'$D(Mpc)$'     ],
                quantiles     = [0.05, 0.5, 0.95],
                show_titles   = True,
                title_kwargs  = {"fontsize": 12},
                use_math_text = True,
                truths        = expected_values_orientation,
                filename      = os.path.join(input_par['output'],'Plots/Parameters/Orientation_corner.png'))
    
    return

#IMPROVEME: generalize this plot for different amplitudes in case of injections
def amplitudes_corner(x, **input_par):

    """

    Create the corner plot of the amplitude parameters.

    Parameters
    ----------

    x : dictionary  
        Dictionary containing the samples of the amplitude parameters.
    input_par : dictionary
        Dictionary containing the input parameters of the run.
    
    Returns
    -------

    Nothing, but saves the plot in the output directory.

    """
    
    params = (np.arccos(x['cosiota']), np.exp(x['logdistance']))
    injected_values = None
    if (input_par['injection-approximant']=='Kerr'):
        injected_values = [np.arccos(input_par['injection-parameters']['cosiota']), np.exp(input_par['injection-parameters']['logdistance'])]
    label_default = [r'$\iota (rad)$', r'$D(Mpc)$']
    for mode in input_par['kerr-modes']:
        s,l,m,n = mode
        if(input_par['amp-non-prec-sym']):
            params = params + (x['A{}{}{}{}'.format(s,l,m,n)],)
            label_default.append('$|A_{%s%d%d%d}|$'%(s,l,m,n))
        else:
            params = params + (x['A{}{}{}{}_1'.format(s,l,m,n)],)+(x['A{}{}{}{}_2'.format(s,l,m,n)],)
            label_default.append('$|A^1_{%s%d%d%d}|$'%(s,l,m,n))
            label_default.append('$|A^2_{%s%d%d%d}|$'%(s,l,m,n))
        if(not(input_par['injection-approximant']=='Damped-sinusoids') and not(input_par['injection-approximant']=='')):
            injected_values.append(np.abs(np.real(input_par['injection-scaling']*(input_par['injection-parameters']['kerr-amplitudes'][(l,m,n)]))))
        pos = np.column_stack(params)
        mode_corner( pos,
                     labels        = label_default,
                     quantiles     = [0.05, 0.5, 0.95],
                     show_titles   = True,
                     title_kwargs  = {"fontsize": 12},
                     use_math_text = True,
                     truths        = injected_values,
                     filename      = os.path.join(input_par['output'], 'Plots/Parameters/Amplitudes_corner.png'))
    
    return

def f_tau_amp_corner(x, **input_par):

    """

    Create the corner plot of the frequency, damping time and amplitude parameters.

    Parameters
    ----------

    x : dictionary
        Dictionary containing the samples of the frequency, damping time and amplitude parameters.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the plot in the output directory.

    """
    
    for pol in input_par['n-ds-modes'].keys():
        for i in range(input_par['n-ds-modes'][pol]):
            pos             = np.column_stack((x['logA_{}_{}'.format(pol,i)],x['f_{}_{}'.format(pol,i)],1e3*x['tau_{}_{}'.format(pol,i)]))
            injected_values = None
            if((i==0) and not(input_par['injection-approximant']=='Damped-sinusoids') and not(input_par['injection-approximant']=='')):
                # Assume that the strongest one is the 220, to predict the other modes, we have the spectroscpy plot.
                injected_values = [None, QNM_fit(2,2,0).f(input_par['injection-parameters']['Mf'],input_par['injection-parameters']['af']), QNM_fit(2,2,0).tau(input_par['injection-parameters']['Mf'],input_par['injection-parameters']['af'])*1e3]
            elif(input_par['injection-approximant']=='Damped-sinusoids'):
                injected_values = [np.log10(input_par['injection-parameters']['A'][pol][i]*input_par['injection-scaling']), input_par['injection-parameters']['f'][pol][i], 1e3*(input_par['injection-parameters']['tau'][pol][i])]
            mode_corner(pos,
                        labels        = [r'$logA_{0}$'.format(i)     ,
                                         r'$f_{0}\,(Hz)$'.format(i)  ,
                                         r'$\tau_{0} (ms)$'.format(i)],
                        quantiles     = [0.05, 0.5, 0.95],
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths        = injected_values,
                        filename      = os.path.join(input_par['output'],'Plots/Parameters/Mode_{}_corner.png'.format(i)))
    return

def tick_function(X): 
    
    """
    
    Function to create the ticks for the time plot.

    Parameters
    ----------

    X : array
        FIXME: Ignored.
    
    Returns
    -------

    Array containing the values of the ticks.
    
    """
    
    ticks = ['10', '11','12', '13', '14', '15', '16', '17', '18', '19', '20']

    return ticks

def t_start_plot(t0_ds, Mf, **input_par):

    """

    Create the plot of the start time of the template.

    Parameters
    ----------

    t0_ds : array
        Array containing the samples of the start time of the template.
    Mf : float
        Final mass of the system.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the plot in the output directory.

    """
    
    init_plotting()
    Mf                 = Mf * lal.MTSUN_SI
    new_tick_locations = np.array([i*Mf for i in range(10,21)])
    output_dir = os.path.join(input_par['output'], 'Plots')

    if(not(input_par['injection-approximant']=='Damped-sinusoids') and not(input_par['injection-approximant']=='')):
        t0_inj = input_par['injection-parameters']['t0']*1e3
        label_t0_inj = 'Injected value'
    else:
        t0_inj = None
        label_t0_inj = None

    plt.figure(figsize=(6,5))
    ax = plt.subplot(1,1,1)
    ax.hist(t0_ds*1e3, bins=70, histtype='step', color = 'firebrick', lw=1.7)
    plt.axvline(np.mean(t0_ds*1e3), label=r'$\mathrm{Median}$', ls='dotted', c='k', lw=0.9)
    plt.axvline(np.percentile(t0_ds*1e3, 5), label=r'$90\% \mathrm{CI}$', ls='dotted', c='k', lw=0.9)
    plt.axvline(np.percentile(t0_ds*1e3, 95), ls='dotted', c='k', lw=0.9)
    if t0_inj:
        plt.axvline(t0_inj, ls='solid', c='royalblue', lw=0.9, label = label_t0_inj)
    ax.set_xlabel(r'$\mathrm{t_{start} - t_{peak} \, [ms]}$')
    ax.set_ylabel(r'$\mathrm{P(t_{start}|D_{ring})}$')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(False)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r'$t_{start} - t_{peak}/M_f$', fontsize=10)
    ax2.grid(False)
    plt.xlim(new_tick_locations.min(),new_tick_locations.max())
    plt.grid(False)
    plt.legend()
    plt.savefig(os.path.join(output_dir,'t_start.pdf') ,bbox_inches='tight')

    return

def TEOBPM_intrinsic_corner(x, **input_par):

    """

    Create the corner plot of the intrinsic parameters of the TEOBPM model.

    Parameters
    ----------

    x : array
        Array containing the samples of the intrinsic parameters of the TEOBPM model.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the plot in the output directory.

    """

    init_plotting()
    pos             = np.column_stack( (x['m1'], x['m2'], x['chi1'], x['chi2'], x['t0']) )
    injected_values = None
    if (input_par['injection-approximant']=='NR'):
        m1   = input_par['injection-parameters']['m1']
        m2   = input_par['injection-parameters']['m2']
        chi1 = input_par['injection-parameters']['s1z']
        chi2 = input_par['injection-parameters']['s2z']
        eta  = q/((1+q)*(1+q))
        injected_values = [m1,  m2, chi1, chi2, None]

    mode_corner(  pos,
                  labels        = [r'$m_1 (M_{\odot})$',
                                   r'$m_2 (M_{\odot})$',
                                   r'$\chi_1$'         ,
                                   r'$\chi_2$'         ,
                                   r'$t_{start}$'      ],
                  quantiles     = [0.05, 0.5, 0.95],
                  show_titles   = True,
                  title_kwargs  = {"fontsize": 12},
                  use_math_text = True,
                  truths        = injected_values,
                  filename      = os.path.join(input_par['output'],'Plots/Parameters/TEOBPM_intrinsic_corner.png'))

    return

def TEOBPM_masses_spins_corner(x, **input_par):

    """

    Create the corner plot of the progenitors masses and spins of the TEOBPM model.

    Parameters
    ----------

    x : array
        Array containing the samples of the intrinsic parameters of the TEOBPM model.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the plot in the output directory.

    """
    
    pos        = np.column_stack( (x['m1'], x['m2'], x['chi1'], x['chi2'] ) )
    output_dir = os.path.join(input_par['output'], 'Plots')

    mode_corner(  pos,
                  labels        = [r'$m_1 (M_{\odot})$',
                                   r'$m_2 (M_{\odot})$',
                                   r'$\chi_1$'         ,
                                   r'$\chi_2$'         ],
                  quantiles     = [0.05, 0.5, 0.95],
                  show_titles   = True,
                  title_kwargs  = {"fontsize": 12},
                  use_math_text = True,
                  truths        = None,
                  filename      = os.path.join(input_par['output'],'Plots/Parameters/corner_masses_spins.png'))

    return

def Kerr_Newman_intrinsic_corner(x, **input_par):

    """

    Create the corner plot of the intrinsic parameters of the Kerr-Newman model.

    Parameters  
    ----------

    x : array
        Array containing the samples of the intrinsic parameters of the Kerr-Newman model.
    input_par : dictionary
        Dictionary containing the input parameters of the run.
    
    Returns
    -------

    Nothing, but saves the plot in the output directory.

    """

    pos             = np.column_stack( (x['Mf'], x['af'], x['Q']) )
    injected_values = None
    mode_corner(  pos,
                  labels        = [r'$M_f (M_{\odot})$',
                                   r'$a_f$'            ,
                                   r'$Q$'              ],
                  quantiles     = [0.05, 0.5, 0.95],
                  show_titles   = True,
                  title_kwargs  = {"fontsize": 12},
                  use_math_text = True,
                  truths        = injected_values,
                  filename      = os.path.join(input_par['output'],'Plots/Parameters/Kerr_Newman_intrinsic_corner.png'))

    return

def Mf_af_plot(samples, Mf_LAL_samples, af_LAL_samples, **input_par):

    """

    Create the plot of the final mass and spin and compare them to IMR samples.

    Parameters
    ----------

    samples : array
        Array containing the samples of final mass and spin.
    Mf_LAL_samples : array
        Array containing the samples of the final mass of the IMR model.
    af_LAL_samples : array
        Array containing the samples of the final spin of the IMR model.
    input_par : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the plot in the output directory.

    """

    init_plotting()
    Mf                  = samples['Mf']
    af                  = samples['af']
    logL                = samples['logL']
    if (not(input_par['injection-approximant']=='Damped-sinusoids') and not(input_par['injection-approximant']=='')):
        Mf_inj = input_par['injection-parameters']['Mf']
        af_inj = input_par['injection-parameters']['af']
    samples_stacked = np.column_stack((Mf, af))
    output_dir = os.path.join(input_par['output'], 'Plots')
    plt.figure()
    plt.scatter(Mf, af, c=logL, cmap='Reds',  marker='.', alpha=1.0, label = r'$Ringdown$')
    plot_contour(samples_stacked, [0.95, 0.5], zorder=-1)
    if((Mf_LAL_samples is not None) and (af_LAL_samples is not None)):
        samples_stacked_LAL = np.column_stack((Mf_LAL_samples, af_LAL_samples))
        plot_contour(samples_stacked_LAL, [0.95], linest = 'solid', label= 'IMR (LVC)')
    if (not(input_par['injection-approximant']=='Damped-sinusoids') and not(input_par['injection-approximant']=='')):
        plt.axvline(Mf_inj, ls='dashed', c='k', label='Injected values')
        plt.axhline(af_inj, ls='dashed', c='k')
    plt.ylim([0,1])
    plt.grid(alpha=0.2,linestyle='dotted', color='k')
    plt.xlabel(r'$\mathrm{M_f(M_{\odot})}$')
    plt.ylabel(r'$\mathrm{a_f}$')
    plt.savefig(os.path.join(output_dir,'Parameters/Mf_af.pdf') ,bbox_inches='tight')

    return

def omega_tau_eff_plot(x, **kwargs):

    """

    Create the plot of the effective frequency and damping time.

    Parameters
    ----------

    x : array
        Array containing the samples of the intrinsic parameters of the TEOBPM model.
    kwargs : dictionary
        Dictionary containing the input parameters of the run.

    Returns
    -------

    Nothing, but saves the plot in the output directory.

    """
    
    Mf         = x['Mf']
    af         = x['af']
    output_dir = os.path.join(kwargs['output'], 'Plots')
    if ((kwargs['domega-tgr-modes'] is not None) and (kwargs['dtau-tgr-modes'] is None)):
        for mode in kwargs['domega-tgr-modes']:
            (l,m,n)   = mode
            domega    = x['domega_{0}{1}{2}'.format(l, m, n)]
            omega_eff = []
            for i in range(len(Mf)):
                omega_eff.append(QNM_fit(l,m,n).f(Mf[i], af[i])*(1.0+domega[i]))
            pos        = np.column_stack((Mf, af, omega_eff))
            mode_corner(pos,
                        labels        = [r'$M_f \,(M_{\odot})$',
                                         r'$a_f$',
                                         r'$\omega_{%d%d%d} [\!eff] \,(Hz)$'%(l,m,n)],
                        quantiles     = [0.05, 0.5, 0.95],
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths        = None,
                        filename      = os.path.join(kwargs['output'],'Plots/omega_eff_{0}{1}{2}_corner.pdf'.format(l,m,n)))
            pos2        = np.column_stack((Mf, af, domega))
            mode_corner(pos2,
                        labels        = [r'$M_f \,(M_{\odot})$',
                                         r'$a_f$',
                                         r'$\delta \omega_{%d%d%d} \, (Hz)$'%(l,m,n)],
                        quantiles     = [0.05, 0.5, 0.95],
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths        = None,
                        filename      = os.path.join(kwargs['output'],'Plots/domega_corner.pdf'))

    elif ((kwargs['dtau-tgr-modes'] is not None) and (kwargs['domega-tgr-modes'] is None)):
        for mode in kwargs['dtau-tgr-modes']:
            (l,m,n)    = mode
            dtau       = x['dtau_{0}{1}{2}'.format(l, m, n)]
            tau_eff    = []
            for i in range(len(Mf)):
                tau_eff.append(QNM_fit(l,m,n).tau(Mf[i], af[i])*1e3*(1.0+dtau[i]) )
            pos        = np.column_stack((Mf, af, tau_eff))
            output_dir = os.path.join(kwargs['output'], 'Plots')
            mode_corner(pos,
                        labels        = [r'$M_f \,(M_{\odot})$',
                                         r'$a_f$',
                                         r'$\tau_{%d%d%d} [\!eff] (ms)$'%(l,m,n)],
                        quantiles     = [0.05, 0.5, 0.95],
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths        = None,
                        filename      = os.path.join(kwargs['output'],'Plots/tau_eff_{0}{1}{2}_corner.pdf'.format(l,m,n)))
            pos2        = np.column_stack((Mf, af, dtau))
            mode_corner(pos2,
                        labels        = [r'$M_f \,(M_{\odot})$',
                                         r'$a_f$',
                                         r'$\delta \tau_{%d%d%d} \, (Hz)$'%(l,m,n)],
                        quantiles     = [0.05, 0.5, 0.95],
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths        = None,
                        filename      = os.path.join(kwargs['output'],'Plots/dtau_corner.pdf'))
    elif ((kwargs['dtau-tgr-modes'] is not None) and (kwargs['domega-tgr-modes'] is not None)):
        for mode in kwargs['dtau-tgr-modes']:
            (l,m,n)   = mode
            domega    = x['domega_{0}{1}{2}'.format(l, m, n)]
            dtau      = x['dtau_{0}{1}{2}'.format(l, m, n)]
            omega_eff = []
            tau_eff   = []
            for i in range(len(Mf)):
                omega_eff.append(QNM_fit(l,m,n).f(Mf[i], af[i])*(1.0+domega[i]))
                tau_eff.append(QNM_fit(l,m,n).tau(Mf[i], af[i])*1e3*(1.0+dtau[i]) )
            pos        = np.column_stack((Mf, af, omega_eff, tau_eff))
            mode_corner(pos,
                        labels        = [r'$M_f \,(M_{\odot})$',
                                         r'$a_f$',
                                         r'$\omega_{%d%d%d} [\!eff] (Hz)$'%(l,m,n),
                                         r'$\tau_{%d%d%d} [\!eff] (ms)$'%(l,m,n)],
                        quantiles     = [0.05, 0.5, 0.95],
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths        = None,
                        filename      = os.path.join(kwargs['output'],'Plots/omega_tau_eff_{0}{1}{2}_corner.pdf'.format(l,m,n)))
            pos2        = np.column_stack((Mf, af, domega))
            mode_corner(pos2,
                        labels        = [r'$M_f \,(M_{\odot})$',
                                         r'$a_f$',
                                         r'$\delta \omega_{%d%d%d} \, (Hz)$'%(l,m,n)],
                        quantiles     = [0.05, 0.5, 0.95],
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths        = None,
                        filename      = os.path.join(kwargs['output'],'Plots/domega_corner.pdf'))
            pos3        = np.column_stack((Mf, af, dtau))
            mode_corner(pos3,
                        labels        = [r'$M_f \,(M_{\odot})$',
                                         r'$a_f$',
                                         r'$\delta \tau_{%d%d%d} \, (Hz)$'%(l,m,n)],
                        quantiles     = [0.05, 0.5, 0.95],
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths        = None,
                        filename      = os.path.join(kwargs['output'],'Plots/dtau_corner.pdf'))
    else:
        raise Exception("Invalid plotting option in omega-tau effective plot.")

    return

def plot_NR_single_mode(t_geom, hr_geom, hi_geom, **kwargs):

    """
    
    Plot the NR waveform in the time domain for a single mode.

    Parameters
    ----------

    t_geom: array
        Time array in geometric units.
    hr_geom: array
        Real part of the waveform in geometric units.
    hi_geom: array
        Imaginary part of the waveform in geometric units.
    kwargs: dict  
        Dictionary of keyword arguments.

    Returns
    -------

    Nothing, but saves the waveform plot to the output directory.
    
    """

    init_plotting()
    output_dir = os.path.join(kwargs['output'], 'Plots')

    dt_geom                  = np.min(np.diff(t_geom)) # the sampling is NOT uniform
    Amp_geom                 = np.sqrt(hr_geom**2+hi_geom**2)
    Phi_geom                 = np.unwrap(np.angle(hr_geom - 1j*hi_geom))

    t_geom_uniform           = np.arange(t_geom[0], t_geom[-1], dt_geom)
    hr_geom_interp           = np.interp(t_geom_uniform, t_geom, hr_geom)
    hi_geom_interp           = np.interp(t_geom_uniform, t_geom, hi_geom)
    Amp_geom_interp          = np.interp(t_geom_uniform, t_geom, Amp_geom)
    Phi_geom_interp          = np.interp(t_geom_uniform, t_geom, Phi_geom)
    omega_geom_interp        = np.gradient(Phi_geom_interp, dt_geom)
    t_peak_geom_uniform      = t_geom_uniform[np.argmax(Amp_geom_interp)]

    l,m         = kwargs['injection-parameters']['fix-NR-mode'][0]
    M_inj_sec   = kwargs['injection-parameters']['M']*lal.MTSUN_SI
    t_phys      = t_geom_uniform*M_inj_sec
    freq_phys   = (omega_geom_interp/(2.*np.pi)) * (M_inj_sec)**(-1)
    t_peak_phys = t_phys[np.argmax(Amp_geom_interp)]

    new_tick_locations = np.array([t_peak_phys+10*i*kwargs['injection-parameters']['Mf']*lal.MTSUN_SI for i in range(0,9)])

    f = plt.figure(figsize=(12,8))
    ax = plt.subplot(2,2,1)
    ax.plot(t_phys, hr_geom_interp, c='firebrick', label=r'$\mathrm{h_r}$')
    ax.axvline(t_peak_phys, ls='dotted', c='k')
    ax.set_xlim([t_peak_phys-50*M_inj_sec, t_peak_phys+130*M_inj_sec])
    ax.legend(loc='best')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r'$t_{start} - t_{peak}/M_f$', fontsize=10)
    ax = plt.subplot(2,2,3)
    ax.plot(t_phys, hi_geom_interp, c='firebrick', label=r'$\mathrm{h_i}$')
    ax.axvline(t_peak_phys, ls='dotted', c='k')
    ax.set_xlim([t_peak_phys-50*M_inj_sec, t_peak_phys+130*M_inj_sec])
    ax.legend(loc='best')
    plt.xlabel('Time (s)')
    ax = plt.subplot(2,2,2)
    ax.plot(t_phys, freq_phys, c='firebrick', label=r'$\mathrm{Freq}$')
    ax.axhline(kwargs['injection-parameters']['f_220'], ls='dotted', c='k', alpha=0.8, label=r'$\mathrm{f_{220}}$' )
    ax.axhline(kwargs['injection-parameters']['f_220_peak'], ls='dashed', c='darkgreen', alpha=0.8, label=r'$\mathrm{f^{peak}_{22}}$')
    ax.axvline(t_peak_phys, ls='dotted', c='k')
    ax.set_ylim([100, kwargs['injection-parameters']['f_220']+150])
    ax.set_xlim([t_peak_phys-50*M_inj_sec, t_peak_phys+130*M_inj_sec])
    ax.legend(loc='upper left')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r'$t_{start} - t_{peak}/M_f$', fontsize=10)
    ax = plt.subplot(2,2,4)
    ax.plot(t_phys, Amp_geom_interp, c='firebrick', label=r'$\mathrm{Amp}$')
    ax.axvline(t_peak_phys, ls='dotted', c='k', label=r'$\mathrm{t^{peak}_{22}}$')
    ax.set_xlim([t_peak_phys-50*M_inj_sec, t_peak_phys+130*M_inj_sec])
    ax.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.suptitle('SXS:BBH:{0}'.format(kwargs['injection-parameters']['SXS-ID']), size=24)
    plt.tight_layout(rect=[0,0,1,0.95])
    f.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(output_dir,'SXS_{0}_N_{1}_l_{2}_m_{3}.pdf'.format(kwargs['injection-parameters']['SXS-ID'], kwargs['injection-parameters']['N'], l, m)), bbox_inches='tight')

    return

def single_time_delay_plots(dt_vec, **input_par):

    """
    
    Plot the time delay posterior distributions and chains for each detector pair.

    Parameters
    ----------

    dt_vec : dict
        Dictionary containing the time delay posterior distributions for each detector pair.
    input_par : dict
        Dictionary containing the input parameters for the run.

    Returns
    -------

    Nothing. Plots are saved to the output directory.
        
    """

    for det in input_par['detectors']:
        if(det==input_par['ref-det']):
            pass
        else:
            plt.figure()
            plt.hist(dt_vec[det], bins=50, alpha=0.8)
            plt.xlabel(r'$\Delta t_{%s-%s}$ (s)'%(input_par['ref-det'], det))
            plt.savefig(os.path.join(input_par['output'],'Plots/Parameters/dt_%s%s_posterior.png'%(input_par['ref-det'], det)))
            plt.figure()
            plt.plot(dt_vec[det],'.')
            plt.xlabel(r'$\Delta t_{%s-%s}$ (s)'%(input_par['ref-det'], det))
            plt.xlabel(r'$N_{samples}$')
            plt.savefig(os.path.join(input_par['output'],'Plots/Parameters/dt_%s%s_chain.png'%(input_par['ref-det'], det)))

    return

def sky_location_plots(x, Config, **input_par):

    """

    Plot the sky location posterior distributions and chains for each detector pair.

    Parameters
    ----------

    x : array
        Array containing the sky location posterior distributions for each detector pair.
    Config : configparser object
        Configparser object containing the input parameters for the run.
    input_par : dict
        Dictionary containing the input parameters for the run.

    Returns
    -------

    Nothing. Plots are saved to the output directory.

    """

    dt_vec       = {}
    ra_vec       = []
    dec_vec      = []
    tg_vec       = []
    non_ref_dets = []
    ref_det      = input_par['ref-det']
    sky_frame    = input_par['sky-frame']

    for det in input_par['detectors']:
        if(det==ref_det):
            pass
        else:
            non_ref_dets.append(det)
    if(input_par['template']=='Damped-sinusoids'):
        sky_loc_header = 'ra\tdec\ttg'
    else:
        sky_loc_header = 'ra\tdec\ttg\tdist'
    for det in non_ref_dets:
        sky_loc_header = sky_loc_header + '\tdt_'+ ref_det + det
        dt_vec[det]    = []

    for par in x:
        if(sky_frame == 'detector'):
            tg, ra, dec = DetFrameToEquatorial(lal.cached_detector_by_prefix[ref_det],
                                               lal.cached_detector_by_prefix[input_par['nonref-det']],
                                               input_par['trigtime'],
                                               np.arccos(par['cos_altitude']),
                                               par['azimuth'])
        elif(sky_frame == 'equatorial'):
            tg, ra, dec  = 0.0, par['ra'], par['dec']
        ra_vec.append(ra)
        dec_vec.append(dec)
        tg_vec.append(tg)
        for d in non_ref_dets:
            dt_vec[d].append(lal.ArrivalTimeDiff(lal.cached_detector_by_prefix[d].location,
                                                 lal.cached_detector_by_prefix[ref_det].location,
                                                 ra,
                                                 dec,
                                                 lal.LIGOTimeGPS(float(input_par['trigtime']))))
    if(len(input_par['detectors']) > 1):
        dt_tuple = np.array([dt_vec[det] for det in non_ref_dets])
        if(input_par['template']=='Damped-sinusoids'):
            np.savetxt(os.path.join(input_par['output'],'Nested_sampler/Sky-loc-samples.txt'), np.column_stack((np.array(ra_vec), np.array(dec_vec), np.array(tg_vec), dt_tuple.transpose())), header=sky_loc_header)
        else:
            np.savetxt(os.path.join(input_par['output'],'Nested_sampler/Sky-loc-samples.txt'), np.column_stack((np.array(ra_vec), np.array(dec_vec), np.array(tg_vec), np.array(np.exp(x['logdistance'])), dt_tuple.transpose())), header=sky_loc_header)
        single_time_delay_plots(dt_vec, **input_par)
    else:
        if(input_par['template']=='Damped-sinusoids'):
            np.savetxt(os.path.join(input_par['output'],'Nested_sampler/Sky-loc-samples.txt'), np.column_stack((np.array(ra_vec), np.array(dec_vec), np.array(tg_vec))), header=sky_loc_header)
        else:
            np.savetxt(os.path.join(input_par['output'],'Nested_sampler/Sky-loc-samples.txt'), np.column_stack((np.array(ra_vec), np.array(dec_vec), np.array(tg_vec), np.array(np.exp(x['logdistance'])))), header=sky_loc_header)

    if not(input_par['injection-approximant']==''):
        if(sky_frame == 'detector'):
            inj_tg, inj_ra, inj_dec = DetFrameToEquatorial(lal.cached_detector_by_prefix[ref_det], lal.cached_detector_by_prefix[input_par['nonref-det']], input_par['trigtime'], np.arccos(input_par['injection-parameters']['cos_altitude']), input_par['injection-parameters']['azimuth'])
        elif(sky_frame == 'equatorial'):
            inj_ra  = input_par['injection-parameters']['ra']
            inj_dec = input_par['injection-parameters']['dec']

        inj_psi = input_par['injection-parameters']['psi']
        inj_time_delay = {'{}_'.format(ref_det)+d2: lal.ArrivalTimeDiff(lal.cached_detector_by_prefix[d2].location,lal.cached_detector_by_prefix['{}'.format(ref_det)].location,inj_ra,inj_dec,lal.LIGOTimeGPS(float(input_par['trigtime']))) for d2 in non_ref_dets}
        expected_values_skypos = [inj_ra, inj_dec, inj_psi ] + [inj_time_delay['{}_{}'.format(ref_det, det)] for det in non_ref_dets]
    else:
        try:
            file = str(Config.get("Plot",'imr-samples'))
            expected_values_skypos = None
            if('GWTC' in file):
                BBH     = h5py.File(file, 'r')['IMRPhenomPv2_posterior']
                IMR_ra  = np.median(BBH['right_ascension'])
                IMR_dec = np.median(BBH['declination'])
                IMR_time_delay = {'{}_'.format(ref_det)+d2: lal.ArrivalTimeDiff(lal.cached_detector_by_prefix[d2].location,lal.cached_detector_by_prefix['{0}'.format(ref_det)].location,IMR_ra,IMR_dec,lal.LIGOTimeGPS(float(input_par['trigtime']))) for d2 in non_ref_dets}
                expected_values_skypos = [IMR_ra, IMR_dec, None] + [IMR_time_delay['{}_{}'.format(ref_det, det)] for det in non_ref_dets]
                sys.stdout.write(('Using %s to get sky position IMR samples.'%(file)))
        except(KeyError,configparser.NoOptionError, configparser.NoSectionError):
            expected_values_skypos = None
    #IMPROVEME: produce the plot properly also when parameters are fixed
    try:
        if(len(input_par['detectors']) > 1):
            mode_corner(np.column_stack((np.array(ra_vec), np.array(dec_vec), np.array(x['psi']), dt_tuple.transpose())),
                    labels        = [r'$ra$', r'$dec$', r'$psi$'] + [r'$\Delta t_{%s-%s}$ (s)'%(ref_det, det) for det in non_ref_dets],
                    quantiles     = [0.05, 0.5, 0.95],
                    show_titles   = True,
                    title_kwargs  = {"fontsize": 12},
                    use_math_text = True,
                    truths        = expected_values_skypos,
                    filename      = os.path.join(input_par['output'],'Plots/Parameters/corner_skypos.png'))
        else:
            mode_corner(np.column_stack((np.array(ra_vec), np.array(dec_vec), np.array(x['psi']))),
                    labels        = [r'$ra$', r'$dec$', r'$psi$'],
                    quantiles     = [0.05, 0.5, 0.95],
                    show_titles   = True,
                    title_kwargs  = {"fontsize": 12},
                    use_math_text = True,
                    truths        = expected_values_skypos,
                    filename      = os.path.join(input_par['output'],'Plots/Parameters/corner_skypos.png'))
    except(ValueError):
        pass

    return

def read_Mf_af_IMR_posterior(Config, **input_par):

    """
    
    Read the IMR posterior samples from the `imr-samples` entry of the `Plot` section of the configuration file, and return the median values of the mass and spin parameters.
    
    Parameters
    ----------

    Config : configparser.ConfigParser
        Configuration file parser.

    input_par : dict
        Dictionary containing the input parameters.

    Returns
    -------

    Mf_d : float
        Median value of the final mass.
    af_d : float
        Median value of the final spin.

    """

    # Get GR prediction (IMR + UIB NR fits applied on real data) to plot expected values. Using non-precessing fits since no phi12 angle was released alongside GWTC-1
    # IMPROVEME: GWTC-1 did not release spin angles, so precessing fits cannot be applited on this set of samples. When GWTC-2 is out, apply precessing fits.
    try:
        file = str(Config.get("Plot",'imr-samples'))
        if('GWTC' in file):
            BBH   = h5py.File(file, 'r')['IMRPhenomPv2_posterior']
            m1_samples    = BBH['m1_detector_frame_Msun']
            m2_samples    = BBH['m2_detector_frame_Msun']
            chi1_samples  = BBH['spin1']
            chi2_samples  = BBH['spin2']
            tilt1_samples = np.arccos(BBH['costilt1'])
            tilt2_samples = np.arccos(BBH['costilt2'])
            Mf_d          = []
            af_d          = []
            for (m1, m2, chi1, chi2, tilt1, tilt2) in zip(m1_samples, m2_samples, chi1_samples, chi2_samples, tilt1_samples, tilt2_samples):
                af_x = bbh_final_spin_projected_spins(m1, m2, chi1, chi2, tilt1, tilt2, 'UIB2016', truncate = bbh_Kerr_trunc_opts.trunc)
                af_d.append(af_x)
                Mf_d.append(bbh_final_mass_projected_spins(m1, m2, chi1, chi2, tilt1, tilt2, 'UIB2016'))
            Mf_d = np.array(Mf_d)
            af_d = np.array(Mf_d)
        else:
            package_datapath = import_datafile_path(file)
            PYRING_PREFIX    = set_prefix(warning_message=False)
            custom_datapath  = os.path.join(PYRING_PREFIX, file)
            if(os.path.exists(package_datapath)):
                print('* Fetching the IMR posterior from the default ones included in the package.\n')
                file = package_datapath
            elif(os.path.exists(custom_datapath)):
                print('* Fetching the IMR posterior relatively to the PYRING_PREFIX.\n')
                file = custom_datapath
            else:
                print('* Fetching the IMR posterior using the provided absolute path.\n')
                pass
            Mf_d = np.genfromtxt(file, names=True)['Mf']
            af_d = np.genfromtxt(file, names=True)['af']
        Mf  = np.median(Mf_d)
        dMf = np.std(Mf_d, ddof=1)
        af  = np.median(af_d)
        daf = np.std(af_d, ddof=1)

    except (KeyError,configparser.NoOptionError, configparser.NoSectionError, OSError):
        try:
            if(not(input_par['injection-approximant']=='Damped-sinusoids') and not(input_par['injection-approximant']=='')):
                Mf   = input_par['injection-parameters']['Mf']
                dMf  = (5./100.)*Mf #IMPROVEME: substitute with an estimate of NR uncertainty
                Mf_d = np.random.normal(Mf, dMf/2, 1000)
                af   = input_par['injection-parameters']['af']
                daf  = (2./100.)*af #IMPROVEME: substitute with an estimate of NR uncertainty
                af_d = np.random.normal(af, daf/2, 1000)
            else:
                Mf   = Config.getfloat("Priors",'mf-time-prior')
                dMf  = Config.getfloat("Plot",'dmf')#Estimate of half the 95% CI, which is 2sigma
                Mf_d = np.random.normal(Mf, dMf/2, 1000)
                af   = Config.getfloat("Plot",'af')
                daf  = Config.getfloat("Plot",'daf')#Estimate of half the 95% CI, which is 2sigma
                af_d = np.random.normal(af, daf/2, 1000)
        except (KeyError,configparser.NoOptionError, configparser.NoSectionError):
            sys.stdout.write('No IMR posterior was passed.\n')
            Mf_d = None
            af_d = None
    if((Mf_d is not None) and (af_d is not None)):
        sys.stdout.write('* To predict the GR spectrum of QNM or final mass and spin, the following parameters will be used:\n\n  Mf: {:.3f}  ({:.3f})\n  af: {:.3f}   ({:.3f})\n\n'.format(Mf, dMf, af, daf))

    return Mf_d, af_d

def plot_ACF(time, acf, label, output_path):

    """
    
    Plot the ACF of the data.

    Parameters
    ----------

    time : array
        Time array.
    acf : array
        ACF array.
    label : str
        Label of the plot.
    output_path : str
        Path to the output file.

    Returns
    -------

    Nothing, but it saves the plot in the output_path.
    
    """

    init_plotting()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(time, acf, linewidth=0.5, color = 'k', label=r'{}'.format(label))
    ax.set_xlabel(r'$\tau\,(s)$', fontsize=18)
    ax.set_ylabel(r'$C(\tau)$'  , fontsize=18)
    ax.legend()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')

    return

def plot_PSD(freqs, psd, label, output_path):

    """

    Plot the PSD of the data.

    Parameters
    ----------

    freqs : array
        Frequency array.
    psd : array
        PSD array.
    label : str
        Label of the plot.
    output_path : str
        Path to the output file.
    
    Returns
    -------

    Nothing, but it saves the plot in the output_path.

    """

    init_plotting()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.loglog(freqs, psd, linewidth=0.5, color = 'k', label=r'{}'.format(label))
    ax.set_xlabel(r'$f\,(Hz)$',         fontsize=18)
    ax.set_ylabel(r'$S(f)\,(Hz^{-1})$', fontsize=18)
    ax.legend()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
    
    return

def plot_ACF_compare(time1, acf1, label1, time2, acf2, label2, output_path):

    """
    
    Plot two different ACF estimates to compare them.

    Parameters
    ----------

    time1 : array
        Time array of the first ACF.
    acf1 : array
        ACF array of the first ACF.
    label1 : str
        Label of the first ACF.
    time2 : array
        Time array of the second ACF.
    acf2 : array
        ACF array of the second ACF.
    label2 : str
        Label of the second ACF.
    output_path : str
        Path to the output file.

    Returns
    -------

    Nothing, but it saves the plot in the output_path.
    
    """

    init_plotting()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(time1, acf1, linewidth=0.5, color = 'k',         label=r'{}'.format(label1), linestyle='dotted')
    ax.plot(time2, acf2, linewidth=0.5, color = 'firebrick', label=r'{}'.format(label2), alpha=0.8)
    ax.set_xlabel(r'$\tau\,(s)$', fontsize=18)
    ax.set_ylabel(r'$C(\tau)$'  , fontsize=18)
    ax.legend()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
    
    return

def plot_PSD_compare(freqs1, psd1, label1, freqs2, psd2, label2, output_path):

    """

    Plot two different PSD estimates to compare them.

    Parameters
    ----------

    freqs1 : array
        Frequency array of the first PSD.
    psd1 : array
        PSD array of the first PSD.
    label1 : str
        Label of the first PSD.
    freqs2 : array
        Frequency array of the second PSD.
    psd2 : array
        PSD array of the second PSD.
    label2 : str    
        Label of the second PSD.    
    output_path : str
        Path to the output file.

    Returns
    ------- 

    Nothing, but it saves the plot in the output_path.

    """

    init_plotting()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.loglog(freqs1, psd1, linewidth=0.5, color = 'firebrick', label=r'{}'.format(label1), alpha=0.8)
    ax.loglog(freqs2, psd2, linewidth=0.5, color = 'k',         label=r'{}'.format(label2), linestyle='dotted')
    ax.set_xlabel(r'$f\,(Hz)$',        fontsize=18)
    ax.set_ylabel(r'$S(f)\,(Hz^{-1})$',fontsize=18)
    ax.legend()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
    
    return

def UNUSED_noise_evidence_density(t0_samples,noise_model):

    """
    
    Compute the noise evidence density for a given set of t0 samples.

    Parameters
    ----------

    t0_samples : array
        Array of t0 samples.
    noise_model : NoiseModel object
        Noise model object.

    Returns
    -------

    logZnoise : array
        Array of logZnoise values for each t0 sample.
    
    """
    
    logZnoise = []
    time = noise_model.times-noise_model.tevent
    for t0 in t0_samples:
        index = np.abs(t0-time).argmin()
        logZnoise.append(np.sum([- 0.5*np.einsum('i, ij, j',s[index:],Cinv[index:,index:],s[index:]) for s,Cinv in zip(noise_model.data,noise_model.inverse_covariance)]))
    
    return np.array(logZnoise)
