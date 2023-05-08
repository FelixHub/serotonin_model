import os
from os.path import join as pjoin
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
from foundation_tables import PhotometrySession
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import linregress

schema = dj.schema('rlmp_gainChange',locals())
plt.rcParams['figure.figsize'] = [10, 7] # Make default figure size larger.

@schema
class PhotometryPreprocessingParam(dj.Lookup):
    definition = """
    # Parameters for photometry preprocessing
    photometry_preprocessing_param_id: int      # unique id for photometry preprocessing parameter set
    ---
    median_filt_kernel_size : int
    lowpass_filter_frequency : int 
    photobleaching_estim_method : varchar(80)
    photobleaching_estim_params : blob 
    baseline_fluorescence_signal : varchar(25)
    baseline_fluorescence_lowpass_freq : float
    """  

@schema
class ProcessedPhotometry(dj.Computed):
    definition = """
    # Processed photometry signal 
    -> PhotometryPreprocessingParam
    -> PhotometrySession
    ---
    sample_times : longblob  # timepoints when we sampled the GCaMP frequency
    df_over_f : longblob     # time series of normalized fluorescence values
    """
    key_source = PhotometrySession() * (PhotometryPreprocessingParam() & "photometry_preprocessing_param_id = 0")
    def make(self,key): 
        print("Processing photometry session from {prefix} {mouse_id} {session_date} \n".format(**key))
        
        # load data and preprocessing parameters
        photometry_path = (PhotometrySession() & key).fetch1('photometry_path') 
        photometry_folder = Path(photometry_path).parent
        preprocessing_param = (PhotometryPreprocessingParam() & key).fetch(as_dict = True)[0]

        # make folder for processing visualizations 
        if not os.path.exists(pjoin(photometry_folder,"param%i"%preprocessing_param["photometry_preprocessing_param_id"])): 
            os.mkdir(pjoin(photometry_folder,"param%i"%preprocessing_param["photometry_preprocessing_param_id"]))

        # load some data
        raw_df = pd.read_csv(photometry_path)
        raw_df = raw_df[6:] # get rid of initial artifact

        # unpack raw data
        sample_wavelength = raw_df['Flags'].to_numpy()
        roi_g = raw_df['Region1G'].to_numpy() 
        roi_r = raw_df['Region0R'].to_numpy() 
        sample_times_all = raw_df['Timestamp'].to_numpy()
        sample_rate = (1 / np.median(np.diff(sample_times_all)))

        # subselect sampled wavelength
        sample415mask = sample_wavelength == 17 
        sample470mask = sample_wavelength == 18
        sample_times415 = sample_times_all[sample415mask]
        sample_times = sample_times_all[sample470mask] # 470 is the official clock
        g415_raw = roi_g[sample415mask]
        g470_raw = roi_g[sample470mask]

        # linear interpolation to get everyone on 470 timeframe
        g415_raw = np.interp(sample_times,sample_times415,g415_raw)  

        # plot raw signals
        plt.figure()
        plt.plot(sample_times,g415_raw,color = 'r',label = '415 signal')
        plt.plot(sample_times,g470_raw,color = 'g',label = '470 signal')
        plt.title("Raw signal from {prefix} {mouse_id} {session_date}".format(**key))
        plt.legend()
        plt.savefig(pjoin(photometry_folder,'raw_signal.png'))
        plt.close()

        # Median filtering to remove electrical artifacts
        g415_denoised = medfilt(g415_raw, kernel_size=preprocessing_param['median_filt_kernel_size'])
        g470_denoised = medfilt(g470_raw, kernel_size=preprocessing_param['median_filt_kernel_size'])
        
        # Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
        b,a = butter(2, preprocessing_param['lowpass_filter_frequency'], btype='low', fs=sample_rate)
        g415_denoised = filtfilt(b,a, g415_denoised)
        g470_denoised = filtfilt(b,a, g470_denoised) 

        # photobleaching baseline estimation 
        baseline_fun = get_baseline_fun(preprocessing_param)
        g415_baseline , g415_bleachCorrected = baseline_fun(sample_rate,sample_times,g415_denoised,**preprocessing_param['photobleaching_estim_params']) 
        g470_baseline , g470_bleachCorrected = baseline_fun(sample_rate,sample_times,g470_denoised,**preprocessing_param['photobleaching_estim_params']) 

        # visualize photobleaching correction 
        if np.isin(preprocessing_param['photobleaching_estim_method'],['polyFit_baseline','expoFit_baseline']): 
            # plot 
            plt.figure()
            plt.plot(sample_times,g415_denoised,color = 'r')
            plt.plot(sample_times,g415_baseline,color = 'k')
            plt.plot(sample_times,g470_denoised,color = 'g')
            plt.plot(sample_times,g470_baseline,color = 'k')
            plt.title("{prefix} {mouse_id} {session_date} photobleaching estimation using {photobleaching_estim_method}".format(**{**key,**preprocessing_param}))
        else: 
            fig,ax = plt.subplots(2,1)
            ax[0].plot(sample_times,g470_denoised,color = 'g')
            ax[0].plot(sample_times,g415_denoised,color = 'r')
            ax[1].plot(sample_times,g470_bleachCorrected,color = 'g')
            ax[1].plot(sample_times,g415_bleachCorrected,color = 'r')
            ax[0].set_title("{prefix} {mouse_id} {session_date} photobleaching estimation using {photobleaching_estim_method}".format(**{**key,**preprocessing_param}))
            plt.tight_layout()
        plt.savefig(pjoin(photometry_folder,"param%i"%preprocessing_param["photometry_preprocessing_param_id"],'photobleaching_estim.png'))
        plt.close()

        # Motion correction 
        # get linear regression betw g415 and g470 signal 
        slope, intercept, r_value, p_value, std_err = linregress(x=g415_bleachCorrected, y=g470_bleachCorrected)
        GCaMP_est_motion = intercept + slope * g415_bleachCorrected
        GCaMP_motionCorrected = g470_bleachCorrected - GCaMP_est_motion

        # visualize motion correction 
        fig,ax = plt.subplots(1,2,figsize = (12,5))
        ax[0].scatter(g415_bleachCorrected[::5], g470_bleachCorrected[::5],alpha=0.1, marker='.')
        x = np.array(ax[0].get_xlim())
        ax[0].plot(x, intercept+slope*x)
        ax[0].set_xlabel('g415')
        ax[0].set_ylabel('g470')
        ax[0].set_title('Slope: %.3f     R-squared: %.3f'%(slope,r_value**2))
        ax[1].plot(sample_times, g470_bleachCorrected  , label='GCaMP - pre motion correction')
        ax[1].plot(sample_times, GCaMP_motionCorrected, 'g', label='GCaMP - motion corrected')
        ax[1].plot(sample_times, GCaMP_est_motion, 'y', label='estimated motion')
        ax[1].set_xlabel('Time (seconds)')
        # ax[1].set_title('Signal correction')
        ax[1].legend()
        ax[1].set_xlim([sample_times[0],sample_times[0]+10])
        fig.suptitle("Motion correction for {prefix} {mouse_id} {session_date}".format(**key))
        fig.tight_layout()
        plt.savefig(pjoin(photometry_folder,"param%i"%preprocessing_param["photometry_preprocessing_param_id"],'motion_correction.png'))
        plt.close()

        # get baseline for dF/F
        b,a = butter(2,preprocessing_param['baseline_fluorescence_lowpass_freq'], btype='low', fs=sample_rate)

        # # visualize dF/F baseline
        plt.figure()
        if preprocessing_param['baseline_fluorescence_signal'] == 'denoised': 
            baseline_fluorescence = filtfilt(b,a, g470_denoised, padtype='even')
            plt.plot(sample_times, g470_denoised       , 'g', label='GCaMP denoised')
            plt.plot(sample_times, baseline_fluorescence, 'k', label='baseline fluorescence')
        elif preprocessing_param['baseline_fluorescence_signal'] == 'motion_corrected': 
            baseline_fluorescence = filtfilt(b,a, GCaMP_motionCorrected, padtype='even')
            plt.plot(sample_times, GCaMP_motionCorrected       , 'g', label='GCaMP motion corrected')
            plt.plot(sample_times, baseline_fluorescence, 'k', label='baseline fluorescence') 
        else: 
            raise NotImplementedError 
        plt.xlabel('Time (seconds)')
        plt.title('{prefix} {mouse_id} {session_date} baseline fluorescence'.format(**key))
        plt.legend();
        plt.savefig(pjoin(photometry_folder,"param%i"%preprocessing_param["photometry_preprocessing_param_id"],'baseline_fluorescence.png'))
        plt.close()

        # calculate dF/F
        GCaMP_dF_F = GCaMP_motionCorrected/baseline_fluorescence

        # visualize final dF/F
        fig,ax = plt.subplots(1,2)
        ax[0].plot(sample_times, GCaMP_dF_F * 100, 'g')
        ax[0].set_xlabel('Time (seconds)')
        ax[0].set_ylabel('GCaMP delta-F/F (%)')
        ax[0].set_title('GCaMP dF/F')
        ax[1].plot(sample_times, GCaMP_dF_F * 100, 'g')
        ax[1].set_xlabel('Time (seconds)')
        ax[1].set_ylabel('GCaMP delta-F/F (%)')
        ax[1].set_title('GCaMP dF/F First 100 seconds')
        ax[1].set_xlim(sample_times[0],sample_times[0] + 100);
        fig.suptitle('Final {prefix} {mouse_id} {session_date} dF/F'.format(**key))
        plt.savefig(pjoin(photometry_folder,"param%i"%preprocessing_param["photometry_preprocessing_param_id"],'final_dF_over_F.png'))
        plt.close()

        # perform insert
        key["sample_times"] = sample_times
        key["df_over_f"] = GCaMP_dF_F
        self.insert1(key)
            



def get_baseline_fun(preprocessing_params): 
    """ 
        Args: preprocessing params dict
        
        Return function that takes in sample_rate,sample_times,signal_denoised,and params

        And returns baseline and bleach_corrected
    """
    photobleaching_estim_method = preprocessing_params['photobleaching_estim_method'] 

    if photobleaching_estim_method == 'highpass_baseline': 
        return highpass_baseline 
    elif photobleaching_estim_method == 'expoFit_baseline': 
        # raise NotImplementedError
        return expoFit_baseline 
    elif photobleaching_estim_method == 'polyFit_baseline': 
        return polyFit_baseline 
    else: 
        raise NotImplementedError

def exp_func(x, a, b, c): # exp fun to fit 
    return a*np.exp(-b*x) + c   

def highpass_baseline(sample_rate,sample_times,signal_denoised,highpass_freq = 0.001): 
    """ 
        Return baseline signal by highpass filtering
    """
    b,a = butter(2, highpass_freq, btype='high', fs=sample_rate)
    signal_bleachCorrected = filtfilt(b,a, signal_denoised, padtype='even')
    return [] , signal_bleachCorrected 

def expoFit_baseline(sample_rate,sample_times,signal_denoised,p0 = [1,1e-3,1]): 
    """ 
        Return baseline signal by expo fit
    """
    params, parm_cov = curve_fit(exp_func, sample_times, signal_denoised, p0=p0,bounds=([0,0,0],[4,0.1,4]), maxfev=1000)
    signal_baseline = exp_func(sample_times, *params)
    signal_bleachCorrected = signal_denoised - signal_baseline
    return signal_baseline , signal_bleachCorrected

def polyFit_baseline(sample_rate,sample_times,signal_denoised,polyfit_deg = 4): 
    """ 
        Return baseline signal by polynomial fit
    """
    coefs = np.polyfit(sample_times, signal_denoised, deg=polyfit_deg)
    signal_baseline = np.polyval(coefs, sample_times)
    signal_bleachCorrected = signal_denoised - signal_baseline
    return signal_baseline , signal_bleachCorrected 