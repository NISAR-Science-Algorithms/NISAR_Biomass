# -*- coding: utf-8 -*-
"""
Created on Wed May 14 18:59:46 2025

@author: naveenr
"""
import os
from scipy.stats import gaussian_kde
from ipywidgets import widgets
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import functions as pp
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from osgeo import gdal
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy
import concurrent.futures
from scipy.optimize import differential_evolution, minimize, shgo, dual_annealing
import seaborn as sns
from scipy.ndimage import generic_filter
from datetime import datetime, UTC

def calculate_density(x, y):
    # Calculate the point density
    kde = gaussian_kde([x, y])
    xy = np.vstack([x, y])
    density = kde(xy)
    return density

def adaptive_smooth(image, thresh_sigma=2.0):
    """
    Only smooths pixels that are more than 2 sigma away from their 3x3 neighborhood.
    This preserves forest edges while removing 'salt and pepper' speckle.
    """
    def filter_func(window):
        center = window[len(window)//2]
        neighborhood = np.delete(window, len(window)//2)
        mean = np.mean(neighborhood)
        std = np.std(neighborhood)
        
        # If the pixel is a statistical outlier, replace it with the neighborhood mean
        if abs(center - mean) > (thresh_sigma * std):
            return mean
        return center

    return generic_filter(image, filter_func, size=3)


def plot_nisar_curve(X, W0, OUT_PREFIX, param0, X_RANGE):
    import math
    W = np.arange(0, 50 * math.ceil(W0.max() / 50), 1)
    
    
    plt.rcParams['font.family'] = 'serif'  # SET DEFAULT FONT FAMILY FOR FIGURE 
    plt.rcParams['font.serif'] = ['DejaVu Serif']  # SET THE PRIMARY FONT FOR ALL SERIF TEXT IN THE FIGURE  
    plt.rcParams['font.size'] = 12  # SET FONT SIZE 
    
    fig, ax = plt.subplots(1, X.shape[-1], figsize=(10, 5))
    
    AHV,  AHH, BHV, BHH, CHV, CHH, alphaHV, alphaHH, deltaHV, deltaHH, DHV, DHH, S   = param0
    
    # Calculate the point density
    IDX = (~np.isnan(X[:, 0])) & (~np.isnan(W0[:, 0]))
    print(IDX.shape,X[:, 0].shape, W0[:, 0].shape)
    x0 = X[:, 0][IDX]
    y0 = W0[:, 0][IDX]
    xy = np.vstack([x0, y0])
    z0 = gaussian_kde(xy)(xy)
    idx = z0.argsort()
    x1, y1, z1 = x0[idx], y0[idx], z0[idx]
    
    ax[0].plot(W, volume(W, AHV, BHV, alphaHV),  linestyle ='--',  markerfacecolor='none', color = 'g', label = 'Vol')
    ax[0].plot(W, double(W, BHV, CHV, deltaHV, S),  linestyle =':',  markerfacecolor='none', color = 'r', label = 'Vol-Surf')
    ax[0].plot(W, surface(W, BHV, DHV,  S),  linestyle='-.',  markerfacecolor='none', color = 'b', label = 'Surf')
    ax[0].plot(W, nisar(W, AHV, BHV, CHV, alphaHV, deltaHV, DHV, S),  linestyle='-',   markerfacecolor='none', color = 'k', label = 'Total')
    ax[0].set_xlabel('Simulated AGB (Mg/ha)')
    
    ax[0].grid(True)
    ax[0].scatter(y1, x1, c=z1, s=10, edgecolor=None, cmap='viridis')
    ax[0].legend()
    ax[0].set_xlim(X_RANGE)
    
    ax[0].set_ylabel('HV Backscattered Power (m2/m2)')
    
    
    # Calculate the point density
    IDX = (~np.isnan(X[:, 1])) & (~np.isnan(W0[:, 0]))
    x0 = X[:, 1][IDX]
    y0 = W0[:, 0][IDX]
    xy = np.vstack([x0, y0])
    z0 = gaussian_kde(xy)(xy)
    idx = z0.argsort()
    x1, y1, z1 = x0[idx], y0[idx], z0[idx]
    
    
    ax[1].plot(W, volume(W, AHH, BHH, alphaHH),  linestyle ='--',  markerfacecolor='none', color = 'g', label = 'Vol')
    ax[1].plot(W, double(W, BHH, CHH, deltaHH, S),  linestyle =':',  markerfacecolor='none', color = 'r', label = 'Vol-Surf')
    ax[1].plot(W, surface(W, BHH, DHH,  S),  linestyle='-.',  markerfacecolor='none', color = 'b', label = 'Surf')
    ax[1].plot(W, nisar(W, AHH, BHH, CHH, alphaHH, deltaHH, DHH, S),  linestyle='-',   markerfacecolor='none', color = 'k', label = 'Total')
    
    
    ax[1].set_xlabel('Simulated AGB (Mg/ha)')
    ax[1].set_ylabel('HH Backscattered Power (m2/m2)')
    ax[1].grid(True)
    ax[1].scatter(y1, x1, c=z1, s=10, edgecolor=None, cmap='viridis')
    ax[1].legend()
    ax[1].set_xlim(X_RANGE)

    plt.tight_layout()
    plt.savefig(OUT_PREFIX + '_nisar_model.png', dpi=600)
    plt.show()
    plt.close()


def r_squared(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value ** 2

from scipy.optimize import curve_fit
# hh/hv retrieval
def linear_agb_model(z, a0, a1, a2, a3, a4):
    hv, hh = z[:,  0].flatten(), z[:, 1].flatten()
    return a0 + a1 * hv + a2 * pow(hv, 2) + a3 * hh + a4 * pow(hh, 2)

def agb_initialization(z, W):
    # Fit curve to the data
    params, params_covariance = curve_fit(linear_agb_model, z, W.flatten(), p0 = [1.0,1.0,1.0,1.0,1.0])
    yfit = linear_agb_model(z, params[0], params[1], params[2], params[3], params[4])
    return yfit[:, np.newaxis], params

def plot_agb_accuracy(OUTNAME, W_MEAN2_TRAIN, Y0,  p2, p1, W_MEAN2_TEST, Y0_TEST, p2_test, p1_test):
    
    NUM_SCENE = len(W_MEAN2_TRAIN)
    for NUM in range(NUM_SCENE):
        
        # Set font properties globally
        sns.set_style("white")
        plt.rcParams['font.family'] = 'serif'  # Options: 'serif', 'sans-serif', 'monospace', etc.
        plt.rcParams['font.serif'] = ['DejaVu Serif']  # Specify serif fonts
        plt.rcParams['font.size'] = 10  # Set default font size
        fig, axs = plt.subplots(1, 2, figsize = (9, 3), facecolor='white')

        ## Get the input data samples
        x_measured = Y0 
        x_pred     = W_MEAN2_TRAIN[NUM]
        
        x_bin = np.arange(0,301,1)
        density =  calculate_density(x_measured, x_pred)
        axs[0].grid(which='both', linestyle='--', linewidth=0.7)  # Show grid lines
        axs[0].scatter(x_measured, x_pred, c = density, cmap='viridis', s=5, edgecolor=None)
        axs[0].plot(x_bin, x_bin, c = 'k', label = '1:1')
        axs[0].plot(x_bin, x_bin+20, c = 'r', linestyle='--')
        axs[0].plot(x_bin, x_bin-20, c = 'r', linestyle='--', label='$\pm$ 20 Mg/ha')
        # axs[0].set_xlim([0,300])
        # axs[0].set_ylim([0,300])
        axs[0].set_xlabel('LIDAR AGB (Mg/ha)')
        if NUM == 0:
            axs[0].set_ylabel('\n Estimated NISAR AGB (Mg/ha) \n from First image')
        else:
            axs[0].set_ylabel('\n Estimated NISAR  AGB (Mg/ha) \n Mean from First ' + str(NUM+1) + 'image')
        axs[0].set_title('Calibration Area')
            
        
        # Add text annotations for statistics
        stats_text = "{:.1f}% within 20 \n{:.1f}% within 10".format(p2[NUM], p1[NUM])
                
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[0].text(0.025, 0.95, stats_text, transform=axs[0].transAxes, fontsize=10,
                               verticalalignment='top', horizontalalignment='left',
                               bbox=props)



        ## Get the input data samples
        x_measured = Y0_TEST
        x_pred     = W_MEAN2_TEST[NUM]
        
        x_bin = np.arange(0,301,1)
        density =  calculate_density(x_measured, x_pred)
        axs[1].grid(which='both', linestyle='--', linewidth=0.7)  # Show grid lines
        axs[1].scatter(x_measured, x_pred, c = density, cmap='viridis', s=5, edgecolor=None)
        axs[1].plot(x_bin, x_bin, c = 'k', label = '1:1')
        axs[1].plot(x_bin, x_bin+20, c = 'r', linestyle='--', label='$\pm$ 20 Mg/ha')
        axs[1].plot(x_bin, x_bin-20, c = 'r', linestyle='--')
        axs[1].set_xlim([0,300])
        axs[1].set_ylim([0,300])
        axs[1].set_xlabel('LIDAR AGB (Mg/ha)')
        axs[1].set_ylabel(None)
        axs[1].set_title('Validation Area')
        
         
        
        # Add text annotations for statistics
        stats_text = "{:.1f}% within 20 \n{:.1f}% within 10".format(p2_test[NUM], p1_test[NUM])
                
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[1].text(0.025, 0.95, stats_text, transform=axs[1].transAxes, fontsize=10,
                               verticalalignment='top', horizontalalignment='left',
                               bbox=props)

        handles, labels = axs[0].get_legend_handles_labels()  
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1), bbox_transform=fig.transFigure )  
        plt.subplots_adjust(wspace=0.07, hspace=0.09)  # Adjust these values as needed           
        plt.tight_layout()
        
        plt.savefig(OUTNAME + '_' +str(NUM) + '.png', dpi=600, bbox_inches='tight')
        #plt.show()
        
    
def plot_accuracy_stats(M_T, RMSE, R2, RMSE_100, R2_100, p2, p1, OUTNAME):
    plt.rcParams['font.family'] = 'serif'  # SET DEFAULT FONT FAMILY FOR FIGURE 
    plt.rcParams['font.serif'] = ['DejaVu Serif']  # SET THE PRIMARY FONT FOR ALL SERIF TEXT IN THE FIGURE  
    plt.rcParams['font.size'] = 12  # SET FONT SIZE 
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    
    axs[0, 0].plot(np.arange(1, M_T+1), RMSE, marker='o', linestyle='-', color='k', label = 'Overall')
    axs[0, 0].plot(np.arange(1, M_T+1), RMSE_100, marker='o', linestyle='-', color='b', label = 'AGB < 100 Mg/ha')
    axs[0, 0].set_xlabel('No. of SAR Images')
    axs[0, 0].set_ylabel('RMSE (Mg/ha)')
    axs[0, 0].set_xlim([0, M_T+2])
    axs[0, 0].set_ylim([min(RMSE_100) - 5, max(RMSE) + 5])
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    axs[0, 1].plot(np.arange(1, M_T+1), R2, marker='o', linestyle='-', color='k', label = 'Overall')
    axs[0, 1].plot(np.arange(1, M_T+1), R2_100, marker='o', linestyle='-', color='b', label = 'AGB < 100 Mg/ha')
    axs[0, 1].set_xlabel('No. of SAR Images')
    axs[0, 1].set_ylabel('R2')
    axs[0, 1].set_xlim([0, M_T+2])
    # axs[0, 1].set_ylim([0, 1])
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    
    axs[1, 0].plot(np.arange(1, M_T+1), p2, marker='o', linestyle='-', color='b')
    axs[1, 0].set_xlabel('No. of SAR Images')
    axs[1, 0].set_ylabel('Percentage of pixels within +/- 20 Mg/ha')
    axs[1, 0].set_xlim([0, M_T+2])
    axs[1, 0].set_ylim([0, max(p2)+10])
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(np.arange(1, M_T+1), p1, marker='o', linestyle='-', color='b')
    axs[1, 1].set_xlabel('No. of SAR Images')
    axs[1, 1].set_ylabel('Percentage of pixels within +/- 10 Mg/ha')
    axs[1, 1].set_xlim([0, M_T+2])
    axs[1, 1].set_ylim([0, max(p1)+10])
    axs[1, 1].grid(True)
    
    
    plt.savefig(OUTNAME, dpi=600, bbox_inches='tight')
    #plt.show()
    
    
    
def density_scatter_plot(
    out_prefix,
    x1,
    y1,
    x_label="W",
    y_label="W_hat",
    x_limit=None,
    y_limit=None,
):
    # Calculate the point density
    x0 = x1[(~np.isnan(x1)) & (~np.isnan(y1))]
    y0 = y1[(~np.isnan(x1)) & (~np.isnan(y1))]
    xy = np.vstack([x0, y0])
    z0 = gaussian_kde(xy)(xy)
    idx = z0.argsort()
    x1, y1, z1 = x0[idx], y0[idx], z0[idx]

    fig0 = plt.figure(figsize=(5, 4))
    plt.scatter(x1, y1, c=z1, s=10, edgecolor=None)
    # plt.scatter(x0, y0, s=4, edgecolor="")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    x1 = plt.xlim(x_limit)
    y1 = plt.ylim(y_limit)
    plt.plot(x1, y1, "k-")
    # plt.plot(x1, np.array(y1) * 0.8, '--', color='0.6')
    # plt.plot(x1, np.array(y1) * 1.2, '--', color='0.6')
    # print('{}% of the data within the range of ±20%.'.format(
    #     np.sum((x0 * 0.8 < y0) & (y0 < x0 * 1.2) & (x0 < 100)) / x0[x0 < 100].size * 100))
    # print('{}% of the data within the range of ±10%.'.format(
    #     np.sum((x0 * 0.9 < y0) & (y0 < x0 * 1.1) & (x0 < 100)) / x0[x0 < 100].size * 100))
    # print('{}% of the data within the range of 20 Mg/ha.'.format(
    #     np.sum((x0 - 20 < y0) & (y0 < x0 + 20) & (x0 < 100)) / x0[x0 < 100].size * 100))
    # print('{}% of the data within the range of 10 Mg/ha.'.format(
    #     np.sum((x0 - 10 < y0) & (y0 < x0 + 10) & (x0 < 100)) / x0[x0 < 100].size * 100))

    # r2 = r2_score(x0[:, None], y0[:, None])
    r2 = r_squared(x0, y0)
    rmse = np.sqrt(mean_squared_error(x0[:, None], y0[:, None]))
    # print("RMSE: {:.5f}".format(rmse))
    plt.text(
        (max(x_limit) - min(x_limit)) * 0.04 + min(x_limit),
        (max(y_limit) - min(y_limit)) * 0.85 + min(y_limit),
        r"$R^2 = {:1.3f}$".format(r2) + "\n" + "$RMSE = {:1.3f}$".format(rmse),
    )
    plt.savefig(out_prefix + "_measured_vs_observed.png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig0)

    return r2, rmse


def initial_agb_estimation(OUT_DIR, SCENE_MASK_FILE_100,  SAR_RES_DATA_LIST, FILE_NAME, TRAIN_FILE = None, OUT_AGB_FILE = None, A_MAX=300):
    
    
    os.chdir(OUT_DIR)
     
    # CREATE OUTPUT NAME 
    OUT_PREFIX = OUT_DIR / FILE_NAME
    
    if TRAIN_FILE is not None:
        ############ estimate AGB initial values ########    
        mdl00 = FieldRetrieval(TRAIN_FILE)
        z1, W1, mask_ws1 = mdl00.inversion_setup(SAR_RES_DATA_LIST, agb_file = OUT_AGB_FILE, mask_file = TRAIN_FILE, n_1 = 0)
        
        z1_dim = z1.shape
        z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        
        z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=TRAIN_FILE)
        
        Z0_TRAIN = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        
        W0_TRAIN = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=TRAIN_FILE)
    
    
    
    mdl00 = FieldRetrieval(SCENE_MASK_FILE_100)
    
    z1_scene, _, mask_ws1_scene = mdl00.inversion_setup(SAR_RES_DATA_LIST, agb_file = None, mask_file = SCENE_MASK_FILE_100, n_1 = 0)
    
    z1_dim_scene = z1_scene.shape
    z1r_scene = z1_scene.reshape([-1, z1_dim_scene[1] * z1_dim_scene[2] * z1_dim_scene[3]])

    z0_scene = mdl00.inversion_return_valid(z1r_scene, mask_ws1_scene, mask_file=SCENE_MASK_FILE_100)
    
    Z0_SCENE = z0_scene.reshape([-1, z1_dim_scene[1], z1_dim_scene[2], z1_dim_scene[3]])
     
    IN0 = gdal.Open(SCENE_MASK_FILE_100, gdal.GA_ReadOnly)
    
    # MEAN_HV_TRAIN = np.nanmean(Z0_TRAIN[:, :, :, 0].reshape([Z0_TRAIN.shape[0], -1]), axis=1) # ESTIMATE THE MEAN VALUE OF HV TIME SERIES
    # MEAN_HH_TRAIN = np.nanmean(Z0_TRAIN[:, :, :, 1].reshape([Z0_TRAIN.shape[0], -1]), axis=1) # ESTIMATE THE MEAN VALUE OF HH TIME SERIES
 
    # # STACK THE HV AND HH VALUES
    # MEAN_X_TRAIN = np.stack([MEAN_HV_TRAIN, MEAN_HH_TRAIN]).T
    
    # # FIT THE QUADRATIC MODEL 
    # param0w, params =  agb_initialization(MEAN_X_TRAIN, W0_TRAIN)
    # para_df = pd.DataFrame()
    
    for NUM in range(0, Z0_SCENE.shape[2]):
        
        # param0w, params =  agb_initialization(Z0_TRAIN[:, 0, NUM, :], W0_TRAIN)
        # hf = pd.DataFrame(params[:,np.newaxis].T, columns=["a0", "a1", "a2", "a3", "a4"])
        # hf['Image'] = 'Scene ' + str(NUM+1)
        # para_df = pd.concat([para_df, hf])
        
        # W0_HAT_TRAIN =  linear_agb_model(Z0_TRAIN[:, 0, NUM, :], params[0], params[1], params[2], params[3], params[4])    
        
        # Clip the data for maximum AGB value in the training data
        if TRAIN_FILE is not None:
            FACTOR  = np.sqrt(np.mean(W0_TRAIN[W0_TRAIN< 100]))/np.mean(Z0_TRAIN[:, :, NUM, 0][W0_TRAIN< 100])
            W0_HAT_TRAIN =  np.power((FACTOR * Z0_TRAIN[:, 0, NUM, 0]), 2)
            W0_HAT_TRAIN = np.clip(W0_HAT_TRAIN, a_min= 0, a_max=W0_TRAIN.max())
        else:
            FACTOR = 100
        
        W0_HAT_SCENE =  np.power((FACTOR * Z0_SCENE[:, 0, NUM, 0]), 2)
        # Clip the data for maximum AGB value in the training data
        W0_HAT_SCENE = np.clip(W0_HAT_SCENE, a_min = 0, a_max = A_MAX)
        #writing the output to geotiff
        AGB_NAME =  OUT_DIR / os.path.basename(SAR_RES_DATA_LIST[::2][NUM]).replace('.tif', '_initial_agb.tif')
        
        # OUT_ARRAY = IN0.GetRasterBand(1).ReadAsArray()
        # OUT_ARRAY[OUT_ARRAY > 0] = W0_HAT_SCENE
        
        # try:
        #     os.remove(AGB_NAME)
        # except OSError:
        #     pass
        # driver = gdal.GetDriverByName("GTiff")
        # ds = driver.CreateCopy(
        #     AGB_NAME,
        #     IN0,
        #     0,
        #     ["COMPRESS=LZW", "PREDICTOR=2"],
        # )
        # ds.GetRasterBand(1).WriteArray(OUT_ARRAY)
        # ds.FlushCache()  # Write to disk.
        # ds = None
        
        # 1. Prepare your array
        OUT_ARRAY = IN0.GetRasterBand(1).ReadAsArray()
        OUT_ARRAY[OUT_ARRAY > 0] = W0_HAT_SCENE
        
        # 2. Create a Workspace in Memory (RAM)
        # This prevents writing a "broken" file to disk
        mem_driver = gdal.GetDriverByName("MEM")
        tmp_ds = mem_driver.Create("", IN0.RasterXSize, IN0.RasterYSize, 1, IN0.GetRasterBand(1).DataType)
        tmp_ds.SetGeoTransform(IN0.GetGeoTransform())
        tmp_ds.SetProjection(IN0.GetProjection())
        
        # 3. Write data to the Memory workspace
        tmp_band = tmp_ds.GetRasterBand(1)
        tmp_band.WriteArray(OUT_ARRAY)
        # Set NoData to NaN (or your preference) to keep stats clean
        tmp_band.SetNoDataValue(0) 
        
        # 4. Define High-Optimization COG Options
        # Use PREDICTOR=3 for Float32, PREDICTOR=2 for Integers (Byte/Int16)
        data_type = tmp_band.DataType
        predictor = "3" if data_type >= 6 else "2"
        
        cog_options = [
            "COMPRESS=DEFLATE",       # Better for cloud storage than LZW
            "LEVEL=9",                # Max compression
            f"PREDICTOR={predictor}", 
            "BLOCKSIZE=512",          # Standard for cloud range-requests
            "BIGTIFF=YES",
            "NUM_THREADS=ALL_CPUS",
            "OVERVIEW_RESAMPLING=AVERAGE"
            ]
        
        # 5. Finalize: Write the COG in one single shot
        try:
            if os.path.exists(AGB_NAME):
                os.remove(AGB_NAME)
        except OSError:
            pass

        cog_driver = gdal.GetDriverByName("COG")
        ds = cog_driver.CreateCopy(AGB_NAME, tmp_ds, options=cog_options)
        
        # 6. Cleanup
        ds = None
        tmp_ds = None
        print(f"Validated COG created: {AGB_NAME}")
        
    
        # plt.plot(W0_TRAIN, W0_HAT_TRAIN, '.')
        
        
    
    
    
    
    
    # # print(f"Retrieved parameters: \n {params[0]}")
    # inparam_file = f"{out_prefix}_initial_quad_model_param.csv"
    # hf = pd.DataFrame(params[:,np.newaxis].T, columns=["a0", "a1", "a2", "a3", "a4"])
    # hf.to_csv(inparam_file)                
    # inparam_test_file = f"{out_prefix}_test_initial_quad_model_param.csv"
    # hf.to_csv(inparam_test_file)    
    
    # ### clean whole image
    # mdl00 = FieldRetrieval(train_file, out_name=out_prefix)
    
    # z1, W1, mask_ws1 = mdl00.inversion_setup(
    #     sar_list, agb_file=agb_file, mask_file=mask_file, n_1=nwin_size
    # )
    # z1_dim = z1.shape
    
    # z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])

    # z = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    
    # W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)

    # z = z.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    # z = np.mean(z, axis=2)
    
    
    # ########## AGB estimation using Sassan Model #######
    # param0w = linear_agb_model(z, params[0], params[1], params[2], params[3], params[4])
    # param0w = param0w[:, np.newaxis]
    
    # mean_hv = np.mean(z, axis=2)
    # if np.any(param0w < 0):
    #     param0w[param0w<0] = (100 * mean_hv[param0w<0]) ** 2
        
    # #writing the output to geotiff
    # in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
    # agb_name = f"{out_prefix}initial_agb.tif"

    # array0 = in0.GetRasterBand(1).ReadAsArray()
    # array0[array0 > 0] = np.mean(param0w, axis=-1)
    # try:
    #     os.remove(agb_name)
    # except OSError:
    #     pass
    # driver = gdal.GetDriverByName("GTiff")
    # ds = driver.CreateCopy(
    #     agb_name,
    #     in0,
    #     0,
    #     ["COMPRESS=LZW", "PREDICTOR=2"],
    # )
    # ds.GetRasterBand(1).WriteArray(array0)
    # ds.FlushCache()  # Write to disk.
    # ds = None

    # hf = pd.DataFrame()
    # hf['Measured_AGB'] =  np.mean(W0, axis=-1)
    # hf['Sassan_AGB'] = np.mean(param0w, axis=-1)
    
    
    # ########## AGB estimation using HV backscatter #######
    # param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, nxn)

    # #writing the output to geotiff
    # in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
    # agb_name = f"{out_prefix}hv.tif"

    # array0 = in0.GetRasterBand(1).ReadAsArray()
    # array0[array0 > 0] = np.mean(param0w, axis=-1)
    # try:
    #     os.remove(agb_name)
    # except OSError:
    #     pass
    # driver = gdal.GetDriverByName("GTiff")
    # ds = driver.CreateCopy(
    #     agb_name,
    #     in0,
    #     0,
    #     ["COMPRESS=LZW", "PREDICTOR=2"],
    # )
    # ds.GetRasterBand(1).WriteArray(array0)
    # ds.FlushCache()  # Write to disk.
    # ds = None
    
    # # Write the values to dataframe
    # hf['HV_AGB'] = np.mean(param0w, axis=-1)
    # inparam_file = f"{out_prefix}initial_agb_estimates.csv"
    # hf.to_csv(inparam_file)
    # print('Done')





def data_clean(X, y, w1_noise=10, w2_noise=20):
    """

    :param X:
    :param y:
    :param w1_noise: absolute noise
    :param w2_noise: relative noise in %
    :return:
    """
    # define noise threshold

    noise_threshold = w1_noise + w2_noise * 0.01 * y

    mdl1 = RandomForestRegressor(
        n_estimators=10,
        max_depth=6,
    )
    mdl2 = KNeighborsRegressor(n_neighbors=12)

    mdl3 = MLPRegressor(
        learning_rate_init=0.01,
        hidden_layer_sizes=(100,)
    )

    mdl_list = [mdl1, mdl2, mdl3]
    noise_list = []
    for mdl in mdl_list:
        estimators = [('scale', StandardScaler()), ('impute', SimpleImputer()), ('learn', mdl)]
        ppl = Pipeline(estimators)
        y_hat = ppl.fit(X, y).predict(X)
        y_c = np.int8(np.abs(y - y_hat) > noise_threshold)
        noise_list.append(y_c)

    noise0 = np.sum(np.stack(noise_list), axis=0)

    # post-preliminary, iteration
    noise = noise0
    noise_pool = []
    iter = 4
    for i in range(iter):
        Xtr = X[noise < 2, :]
        ytr = y[noise < 2]
        noise_list = []
        for mdl in mdl_list:
            estimators = [('scale', StandardScaler()), ('impute', SimpleImputer()), ('learn', mdl)]
            ppl = Pipeline(estimators)
            y_hat = ppl.fit(Xtr, ytr).predict(X)
            y_c = np.int8(np.abs(y - y_hat) > noise_threshold)
            noise_list.append(y_c)
            noise_pool.append(y_c)
        noise = np.sum(np.stack(noise_list), axis=0)

    noise = np.sum(np.stack(noise_pool), axis=0)
    idx = noise < 2 * iter
    X_clean = X[noise < 2 * iter, :]
    y_clean = y[noise < 2 * iter]

    # save results
    return X_clean, y_clean, idx

def data_clean_2(X, y, w1_noise=10, w2_noise=20):
    """

    :param X:
    :param y:
    :param w1_noise: absolute noise
    :param w2_noise: relative noise in %
    :return:
    """
   
    # define noise threshold
    noise_threshold = w1_noise + w2_noise * 0.01 * y

    mdl1 = RandomForestRegressor(
        n_estimators=10,
        max_depth=6,
    )
    mdl2 = KNeighborsRegressor(n_neighbors=12)

    mdl3 = MLPRegressor(learning_rate_init=0.01, hidden_layer_sizes=(100,))

    mdl_list = [mdl1, mdl2, mdl3]
    noise_list = []
    for mdl in mdl_list:
        estimators = [
            ("scale", StandardScaler()),
            ("impute", SimpleImputer()),
            ("learn", mdl),
        ]
        ppl = Pipeline(estimators)
        y_hat = ppl.fit(X, y).predict(X)
        y_c = np.int8(np.abs(y - y_hat) > noise_threshold)
        noise_list.append(y_c)

    noise0 = np.sum(np.stack(noise_list), axis=0)

    # post-preliminary, iteration
    noise = noise0
    noise_pool = []
    iter = 4
    for i in range(iter):
        Xtr = X[noise < 2, :]
        ytr = y[noise < 2]
        noise_list = []
        for mdl in mdl_list:
            estimators = [
                ("scale", StandardScaler()),
                ("impute", SimpleImputer()),
                ("learn", mdl),
            ]
            ppl = Pipeline(estimators)
            y_hat = ppl.fit(Xtr, ytr).predict(X)
            y_c = np.int8(np.abs(y - y_hat) > noise_threshold)
            noise_list.append(y_c)
            noise_pool.append(y_c)
        noise = np.sum(np.stack(noise_list), axis=0)

    noise = np.sum(np.stack(noise_pool), axis=0)
    idx = noise < 2 * iter
    X_clean = X[noise < 2 * iter, :]
    y_clean = y[noise < 2 * iter]
    
    
    
    idx1 = X_clean[:,1::2] < np.mean(X_clean[:,1::2].ravel())+np.std(X_clean[:,1::2] .ravel())
    idx1 = np.all(idx1 == True,axis=1)
    idx[idx] = idx1
    
    X_clean = X_clean[idx1] 
    y_clean = y_clean[idx1] 
    
    
    mean_hv = np.mean(X_clean[:,::2], axis=1)
    mean_hh = np.mean(X_clean[:,1::2], axis=1)
    mean_x = np.stack([mean_hv, mean_hh]).T
        
    # print(mean_x.shape)
    model_00 = Retrieval(2, 1, mean_x.shape[0])
        
    if len(y.shape) == 1:
        y = y_clean[:, None]
               
    W = model_00.x_S(y.T)
        
    x = mean_x.reshape([1, -1])
    bounds = (
            (
                (0.0001, 0.5),
                (0.0001, 0.5),
            )
            + 2 * ((0.01, 0.05),)
            + 2 * ((0.0001, 0.9999),)
            + 2 * ((0.0001, 0.5),)
            + 2 * ((0.0001, 0.8),)
            + (
                (1, 1.0001),
                (0, 25),
            )
            + 1 * ((0.0001, 0.5),)
        )
     
        
    params, y_hat = model_00.model_inverse_03a(x, W, bounds, init_weights=[4, 1])
    
    y_hat_hv = y_hat[:,::2]
    y_hat_hh = y_hat[:,1::2]
    
    diff_hv = np.abs(mean_hv - y_hat_hv)
    diff_hh = np.abs(mean_hh - y_hat_hh)
    
    mean_hh_value = np.mean(diff_hh)
    mean_hv_value = np.mean(diff_hv)
    
    std_hv_value = np.std(diff_hv)
    std_hh_value = np.std(diff_hh)
    
    hh_range = [mean_hh_value - 3 * std_hh_value, mean_hh_value + 3 * std_hh_value]
    hv_range = [mean_hv_value - 3 * std_hv_value, mean_hv_value + 3 * std_hv_value]

    # hh_indx = (diff_hh > hh_range[0]) & (diff_hh < hh_range[1])
    # hv_indx = (diff_hv > hv_range[0]) & (diff_hv < hv_range[1])
    
    hh_indx =  diff_hh < hh_range[1]
    hv_indx =  diff_hv < hv_range[1]
    
    indx =  hh_indx & hv_indx 
    idx[idx] = indx[0]
    
    X_clean = X_clean[indx[0]]
    y_clean = y_clean[indx[0]]
    print(y_clean[y_clean<100].shape,y_clean[y_clean>100].shape )
    
    return X_clean, y_clean, idx

class Retrieval:
    def __init__(self, size_p, size_t, size_s):
        """

        Args:
            size_p: num of pols
            size_t: num of temp. obs.
            size_s: num of spatial obs. (nxn)
        """
        self.size_p = size_p
        self.size_t = size_t
        self.size_s = size_s
        self.param_A = np.array([[0.18, 0.04]])
        self.param_B = np.array([[0.06, 0.12]])
        self.param_C = np.array([[0.4, 0.2]])
        self.param_D = np.array([[10, 1]])
        self.param_a = np.array([[0.14, 0.22]])
        self.param_b = np.array([[1, 1]])
        self.param_c = np.array([[0.4, 0.6]])

    def x_P(self, x):
        # x has size (size_p, )
        return np.tile(x, [1, self.size_t * self.size_s])

    def x_S(self, x):
        # x has size (size_n, size_s)
        return np.repeat(x, self.size_p * self.size_t, axis=1)

    def x_T(self, x):
        # x has size (size_t, )
        return np.tile(np.repeat(x, self.size_p, axis=1), [1, self.size_s])

    # parameter calibration
    def model_03a_fun(self, W, x):
        A = self.x_P(x[0 * self.size_p: 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p: 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p: 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p: 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p: 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p: 6 * self.size_p][None, :])
        S = self.x_T(x[6 * self.size_p: 6 * self.size_p + self.size_t][None, :])

        model_fun = A * W ** a * (1 - np.exp(-B * W)) + (C * W ** c + D) * S * np.exp(
            -B * W
        )
        return model_fun

    def model_inverse_03a(self, y1, W1, bounds, name="tmp", init_weights=[4, 1]):
        """
        Model inversion 03a (A, B, C, D, a, c change with p, S change with t) - given y, W, retrieve A-c
        y = A*W^a (1-exp(-BW)) + (C*W^c + D)*S*exp(-BW)
        p order: HV, HH  --- modified 05/09/2018
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param inc_angle1: Incidence angle in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """

        def model_inversion(instance):
            y = y1[instance, :]
            W = W1[instance, :]

            wt = np.array(init_weights)

            result = differential_evolution(
                lambda x: np.sum(
                    (
                            (
                                    self.model_03a_fun(W, x).reshape(-1, self.size_p)
                                    - y.reshape(-1, self.size_p)
                            )
                            * wt[-self.size_p:]
                    )
                    ** 2
                ),
                bounds,
            )
            param0 = result.x

            y_hat0 = self.model_03a_fun(W, param0)
            return y_hat0, param0

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat
    
    # Inversion with only SAR
    def sar_model_03_fun(self, x):
        A = self.x_P(x[0 * self.size_p: 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p: 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p: 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p: 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p: 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p: 6 * self.size_p][None, :])
        S = self.x_T(x[6 * self.size_p: 6 * self.size_p + self.size_t][None, :])
        W = self.x_S(
            x[
            6 * self.size_p
            + self.size_t: 6 * self.size_p
                           + self.size_t
                           + self.size_s
            ][None, :]
        )

        model_fun = A * W ** a * (1 - np.exp(-B * W)) + (C * W ** c + D) * S * np.exp(
            -B * W
        )
        return model_fun

    def sar_model_inverse_03(self, param0, y1, bounds, name="tmp", init_weights=[4, 1]):
        """
        Model inversion 03a (A, B, C, D, a, c change with p, S change with t) - given y, W, retrieve A-c
        y = A*W^a (1-exp(-BW)) + (C*W^c + D)*S*exp(-BW)
        p order: HV, HH
        5x5 approach - parameter being all LOCAL
        --- modified 05/10/2018
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param inc_angle1: Incidence angle in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """

        param0w = param0
        bounds0w = bounds

        def model_inversion(instance):
            y = y1[instance, :]
            param0aw = param0w[instance, :]
            bounds0aw = bounds0w[instance]

            wt = np.array(init_weights)
            result = minimize(
                lambda x: np.sum(
                    (
                            (
                                    self.sar_model_03_fun(x).reshape(-1, self.size_p)
                                    - y.reshape(-1, self.size_p)
                            )
                            * wt[-self.size_p:]
                    )
                    ** 2
                ),
                param0aw,
                method="L-BFGS-B",
                bounds=bounds0aw,
            )
            # result = differential_evolution(lambda x: np.sum(
            #     ((self.model_03a_fun(W, x).reshape(-1, self.size_p) - y.reshape(-1, self.size_p)) * wt) ** 2
            # ), bounds)
            # print(instance)
            # print(result.success)
            param1 = result.x

            y_hat1 = self.sar_model_03_fun(param1)

            return y_hat1, param1

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat
    
    def optimize_two_stage_worker(self, task_data):
        """
        Handles the inversion for a single pixel.
        This is the function executed by the worker processes.
        """
        # NOW: bounds_cfg is passed in!
        y_obs, p0_full, weights, bounds_cfg = task_data
        
        # bounds_cfg mapping: 
        # [0:6] -> A,B,C | [6:9] -> D,S | [9] -> W
    
        # --- STAGE 1: RETRIEVE BIOMASS (W) ---
        p0_w = p0_full[13:14]
        w_mult = bounds_cfg[9] # e.g., (0.2, 1.8)
        bounds_w = [(p0_w[0] * w_mult[0], p0_w[0] * w_mult[1])] 
        
        def obj_stage1(w_val):
            x_recon = np.concatenate([p0_full[:13], w_val])
            y_hat = self.sar_model_03_fun(x_recon).reshape(-1, self.size_p)
            return np.sum((( 10*np.log10(y_hat) - 10*np.log10(y_obs.reshape(-1, self.size_p)) ) * weights)**2)

        res_w = minimize(obj_stage1, p0_w, method='L-BFGS-B', bounds=bounds_w)
        w_curr = res_w.x
        
        # --- STAGE 2: RETRIEVE SOIL (S) & COEFFICIENTS ---
        p0_s2 = np.concatenate([p0_full[0:6], p0_full[10:13]])
            
        # Combine bounds for Stage 2 (A,B,C + D,S)
        # This grabs the specific multipliers you provided for those 9 active params
        active_mults = bounds_cfg[0:6] + bounds_cfg[6:9]
        bounds_s2 = [(p0_s2[i] * m[0], p0_s2[i] * m[1]) for i, m in enumerate(active_mults)]
        
        def obj_stage2(x9):
            x_recon = np.concatenate([x9[:6], p0_full[6:10], x9[6:], w_curr])
            y_hat = self.sar_model_03_fun(x_recon).reshape(-1, self.size_p)
            return np.sum(((10*np.log10(y_hat) - 10*np.log10(y_obs.reshape(-1, self.size_p))) * weights)**2)
        
        res_s2 = minimize(obj_stage2, p0_s2, method='L-BFGS-B', bounds=bounds_s2)
        
        final_p = np.concatenate([res_s2.x[:6], p0_full[6:10], res_s2.x[6:], w_curr])
        return final_p, self.sar_model_03_fun(final_p).flatten()
    
    def sar_model_inverse_03_v2(self, param0c, z, bounds_cfg, init_weights=[4, 1]):
        """
        MODIFIED: Now accepts bounds_cfg and passes it to the parallel workers.
        """
        n_samples = z.shape[0]
        wt = np.array(init_weights)
        
        # --- THE CRITICAL CHANGE ---
        # We add bounds_cfg to the tuple so each worker knows its specific constraints.
        tasks = [(z[i], param0c[i], wt, bounds_cfg) for i in range(n_samples)]
        
        # ProcessPoolExecutor distributes these tasks across your CPU cores
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Each worker now receives (z, p0, wt, bounds_cfg)
            results = list(executor.map(self.optimize_two_stage_worker, tasks))

        params, y_hat = zip(*results)
        return np.array(params), np.array(y_hat)
    
    
    
    
def volume(x, A, B, a):
    return A * pow(x, a) * (1 - np.exp(-B * x)) 

def double(x, B, C, d, S):
    return C * S * pow(x, d) * np.exp(-B * x)

def surface(x, B, D, S):
    return D * S * np.exp(-B * x)

def nisar(x, A, B, C,  a, d, D, S):
    return  volume(x, A, B, a) + double(x, B, C, d, S) + surface(x, B, D, S)    
    
    
def nisar_law_model(parameters, *data):
    #we have 2 parameters which will be passed as parameters and W 
    # Model parameters
    A, B, C, a, d, D, S = parameters
    # Biomass 
    W = data[0]
 
    return nisar(W, A, B, C,  a, d, D, S)



def nisar_model_objective(parameters, *data):

    #we have 4 parameters which will be passed as parameters and
    #"experimental" x,y which will be passed as data
    
    # Model parameters
    AHV, AHH, BHV, BHH, CHV, CHH, alphaHV, alphaHH,  deltaHV, deltaHH, DHV, DHH, S = parameters
    
    # Backscatter and BIOMASS  
    Y_HV, Y_HH, W = data
    
    Y_PRED_HV = nisar_law_model([AHV, BHV, CHV, alphaHV, deltaHV, DHV, S], W)
    Y_PRED_HH = nisar_law_model([AHH, BHH, CHH, alphaHH, deltaHH, DHH, S], W)
    
    
    # Constraint 1: x1 + x2 >= 3
    if not (AHH - AHV > 0):
        return 10e20 
    # Constraint 1: x1 + x2 >= 3
    if not (CHH - CHV > 0):
        return 10e20 

    # Constraint 1: x1 + x2 >= 3
    if not (DHH - DHV > 0):
        return 10e20 
    
    # Constraint 1: x1 + x2 >= 3
    if not (BHV - BHH > 0):
        return 10e20 
    
    # Constraint 1: x1 + x2 >= 3
    if not (alphaHV - alphaHH > 0):
        return 10e20 
    
    # Constraint 1: x1 + x2 >= 3
    if not (deltaHH - deltaHV > 0):
        return 10e20 
    
     
    if (AHH - AHV > 0) & (alphaHV - alphaHH > 0) & (CHH - CHV > 0) & (DHH - DHV > 0) & (BHV - BHH > 0)  &  (deltaHH - deltaHV > 0): #&  (AHH - CHH > 0) & (AHV - CHV > 0) & (deltaHH - alphaHH > 0) & (deltaHV - alphaHV > 0)
        return np.sum( (4 * pow((10*np.log10(Y_PRED_HV) - 10*np.log10(Y_HV)), 2)) + (pow((10*np.log10(Y_PRED_HH) - 10*np.log10(Y_HH)),2)))    
    

    
class FieldRetrieval:
    def __init__(self, z0_file, valid_min=0, out_name="tmp"):
        """

        :param z0_file:
        :param z1_file:
        :param valid_min:
        :param out_name:
        """
        self.mask_file = z0_file
        self.valid_min = valid_min
        self.out_name = out_name
        self.out_agb = "{}_agb_0.tif".format(self.out_name)

    def data_cleaner(
            self, out_radar_list, agb_file=None, mask_file=None, w1_noise=10, w2_noise=20
    ):
        """

        :param out_radar_list: time series of HV/HH files (hv_t0, hh_t0, hv_t1, hh_t1, ...)
        :param agb_file: agb measurements, if any
        :param mask_file:
        :param n_1:
        :return:
        """
        if mask_file is None:
            mask_file = self.mask_file
        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        m_t = len(out_radar_list) // 2
        df_alos = pp.array_reshape_rolling(
            out_radar_list,
            mask_file,
            name=f"tmp/tmp_alos_{os.path.basename(self.out_name)}",
            m=0,
            n=0,
            valid_min=self.valid_min,
        )
        # print(df_alos.iloc[:, 1:])
        z_alos = df_alos.iloc[:, 1:].values
        if str(out_radar_list[0])[:3] == "./A":
            z_alos = z_alos ** 2 / 199526231
        z1 = z_alos.reshape([-1, m_t * 2])

        if agb_file is not None:
            df_agb = pp.array_reshape_rolling(
                [agb_file],
                mask_file,
                name=f"tmp/tmp_w_{os.path.basename(self.out_name)}",
                m=0,
                n=0,
                valid_min=self.valid_min,
            )
            W1 = df_agb.iloc[:, 1].values
        else:
            W1 = None

        x0, y0, idx = data_clean(
            z1, W1, w1_noise=w1_noise, w2_noise=w2_noise
        )

        in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        lc = array0[array0 > self.valid_min]
        lc[~idx] = self.valid_min
        array0[array0 > self.valid_min] = lc

        # mask_ws0 = "{}_mask_ws0.tif".format(self.out_name)
        # driver = gdal.GetDriverByName("GTiff")
        # ds = driver.CreateCopy(mask_ws0, in0, 0, ["COMPRESS=LZW", "PREDICTOR=2"])
        # ds.GetRasterBand(1).WriteArray(array0)
        # ds.FlushCache()  # Write to disk.
        # ds = None
        # in0 = None
        
        mask_ws0 = "{}_mask_ws0.tif".format(self.out_name)

        # 1. Prepare your data in a Memory Dataset first
        mem_driver = gdal.GetDriverByName("MEM")
        tmp_ds = mem_driver.Create("", in0.RasterXSize, in0.RasterYSize, 1, in0.GetRasterBand(1).DataType)
        tmp_ds.SetGeoTransform(in0.GetGeoTransform())
        tmp_ds.SetProjection(in0.GetProjection())
        
        # 2. Apply your modifications here
        tmp_band = tmp_ds.GetRasterBand(1)
        tmp_band.WriteArray(array0)
        tmp_band.SetNoDataValue(0) # Or nan
        
        # 3. Finalize: Write to COG in one single operation
        cog_driver = gdal.GetDriverByName("COG")
        cog_options = [
            "COMPRESS=LZW",
            "PREDICTOR=2",
            "BLOCKSIZE=512",
            "NUM_THREADS=ALL_CPUS"
            ]
        
        # This creates a perfectly optimized file that will NOT trigger the warning
        final_ds = cog_driver.CreateCopy(mask_ws0, tmp_ds, options=cog_options)
        
        # Cleanup
        final_ds = None
        tmp_ds = None
            
        return x0, y0, mask_ws0
    
    
    def data_cleaner_2(
        self, out_radar_list, agb_file=None, mask_file=None, w1_noise=10, w2_noise=20
    ):
        """

        :param out_radar_list: time series of HV/HH files (hv_t0, hh_t0, hv_t1, hh_t1, ...)
        :param agb_file: agb measurements, if any
        :param mask_file:
        :param n_1:
        :return:
        """
        if mask_file is None:
            mask_file = self.mask_file
        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        m_t = len(out_radar_list) // 2
        df_alos = pp.array_reshape_rolling(
            out_radar_list,
            mask_file,
            name=f"tmp/tmp_alos_{os.path.basename(self.out_name)}",
            m=0,
            n=0,
            valid_min=self.valid_min,
        )
        # print(df_alos.iloc[:, 1:])
        z_alos = df_alos.iloc[:, 1:].values
        if out_radar_list[0][:3] == "./A":
            z_alos = z_alos**2 / 199526231
        z1 = z_alos.reshape([-1, m_t * 2])

        if agb_file is not None:
            df_agb = pp.array_reshape_rolling(
                [agb_file],
                mask_file,
                name=f"tmp/tmp_w_{os.path.basename(self.out_name)}",
                m=0,
                n=0,
                valid_min=self.valid_min,
            )
            W1 = df_agb.iloc[:, 1].values
        else:
            W1 = None

        x0, y0, idx = data_clean_2(z1, W1, w1_noise=w1_noise, w2_noise=w2_noise)

        in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        lc = array0[array0 > self.valid_min]
        lc[~idx] = self.valid_min
        array0[array0 > self.valid_min] = lc

        mask_ws0 = "{}_mask_ws0.tif".format(self.out_name)
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(mask_ws0, in0, 0, ["COMPRESS=LZW", "PREDICTOR=2"])
        ds.GetRasterBand(1).WriteArray(array0)
        ds.FlushCache()  # Write to disk.
        ds = None
        in0 = None

        return x0, y0, mask_ws0
    
     
    
    
    
    def inversion_setup(self, out_radar_list, agb_file=None, mask_file=None, n_1=0):
        """
        set up the inversion of AGB retrieval.
        :param out_radar_list: time series of HV/HH files (hv_t0, hh_t0, hv_t1, hh_t1, ...)
        :param agb_file: agb measurements, if any
        :param mask_file:
        :param n_1:
        :return:
            z1 (n_obs, nxn, n_time, 2_pols),
            W1 (if agb exists, central pixel of nxn),
            mask_ws1 (dilated mask depending on nxn)
        """
        if mask_file is None:
            mask_file = self.mask_file
        n_2 = n_1 * 2 + 1
        in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        struct2 = ndimage.generate_binary_structure(2, 2)
        array0 = ndimage.binary_dilation(
            array0, structure=struct2, iterations=n_2
        ).astype(array0.dtype)
        # mask_ws1 = "{}_mask_ws1.tif".format(self.out_name)
        # driver = gdal.GetDriverByName("GTiff")
        # ds = driver.CreateCopy(mask_ws1, in0, 0, ["COMPRESS=LZW", "PREDICTOR=2"])
        # ds.GetRasterBand(1).WriteArray(array0)
        # ds.FlushCache()  # Write to disk.
        # ds = None
        # in0 = None
        
        
        mask_ws1 = "{}_mask_ws1.tif".format(self.out_name)

        # 1. Prepare your data in a Memory Dataset first
        mem_driver = gdal.GetDriverByName("MEM")
        tmp_ds = mem_driver.Create("", in0.RasterXSize, in0.RasterYSize, 1, in0.GetRasterBand(1).DataType)
        tmp_ds.SetGeoTransform(in0.GetGeoTransform())
        tmp_ds.SetProjection(in0.GetProjection())
        
        # 2. Apply your modifications here
        tmp_band = tmp_ds.GetRasterBand(1)
        tmp_band.WriteArray(array0)
        tmp_band.SetNoDataValue(0) # Or nan
        
        # 3. Finalize: Write to COG in one single operation
        cog_driver = gdal.GetDriverByName("COG")
        cog_options = [
            "COMPRESS=LZW",
            "PREDICTOR=2",
            "BLOCKSIZE=512",
            "NUM_THREADS=ALL_CPUS"
            ]
        # This creates a perfectly optimized file that will NOT trigger the warning
        final_ds = cog_driver.CreateCopy(mask_ws1, tmp_ds, options=cog_options)
        
        # Cleanup
        final_ds = None
        tmp_ds = None
        
        
        

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        m_t = len(out_radar_list) // 2
        nxn = (n_1 * 2 + 1) ** 2
        df_alos = pp.array_reshape_rolling(
            out_radar_list,
            mask_ws1,
            name=f"tmp/tmp_alos_{os.path.basename(self.out_name)}",
            m=n_1,
            n=n_1,
            valid_min=self.valid_min,
        )
        z_alos = df_alos.iloc[:, 1:].values
        if str(out_radar_list[0])[:3] == "./A":
            z_alos = z_alos ** 2 / 199526231
        z1 = z_alos.reshape([-1, nxn, m_t, 2])

        if agb_file is not None:
            df_agb = pp.array_reshape_rolling(
                [agb_file],
                mask_ws1,
                name=f"tmp/tmp_w_{os.path.basename(self.out_name)}",
                m=n_1,
                n=n_1,
                valid_min=self.valid_min,
            )
            W1 = df_agb.iloc[:, 1:].values
            W1 = W1.reshape((-1, nxn))[:, -nxn // 2][:, None]
        else:
            W1 = None

        return z1, W1, mask_ws1

    def inversion_return_valid(self, W1, mask_ws1, mask_file=None):
        """
        reshape n_obs -> n_obs2 for valid pixels using the mask file
        :param W1: reshaped input signal (n_obs, bands)
        :param mask_ws1: mask from inversion_setup (# of valid pixels: n_obs)
        :param mask_file: original mask file (# of valid pixels: n_obs2)
        :return: W0 following original mask (n_obs2, bands)
        """
        if mask_file is None:
            mask_file = self.mask_file

        in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        in1 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        array1 = in1.GetRasterBand(1).ReadAsArray()

        if len(W1.shape) == 1:
            W1 = W1[:, None]
        print(W1.shape)

        W0 = []
        for i in range(W1.shape[1]):
            array0 = in0.GetRasterBand(1).ReadAsArray().astype(float)
            array0[array0 > self.valid_min] = W1[:, i]
            W = array0[array1 > self.valid_min]
            W0.append(W)
        W0 = np.array(W0).T
        # print(W0.shape)

        in0 = None
        in1 = None

        return W0
    
    def params_calibration(self, z1, W1, out_prefix, x_range, init_weights=[4, 1]):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_hh = np.nanmean(z1[:, :, :, 1].reshape([z1.shape[0], -1]), axis=1)
        
        
        # Set font properties globally
        sns.set_style("white")
        plt.rcParams['font.family'] = 'serif'  # Options: 'serif', 'sans-serif', 'monospace', etc.
        plt.rcParams['font.serif'] = ['DejaVu Serif']  # Specify serif fonts
        plt.rcParams['font.size'] = 10  # Set default font size
        fig, axs = plt.subplots(1, 2, figsize = (9, 3), facecolor='white')

         
        
        density =  calculate_density(W1[:, 0], mean_hv)
        axs[0].grid(which='both', linestyle='--', linewidth=0.7)  # Show grid lines
        axs[0].scatter(W1, mean_hv, c = density, cmap='viridis', s=5, edgecolor=None)
        axs[0].set_xlabel('LIDAR AGB (Mg/ha)')
        axs[0].set_ylabel('Mean HV Bacscatter (m2/m2)')
            
         
        density =  calculate_density(W1[:, 0], mean_hh)
        axs[1].grid(which='both', linestyle='--', linewidth=0.7)  # Show grid lines
        axs[1].scatter(W1, mean_hh, c = density, cmap='viridis', s=5, edgecolor=None)
        axs[1].set_xlabel('LIDAR AGB (Mg/ha)')
        axs[1].set_ylabel('Mean HH Bacscatter (m2/m2)')
        
        plt.subplots_adjust(wspace=0.07, hspace=0.09)  # Adjust these values as needed           
        plt.tight_layout()
        
        plt.savefig(out_prefix + 'agb_vs_backscatter.png', dpi=600)
        plt.show()
        plt.close()
        
        
        mean_x = np.stack([mean_hv, mean_hh]).T
        # print(mean_x.shape)
        model_00 = Retrieval(2, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
            
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        
        
        bounds = (
                ((0.0001, float(np.amax(mean_hv))), (0.0001, float(np.amax(mean_hh))),)
                + 2 * ((0.009, 0.05),)
                +  ((0.0001, float(np.amax(mean_hv))), (0.0001, float(np.amax(mean_hh))),)
                + 2 * ((0.6, 0.9),)
                + 2 * ((0.1, 1.5),)
                + ((1, 1.0001), (0, float(np.nanmax(mean_hh)/float(np.nanmin(mean_hv)))),)
                + 1 * ((float(np.nanmin(mean_hv)), float(np.nanmin(mean_hv))+1),)
        )
        
        result = differential_evolution(
                                        func=nisar_model_objective,
                                        bounds=bounds,
                                        args=(mean_hv, mean_hh, W1[:, 0]),
                                        popsize=50,  # A larger population is better for a complex landscape
                                        maxiter=10000, # More iterations for a harder search
                                        disp=True,
                                        seed=42
                                        )   
        
        params = result.x
        
        Y_HAT_HV = nisar_law_model([params[0], params[2], params[4], params[6], params[8], params[10], params[12]], W1[:, 0]) 
        Y_HAT_HH = nisar_law_model([params[1], params[3], params[5], params[7], params[9], params[11], params[12]], W1[:, 0]) 
        
        # print(y.shape)
        # print(W.shape)
        # params, y_hat = model_00.model_inverse_03a(y, W, bounds, init_weights=[4, 1])
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
                          "A_HV",
                          "A_HH",
                          "B_HV",
                          "B_HH",
                          "C_HV",
                          "C_HH",
                          "alpha_HV",
                          "alpha_HH",
                          "gamma_HV",
                          "gamma_HH",
                          "D_HV",
                          "D_HH",
                      ] + ["S"]

        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        
        df = pd.DataFrame(params[:, None].T, columns=param_names)
        df.to_csv(param0_file)
         
        density_scatter_plot(out_prefix + '_hh_',
            mean_hh.flatten(),
            Y_HAT_HH.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 1),
            y_limit=(0, 1),
        )
        
        density_scatter_plot(out_prefix + '_hv_',
            mean_hv.flatten(),
            Y_HAT_HV.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 0.4),
            y_limit=(0, 0.4),
        )
        
        
        
        plot_nisar_curve(mean_x, W1, out_prefix, params, x_range)
        return param0_file
    
    def inversion_recursive_ws_v3(self, z1, w1, mask_ws1, param0_file=None):
        """

        Args:
            z1: input signal (n_obs, nxn, n_time, 2_pols)
            mask_ws1: raster mask file used to define valid pixels (# of valid: n_obs)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        kn = 10
        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1:].values

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_1 = (np.sqrt(nxn) - 1) // 2
        in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        W_mean = []
        S_mean = []
        for i in range(m_t):
            print("Scene {} of {}".format(i + 1, m_t))
            z = z1[:, :, i, :]
            mean_hv = z[:, :, 0]
            param0w = w1[:, i]            
            param0w = param0w[:, np.newaxis]
            
            param0c = np.tile(param0, (n_obs, 1))
            param0c = np.concatenate((param0c, param0w), axis=1)

            for k in range(kn):
                bound0c = [
                    [(w0 - 0.0001, w0 + 0.0001) for w0 in param0c[iw, : 13 + nxn // 2]]
                    + [
                        (w0 * 0.2, w0 * 1.8)
                        for w0 in param0c[iw, 13 + nxn // 2: 13 + nxn // 2 + 1]
                    ]
                    + [
                        (w0 - 0.0001, w0 + 0.0001)
                        for w0 in param0c[iw, 13 + nxn // 2 + 1:]
                    ]
                    for iw in range(param0c.shape[0])
                ]
                
                model_00 = Retrieval(2, 1, nxn)
                params, z_hat = model_00.sar_model_inverse_03(param0c, z, bound0c)
                w0 = params[:, -nxn // 2]
                w0sum = np.sqrt(np.mean((w0 - param0c[:, -nxn // 2]) ** 2))

                array0 = in0.GetRasterBand(1).ReadAsArray()
                array0[array0 > self.valid_min] = w0
                if not os.path.exists("tmp"):
                    os.mkdir("tmp")
                try:
                    os.remove("tmp/tmp_agb_{}_{}_{}.tif".format(os.path.basename(self.out_name), k, i))
                except OSError:
                    pass
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.CreateCopy(
                    "tmp/tmp_agb_{}_{}_{}.tif".format(os.path.basename(self.out_name), k, i),
                    in0,
                    0,
                    ["COMPRESS=LZW", "PREDICTOR=2"],
                )
                ds.GetRasterBand(1).WriteArray(array0)
                ds.FlushCache()  # Write to disk.
                ds = None

                df_agb = pp.array_reshape_rolling(
                    ["tmp/tmp_agb_{}_{}_{}.tif".format(os.path.basename(self.out_name), k, i)],
                    mask_ws1,
                    name="tmp/tmp_agb_nxn_{}_{}_{}".format(os.path.basename(self.out_name), k, i),
                    m=n_1,
                    n=n_1,
                    valid_min=self.valid_min,
                )
                param0w = df_agb.iloc[:, 1:].values
                param0w[np.isnan(param0w)] = (100 * mean_hv[np.isnan(param0w)]) ** 2

                param0c = np.concatenate((param0c[:, :13], param0w), axis=1)

                """
                Update Parameters...
                """
                bound0c = [
                    [(w0 - 0.0001, w0 + 0.0001) for w0 in param0c[iw, :6]]
                    # [(w0 * 0.8, w0 * 1.2) for w0 in param0c[iw, :6]]
                    + [(w0 - 0.0001, w0 + 0.0001) for w0 in param0c[iw, 6:12]]
                    + [(w0 * 0.6, w0 * 1.4) for w0 in param0c[iw, 12:13]]
                    + [(w0 - 0.0001, w0 + 0.0001) for w0 in param0c[iw, 13:]]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, nxn)
                param0c, y_hat = model_00.sar_model_inverse_03(param0c, z, bound0c)
                s0 = params[:, 12]
                # density_scatter_plot(
                #     z.flatten(),
                #     y_hat.flatten(),
                #     x_label="Measured Backscatter",
                #     y_label="Predicted Backscatter",
                #     x_limit=(0, 0.6),
                #     y_limit=(0, 0.6),
                #     file_name=(self.out_name + "_ws1_y_s{}_k{}.png").format(i, k),
                # )

                # print(f'k = {k}, w_res = {w0sum}')
                if w0sum < 1:
                    # print(f"early stop of w0 at k = {k}")
                    break

            W_mean.append(w0)
            S_mean.append(s0)

        W_mean = np.array(W_mean).T
        S_mean = np.array(S_mean).T
        print("Dimension of W: {}".format(W_mean.shape))  # should be (n_obs, m_t)

        in0 = None

        return W_mean, S_mean
    
    
    def inversion_recursive_ws_v4(self, z1, w1, mask_ws1, param0_file=None):
        """

        Args:
            z1: input signal (n_obs, nxn, n_time, 2_pols)
            mask_ws1: raster mask file used to define valid pixels (# of valid: n_obs)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """
        
        # User-defined slack multipliers
        bounds_cfg = [
            (0.8, 1.2), (0.8, 1.2), # A
            (0.8, 1.2), (0.8, 1.2), # B
            (0.8, 1.2), (0.8, 1.2), # C
            (0.8, 1.2), (0.8, 1.2), # D
            (0.4, 1.6),             # S
            (0.2, 1.8)              # W
            ]

        kn = 10  # Max recursive iterations
        
        # Load initial coefficients from CSV
        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = f"{in_path}/model_sim_param0_s13.csv"
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1:].values  # Standard model coefficients
        
        
    
        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        model_00 = Retrieval(2, 1, nxn)
        
        # n_1 = (np.sqrt(nxn) - 1) // 2
        # in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        W_mean = []
        S_mean = []
        
        for i in range(m_t):
            print(f"\n--- Processing Scene {i+1} of {m_t} ---")
            z_scene = z1[:, :, i, :]
            param0w = w1[:, i][:, np.newaxis]
            param0c = np.tile(param0, (n_obs, 1))
            param0c = np.concatenate((param0c, param0w), axis=1)
            
            
        
        
            for k in range(kn):
                params, y_hat = model_00.sar_model_inverse_03_v2(param0c, z_scene, bounds_cfg)
                w_new = params[:, -1]
                w0sum = np.sqrt(np.mean((w_new - param0c[:, -1])**2))
                param0c = params.copy()
                # Update for next iteration
                
                # density_scatter_plot((self.out_name + "_ws1_y_s{}_k{}.png").format(i, k),
                #     z_scene.flatten(),
                #     y_hat.flatten(),
                #     x_label="Measured Backscatter",
                #     y_label="Predicted Backscatter",
                #     x_limit=(0, 0.6),
                #     y_limit=(0, 0.6),
                # )
                
                
                print(f'k = {k}, w_res = {w0sum}')
                
                if w0sum < 1.0: break
            
            W_mean.append(w_new); S_mean.append(params[:, 12])

        W_mean = np.array(W_mean).T
        S_mean = np.array(S_mean).T
        print("Dimension of W: {}".format(W_mean.shape))  # should be (n_obs, m_t)

 
        return W_mean, S_mean
    

    
    
# scatter 0
def scatter_plot_radar_agb(wd, mask_file, agb_file, sar_list, out_name,  x_txt="AGB (Mg/ha)"):
    os.chdir(wd)

    w1_slider = widgets.FloatSlider(
        value=5.0,
        min=0.0,
        max=100.0,
        step=5,
        description=f"Absolute Noise in {x_txt}: ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )
    w2_slider = widgets.FloatSlider(
        value=5.0,
        min=0.0,
        max=100.0,
        step=5,
        description=f"Relative Noise (%) in {x_txt}: ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(w1_noise=20, w2_noise=20):
        out_prefix = f"{out_name}"
        mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
        x0, y0, mask0 = mdl00.data_cleaner(
            sar_list,
            agb_file=agb_file,
            mask_file=mask_file,
            w1_noise=w1_noise,
            w2_noise=w2_noise,
        )
        # print(mask0)
        x0 = x0.reshape([y0.shape[0], -1, 2])
        x0v = x0[:, :, 0]
        x0h = x0[:, :, 1]

        # r2 = r2_score(
        #     scipy.stats.zscore(y0[:, None]),
        #     scipy.stats.zscore(np.mean(x0v, axis=1)[:, None]),
        # )
        # print("Variance score 1: {:.2f}".format(r2))
        # rmse = np.sqrt(
        #     mean_squared_error(
        #         scipy.stats.zscore(y0[:, None]),
        #         scipy.stats.zscore(np.mean(x0v, axis=1)[:, None]),
        #     )
        # )
        # print("RMSE: {:.5f}".format(rmse))
        # r2 = r2_score(
        #     scipy.stats.zscore(y0[:, None]),
        #     scipy.stats.zscore(np.mean(x0h, axis=1)[:, None]),
        # )
        # print("Variance score 1: {:.2f}".format(r2))
        # rmse = np.sqrt(
        #     mean_squared_error(
        #         scipy.stats.zscore(y0[:, None]),
        #         scipy.stats.zscore(np.mean(x0h, axis=1)[:, None]),
        #     )
        # )
        # print("RMSE: {:.5f}".format(rmse))

        df = pd.DataFrame({'AGB_measured': y0.ravel(),
                           'VH_mean': np.nanmean(x0v, axis=-1),
                           'VV_mean': np.nanmean(x0h, axis=-1)})
        df.to_csv(f"{out_prefix}_mean_measured.csv")
        return y0, x0v, x0h

    y0, x0v, x0h = update_numbers()

    trace1 = go.Scatter(
        x=y0,
        y=np.nanmean(x0v, axis=1),
        mode="markers",
        name="HV",
        error_y=dict(
            type="data", array=np.nanstd(x0v, axis=1), thickness=0.5, width=2,
        ),
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=y0,
        y=np.nanmean(x0h, axis=1),
        mode="markers",
        name="HH",
        error_y=dict(
            type="data",
            array=np.nanstd(x0h, axis=1),
            thickness=0.5,
            width=2,
            color="rgb(255, 127, 14)",
        ),
        marker=dict(size=10, color="rgb(255, 127, 14)", ),
    )
    layout1 = go.Layout(
        title=f"HV vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter HV",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    layout2 = go.Layout(
        title=f"HH vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter HH",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )

    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    label1 = widgets.Label(f"Number of Valid Obs. ({y0.shape[0]})")

    def response(change):
        y0, x0v, x0h = update_numbers(w1_slider.value, w2_slider.value)
        with g1.batch_update():
            g1.data[0].x = y0
            g1.data[0].y = np.nanmean(x0v, axis=1)
            g1.data[0].error_y = dict(
                type="data", array=np.nanstd(x0v, axis=1), thickness=0.5, width=2,
            )
        with g2.batch_update():
            g2.data[0].x = y0
            g2.data[0].y = np.nanmean(x0h, axis=1)
            g2.data[0].error_y = dict(
                type="data",
                array=np.nanstd(x0h, axis=1),
                thickness=0.5,
                width=2,
                color="rgb(255, 127, 14)",
            )
        label1.value = f"Number of Valid Obs. ({y0.shape[0]})"

    w1_slider.observe(response, names="value")
    w2_slider.observe(response, names="value")

    container4 = widgets.HBox([g1, g2])
    app = widgets.VBox([w1_slider, w2_slider, label1, container4])

    return app


# scatter 0
def scatter_plot_radar_agb_v2(wd, mask_file, agb_file, sar_list, out_name, x_txt="AGB (Mg/ha)"):
    os.chdir(wd)

    w1_slider = widgets.FloatSlider(
        value=5.0,
        min=0.0,
        max=100.0,
        step=5,
        description=f"Absolute Noise in {x_txt}: ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )
    w2_slider = widgets.FloatSlider(
        value=5.0,
        min=0.0,
        max=100.0,
        step=5,
        description=f"Relative Noise (%) in {x_txt}: ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(w1_noise=10, w2_noise=10):
        if not os.path.exists("output"):
            os.mkdir("output")
        out_prefix = f"output/{out_name}"
        mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
        x0, y0, mask0 = mdl00.data_cleaner_2(
            sar_list,
            agb_file=agb_file,
            mask_file=mask_file,
            w1_noise=w1_noise,
            w2_noise=w2_noise,
        )
        # print(mask0)
        x0 = x0.reshape([y0.shape[0], -1, 2])
        x0v = x0[:, :, 0]
        x0h = x0[:, :, 1]

        # r2 = r2_score(
        #     scipy.stats.zscore(y0[:, None]),
        #     scipy.stats.zscore(np.mean(x0v, axis=1)[:, None]),
        # )
        # print("Variance score 1: {:.2f}".format(r2))
        # rmse = np.sqrt(
        #     mean_squared_error(
        #         scipy.stats.zscore(y0[:, None]),
        #         scipy.stats.zscore(np.mean(x0v, axis=1)[:, None]),
        #     )
        # )
        # print("RMSE: {:.5f}".format(rmse))
        # r2 = r2_score(
        #     scipy.stats.zscore(y0[:, None]),
        #     scipy.stats.zscore(np.mean(x0h, axis=1)[:, None]),
        # )
        # print("Variance score 1: {:.2f}".format(r2))
        # rmse = np.sqrt(
        #     mean_squared_error(
        #         scipy.stats.zscore(y0[:, None]),
        #         scipy.stats.zscore(np.mean(x0h, axis=1)[:, None]),
        #     )
        # )
        # print("RMSE: {:.5f}".format(rmse))

        df = pd.DataFrame({'AGB_measured': y0.ravel(),
                           'VH_mean': np.nanmean(x0v, axis=-1),
                           'VV_mean': np.nanmean(x0h, axis=-1)})
        df.to_csv(f"{out_prefix}_mean_measured.csv")
        return y0, x0v, x0h

    y0, x0v, x0h = update_numbers()

    trace1 = go.Scatter(
        x=y0,
        y=np.nanmean(x0v, axis=1),
        mode="markers",
        name="HV",
        error_y=dict(
            type="data", array=np.nanstd(x0v, axis=1), thickness=0.5, width=2,
        ),
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=y0,
        y=np.nanmean(x0h, axis=1),
        mode="markers",
        name="HH",
        error_y=dict(
            type="data",
            array=np.nanstd(x0h, axis=1),
            thickness=0.5,
            width=2,
            color="rgb(255, 127, 14)",
        ),
        marker=dict(size=10, color="rgb(255, 127, 14)", ),
    )
    layout1 = go.Layout(
        title=f"HV vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter HV",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    layout2 = go.Layout(
        title=f"HH vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter HH",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )

    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    label1 = widgets.Label(f"Number of Valid Obs. ({y0.shape[0]})")

    def response(change):
        y0, x0v, x0h = update_numbers(w1_slider.value, w2_slider.value)
        with g1.batch_update():
            g1.data[0].x = y0
            g1.data[0].y = np.nanmean(x0v, axis=1)
            g1.data[0].error_y = dict(
                type="data", array=np.nanstd(x0v, axis=1), thickness=0.5, width=2,
            )
        with g2.batch_update():
            g2.data[0].x = y0
            g2.data[0].y = np.nanmean(x0h, axis=1)
            g2.data[0].error_y = dict(
                type="data",
                array=np.nanstd(x0h, axis=1),
                thickness=0.5,
                width=2,
                color="rgb(255, 127, 14)",
            )
        label1.value = f"Number of Valid Obs. ({y0.shape[0]})"

    w1_slider.observe(response, names="value")
    w2_slider.observe(response, names="value")

    container4 = widgets.HBox([g1, g2])
    app = widgets.VBox([w1_slider, w2_slider, label1, container4])

    return app

def plot_nisar_curve(X, W0, OUT_PREFIX, param0, X_RANGE):
    import math
    W = np.arange(0, 50 * math.ceil(W0.max() / 50), 1)
    
    
    plt.rcParams['font.family'] = 'serif'  # SET DEFAULT FONT FAMILY FOR FIGURE 
    plt.rcParams['font.serif'] = ['DejaVu Serif']  # SET THE PRIMARY FONT FOR ALL SERIF TEXT IN THE FIGURE  
    plt.rcParams['font.size'] = 12  # SET FONT SIZE 
    
    fig, ax = plt.subplots(1, X.shape[-1], figsize=(10, 5))
    
    AHV,  AHH, BHV, BHH, CHV, CHH, alphaHV, alphaHH, deltaHV, deltaHH, DHV, DHH, S   = param0
    
    # Calculate the point density
    IDX = (~np.isnan(X[:, 0])) & (~np.isnan(W0[:, 0]))
    print(IDX.shape,X[:, 0].shape, W0[:, 0].shape)
    x0 = X[:, 0][IDX]
    y0 = W0[:, 0][IDX]
    xy = np.vstack([x0, y0])
    z0 = gaussian_kde(xy)(xy)
    idx = z0.argsort()
    x1, y1, z1 = x0[idx], y0[idx], z0[idx]
    
    ax[0].plot(W, volume(W, AHV, BHV, alphaHV),  linestyle ='--',  markerfacecolor='none', color = 'g', label = 'Vol')
    ax[0].plot(W, double(W, BHV, CHV, deltaHV, S),  linestyle =':',  markerfacecolor='none', color = 'r', label = 'Vol-Surf')
    ax[0].plot(W, surface(W, BHV, DHV,  S),  linestyle='-.',  markerfacecolor='none', color = 'b', label = 'Surf')
    ax[0].plot(W, nisar(W, AHV, BHV, CHV, alphaHV, deltaHV, DHV, S),  linestyle='-',   markerfacecolor='none', color = 'k', label = 'Total')
    ax[0].set_xlabel('Simulated AGB (Mg/ha)')
    
    ax[0].grid(True)
    ax[0].scatter(y1, x1, c=z1, s=10, edgecolor=None, cmap='viridis')
    ax[0].legend()
    ax[0].set_xlim(X_RANGE)
    
    ax[0].set_ylabel('HV Backscattered Power (m2/m2)')
    
    
    # Calculate the point density
    IDX = (~np.isnan(X[:, 1])) & (~np.isnan(W0[:, 0]))
    x0 = X[:, 1][IDX]
    y0 = W0[:, 0][IDX]
    xy = np.vstack([x0, y0])
    z0 = gaussian_kde(xy)(xy)
    idx = z0.argsort()
    x1, y1, z1 = x0[idx], y0[idx], z0[idx]
    
    
    ax[1].plot(W, volume(W, AHH, BHH, alphaHH),  linestyle ='--',  markerfacecolor='none', color = 'g', label = 'Vol')
    ax[1].plot(W, double(W, BHH, CHH, deltaHH, S),  linestyle =':',  markerfacecolor='none', color = 'r', label = 'Vol-Surf')
    ax[1].plot(W, surface(W, BHH, DHH,  S),  linestyle='-.',  markerfacecolor='none', color = 'b', label = 'Surf')
    ax[1].plot(W, nisar(W, AHH, BHH, CHH, alphaHH, deltaHH, DHH, S),  linestyle='-',   markerfacecolor='none', color = 'k', label = 'Total')
    
    
    ax[1].set_xlabel('Simulated AGB (Mg/ha)')
    ax[1].set_ylabel('HH Backscattered Power (m2/m2)')
    ax[1].grid(True)
    ax[1].scatter(y1, x1, c=z1, s=10, edgecolor=None, cmap='viridis')
    ax[1].legend()
    ax[1].set_xlim(X_RANGE)

    plt.tight_layout()
    plt.savefig(OUT_PREFIX + '_nisar_model.png', dpi=600)
    plt.show()
    plt.close()

def nisar_parameter_estimation(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    nwin_size = 0

    
    out_prefix = out_name 
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=nwin_size
    )
    z1_dim = z1.shape
    
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    
    
    
    df = pd.DataFrame(
        {
            "AGB": W0[:,0],
            "HH": z0[:,1],
            "HV": z0[:,0],
        }
    )
    df.to_csv(f"{out_prefix}_para_cal_data.csv")
    
    
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    
    import pickle

    with open(out_prefix + "_z0_W0.bin", "wb") as f:
        pickle.dump([z0, W0], f)

    param0_file = mdl00.params_calibration(z0, W0, out_prefix, x_range)
    
    
 
import time
# hh/hv retrieval
def scatter_plot_agb_retrieval0_v3(OUT_DIR,
                                   FILE_NAME,
                                   TRAIN_FILE,
                                   OUT_AGB_FILE,
                                   SAR_RES_DATA_LIST,
                                   TEST_FILE = None,
                                   X_RANGE = [0, 200],
                                   X_TXT = "AGB (Mg/ha)",
                                   PARAM0_FILE = None
                                   ):    
    
    start_time = time.time()
    nwin_size = 0
    
    os.chdir(OUT_DIR)
    
    
    
    # CREATE OUTPUT NAME 
    OUT_PREFIX = OUT_DIR / FILE_NAME
    
    mdl00 = FieldRetrieval(TRAIN_FILE, out_name=FILE_NAME)
    
    z1, W1, mask_ws1 = mdl00.inversion_setup(SAR_RES_DATA_LIST, agb_file = OUT_AGB_FILE, mask_file = TRAIN_FILE, n_1 = nwin_size)
    
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])

    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=TRAIN_FILE)
    
    Z0_TRAIN = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    
    W0_TRAIN = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=TRAIN_FILE)
    
    M_T = Z0_TRAIN.shape[2]
    # GET THE INITIAL AGB FILES
    OUT_AGB_INIT_FILE = []
    for NUM in range(0, M_T):
        for NUMJ in range(0, 2):
            OUT_AGB_INIT_FILE.append(OUT_DIR / os.path.basename(SAR_RES_DATA_LIST[::2][NUM]).replace('.tif', '_initial_agb.tif'))
    
    
    
    W1_INIT, _, mask_ws1_INIT = mdl00.inversion_setup(OUT_AGB_INIT_FILE, agb_file = OUT_AGB_FILE, mask_file = TRAIN_FILE, n_1 = nwin_size)
    
    W1_INIT = W1_INIT[:, 0, :, 0]
    # w1_dim_INIT = W1_INIT.shape
    # W1r_INIT = W1_INIT.reshape([-1, w1_dim_INIT[1] * w1_dim_INIT[2] * w1_dim_INIT[3]])
    W1r_INIT = W1_INIT
    W0_INIT_TRAIN = mdl00.inversion_return_valid(W1r_INIT, mask_ws1_INIT, mask_file=TRAIN_FILE)
    
    
    
    import pickle

    with open(str(OUT_PREFIX) + "_z0_W0.bin", "wb") as f:
        pickle.dump([Z0_TRAIN, W0_TRAIN], f)

    W_MEAN, _ = mdl00.inversion_recursive_ws_v3(Z0_TRAIN, W0_INIT_TRAIN, TRAIN_FILE, param0_file=PARAM0_FILE)
    
    W_MEAN = np.clip(W_MEAN, a_min= 0, a_max=W0_TRAIN.max())


    Y0 = W0_TRAIN
    Y0 = Y0.flatten()
     
    
    R2_TRAIN = []
    RMSE_TRAIN = []
    W_MEAN2_TRAIN = []
    for i in range(M_T):
        X0 = np.mean(W_MEAN[:, : i + 1], axis=1)
        R2_1 = r2_score(Y0[~np.isnan(X0), None], X0[~np.isnan(X0)])
        RMSE_1 = np.sqrt(
            mean_squared_error(Y0[~np.isnan(X0), None], X0[~np.isnan(Y0), None])
        )

        W_MEAN2_TRAIN.append(X0)
        R2_TRAIN.append(R2_1)
        RMSE_TRAIN.append(RMSE_1)
    
    
    R2_TRAIN_100 = []
    RMSE_TRAIN_100 = []
    for NUM in range(M_T):
        X0 = np.nanmean(W_MEAN[:, : NUM + 1], axis=1)
        # R2_1 = r2_score(Y0[~np.isnan(X0), None][Y0<100] , X0[~np.isnan(X0), None][Y0<100])
        R2_1 = r2_score(Y0[~np.isnan(X0), None][Y0<100], X0[~np.isnan(X0), None][Y0<100])
        # print("Variance score 1: {:.2f}".format(r2_1))
        RMSE_1 = np.sqrt(
            mean_squared_error(Y0[~np.isnan(X0), None][Y0<100], X0[~np.isnan(X0), None][Y0<100])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        R2_TRAIN_100.append(R2_1)
        RMSE_TRAIN_100.append(RMSE_1)
    R2_TRAIN_100 = np.array(R2_TRAIN_100)
    RMSE_TRAIN_100 = np.array(RMSE_TRAIN_100)
    
    
    df = pd.DataFrame()
    df["AGB_measured"] = W0_TRAIN.ravel()
    for i in range(len(W_MEAN2_TRAIN)):
        df["AGB_predicted_" + str(i)] = W_MEAN2_TRAIN[i] 
    df.to_csv(f"{OUT_PREFIX}_independent_AGB_train.csv")
    
    print(f"{OUT_PREFIX}_independent_AGB_train.csv")
    
    df = pd.DataFrame(
        {
            "AGB_measured": W0_TRAIN.ravel(),
            "AGB_predicted": W_MEAN2_TRAIN[-1],
            "HH_mean": np.nanmean(Z0_TRAIN[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(Z0_TRAIN[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{OUT_PREFIX}_mean_data.csv")

    trace1 = go.Scatter(
        x = np.arange(M_T) + 1,
        y = R2_TRAIN,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x = np.arange(M_T) + 1,
        y = RMSE_TRAIN,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=M_T,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=-1):
        y = W_MEAN2_TRAIN[m_ti]
        x = Y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    
    x, y, z0, p1, p2 = update_numbers()
    
    
    
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x = X_RANGE,
        y = X_RANGE,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {X_TXT}",
            "range": X_RANGE,
        },
        yaxis={
            "title": f"Predicted {X_TXT}",
            "range": X_RANGE,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=X_RANGE[0] + 0.2 * (X_RANGE[1] - X_RANGE[0]),
                y=X_RANGE[0] + 0.9 * (X_RANGE[1] - X_RANGE[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)
    
    p1 = []; p2= []
    for i in range(M_T):
        x, y, z0, a, b = update_numbers(i)
        p1.append(a.round(2)); p2.append(b.round(2))
        
    df = pd.DataFrame(
        {
            "RMSE": RMSE_TRAIN,
            "r2": R2_TRAIN,
            "RMSE_100": RMSE_TRAIN_100,
            "r2_100": R2_TRAIN_100,
            "per20": p2,
            "per10": p1 
        }
    )
    df.to_csv(f"{OUT_PREFIX}_train_stats.csv")
    
    plot_accuracy_stats(M_T, RMSE_TRAIN, R2_TRAIN, RMSE_TRAIN_100, R2_TRAIN_100, p2, p1, str(OUT_PREFIX) + "_train_stats.png")
    
    if TEST_FILE is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        
        mdl00 = FieldRetrieval(TEST_FILE, out_name = FILE_NAME + '_test')
        
        z1, W1, mask_ws1_test = mdl00.inversion_setup(SAR_RES_DATA_LIST, agb_file = OUT_AGB_FILE, mask_file = TEST_FILE, n_1 = nwin_size)
        
        z1_dim = z1.shape
        z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])

        z0 = mdl00.inversion_return_valid(z1r, mask_ws1_test, mask_file=TEST_FILE)
        
        Z0_TEST = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        
        W0_TEST = mdl00.inversion_return_valid(W1, mask_ws1_test, mask_file=TEST_FILE)
        
        W1_INIT, _, mask_ws1_INIT = mdl00.inversion_setup(OUT_AGB_INIT_FILE, mask_file = TEST_FILE, n_1 = nwin_size)
        
        W1_INIT = W1_INIT[:, 0, :, 0]
        # w1_dim_INIT = W1_INIT.shape
        # W1r_INIT = W1_INIT.reshape([-1, w1_dim_INIT[1] * w1_dim_INIT[2] * w1_dim_INIT[3]])
        W1r_INIT = W1_INIT
        
        # w1_dim_INIT = W1_INIT.shape
        # W1r_INIT = W1_INIT.reshape([-1, w1_dim_INIT[1] * w1_dim_INIT[2] * w1_dim_INIT[3]])

        W0_INIT_TEST = mdl00.inversion_return_valid(W1r_INIT, mask_ws1_INIT, mask_file=TEST_FILE)
        
        
        
        import pickle

        with open(str(OUT_PREFIX) + "_z0_W0_test.bin", "wb") as f:
            pickle.dump([Z0_TEST, W0_TEST], f)


        W_MEAN_TEST, _ = mdl00.inversion_recursive_ws_v3(Z0_TEST, W0_INIT_TEST, TEST_FILE, param0_file=PARAM0_FILE)
        W_MEAN_TEST = np.clip(W_MEAN_TEST, a_min= 0, a_max=W0_TRAIN.max())
        
        Y0_TEST = W0_TEST
        Y0_TEST = Y0_TEST.flatten()
        
        
        R2_TEST = []
        RMSE_TEST = []
        W_MEAN2_TEST = []
        for i in range(M_T):
            X0 = np.mean(W_MEAN_TEST[:, : i + 1], axis=1)
            R2_1 = r2_score(Y0_TEST[~np.isnan(X0), None], X0[~np.isnan(X0), None])
            RMSE_1 = np.sqrt(
                mean_squared_error(Y0_TEST[~np.isnan(X0), None], X0[~np.isnan(X0), None])
            )

            W_MEAN2_TEST.append(X0)
            R2_TEST.append(R2_1)
            RMSE_TEST.append(RMSE_1)

        R2_TEST_100 = []
        RMSE_TEST_100 = []
        for NUM in range(M_T):
            X0 = np.nanmean(W_MEAN_TEST[:, : NUM + 1], axis=1)
            R2_1 = r2_score(Y0_TEST[~np.isnan(X0), None][Y0_TEST<100], X0[~np.isnan(X0), None][Y0_TEST<100])
            # print("Variance score 1: {:.2f}".format(r2_1))
            RMSE_1 = np.sqrt(
                mean_squared_error(Y0_TEST[~np.isnan(X0), None][Y0_TEST<100], X0[~np.isnan(X0), None][Y0_TEST<100])
            )
            # print("RMSE: {:.5f}".format(rmse_1))
            R2_TEST_100.append(R2_1)
            RMSE_TEST_100.append(RMSE_1)
        R2_TEST_100 = np.array(R2_TEST_100)
        RMSE_TEST_100 = np.array(RMSE_TEST_100)
           
        df = pd.DataFrame()
        df["AGB_measured"] = W0_TEST.ravel()
        for i in range(len(W_MEAN2_TEST)):
            df["AGB_predicted_" + str(i)] = W_MEAN2_TEST[i] 
        df.to_csv(f"{OUT_PREFIX}_independent_AGB_test.csv")
        
        print(os.path.dirname(f"{OUT_PREFIX}_independent_AGB_test.csv"))
        
        df = pd.DataFrame(
            {
                "AGB_measured": Y0_TEST,
                "AGB_predicted": W_MEAN2_TEST[-1],
                "HH_mean": np.nanmean(Z0_TEST[:, 0, :, 1], axis=-1),
                "HV_mean": np.nanmean(Z0_TEST[:, 0, :, 0], axis=-1),
            }
        )
        df.to_csv(f"{OUT_PREFIX}_mean_data_test.csv")

        def update_numbers_test(m_ti=-1):
            y = W_MEAN2_TEST[m_ti]
            x = Y0_TEST
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x = X_RANGE,
            y = X_RANGE,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {X_TXT} - Test",
                "range": X_RANGE,
            },
            yaxis={
                "title": f"Predicted {X_TXT}",
                "range": X_RANGE,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=X_RANGE[0] + 0.2 * (X_RANGE[1] - X_RANGE[0]),
                    y=X_RANGE[0] + 0.9 * (X_RANGE[1] - X_RANGE[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)
        
        

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)
            
            
        p1_test = []; p2_test = []
        for i in range(M_T):
            x, y, z0, a, b = update_numbers_test(i)
            p1_test.append(a.round(2)); p2_test.append(b.round(2))
        
        df = pd.DataFrame(
            {
                "RMSE": RMSE_TEST,
                "r2": R2_TEST,
                "RMSE_100": RMSE_TEST_100,
                "r2_100": R2_TEST_100,
                "per20": p2_test,
                "per10": p1_test 
                }
            )
        df.to_csv(f"{OUT_PREFIX}_test_stats.csv")
        plot_accuracy_stats(M_T, RMSE_TEST, R2_TEST, RMSE_TEST_100, R2_TEST_100, p2_test, p1_test, str(OUT_PREFIX) + "_test_stats.png")
        plot_agb_accuracy(str(OUT_PREFIX) + "_agb_measured_vs_pred", W_MEAN2_TRAIN, Y0,  p2, p1, W_MEAN2_TEST, Y0_TEST, p2_test, p1_test)
        
        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
        
        
    end_module = time.time()
    duration_mins = (end_module - start_time) / 60
    print(f"Module Computation Time: {duration_mins:.2f} minutes")

    return app

# hh/hv retrieval
def scatter_plot_agb_retrieval0_v4(OUT_DIR,
                                   FILE_NAME,
                                   TRAIN_FILE,
                                   OUT_AGB_FILE,
                                   SAR_RES_DATA_LIST,
                                   TEST_FILE = None,
                                   X_RANGE = [0, 200],
                                   X_TXT = "AGB (Mg/ha)",
                                   PARAM0_FILE = None
                                   ):    
    
    start_time = time.time()
    
    nwin_size = 0
    
    os.chdir(OUT_DIR)
    
    
    
    # CREATE OUTPUT NAME 
    OUT_PREFIX = OUT_DIR / FILE_NAME
    
    mdl00 = FieldRetrieval(TRAIN_FILE, out_name=FILE_NAME)
    
    z1, W1, mask_ws1 = mdl00.inversion_setup(SAR_RES_DATA_LIST, agb_file = OUT_AGB_FILE, mask_file = TRAIN_FILE, n_1 = nwin_size)
    
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])

    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=TRAIN_FILE)
    
    Z0_TRAIN = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    
    W0_TRAIN = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=TRAIN_FILE)
    
    M_T = Z0_TRAIN.shape[2]
    # GET THE INITIAL AGB FILES
    OUT_AGB_INIT_FILE = []
    for NUM in range(0, M_T):
        for NUMJ in range(0, 2):
            OUT_AGB_INIT_FILE.append(OUT_DIR / os.path.basename(SAR_RES_DATA_LIST[::2][NUM]).replace('.tif', '_initial_agb.tif'))
    
    
    
    W1_INIT, _, mask_ws1_INIT = mdl00.inversion_setup(OUT_AGB_INIT_FILE, agb_file = OUT_AGB_FILE, mask_file = TRAIN_FILE, n_1 = nwin_size)
    
    W1_INIT = W1_INIT[:, 0, :, 0]
    # w1_dim_INIT = W1_INIT.shape
    # W1r_INIT = W1_INIT.reshape([-1, w1_dim_INIT[1] * w1_dim_INIT[2] * w1_dim_INIT[3]])
    W1r_INIT = W1_INIT
    W0_INIT_TRAIN = mdl00.inversion_return_valid(W1r_INIT, mask_ws1_INIT, mask_file=TRAIN_FILE)
    
    
    
    import pickle

    with open(str(OUT_PREFIX) + "_z0_W0.bin", "wb") as f:
        pickle.dump([Z0_TRAIN, W0_TRAIN], f)

    W_MEAN, _ = mdl00.inversion_recursive_ws_v4(Z0_TRAIN, W0_INIT_TRAIN, TRAIN_FILE, param0_file=PARAM0_FILE)
    
    W_MEAN = np.clip(W_MEAN, a_min= 0, a_max=W0_TRAIN.max())


    Y0 = W0_TRAIN
    Y0 = Y0.flatten()
     
    
    R2_TRAIN = []
    RMSE_TRAIN = []
    W_MEAN2_TRAIN = []
    for i in range(M_T):
        X0 = np.mean(W_MEAN[:, : i + 1], axis=1)
        R2_1 = r2_score(Y0[~np.isnan(X0), None], X0[~np.isnan(X0)])
        RMSE_1 = np.sqrt(
            mean_squared_error(Y0[~np.isnan(X0), None], X0[~np.isnan(Y0), None])
        )

        W_MEAN2_TRAIN.append(X0)
        R2_TRAIN.append(R2_1)
        RMSE_TRAIN.append(RMSE_1)
    
    
    R2_TRAIN_100 = []
    RMSE_TRAIN_100 = []
    for NUM in range(M_T):
        X0 = np.nanmean(W_MEAN[:, : NUM + 1], axis=1)
        # R2_1 = r2_score(Y0[~np.isnan(X0), None][Y0<100] , X0[~np.isnan(X0), None][Y0<100])
        R2_1 = r2_score(Y0[~np.isnan(X0), None][Y0<100], X0[~np.isnan(X0), None][Y0<100])
        # print("Variance score 1: {:.2f}".format(r2_1))
        RMSE_1 = np.sqrt(
            mean_squared_error(Y0[~np.isnan(X0), None][Y0<100], X0[~np.isnan(X0), None][Y0<100])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        R2_TRAIN_100.append(R2_1)
        RMSE_TRAIN_100.append(RMSE_1)
    R2_TRAIN_100 = np.array(R2_TRAIN_100)
    RMSE_TRAIN_100 = np.array(RMSE_TRAIN_100)
    
    
    df = pd.DataFrame()
    df["AGB_measured"] = W0_TRAIN.ravel()
    for i in range(len(W_MEAN2_TRAIN)):
        df["AGB_predicted_" + str(i)] = W_MEAN2_TRAIN[i] 
    df.to_csv(f"{OUT_PREFIX}_independent_AGB_train.csv")
    
    print(f"{OUT_PREFIX}_independent_AGB_train.csv")
    
    df = pd.DataFrame(
        {
            "AGB_measured": W0_TRAIN.ravel(),
            "AGB_predicted": W_MEAN2_TRAIN[-1],
            "HH_mean": np.nanmean(Z0_TRAIN[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(Z0_TRAIN[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{OUT_PREFIX}_mean_data.csv")

    trace1 = go.Scatter(
        x = np.arange(M_T) + 1,
        y = R2_TRAIN,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x = np.arange(M_T) + 1,
        y = RMSE_TRAIN,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=M_T,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=-1):
        y = W_MEAN2_TRAIN[m_ti]
        x = Y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    
    x, y, z0, p1, p2 = update_numbers()
    
    
    
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x = X_RANGE,
        y = X_RANGE,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {X_TXT}",
            "range": X_RANGE,
        },
        yaxis={
            "title": f"Predicted {X_TXT}",
            "range": X_RANGE,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=X_RANGE[0] + 0.2 * (X_RANGE[1] - X_RANGE[0]),
                y=X_RANGE[0] + 0.9 * (X_RANGE[1] - X_RANGE[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)
    
    p1 = []; p2= []
    for i in range(M_T):
        x, y, z0, a, b = update_numbers(i)
        p1.append(a.round(2)); p2.append(b.round(2))
        
    df = pd.DataFrame(
        {
            "RMSE": RMSE_TRAIN,
            "r2": R2_TRAIN,
            "RMSE_100": RMSE_TRAIN_100,
            "r2_100": R2_TRAIN_100,
            "per20": p2,
            "per10": p1 
        }
    )
    df.to_csv(f"{OUT_PREFIX}_train_stats.csv")
    
    #plot_accuracy_stats(M_T, RMSE_TRAIN, R2_TRAIN, RMSE_TRAIN_100, R2_TRAIN_100, p2, p1, str(OUT_PREFIX) + "_train_stats.png")
    
    if TEST_FILE is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        
        mdl00 = FieldRetrieval(TEST_FILE, out_name = FILE_NAME + '_test')
        
        z1, W1, mask_ws1_test = mdl00.inversion_setup(SAR_RES_DATA_LIST, agb_file = OUT_AGB_FILE, mask_file = TEST_FILE, n_1 = nwin_size)
        
        z1_dim = z1.shape
        z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])

        z0 = mdl00.inversion_return_valid(z1r, mask_ws1_test, mask_file=TEST_FILE)
        
        Z0_TEST = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        
        W0_TEST = mdl00.inversion_return_valid(W1, mask_ws1_test, mask_file=TEST_FILE)
        
        W1_INIT, _, mask_ws1_INIT = mdl00.inversion_setup(OUT_AGB_INIT_FILE, mask_file = TEST_FILE, n_1 = nwin_size)
        
        W1_INIT = W1_INIT[:, 0, :, 0]
        # w1_dim_INIT = W1_INIT.shape
        # W1r_INIT = W1_INIT.reshape([-1, w1_dim_INIT[1] * w1_dim_INIT[2] * w1_dim_INIT[3]])
        W1r_INIT = W1_INIT
        
        # w1_dim_INIT = W1_INIT.shape
        # W1r_INIT = W1_INIT.reshape([-1, w1_dim_INIT[1] * w1_dim_INIT[2] * w1_dim_INIT[3]])

        W0_INIT_TEST = mdl00.inversion_return_valid(W1r_INIT, mask_ws1_INIT, mask_file=TEST_FILE)
        
        
        
        import pickle

        with open(str(OUT_PREFIX) + "_z0_W0_test.bin", "wb") as f:
            pickle.dump([Z0_TEST, W0_TEST], f)


        W_MEAN_TEST, _ = mdl00.inversion_recursive_ws_v4(Z0_TEST, W0_INIT_TEST, TEST_FILE, param0_file=PARAM0_FILE)
        W_MEAN_TEST = np.clip(W_MEAN_TEST, a_min= 0, a_max=W0_TRAIN.max())
        
        Y0_TEST = W0_TEST
        Y0_TEST = Y0_TEST.flatten()
        
        
        R2_TEST = []
        RMSE_TEST = []
        W_MEAN2_TEST = []
        for i in range(M_T):
            X0 = np.mean(W_MEAN_TEST[:, : i + 1], axis=1)
            R2_1 = r2_score(Y0_TEST[~np.isnan(X0), None], X0[~np.isnan(X0), None])
            RMSE_1 = np.sqrt(
                mean_squared_error(Y0_TEST[~np.isnan(X0), None], X0[~np.isnan(X0), None])
            )

            W_MEAN2_TEST.append(X0)
            R2_TEST.append(R2_1)
            RMSE_TEST.append(RMSE_1)

        R2_TEST_100 = []
        RMSE_TEST_100 = []
        for NUM in range(M_T):
            X0 = np.nanmean(W_MEAN_TEST[:, : NUM + 1], axis=1)
            R2_1 = r2_score(Y0_TEST[~np.isnan(X0), None][Y0_TEST<100], X0[~np.isnan(X0), None][Y0_TEST<100])
            # print("Variance score 1: {:.2f}".format(r2_1))
            RMSE_1 = np.sqrt(
                mean_squared_error(Y0_TEST[~np.isnan(X0), None][Y0_TEST<100], X0[~np.isnan(X0), None][Y0_TEST<100])
            )
            # print("RMSE: {:.5f}".format(rmse_1))
            R2_TEST_100.append(R2_1)
            RMSE_TEST_100.append(RMSE_1)
        R2_TEST_100 = np.array(R2_TEST_100)
        RMSE_TEST_100 = np.array(RMSE_TEST_100)
           
        df = pd.DataFrame()
        df["AGB_measured"] = W0_TEST.ravel()
        for i in range(len(W_MEAN2_TEST)):
            df["AGB_predicted_" + str(i)] = W_MEAN2_TEST[i] 
        df.to_csv(f"{OUT_PREFIX}_independent_AGB_test.csv")
        
        print(os.path.dirname(f"{OUT_PREFIX}_independent_AGB_test.csv"))
        
        df = pd.DataFrame(
            {
                "AGB_measured": Y0_TEST,
                "AGB_predicted": W_MEAN2_TEST[-1],
                "HH_mean": np.nanmean(Z0_TEST[:, 0, :, 1], axis=-1),
                "HV_mean": np.nanmean(Z0_TEST[:, 0, :, 0], axis=-1),
            }
        )
        df.to_csv(f"{OUT_PREFIX}_mean_data_test.csv")

        def update_numbers_test(m_ti=-1):
            y = W_MEAN2_TEST[m_ti]
            x = Y0_TEST
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x = X_RANGE,
            y = X_RANGE,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {X_TXT} - Test",
                "range": X_RANGE,
            },
            yaxis={
                "title": f"Predicted {X_TXT}",
                "range": X_RANGE,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=X_RANGE[0] + 0.2 * (X_RANGE[1] - X_RANGE[0]),
                    y=X_RANGE[0] + 0.9 * (X_RANGE[1] - X_RANGE[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)
        
        

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)
            
            
        p1_test = []; p2_test = []
        for i in range(M_T):
            x, y, z0, a, b = update_numbers_test(i)
            p1_test.append(a.round(2)); p2_test.append(b.round(2))
        
        df = pd.DataFrame(
            {
                "RMSE": RMSE_TEST,
                "r2": R2_TEST,
                "RMSE_100": RMSE_TEST_100,
                "r2_100": R2_TEST_100,
                "per20": p2_test,
                "per10": p1_test 
                }
            )
        df.to_csv(f"{OUT_PREFIX}_test_stats.csv")
        #plot_accuracy_stats(M_T, RMSE_TEST, R2_TEST, RMSE_TEST_100, R2_TEST_100, p2_test, p1_test, str(OUT_PREFIX) + "_test_stats.png")
        #plot_agb_accuracy(str(OUT_PREFIX) + "_agb_measured_vs_pred", W_MEAN2_TRAIN, Y0,  p2, p1, W_MEAN2_TEST, Y0_TEST, p2_test, p1_test)
        
        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
        
        
        
    end_module = time.time()
    duration_mins = (end_module - start_time) / 60
    print(f"Module Computation Time: {duration_mins:.2f} minutes")
    
    return app


def colormap_plot_agb_prediction(OUTDIR_DIR, SAR_RES_DATA_LIST, SCENE_MASK_FILE, param0_file, FILE_NAME,  ab_range=[43,51]):
    
    
    # CREATE A OUTPUT DIRECTORY, IF IT DOES NOT EXISIT   
    if not os.path.exists(OUTDIR_DIR + '/output'):
        os.mkdir(OUTDIR_DIR + '/output')
    
    # CREATE OUTPUT NAME 
    OUT_PREFIX = OUTDIR_DIR + '/output/' + FILE_NAME
    
    
    mdl00 = FieldRetrieval(SCENE_MASK_FILE, out_name=OUT_PREFIX)
    z1, _, mask_ws1 = mdl00.inversion_setup(
        SAR_RES_DATA_LIST, mask_file=SCENE_MASK_FILE, n_1=0
    )
    
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=SCENE_MASK_FILE)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    
    
    M_T = z0.shape[2]
    # GET THE INITIAL AGB FILES
    OUT_AGB_INIT_FILE = []
    for NUM in range(0, M_T):
        OUT_AGB_INIT_FILE.append(OUTDIR_DIR +  '/' + os.path.basename(SAR_RES_DATA_LIST[::2][NUM]).replace('.tif', '_initial_agb.tif'))
        OUT_AGB_INIT_FILE.append(OUTDIR_DIR +  '/' + os.path.basename(SAR_RES_DATA_LIST[::2][NUM]).replace('.tif', '_initial_agb.tif'))
        
    
    
    
    W1_INIT, _, mask_ws1_INIT = mdl00.inversion_setup(
        OUT_AGB_INIT_FILE, mask_file=SCENE_MASK_FILE, n_1=0
    )
    
    
    w1_dim_INIT = W1_INIT.shape
    W1r_INIT = W1_INIT.reshape([-1, w1_dim_INIT[1] * w1_dim_INIT[2] * w1_dim_INIT[3]])

    W0_INIT_TRAIN = mdl00.inversion_return_valid(W1r_INIT, mask_ws1_INIT, mask_file=SCENE_MASK_FILE)
    W0_INIT_TRAIN = W0_INIT_TRAIN[:,  ::2]
    
    
    W_mean, S_mean = mdl00.inversion_recursive_ws_v4(z0, W0_INIT_TRAIN, SCENE_MASK_FILE, param0_file = param0_file)
    
    
    
    in0 = gdal.Open(SCENE_MASK_FILE, gdal.GA_ReadOnly)
    array0 = in0.GetRasterBand(1).ReadAsArray()
    array0[array0 > 0] = np.mean(W_mean, axis=-1)
    
    agb_name = f"{OUT_PREFIX}.tif"
    try:
        os.remove(agb_name)
    except OSError:
        pass
    
    # Force GDAL to show detailed errors in the console
    gdal.UseExceptions()

    mem_driver = gdal.GetDriverByName("MEM")
    ds_update = mem_driver.Create("", in0.RasterXSize, in0.RasterYSize, 1, gdal.GDT_Float32)
    ds_update.SetProjection(in0.GetProjection())
    ds_update.SetGeoTransform(in0.GetGeoTransform())
    ds_update.GetRasterBand(1).WriteArray(array0)
    
    
    # --- 4. UPDATE METADATA LATER ---
    
    source_collectiontime_first = os.path.basename(SAR_RES_DATA_LIST[0]).split('_HVHV')[0]
    source_collectiontime_last = os.path.basename(SAR_RES_DATA_LIST[-1]).split('_HHHH')[0]
    Nlen = str(int(len(SAR_RES_DATA_LIST)//2))
    GRANULE_LIST = [os.path.basename(name).split('_HVHV')[0] for name in SAR_RES_DATA_LIST[::2]]
    GRANULE_LIST = ", ".join(GRANULE_LIST)
    
    ds_update.SetMetadataItem('AREA_OR_POINT', "Area")          # pixel is point or pixel is area
    ds_update.SetMetadataItem('calval_accuracy', '')             # overall accuracy of the L3 Science Product if the full cal/val workflow was implemented. This relates to science requirements
    ds_update.SetMetadataItem('calval_processing_facility', '')  # Processing facility that ran the cal/val workflow
    ds_update.SetMetadataItem('calval_source', '')	              # Data source that was used for calibration and validation of the L3 science product
    ds_update.SetMetadataItem('calval_status', '')               # Current status of the L3 science product
    ds_update.SetMetadataItem('CEOS_docid', 'https://ceos.org/ard/files/PFS/SAR/v1.1/CEOS-ARD_PFS_Synthetic_Aperture_Radar_v1.1.pdf')
    
    ds_update.SetMetadataItem('prod_access_location', '')
    ds_update.SetMetadataItem('prod_descriptio', '')	              # 1 sentence on process for generating the product
    ds_update.SetMetadataItem('prod_agb_param_init', '')	          # initial model parameters for agb
    ds_update.SetMetadataItem('prod_name', 'AGB')                # name of L3 science product
    ds_update.SetMetadataItem('prod_agb_param_final', '' )          # final model parameters for agb
    ds_update.SetMetadataItem('prod_processing_date', datetime.now(UTC).strftime("%Y%m%dT%H%M%S"))	## Date the L3 science products was processed
    ds_update.SetMetadataItem('prod_processing_facility', '')	     # Where the L3 science product was processed
    ds_update.SetMetadataItem('prod_software_version', "1.0")      # software version used to create the L3 science product
    ds_update.SetMetadataItem('prod_speckle_filter', "FALSE")       # Info on whether speckle filtering was applied during product generation
    ds_update.SetMetadataItem('prod_resample','' )                #The number of pixels aggregated during resampling from the 20m L2 GCOV product to the 100m L3 Science product
    ds_update.SetMetadataItem('prod_type', "SCIENCE")	         # product type
    ds_update.SetMetadataItem('prod_units', 'Mg/ha')                # product units
        
    ds_update.SetMetadataItem('source_band_width', "20")	# NISAR bandwidth in MHz
    ds_update.SetMetadataItem('source_center_wavelength', "0.238403545 m")	# NISAR center wavelength in meters
    ds_update.SetMetadataItem('source_collectiontime_last', source_collectiontime_last)	# NISAR acquisiiton time of last image in time series stack used for evaluation
    ds_update.SetMetadataItem('source_collectiontime_first', source_collectiontime_first) # NISAR acquisiiton time of first image in time series stack used for evaluation
    ds_update.SetMetadataItem('source_baseline_last', '')	# NISAR acquisiiton time of last image in time series stack used for the baseline
    ds_update.SetMetadataItem('source_baseline_first', '')	# NISAR acquisiiton time of first image in time series stack used for the baseline
    ds_update.SetMetadataItem('source_data_access', '')  #	URL where L3 science product will be stored
    ds_update.SetMetadataItem('source_instrument', "L-SAR")	# NISAR instrument name
    ds_update.SetMetadataItem('source_number_of_acquisitions', Nlen)	# Number of NISAR acquisitions in the time series stack
    ds_update.SetMetadataItem('source_radar_pointing', "left")
    ds_update.SetMetadataItem('source_radar_pol', 'HH+HV') 	#NISAR polariazation
    ds_update.SetMetadataItem('source_radar_az_looks', "22")	# NISAR number of looks in azimuth
    ds_update.SetMetadataItem('source_radar_range_looks', "5")	# NISAR Number of looks in range
    ds_update.SetMetadataItem('source_radar_band', "L")	# NISAR radar band
    ds_update.SetMetadataItem('source_radar_east_pixel_spacing', "20") #	NISAR east pixel spacing in meters
    ds_update.SetMetadataItem('source_radar_look_look_direction', "B")	 # NISAR source data look Direction
    ds_update.SetMetadataItem('source_radar_mode', "polsar")	 # NISAR source data radar mode
    ds_update.SetMetadataItem('source_radar_nes0', "-30")	
    ds_update.SetMetadataItem('source_radar_north_pixel_spacing', "100") #in meters		
    ds_update.SetMetadataItem('source_radar_processing_facility', "NASA Jet Propulsion Laboratory") #Facility where the NISAR L2 products were processed
    ds_update.SetMetadataItem('source_radar_nisar_granules_evaluation', GRANULE_LIST) #list of NISAR L2 products (full granule name) that were used for generating the L3 Science Product
    ds_update.SetMetadataItem('source_radar_software_version', '83634531328a6b1fb3991f252a5c6d52b7a9ed4b')
    ds_update.SetMetadataItem('source_radar_product_level', "L2") # The NISAR source data level used to generate the L3 Science Product
    ds_update.SetMetadataItem('source_radar_software_version', '1.0') #Software version used to generate the NISAR source data
    ds_update.SetMetadataItem('source_radar_satellite', 'NISAR') #	Name of the satellite

    ds_update.SetMetadataItem('cal_20', '')
    ds_update.SetMetadataItem('cal_10', '')
    ds_update.SetMetadataItem('val_rmse', '')
    ds_update.SetMetadataItem('val_r2', '') 
    ds_update.SetMetadataItem('val_20', '') 
    ds_update.SetMetadataItem('val_10', '') 
    ds_update.SetMetadataItem('cal_val_note', '') 
    
     
    print("Metadata successfully updated.")
    
    
    
    
    # Optional: Set the Color Table on the memory dataset if you are using one
    # tmp_ds.GetRasterBand(1).SetColorTable(ct)
    
    cog_driver = gdal.GetDriverByName("COG")
    if cog_driver is None:
        raise RuntimeError("COG driver not found. Check your GDAL version (requires 3.1+).")

    try:
        # Remove existing file if it exists to avoid lock errors
        if os.path.exists(agb_name):
            os.remove(agb_name)
            
        ds = cog_driver.CreateCopy(
            agb_name,
            ds_update,
            strict=0,
            options=[
                "COMPRESS=LZW",
                "PREDICTOR=YES",
                "BLOCKSIZE=512",
                "OVERVIEWS=AUTO" # Let the COG driver build the overviews here!
                ]
            )
        
        # Double check ds is not None before calling methods
        if ds:
            ds.FlushCache()
            ds = None
        else:
            print("Failed to create COG dataset.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        ds_update = None
        
    
    
    
    
    print(f"Created True Value Raster: {agb_name}")

    # 3. CREATE THE QML STYLE (Visuals)
    # This XML tells QGIS: 0=Black, 1=Light Green, 100+=Dark Green
    qml_content = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
    <qgis version="3.28.0">
      <pipe>
        <rasterrenderer type="singlebandpseudocolor" opacity="1" classificationMin="0" classificationMax="100" band="1">
          <rastershader>
            <colorrampshader colorRampType="INTERPOLATED" clip="0">
              <item label="0 (None)" value="0" color="#000000" alpha="255"/>
              <item label="1 (Low)" value="1" color="#f7fcf5" alpha="255"/>
              <item label="50 (Med)" value="50" color="#74c476" alpha="255"/>
              <item label="100 (High)" value="100" color="#00441b" alpha="255"/>
              <item label=">100 (Overflow)" value="10000" color="#00441b" alpha="255"/>
            </colorrampshader>
          </rastershader>
        </rasterrenderer>
      </pipe>
    </qgis>
    """

    with open(agb_name.replace('.tif', '.qml'), "w") as f:
        f.write(qml_content)

    
    for k in range(len(SAR_RES_DATA_LIST) // 2):
        basename = os.path.basename(SAR_RES_DATA_LIST[k * 2])
        agb_name = f"{OUT_PREFIX}_agb_predictions_{basename[ab_range[0]:ab_range[1]]}.tif"
        print(agb_name)
        try:
            os.remove(agb_name)
        except OSError:
            pass

        array0 = in0.GetRasterBand(1).ReadAsArray()
        array0[array0 > 0] = W_mean[:, k]
        
        # Force GDAL to show detailed errors in the console
        gdal.UseExceptions()

        mem_driver = gdal.GetDriverByName("MEM")
        ds_update = mem_driver.Create("", in0.RasterXSize, in0.RasterYSize, 1, gdal.GDT_Float32)
        ds_update.SetProjection(in0.GetProjection())
        ds_update.SetGeoTransform(in0.GetGeoTransform())
        ds_update.GetRasterBand(1).WriteArray(array0)


        # Optional: Set the Color Table on the memory dataset if you are using one
        # tmp_ds.GetRasterBand(1).SetColorTable(ct)
        
        cog_driver = gdal.GetDriverByName("COG")
        if cog_driver is None:
            raise RuntimeError("COG driver not found. Check your GDAL version (requires 3.1+).")

        try:
            # Remove existing file if it exists to avoid lock errors
            if os.path.exists(agb_name):
                os.remove(agb_name)
                
            ds = cog_driver.CreateCopy(
                agb_name,
                ds_update,
                strict=0,
                options=[
                    "COMPRESS=LZW",
                    "PREDICTOR=YES",
                    "BLOCKSIZE=512",
                    "OVERVIEWS=AUTO" # Let the COG driver build the overviews here!
                    ]
                )
            
            # Double check ds is not None before calling methods
            if ds:
                ds.FlushCache()
                ds = None
            else:
                print("Failed to create COG dataset.")

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            ds_update = None




    in0 = None