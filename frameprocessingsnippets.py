import matplotlib.pyplot as plt
import numpy as np

import time

import plotly.offline as py
import plotly.tools as tls
py.init_notebook_mode()

def plotly_show():
    #get fig and convert to plotly
    fig = plt.gcf()
    plotlyfig = tls.mpl_to_plotly(fig, resize=True)
    
    #fix dumb automatic formatting choices
    plotlyfig['layout']['xaxis1']['tickfont']['size']=14
    plotlyfig['layout']['xaxis1']['titlefont']['size']=16
    plotlyfig['layout']['yaxis1']['tickfont']['size']=14
    plotlyfig['layout']['yaxis1']['titlefont']['size']=16
    plotlyfig['layout']['showlegend'] = True
    
    #add a fix to bring back automatic sizing
    plotlyfig['layout']['height'] = None
    plotlyfig['layout']['width'] = None
    plotlyfig['layout']['autosize'] = True
    
    # plot
    py.iplot(plotlyfig)

def resample_array(arr,pxsize=100):
    kromopxsize = 2.9 #microns
    factor = int(np.round(pxsize/kromopxsize))
    print('resample will give pixels of size {:.2f} microns'.format(factor*kromopxsize))
    row, cols = arr.shape
    grouped = arr[:row//factor*factor,:cols//factor*factor].reshape(row//factor, factor, cols//factor, factor)
    return grouped.sum(axis=3).sum(axis=1)    

def plot_histogram(datarun, xmin=20, xmax=255, show=True):
    plt.bar(np.arange(1,len(datarun.get_histogram())+1)[xmin:xmax],datarun.get_histogram()[xmin:xmax],width=1)
    if show:
        plotly_show()

def show_array(datarun, pxsize=50):
    arr = try_getting_array(datarun)
    kromopxsize = 2.9 #microns
    kromoshape = (1936,1096)
    factor = int(np.round(pxsize/kromopxsize))
    resampled = resample_array(arr/try_getting_photon_value(datarun),pxsize=pxsize)
    h = resampled.shape[0]*kromopxsize*1e-3*factor
    w = resampled.shape[1]*kromopxsize*1e-3*factor
    plt.imshow(resampled,extent=[0,w,0,h])
    plt.show()

def lineout(datarun, direction='h', show=True):
    arr = try_getting_array(datarun)
    if direction == 'h':
        ax = 0
    elif direction == 'v':
        ax = 1
    plt.plot(np.sum(arr/try_getting_photon_value(datarun),axis=ax))
    if show:
        plotly_show()

def try_getting_photon_value(datarun):
    try:
        photon_value = datarun.photon_value
    except AttributeError:
        photon_value = 1
    return photon_value

def try_getting_array(datarun):
    try:
        arr = datarun.get_array()
    except AttributeError:
        arr = datarun
    return arr

def combine_frames(dataruns, pxsize=50, setempty=False, plot=True, colorbar=True, rotate = False):
    if rotate:
        camerax, cameray = (1096, 1936)
    else:
        camerax, cameray = (1936, 1096)

    xs = np.array([dr.x for dr in dataruns])
    ys = np.array([dr.y for dr in dataruns])

    if rotate:
        arrs = np.array([np.rot90(dr.get_array()/dr.photon_value) for dr in dataruns])
    else:
        arrs = np.array([(dr.get_array()/dr.photon_value) for dr in dataruns])
    left = xs.max()
    right = xs.min()-camerax*2.9e-3
    top = ys.max()
    bottom = ys.min()-cameray*2.9e-3
    height = np.around((top-bottom)/2.9e-3).astype('int')
    width = -np.around((right-left)/2.9e-3).astype('int')
    image = np.zeros((height,width))
    image = image - 1
    h, w = image.shape
    for x, y, a in zip(xs,ys,arrs):
        xi, yi = np.around((np.array([left-x, y-bottom])/2.9e-3)).astype('int')
        image[h-yi:h-yi+cameray,xi:xi+camerax] = a
    if pxsize != -1:
        resampledimage = resample_array(image, pxsize=pxsize)
        image = resampledimage
    if setempty is not False:
        resampledimage[resampledimage<0] = setempty#np.min(resampledimage)+1] = setempty
    if plot:
        plt.imshow(image,interpolation='none', extent=(0,w*2.9e-3,0,h*2.9e-3))
        plt.xlabel('mm in Rowland Plane')
        plt.ylabel('mm out of Rowland Plane')
        if colorbar:
            plt.colorbar()
        plt.show()
    return image, w, h

def rowland_plane_profile_lineout(arr, xshift=0, pxsize=50, takeyregion=None, normalize=None, plot=False, show=False, **plotkwargs):
    arr = resample_array(arr, pxsize=pxsize)
    profile = np.sum(arr, axis=0)
    pixels = 2.9 #microns
    factor = int(np.round(pxsize/pixels))
    if takeyregion is not None:
        ylength = arr.shape[0]*factor*pixels*1e-3
        y1 = int((takeyregion[0]/ylength)*arr.shape[0])
        y2 = int((takeyregion[1]/ylength)*arr.shape[0])
        profile = np.sum(arr[y1:y2],axis=0)
    if normalize == 'integral':
        profile = profile/np.sum(profile[np.logical_not(np.isnan(profile))])
    x = np.arange(len(profile))*factor*pixels*1e-3
    if plot:
        plt.plot(x+xshift,profile,**plotkwargs)
        if show:
            plt.show()
    return np.array([x+xshift, profile])

def out_of_rowland_plane_profile_lineout(arr, xshift=0, pxsize=50, takeyregion=None, normalize=None, plot=False, show=False, **plotkwargs):
    arr = resample_array(arr, pxsize=pxsize)
    profile = np.sum(arr, axis=0)
    pixels = 2.9 #microns
    factor = int(np.round(pxsize/pixels))
    if takeyregion is not None:
        ylength = arr.shape[0]*factor*pixels*1e-3
        y1 = int((takeyregion[0]/ylength)*arr.shape[0])
        y2 = int((takeyregion[1]/ylength)*arr.shape[0])
        profile = np.sum(arr[y1:y2],axis=0)
    if normalize == 'integral':
        profile = profile/np.sum(profile[np.logical_not(np.isnan(profile))])
    x = np.arange(len(profile))*factor*pixels*1e-3
    if plot:
        plt.plot(x+xshift,profile,**plotkwargs)
        if show:
            plt.show()
    return np.array([x+xshift, profile])

def calc_photon_value(datarun, window_min=50, window_max=150):
    """Watch the window!  Specific to this study."""
    x = datarun.get_array()
    return np.mean(x[np.logical_and(x>window_min,x<window_max)])

def check_processes_alive():
    return [x.is_alive() for x in [kromowindows.camera]+[kromowindows.capture.sink]+kromowindows.capture.workers]

def fraction_dropped():
    settings = kromowindows.get_settings()
    return settings['NumDroppedFrames']/settings['TotalFramesProcessed']

def monitor_datarun_time_left(dr, runlength=180):
    import threading
    import time
    stopevent = threading.Event()
    def _time_left(dr, evt):
        while not evt.is_set() and (runlength-dr.acquisition_time())>0.5:
            print('Time left in acquisition: {:.0f}              \r'.format(runlength-dr.acquisition_time()),end='')
            time.sleep(1)
        print('Done!')
    t = threading.Thread(target=_time_left,args=(dr,stopevent))
    t.start()
    return t, stopevent

def histogram_of_array(dr, show=True):
    n, bins = np.histogram(dr.get_array().flatten(),bins=254,range=(1,255))
    plt.bar(bins[:-1],n,width=1)
    if show:
        plotly_show()

def window_shown_on_histogram(dr, show='plt'):
    plot_histogram(dr,20,255,False)
    histogram_of_array(dr,False)
    if show == 'plotly':
        plotly_show()
    elif show == 'plt':
        plt.show()

def change_window(new_min, new_max):
    print('Killing current process...')
    kromowindows.shutdown()
    print('Wait a smidge.')
    time.sleep(5)
    assert all([not x for x in [False,False,False]]), "Process didn't die!!!"
    kromowindows.start(window_min=new_min, window_max=new_max)
    print('restarting with window: [{},{}]'.format(new_min,new_max))
    return kromowindows.get_settings()
