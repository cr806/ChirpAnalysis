import csv
import logging
from PIL import Image, ImageOps
import pandas as pd
import json
import timeit
import configparser
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join
from ROImulti import ROImulti as ROI


def setup_logger(log_filepath, log_level):
    """ Sets up logging facilities

        Attributes:
        log_filepath (str) :    String indicating log filepath
        log_level (str) :        Level of logging required (e.g. INFO, DEBUG)

        Returns:
        logger (obj)
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    fh = logging.FileHandler(filename=log_filepath, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : '
                                  '%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_simulation_params(logger, config_filepath):
    """ Opens, parses and returns user parameters from configuration file

        Attributes:
        logger (obj) :            Logger object (see "setup_logger" function)
        config_filepath (str) :   Filepath to configuration file. Configuration
                                  file should be of .ini format

        Returns:
        params (dict) :           Dictionary containing user parameters
    """
    def get_param(d, p):
        a = config[d].getfloat(p)
        if a is None:
            logger.info(f'No {d:p} provided')
        return a

    params = {}
    try:
        config = configparser.ConfigParser()
        config.read(config_filepath)
    except FileNotFoundError:
        print('Configuration file can not be found')
        logger.error('Configuration file can not be found')
        quit()

    try:
        params['save_path'] = config['Data']['save_path']
        params['save_name'] = config['Data']['save_name']

        params['image_folder'] = config['Data']['image_folder']
        params['image_type'] = config['Data']['image_type']
        params['image_align'] = config['Data']['image_align']
        params['save_figure_filename'] = config['Data']['save_figure_filename']
        params['analysis'] = config['Data']['analysis_method']

        params['sleep_time'] = 0.33

        if config['Data'].getboolean('live_watch'):
            try:
                params['sleep_time'] = config['Loop'].getfloat('sleep_time')
            except KeyError:
                print("No sleep time provided, it is now set to 2 seconds")
                logger.info('No sleep time provided')
        else:
            logger.info('Live watch not requested')
    except KeyError:
        print('Configuration file incorrectly formatted')
        logger.error('No [Data] section provided within configuration file')
        quit()

    params['num_of_ROIs'] = 1
    params['num_of_subROIs'] = 1
    params['image_interval'] = 1.0
    params['angle'] = None
    params['roi_ranges'] = None
    params['res_gamma'] = None

    try:
        params['num_of_ROIs'] = int(get_param('Image', 'num_of_ROIs'))
        params['num_of_subROIs'] = int(get_param('Image', 'num_of_subROIs'))

        if config['Data'].getboolean('initialise_image'):
            params['image_interval'] = get_param('Image', 'image_interval')

            try:
                params['angle'] = config['Image'].getfloat('image_angle')
            except KeyError:
                logger.info('No angle provided')
            try:
                params['roi_ranges'] = json.loads(config['Image']['rois'])
            except KeyError:
                logger.info('No ROI coordinates provided')
            try:
                params['res_gamma'] = json.loads(
                    config['Image']['resonance_gamma'])
            except KeyError:
                logger.info('No resonance/gamma data provided')
        else:
            logger.log('Image initialised interactively')
    except KeyError:
        logger.info('No [Image] section provided within configuration file')
    return params


def register_ROIs(logger, params, image_files):
    """ Calls ROI constructors and stores resulting ROI object in array

        Attributes:
        params (dict) :    Dictionary containing user parameters

        Returns:
        roi_array (list) : List of ROI objects
    """
    im = get_image(logger, params, image_files[0])
    roi_array = []
    if params['num_of_ROIs'] > 0:
        for roi in range(params['num_of_ROIs']):
            try:
                r_g = params['res_gamma'][roi]
            except (IndexError, TypeError):
                r_g = None
            try:
                roi_range = params['roi_ranges'][roi]
            except (IndexError, TypeError):
                roi_range = None
                r_g = None
            roi = ROI([params['log_filepath'], params['log_level']],
                                subROIs=params['num_of_subROIs'],
                                angle=params['angle'],
                                roi=roi_range,
                                res=r_g)
            details = {}
            roi.set_initial_ROI(im, details)
            roi.create_ROI_data(im)
            roi_array.append(roi)
    else:
        while True:
            roi = ROI([params['log_filepath'], params['log_level']],
                                subROIs=params['num_of_subROIs'],
                                angle=params['angle'],
                                roi=None,
                                res=None)
            rectangles = []
            if roi_array:
                for r in roi_array:
                    rectangles.append(r.get_roi())
            details = {
                'message': (f'Press "q" after last ROI. - ROI {len(roi_array)}'),
                'patches': rectangles,
            }
            roi.set_initial_ROI(im, details)
            roi.create_ROI_data(im)
            if roi.was_last_call() is True:
                params['num_of_ROIs'] = len(roi_array)
                break
            roi_array.append(roi)
    return roi_array


def get_image_filenames(logger, params, filenames_old):
    """ Gets all image filenames from provided filepath, removes filenames that
        have been processed previously and sorts resulting list

        Attributes:
        logger (obj) :         Logger information
        params (dict) :        Dictionary containing user parameters
        filenames_old (list) : List of filenames that were previously processed

        Returns:
        image_files (list) :   List of image filepaths that have not been
                               processed previously
    """
    logger.debug(f'...Files already processed : {len(filenames_old)}')
    image_path = params['image_folder']
    to_be_processed = [item for item in listdir(image_path)
                       if join(image_path, item) not in filenames_old
                       and item[-3:] == params['image_type']]
    logger.debug(f'...Files to be processed : {len(to_be_processed)}')

    image_files = [join(image_path, f)
                   for f in sorted(to_be_processed)]
    return image_files


def process_images(logger, params, image_files, roi_array):
    """ Processes each image. First converts CSV data (optional) into an image.
        Then creates an Image object for each file, which is then passed to the
        ROI objects for rotating, cropping, collapsing and fitting of a Fano
        funtion to the data (see ROImulti.py)

        Attributes:
        logger (obj) :         Logger information
        params (dict) :        Dictionary containing user parameters
        image_files (list) :   List of filenames that were previously processed
        roi_array (list) :     List of ROI objects
    """
    for idx, im_path in enumerate(image_files):
        print(f'Processing image {idx + 1}')
        im = get_image(logger, params, im_path)
        for roi in roi_array:
            # roi.set_initial_ROI(im)
            roi.create_ROI_data(im)
            roi.process_ROI(params['analysis'], idx, im_path)


def save_data(logger, params, roi_array):
    """ Saves ROI data to a CSV file.

        Attributes:
        logger (obj) :         Logger information
        params (dict) :        Dictionary containing user parameters
        roi_array (list) :     List of ROI objects
    """
    all_data = []

    for idx, roi in enumerate(roi_array):
        output_raw = roi.get_save_data()
        for d in output_raw:
            d['ROI'] = idx
            all_data.append(d)

    save_path = params['save_path']
    save_name = params['save_name']

    df = pd.DataFrame(all_data)
    df = df.sort_values(by=['ROI', 'ID', 'subROI'])
    df = df.reset_index(drop=True)
    
    initial_cols = ['ROI', 'ID', 'subROI']
    new_col_order = initial_cols + [col for col in df.columns
                                    if col not in initial_cols]
    df = df[new_col_order]

    df.to_csv(join(save_path, f'{save_name}_NEWDATA.csv'),
              sep=',',
              index=False)
    
    logger.info('...Data saved')


def plot_data(params, filenames_old, roi_array):
    """ Plots relevant data from each ROI, and saves plot to file - one file per
        ROI.  This plot is aimed at allowing users to check on the progress of
        long running experiments - it should not be used as the final data
        analysis

        Attributes:
        params (dict) :        Dictionary containing user parameters
        filenames_old (list) :  List of filenames that were previously processed
        roi_array (list) :     List of ROI objects
    """

    to_plot = {
        'gaussian of mean': 'Mu',
        'fano of mean': 'Resonance',
        'centre of peak': 'Centre',
        'median of gaussians': 'Median',
        'median of fanos': 'Median',
        'median of centres': 'Median',
    }
    ncols = 4
    nrows = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=False,
                          figsize=(11.7, 8.3))
    time_axis = [(y * params['image_interval'])
                 for y in range(len(filenames_old))]
    
    current_roi = 0
    stop_plotting = False
    for r in range(nrows):
        if stop_plotting: break
        for c in range(ncols):
            data = pd.DataFrame(roi_array[current_roi].get_resonance_data())
            for i in range(roi_array[current_roi].get_num_subROIs()):
                plot_data = data[data['subROI'] == i]
                ax[r, c].plot(time_axis, plot_data[to_plot[params['analysis']]])
            ax[r, c].set_title(f'ROI {current_roi}')
            current_roi += 1
            if current_roi >= len(roi_array):
                stop_plotting = True
                break
        fig.text(0.5, 0.02, 'Time (minutes)',
                 ha='center', va='center', fontsize=24)
        fig.text(0.02, 0.5, 'Resonance (px)',
                 ha='center', va='center', rotation='vertical', fontsize=24)

        plt.savefig(f'{params['save_figure_filename']}')


def get_image(logger, params, im_path):
    """ Takes a filepath to an 'image' file, if the image file is in CSV format
        converts file to a PIL image object, otherwise simply opens image file
        directly.  Also rotates image if required.

        Attributes:
        logger (obj) :  Logger information
        params (dict) : Dictionary containing user parameters
        im_path (str) : Filepath to image file

        Returns:
        image (Image obj) : PIL Image object
    """
    if im_path.split('.')[-1] == 'png':
        im = Image.open(im_path)
        # Check image mode and convert to 8-Bit if necessary
        if im.mode == 'I':
            im32 = np.array(im)
            scale = np.iinfo(np.uint8).max + 1
            im8 = (im32/scale).astype('uint8')
            im = Image.fromarray(im8)
        elif '16' in im.mode:
            im16 = np.array(im)
            scale = np.iinfo(np.uint8).max + 1
            im8 = (im16/scale).astype('uint8')
            im = Image.fromarray(im8)
        im = ImageOps.grayscale(im)
        logger.debug('Images found to PNG format')
    elif im_path.split('.')[-1] == 'tif' or im_path.split('.')[1] == 'tiff':
        im = Image.open(im_path)
        logger.debug('Images found to TIFF format')
    elif im_path.split('.')[-1] == 'csv':
        im = convert_CSV_to_Image(logger, im_path)
        logger.debug('Images found to CSV format')

    # im = ImageEnhance.Contrast(im).enhance(10.0)

    if params['image_align'] == 'horizontal':
        im = im.rotate(90, expand=True)

    return im


def convert_CSV_to_Image(logger, path_to_csv):
    """ Takes a CSV file, containing image information, and converts it to an
        PIL Image object for further processing

        Attributes:
        logger (obj) :      Logger information
        path_to_csv (str) : Filepath to location of 'image' CSV file

        Returns:
        image (Image obj) : PIL Image object
    """
    logger.debug(f'...convert_CSV_to_Image({path_to_csv})')
    data = np.asarray(np.genfromtxt(path_to_csv, delimiter=','))
    image = Image.fromarray((data).astype(np.float64))
    return image


def fano(x, amp, assym, res, gamma, off):
    """ Fano function used for curve-fitting

        Attributes:
        x (float) :    Independant data value (i.e. x-value, in this
                        case pixel number)
        amp (float):   Amplitude
        assym (float): Assymetry
        res (float):   Resonance
        gamma (float): Gamma
        off (float):   Offset (i.e. function bias away from zero)

        Returns:
        float: Dependant data value (i.e. y-value)
    """
    num = ((assym * gamma) + (x - res)) * ((assym * gamma) + (x - res))
    den = (gamma * gamma) + ((x - res)*(x - res))
    return (amp * (num / den)) + off


def fano_array(xdata, amp, assym, res, gamma, off):
    """ Fano function used for curve-fitting

        Attributes:
        x (list) :     Independant data values (i.e. x-values)
        amp (float):   Amplitude
        assym (float): Assymetry
        res (float):   Resonance
        gamma (float): Gamma
        off (float):   Offset (i.e. function bias away from zero)

        Returns:
        list: Dependant data values (i.e. y-values)
    """
    ydata = []
    for x in xdata:
        num = ((assym * gamma) + (x - res)) * ((assym * gamma) + (x - res))
        den = (gamma * gamma) + ((x - res)*(x - res))
        ydata.append((amp * (num / den)) + off)
    return ydata
