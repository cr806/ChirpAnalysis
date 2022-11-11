import csv
import logging
from PIL import Image, ImageOps
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


def register_ROIs(params):
    """ Calls ROI constructors and stores resulting ROI object in array

        Attributes:
        params (dict) :    Dictionary containing user parameters

        Returns:
        roi_array (list) : List of ROI objects
    """
    roi_array = []
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
        roi_array.append(ROI([params['log_filepath'], params['log_level']],
                             subROIs=params['num_of_subROIs'],
                             angle=params['angle'],
                             roi=roi_range,
                             res=r_g))
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

        start = timeit.timeit()
        im = get_image(logger, params, im_path)
        # if im_path[-3:] == 'png':
        #     im = Image.open(im_path)
        #     logger.debug('Images found to PNG format')
        # elif im_path[-3:] == 'csv':
        #     im = convert_CSV_to_Image(logger, im_path)
        #     logger.debug('Images found to CSV format')

        # if params['image_align'] == 'horizontal':
        #     im = im.rotate(90, expand=True)

        # im = ImageEnhance.Contrast(im).enhance(10.0)

        end = timeit.timeit()
        logger.info(f'Loading data took: {(end - start):.4}s')

        for idx, roi in enumerate(roi_array):
            roi.set_initial_ROI(im)
            roi.create_ROI_data(im)
            roi.process_ROI()


def save_data(logger, params, roi_array, filenames_old):
    """ Saves ROI data to a number of CSV files. One file will contain all data
        in a format that is easy to read in as a dataframe.  Also two CSVs are
        produced per ROI, one containing "GOOD" results, the other "BAD"
        results, good and bad determined by the r**2 value of the Fano fit and
        the threshold level set by the user within the configuration file

        Attributes:
        logger (obj) :         Logger information
        params (dict) :        Dictionary containing user parameters
        roi_array (list) :     List of ROI objects
        filenames_old (list) : List of filenames that were previously processed
    """
    # Filter and save resulting data to CSV file
    output_all_header = ['ID', 'ROI', 'subROI', 'ROI_x1',
                         'ROI_y1', 'ROI_x2', 'ROI_y2', 'angle',
                         'amp', 'assym', 'res', 'gamma', 'off',
                         'FWHM', 'r2', 'image-path']
    output_all = [output_all_header]
    data_idx = [5 + (i * params['num_of_subROIs']) for i in range(7)]
    for idx, roi in enumerate(roi_array):
        output_raw = roi.get_save_data()
        header = ['ID']
        header.extend(output_raw[0])
        header.append('Image Path')
        output_good = [header]

        for i, row in enumerate(output_raw[1:]):
            for j in range(params['num_of_subROIs']):
                output_all_row = [i, idx, j]
                output_all_row.extend(row[:5])

                data_i = [d+j for d in data_idx]
                subROIdata = [row[z] for z in data_i]

                output_all_row.extend(subROIdata)
                output_all_row.append(filenames_old[i])
                output_all.append(output_all_row)

        for i, row in enumerate(output_raw[1:]):
            output_row = [i]
            output_row.extend(row)
            output_row.append(filenames_old[i])
            # ~(tilde)num -> count from end starting at 0
            output_good.append(output_row)

        save_path = params['save_path']
        save_name = params['save_name']
        output_good_str = [str(o)[1:-1].replace("'", "") for o in output_good]
        with open(join(save_path, f'{save_name}_ROI_{idx}.csv'), 'w') as f:
            f.write('\n'.join(output_good_str))

        logger.info('...Data saved')

    output_all_str = [str(o)[1:-1].replace("'", "") for o in output_all]
    with open(join(save_path, f'{save_name}_ALLDATA.csv'), 'w') as f:
        f.write('\n'.join(output_all_str))
    logger.info('...All raw data saved')


def plot_data(params, filenames_old, roi_array):
    """ Plots relevant data from each ROI, and saves plot to file.  This plot
        is aimed at allowing users to check on the progress of long running
        experiments - it should not be used as the final data analysis

        Attributes:
        params (dict) :        Dictionary containing user parameters
        filenames_old (list) : List of filenames that were previously processed
        roi_array (list) :     List of ROI objects
    """
    # Plot and save data as it is processed
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    time_axis = [(y * params['image_interval'])
                 for y in range(len(filenames_old))]
    for roi in roi_array:
        ref_data = roi.get_reference_data()
        res_data = [np.mean(np.subtract(d['res'], ref_data[2]))
                    for d in roi.get_resonance_data()]

        FWHM_data = [np.mean(np.subtract(d['FWHM'], ref_data[5]))
                     for d in roi.get_resonance_data()]

        r2_data = [np.mean(d['r2']) for d in roi.get_resonance_data()]

        ax1.plot(time_axis, res_data)
        ax2.plot(time_axis, FWHM_data)
        ax3.plot(time_axis, r2_data)
        ax1.set_ylabel('Resonance (px)')
        ax2.set_ylabel('FWHM (px)')
        ax3.set_ylabel('R^2')
        ax1.set_xlabel('Time (minutes)')
        ax2.set_xlabel('Time (minutes)')
        ax3.set_xlabel('Time (minutes)')
        plt.savefig(params['save_figure_filename'])


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
        im = ImageOps.grayscale(im)
        logger.debug('Images found to PNG format')
    elif im_path.split('.')[-1] == 'tif' or im_path.split('.')[1] == 'tiff':
        im = Image.open(im_path)
        logger.debug('Images found to TIFF format')
    elif im_path.split('.')[-1] == 'csv':
        im = convert_CSV_to_Image(logger, im_path)
        logger.debug('Images found to CSV format')

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


def get_data(data_filename):
    """ Open and parses CSV data file.  Creates dictionary object with parsed
        data.  Also extracts number of subROIs that are listed within the data
        file.

        Attributes:
        data_filename (str) :  Filepath to CSV data file

        Returns:
        data_dict (dict) :     Dictionary of data parsed from CSV data file
        num_subROIS (int) :    Number of subROIs as parsed from CSV data file
    """
    data = []
    with open(data_filename, mode='r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            data.append([r.strip() for r in row])

    header = data[0]
    data = data[1:]

    num_subROIs = len([h for h in header if 'amp' in h])

    data_dict = {}
    for idx, h in enumerate(header):
        data_dict[h] = []
        for d in data:
            if h != 'Image Path':
                data_dict[h].append(float(d[idx]))
            else:
                data_dict[h].append(d[idx])

    return data_dict, num_subROIs
