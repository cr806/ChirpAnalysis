import time
import json
import timeit
import logging
import configparser
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # , ImageEnhance
from os import listdir
from os.path import join
from Functions import convert_CSV_to_Image
from ROImulti import ROImulti as ROI


config_filepath = './CK_configuration_multi.ini'
log_filepath = 'ChirpAnalysis_multi.log'

logging.basicConfig(filename=log_filepath,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    encoding='utf-8',
                    level=logging.INFO)

try:
    with open(config_filepath, 'r') as f:
        config = configparser.ConfigParser()
        config.read(config_filepath)
except FileNotFoundError:
    print('Configuration file can not be found')
    logging.error('Configuration file can not be found')
    quit()

try:
    save_path = config['Data']['save_path']
    save_name = config['Data']['save_name']

    image_folder = config['Data']['image_folder']
    image_type = config['Data']['image_type']
    image_align = config['Data']['image_align']
    save_figure_filename = config['Data']['save_figure_filename']

    sleep_time = 0.33

    if config['Data'].getboolean('live_watch'):
        try:
            sleep_time = config['Loop'].getfloat('sleep_time')
        except KeyError:
            print("No sleep time provided, it is now set to 2 seconds")
            logging.info('No sleep time provided')
    else:
        logging.info('Live watch not requested')
except KeyError:
    print('Configuration file incorrectly formatted')
    logging.error('No [Data] section provided within configuration file')
    quit()


num_of_ROIs = 1
num_of_subROIs = 1
r2_threshold = 0.8
image_interval = 1.0
angle = None
roi_ranges = None
res_gamma = None

try:
    num_of_ROIs = config['Image'].getint('num_of_ROIs')
    if num_of_ROIs is None:
        num_of_ROIs = 1
        logging.info('No ROI count provided')

    num_of_subROIs = config['Image'].getint('num_of_subROIs')
    if num_of_subROIs is None:
        num_of_subROIs = 1
        logging.info('No subROI count provided')

    r2_threshold = config['Image'].getfloat('r2_threshold')
    if r2_threshold is None:
        r2_threshold = 0.8
        logging.info('No r2 provided')

    if config['Data'].getboolean('initialise_image'):
        image_interval = config['Image'].getfloat('image_interval')
        if image_interval is None:
            image_interval = 1.0
            logging.info('No image interval provided')

        try:
            angle = config['Image'].getfloat('image_angle')
        except KeyError:
            angle = None
            logging.info('No angle provided')
        try:
            roi_ranges = json.loads(config['Image']['rois'])
        except KeyError:
            roi_ranges = None
            logging.info('No ROI coordinates provided')
        try:
            res_gamma = json.loads(config['Image']['resonance_gamma'])
        except KeyError:
            res_gamma = None
            logging.info('No resonance/gamma data provided')
    else:
        logging.log('Image initialised interactively')

except KeyError:
    logging.info('No [Image] section provided within configuration file')

roi_array = []
for roi in range(num_of_ROIs):
    try:
        r_g = res_gamma[roi]
    except (IndexError, TypeError):
        r_g = None
    try:
        roi_range = roi_ranges[roi]
    except (IndexError, TypeError):
        roi_range = None
        r_g = None
    roi_array.append(ROI(subROIs=num_of_subROIs,
                         angle=angle,
                         roi=roi_range,
                         res=r_g))

filenames_old = list()
# for i in range(1):
while True:
    to_be_processed = [item for item in listdir(image_folder)
                       if item not in filenames_old]
    filenames_old.extend(to_be_processed)

    image_files = [join(image_folder, f) for f in sorted(to_be_processed)
                   if f[-3:] == image_type]

    if image_files:  # Check if there are images ready to be processed
        print(f'{len(image_files)} files will now be processed')
        logging.info(f'{len(image_files)} files processed')

        for idx, im_path in enumerate(image_files):
            print(f'Processing image {idx}')

            start = timeit.timeit()
            if im_path[-3:] == 'png':
                im = Image.open(im_path)
                logging.debug('Images found to PNG format')
            elif im_path[-3:] == 'csv':
                im = convert_CSV_to_Image(im_path)
                logging.debug('Images found to CSV format')

            if image_align == 'horizontal':
                im = im.rotate(90, expand=True)

            # im = ImageEnhance.Contrast(im).enhance(10.0)

            end = timeit.timeit()
            logging.info(f'Loading data took: {(end - start):.4}s')

            for idx, roi in enumerate(roi_array):
                roi.set_initial_ROI(im)
                roi.create_ROI_data(im)
                roi.process_ROI()

        # Filter and save resulting data to CSV file
        for idx, roi in enumerate(roi_array):
            output_raw = roi.get_save_data()
            header = output_raw[0]
            header.append('Image Path')
            output_good = [header]
            output_bad = [header]

            for i, row in enumerate(output_raw[1:]):
                row.append(image_files[i])
                # ~(tilde)num -> count from end starting at 0
                if np.mean(row[~num_of_subROIs:~0]) < r2_threshold:
                    output_bad.append(row)
                else:
                    output_good.append(row)

            output_good_str = [str(o)[1:-1].replace("'", "")
                               for o in output_good]
            with open(join(save_path, f'{save_name}_ROI_{idx}.csv'),
                      'w') as f:
                f.write('\n'.join(output_good_str))

            output_bad_str = [str(o)[1:-1].replace("'", "")
                              for o in output_bad]
            with open(join(save_path, f'{save_name}_ROI_{idx}_BAD.csv'),
                      'w') as f:
                f.write('\n'.join(output_bad_str))
            logging.info('...Data saved')

        # Plot and save data as it is processed
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
        time_axis = [(y * image_interval) for y in range(len(filenames_old))]
        for roi in roi_array:
            res_data = [np.mean(d['res']) for d in roi.get_resonance_data()]
            FWHM_data = [np.mean(d['FWHM']) for d in roi.get_resonance_data()]
            r2_data = [np.mean(d['r2']) for d in roi.get_resonance_data()]
            ax1.plot(time_axis, res_data)
            ax2.plot(time_axis, FWHM_data)
            ax3.plot(time_axis, r2_data)
            ax1.set_xlabel('Time (minutes)')
            ax2.set_xlabel('Time (minutes)')
            ax3.set_xlabel('Time (minutes)')
            plt.savefig(save_figure_filename)

    if sleep_time > 0:
        print(f'Waiting for {sleep_time} minutes')
        time.sleep(sleep_time * 60)
