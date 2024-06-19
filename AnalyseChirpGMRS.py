import time
import logging
import os
import Functions as func
from datetime import datetime

config_filepath = './Configuration.ini'
dt_string = datetime.now().strftime('%Y%m%d-%H%M%S')
log_filepath = f'Analysis_{dt_string}.log'
log_level = logging.INFO
logger = func.setup_logger(log_filepath, log_level)

params = func.get_simulation_params(logger, config_filepath)
params['log_filepath'] = log_filepath
params['log_level'] = log_level

params['save_path']

# Creates the save directory if it doesn't already exist.
try:
    os.makedirs(params['save_path'], exist_ok=True)
    print(f'Directory {params['save_path']} available.')
    logger.info(f'Directory {params['save_path']} available.')
except OSError as e:
    print(f'Error with save directory {params['save_path']}: {e}')
    logger.error(f'Error with save directory {params['save_path']}: {e}')
    exit(1)

roi_array = None

filenames_old = list()
while True:
    print('Waiting for image files to be ready ...')
    image_files = func.get_image_filenames(logger, params, filenames_old)
    filenames_old.extend(image_files)

    if image_files:  # Check if there are images ready to be processed
        print(f'{len(image_files)} files will now be processed')
        logger.info(f'{len(image_files)} files to be processed')
        if not roi_array:
            roi_array = func.register_ROIs(logger, params, image_files)
        func.process_images(logger, params, image_files, roi_array)
        func.save_data(logger, params, roi_array)
        func.plot_data(params, filenames_old, roi_array)
        break

    if params['sleep_time'] > 0:
        print(f"Waiting for {params['sleep_time']} minutes")
        logger.info(f"...Sleeping for {params['sleep_time']} minutes")
        time.sleep(params['sleep_time'] * 60)
