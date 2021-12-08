import matplotlib.pyplot as plt
import time
import timeit
import configparser
import json
from PIL import Image  # , ImageEnhance
from os import listdir
from os.path import join
from Functions import ROI, convert_CSV_to_Image


config = configparser.ConfigParser()
config.read('./CK_configuration.ini')

save_path = config['Data']['save_path']
save_name = config['Data']['save_name']

image_folder = config['Data']['image_folder']
image_type = config['Data']['image_type']
image_align = config['Data']['image_align']
save_figure_filename = config['Data']['save_figure_filename']

if config['Data'].getboolean('live_watch'):
    try:
        sleep_time = config['Loop'].getfloat('sleep_time')
    except KeyError:
        print("No sleep time provided, it is now set to 10 minutes")
        sleep_time = 0

num_of_ROIs = 1
image_interval = 1.0
angle = None
roi_ranges = None
res_gamma = None

try:
    num_of_ROIs = config['Image'].getint('num_of_ROIs')
    if num_of_ROIs is None:
        num_of_ROIs = 1

    if config['Data'].getboolean('initialise_image'):
        image_interval = config['Image'].getfloat('image_interval')
        if image_interval is None:
            image_interval = 1.0

        if config['Data'].getboolean('initialise_image'):
            try:
                angle = config['Image'].getfloat('image_angle')
            except KeyError:
                angle = None
            try:
                roi_ranges = json.loads(config['Image']['rois'])
            except KeyError:
                roi_ranges = None
            try:
                res_gamma = json.loads(config['Image']['resonance_gamma'])
            except KeyError:
                res_gamma = None
except KeyError:
    pass

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
    roi_array.append(ROI(angle=angle, roi=roi_range, res=r_g))

# filenames_old = listdir(image_folder)
# print(f'Original filenames -> \n {sorted(filenames_old)}')
filenames_old = []
for i in range(1):
    to_be_processed = [item for item in listdir(image_folder)
                       if item not in filenames_old]
    filenames_old = listdir(image_folder)

    image_files = [join(image_folder, f) for f in sorted(to_be_processed)
                   if f[-3:] == image_type]

    # print(to_be_processed)
    # print(image_files)
    # image_files = image_files[:3]
    print(f'{len(image_files)} files will now be processed')

    output_err = ''
    for idx, im_path in enumerate(image_files):
        print(f'Processing image {idx}')

        start = timeit.timeit()
        if im_path[-3:] == 'png':
            im = Image.open(im_path)
        elif im_path[-3:] == 'csv':
            im = convert_CSV_to_Image(im_path)

        if image_align == 'horizontal':
            im = im.rotate(90, expand=True)

        # im = ImageEnhance.Contrast(im).enhance(10.0)

        end = timeit.timeit()
        print(f'Loading data took: {(end - start):.4} s')

        for idx, roi in enumerate(roi_array):
            roi.set_initial_ROI(im)
            roi.create_ROI_data(im)
            roi.process_ROI()
            r2 = roi.get_resonance_data()[-1]['r2']
            if r2 < 0.8:
                output_err = f'{output_err}\n{r2},{idx},{im_path}'
    with open(join(save_path, f'{save_name}_BAD_R2.csv'), 'a') as f:
        f.write(output_err)

    for idx, roi in enumerate(roi_array):
        roi.save_data(save_path, f'{save_name}_ROI_{idx}')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    time_axis = [(y * image_interval) for y in range(len(filenames_old)-1)]
    for roi in roi_array:
        res_data = [d['res'] for d in roi.get_resonance_data()]
        FWHM_data = [d['FWHM'] for d in roi.get_resonance_data()]
        r2_data = [round(d['r2'], 4) for d in roi.get_resonance_data()]
        ax1.plot(time_axis, res_data)
        ax2.plot(time_axis, FWHM_data)
        ax3.plot(time_axis, r2_data)
        ax1.set_xlabel('Time (minutes)')
        ax2.set_xlabel('Time (minutes)')
        ax3.set_xlabel('Time (minutes)')
        plt.savefig(save_figure_filename)

    if sleep_time > 0:
        print('Waiting ...')
        time.sleep(sleep_time * 60)
