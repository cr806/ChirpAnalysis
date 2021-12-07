import matplotlib.pyplot as plt
import time
import json
import timeit
import configparser
from PIL import Image  # , ImageEnhance
from os import listdir
from os.path import join
from Functions import ROI, convert_CSV_to_Image


config = configparser.ConfigParser()
config.read('./CK_configuration.ini')
print(config)

save_path = config['Config']['save_path']
save_name = config['Config']['save_name']

image_folder = config['Config']['image_folder']
image_type = config['Config']['image_type']
image_align = config['Config']['image_align']
save_figure_filename = config['Config']['save_figure_filename']

sleep_time = config['Config'].getfloat('sleep_time')

num_of_ROIs = config['Config'].getint('num_of_ROIs')

roi_array = []
first_image_processed = []
for roi in range(num_of_ROIs):
    roi_array.append(ROI())
    first_image_processed.append(False)

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
    print(f'{len(image_files)} files will now be processed')
    image_files = image_files[:1]

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
            if not first_image_processed[idx]:
                roi.set_initial_ROI(im)
                roi.create_ROI_data(im)
                first_image_processed[idx] = True
            roi.create_ROI_data(im)
            roi.process_ROI()
            # roi.get_resonance_data(disp=True)

    for idx, roi in enumerate(roi_array):
        roi.save_data(save_path, f'{save_name}_ROI_{idx}')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    for roi in roi_array:
        res_data = [d['res'] for d in roi.get_resonance_data()]
        FWHM_data = [d['FWHM'] for d in roi.get_resonance_data()]
        r2_data = [round(d['r2'], 2) for d in roi.get_resonance_data()]
        ax1.plot(res_data)
        ax2.plot(FWHM_data)
        ax3.plot(r2_data)
        # ax1.set_xlim(0, 50)
        # ax1.set_ylim(0, 300)
        # ax2.set_xlim(0, 50)
        # ax2.set_ylim(60, 150)
        # ax3.set_xlim(0, 50)
        # ax3.set_ylim(0.98, 1.01)
        plt.savefig(save_figure_filename)

    print('Waiting ...')
    time.sleep(sleep_time * 60)
