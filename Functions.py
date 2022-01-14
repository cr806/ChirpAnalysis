import numpy as np
import csv
import logging
from PIL import Image
# import json
# import timeit


def convert_CSV_to_Image(path_to_csv):
    logging.debug(f'...convert_CSV_to_Image({path_to_csv})')
    data = np.asarray(np.genfromtxt(path_to_csv, delimiter=','))
    return Image.fromarray((data).astype(np.float64))


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


def get_image(im_path, im_align):
    if im_path[-3:] == 'png':
        im = Image.open(im_path)
    elif im_path[-3:] == 'csv':
        im = convert_CSV_to_Image(im_path)

    if im_align == 'horizontal':
        im = im.rotate(90, expand=True)

    return im
