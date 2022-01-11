import numpy as np
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
