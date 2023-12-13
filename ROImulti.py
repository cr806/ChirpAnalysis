import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.optimize import curve_fit
from ImageProcessor import ImageProcessor

from typing import List, Tuple


class ROImulti:
    """ Region of Interest class - manages ROI data for images as they
        are passed in.  Including fitting relevant curves and storing
        and saving out results.

        Attributes:
        im (PIL image):        Used to store image data for processing
        first (bool):          Flag indicating first image has not been
                               processed
        roi (tuple):           Tuple containing tuple(x1, y1) and
                               tuple(x2, y2) the top-left and bottom-right
                               coordinates of the ROI chosen by the user
        angle (float):         Image rotation angle
        initial_values (list): List of floats containing initial guess values
                               for curve fitting, fit parameters are:
                               (amplitude, assymetry, resonance value, gamma,
                               offset)
        resonance_data (list): List of dictionaries containing results from
                               curve fit, dictionaries consist of {amplitude,
                               assymetry, resonance value, gamma, offset, FWHM,
                               r-squared value}
        fig (figure handle):   Used to hold figure data for plotting
        ax (axis handle):      Used to hold axis data for plotting
    """

    def __init__(self, log_data, subROIs=1, angle=None, roi=None, res=None):
        self.log_data = log_data
        self.im = None
        self.first = True
        self.last_call_status = False
        self.subROIs = subROIs
        self.roi = roi
        self.angle = angle
        self.res = res
        # self.initial_values = [0, 0, 0, 0, 0]
        # if res is not None:
        #     self.initial_values = [0, 0, res[0], res[1], 0]
        self.resonance_data = []
        self.reference_data = None
        self.fig = None
        self.ax = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_data[1])
        fh = logging.FileHandler(filename=self.log_data[0], encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s : %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info(f'__init__({self}, angle={angle}, roi={roi}, '
                         'res={res})')

    def set_initial_ROI(self, im, info):
        """ Method called to set-up ROI data from first image

            Usage:
            Call this method if ROI coordinates and correction angle are
            unknown, method will plot image and ask user to select points for
            anlge correction, it will rotate the image (the image data is not
            overwritten however) and then ask the user to select points to
            define the ROI (the image will will not be cropped at this point
            only the ROI data will be stored), finally the user will be asked
            to select the resonance position and width (this data is used as
            the initial guess for the curve fitting)

            Attributes:
            im (PIL image): Used to store image data
        """
        self.logger.debug('...set_initial_ROI({self}, {im})')
        self.im = im

        if self.first:
            if self.angle is None:
                figure = plt.subplots(1, 2, figsize=(12, 6))
                details = {
                    'mode': 'rotate',
                    'figure': figure,
                    'im': self.im,
                }
                image_rotator = ImageProcessor(self.log_data, details)
                self.angle = image_rotator.get_angle()
                self.logger.debug('Angle correction')

            if self.roi is None:
                figure = plt.subplots(1, 2, figsize=(12, 6),
                                      gridspec_kw={'width_ratios': [3, 1]})
                temp = self.im.rotate(self.angle)
                details = {
                    'mode': 'crop',
                    'figure': figure,
                    'im': temp,
                    'message': info['message'],
                    'patches': info['patches'],
                }
                image_cropper = ImageProcessor(self.log_data, details)
                self.roi = tuple(image_cropper.get_coords())
                self.last_call_status = image_cropper.was_last_call()
                self.logger.debug('ROI selection')

            self.im = self.im.crop((self.roi[0][0], self.roi[0][1],
                                    self.roi[1][0], self.roi[1][1]))
            # if self.res is None:
            #     figure = plt.subplots(1, 2, figsize=(12, 6))
            #     details = {
            #         'mode': 'resonance',
            #         'figure': figure,
            #         'im': self.im,
            #     }
            #     p0 = ImageProcessor(self.log_data, details)
            #     # self.set_initial_values(amp=0.2, assym=-100,
            #     #                         res=None, gamma=None, off=40)
            #     self.initial_values[2] = p0.get_initial_values()[0]
            #     self.initial_values[3] = p0.get_initial_values()[1]
            #     self.logger.debug('Resonance/Gamma selection')
            self.first = False

    def create_ROI_data(self, im, plot=False):
        """ Method used to create ROI data from full image.

            Usage:
            This method is used to rotate and crop a full image according to
            the angle and ROI data stored within the class instance (and
            created using the set_initial_ROI() method).  After adjusting the
            image this method then calls the process_ROI() method which
            performs the curve-fitting

            Attributes:
            im (PIL image): Used to store image data, if an image is passed to
                            this method then the method uses the passed image
                            and the already stored angle and ROI data to rotate
                            and crop the image.  If no image is passed then the
                            original image (stored by the set_initial_ROI()
                            method) is cropped and rotated.
        """

        self.logger.debug(f'...create_ROI_data({self}, {im}, plot={plot})')
        self.im = im
        temp = self.im.rotate(self.angle)
        (x1, y1), (x2, y2) = self.roi
        self.im = np.array(temp.crop((x1, y1, x2, y2)))
        # self.process_ROI(plot=plot)

    # def determine_assym(self, plot=False):
    #     """ CURRENTLY NOT REQUIRED BY SETTING INITIAL VALUE TO 0
    #         Determine positive or negative assymetry value from initial
    #         image.
    #         Curve-fit algorithm seems to requires a positive or negative
    #         initial guess for the fitting of the Fano function, if the guess
    #         is the wrong sign then curve-fit will converge to a "bad" fit
    #     """

    #     self.logger.debug(f'...determine_assym({self}, plot={plot})')
    #     self.initial_values[1] = 100
    #     self.process_ROI(plot=plot)
    #     try:
    #         r2pos = self.resonance_data[-1]['r2']
    #     except IndexError:
    #         r2pos = 0
    #     self.initial_values[1] = -100
    #     self.process_ROI(plot=plot)
    #     try:
    #         r2neg = self.resonance_data[-1]['r2']
    #     except IndexError:
    #         r2neg = 0
    #     print(f'Positive r2 -> {r2pos}  Negative r2 -> {r2neg}')
    #     if r2pos > r2neg:
    #         self.initial_values[1] = 100
    #     self.resonance_data = []
    
    def fit_gaussian(self, x_data: np.ndarray, y_data: np.ndarray, plot=False):
        '''Function to fit a Gaussian to the data, returning the fitted parameters
        (or zeros if no fit possible)'''
        def gaussian(x, mu, A, sigma, bias):
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + bias

        initial_guess = [np.argmax(y_data), 200,
                         len(y_data)/4, np.amin(y_data)]
        bounds = ([0,           0,     0,           0],
                  [len(y_data), 2**16, len(y_data), 2**16])
        popt_dict = {
            'Mu': 0,
            'Amplitude': 0,
            'Sigma': 0,
            'Bias': 0,
            'R^2': 0,
        }
        try:
            popt, _ = curve_fit(gaussian, x_data, y_data,
                               p0=initial_guess, bounds=bounds)
            mu, A, sigma, bias = popt

            ss_res = np.sum(gaussian(x_data, *popt)**2)
            ss_tot = np.sum(y_data**2)

            popt_dict['Mu'] = mu
            popt_dict['Amplitude'] = A
            popt_dict['Sigma'] = sigma
            popt_dict['Bias'] = bias
            popt_dict['R^2'] = ss_res / ss_tot

            if plot:
                _, ax = plt.subplots(1, figsize=(12, 6))
                ax.plot(x_data, y_data, 'b')
                ax.plot(x_data, gaussian(x_data, *popt), 'r')
                plt.show()

        except RuntimeError as e:
            print(f'Curve fitting did not converge: {e}')
            self.logger.info(f'Curve fitting did not converge: {e}')
        except ValueError as e:
            print(f'Curve fitting failed: {e}')
            self.logger.info(f'Curve fitting failed: {e}')
        return popt_dict

    def fit_fano(self, x_data, y_data, plot=False):
        def fano(x, amp, assym, res, gamma, bias):
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
            return (amp * (num / den)) + bias

        # initial_guess = np.array(self.initial_values)
        # param_scale = [10, 1, self.initial_values[2],
        #                self.initial_values[3], np.amin(y_data)]
        initial_guess = [np.max(y_data), 0, np.argmax(y_data),
                         len(y_data)/4, 0]
        bounds = ([-1000, -100,  0,   0,    0],
                  [1000, 100,  500, 500,  500])
        param_scale = [10, 1, np.argmax(y_data),
                       len(y_data)/4, np.amin(y_data)]

        popt_dict = {
            'Amplitude': 0,
            'Assymetry': 0,
            'Resonance': 0,
            'Gamma': 0,
            'Bias': 0,
            'FWHM': 0,
            'R^2': 0,
        }
        try:
            popt, _ = curve_fit(fano, x_data, y_data,
                               p0=initial_guess, bounds=bounds,
                               x_scale=param_scale)

            A, b, c, d, e = popt

            FWHM = ((2 * np.sqrt(4 * ((b*d)**2) * (b**2 + 2))) /
                    ((2 * b**2) - 4))

            ss_res = np.sum(fano(x_data, *popt)**2)
            ss_tot = np.sum(y_data**2)
            
            popt_dict['Amplitude'] = A
            popt_dict['Assymetry'] = b
            popt_dict['Resonance'] = c
            popt_dict['Gamma'] = d
            popt_dict['Bias'] = e
            popt_dict['FWHM'] = FWHM
            popt_dict['R^2'] = ss_res / ss_tot

            if plot:
                _, ax = plt.subplots(1, figsize=(12, 6))
                ax.plot(x_data, y_data, 'b')
                ax.plot(x_data, fano(x_data, *popt), 'r')
                plt.show()

        except RuntimeError as e:
            print(f'Curve fitting did not converge: {e}')
            self.logger.info(f'Curve fitting did not converge: {e}')
        except ValueError as e:
            print(f'Curve fitting failed: {e}')
            self.logger.info(f'Curve fitting failed: {e}')

        return popt_dict

    def get_quartiles(self, data: np.ndarray) -> Tuple[float, float, float]:
        if len(data) == 0:
            return (0, 0, 0)
        # Calculate the CDF and normalise
        cdf = np.cumsum(data)
        cdf_norm = cdf / cdf[-1]

        # Find the values at the specified percentiles
        return tuple(np.interp(np.array([0.25, 0.50, 0.75]),
                               cdf_norm,
                               data))
    def filter_data(self, data: np.ndarray, threshold: float=0.0) -> np.ndarray:
        # Sort data
        data = np.sort(data)
        
        # Calculate the CDF and normalise
        cdf = np.cumsum(data)
        cdf_norm = cdf / cdf[-1]

        # Filter data by threshold
        Lo = np.interp(threshold, cdf_norm, data)
        Hi = np.interp((1-threshold), cdf_norm, data)
        data = data[np.where(data >= min(Lo, Hi))]
        data = data[np.where(data <= max(Lo, Hi))]
        return data

    def get_centre_of_peak(self, y_data: np.ndarray, filter: float):
        '''Function to return the 'centre' of the peak within the data.
        Accepts a 1D data array'''
        max_idx = np.argmax(y_data)
        threshold = y_data[max_idx] * filter
        idxs = np.where(y_data > threshold)[0]
        return {
            'Analysis Method': 'Centre of peak',
            'Lower Threshold': idxs[0],
            'Maximum': max_idx,
            'Centre': np.mean(idxs),
            'Upper Threshold': idxs[-1],
        }

    def get_fano_of_mean(self, data: np.ndarray, plot=False):
        x_data = np.arange(0, data.shape[0])
        y_data = np.mean(data, axis=1)
        return self.fit_fano(x_data, y_data, plot)

    def get_gaussian_of_mean(self, data: np.ndarray, plot=False):
        x_data = np.arange(0, data.shape[0])
        y_data = np.mean(data, axis=1)
        return self.fit_gaussian(x_data, y_data, plot)

    def get_median_of_centres(self, data: np.ndarray,
                              threshold: float = 0.75/2):
        centres = []
        x_data = np.arange(0, data.shape[0])
        for i in range(0, data.shape[1]):
            y_data = data[:, i]
            centre = self.fit_fano(x_data, y_data)['Centre']
            centres.append(centre)

        centres = self.filter_data(centres, threshold)
        F, M, T = self.get_quartiles(centres)
        return {
            'Analysis Method': 'Centre of Peak',
            'First Quartile': F,
            'Median': M,
            'Third Quartile': T,
        }

    def get_median_of_gaussians(self, data: np.ndarray,
                                threshold: float = 0.75/2):
        mus = []
        x_data = np.arange(0, data.shape[0])
        for i in range(0, data.shape[1]):
            y_data = data[:, i]
            mu = self.fit_gaussian(x_data, y_data)['Mu']
            mus.append(mu)

        mus = self.filter_data(mus, threshold)
        F, M, T = self.get_quartiles(mus)
        return {
            'Analysis Method': 'Gaussian',
            'First Quartile': F,
            'Median': M,
            'Third Quartile': T,
        }

    def get_median_of_fanos(self, data: np.ndarray, threshold: float = 0.75/2):
        res = []
        x_data = np.arange(0, data.shape[0])
        for i in range(0, data.shape[1]):
            y_data = data[:, i]
            r = self.fit_fano(x_data, y_data)['Resonance']
            res.append(r)

        res = self.filter_data(res, threshold)
        F, M, T = self.get_quartiles(res)
        return {
            'Analysis Method': 'Fano',
            'First Quartile': F,
            'Median': M,
            'Third Quartile': T,
        }

    def process_ROI(self, method, idx, im_path, plot=False):
        """ Processes ROI from initial image, first collapsing image into 1D,
            then applies curve_fit algorithm - updating resonance_data with
            the results
        """
        self.logger.debug(f'...process_ROI({self}, plot={plot})')

        """
        get ROI dimensions (shape of im)
        integer-divide dimension by number of subROIs
        transpose image array
        slice image array im[0:20]
        transpose slice
        """
        if method == 'median_gaussian-2D':
            self.subROIs = 1
        
        ROI_x_size = np.shape(self.im)[1]
        subROI_size = ROI_x_size // self.subROIs
        transposed_im = np.transpose(self.im)

        for i in range(self.subROIs):
            result = {}
            result['ID'] = idx
            result['subROI'] = i
            result['ROI_x1'] = self.roi[0][0]
            result['ROI_y1'] = self.roi[0][1]
            result['ROI_x2'] = self.roi[1][0]
            result['ROI_y2'] = self.roi[1][1]
            result['Angle'] = self.angle

            subROI = np.transpose(
                    transposed_im[(i*subROI_size):((i+1)*subROI_size)])

            if method == 'fano of mean':
                self.logger.info(f'... using simple Fano fit')
                result.update(self.get_fano_of_mean(subROI))
            elif method == 'gaussian of mean':
                self.logger.info(f'... using simple Gaussian fit')
                result.update(self.get_gaussian_of_mean(subROI))
            elif method == 'centre of peak':
                self.logger.info(f'... using simple maximum of centre fit')
                y_data = np.mean(subROI, axis=1)
                result.update(self.get_centre_of_peak(y_data, 0.75))
            elif method == 'median of gaussians':
                self.logger.info(f'... using multi-Fano fit')
                result.update(self.get_median_of_gaussians(subROI, 0.75/2))
            elif method == 'median of fanos':
                self.logger.info(f'... using multi-Gaussian fit')
                result.update(self.get_median_of_gaussians(subROI, 0.75/2))
            elif method == 'median of centres':
                self.logger.info(f'... using multi-Centre fit')
                result.update(self.get_median_of_gaussians(subROI, 0.75/2))
            else:
                self.logger.info((f'... no or incorrect analsysis method '
                                  f'selected, using find maximum analysis '
                                  f'method'))
                y_data = np.mean(subROI, axis=1)
                result.update(self.get_centre_of_peak(y_data, 0.75))

            result['image path'] = im_path

            self.resonance_data.append(result)

    def set_initial_values(self, amp=None, assym=None,
                           res=None, gamma=None, off=None,
                           FWHM=None, r2=None):
        """ Setter method for directly entering initial values for the
            class instance.

            Attributes:
            amp (float):   Amplitude
            assym (float): Assymetry
            res (float):   Resonance
            gamma (float): Gamma
            off (float):   Offset (i.e. function bias away from zero)
            FWHM (float):  Required in method call due to results dictionary
                          however, not used in curve-fit algorithm
            r2 (float):    Required in method call due to results dictionary
                          however, not used in curve-fit algorithm
        """
        self.logger.debug(f"""set_initial_values({self}, amp={amp},
                          assym={assym}, res={res}, gamma={gamma}, off={off},
                          FWHM={FWHM}, r2={r2})""")
        self.initial_values[0] = np.mean(amp)
        self.initial_values[1] = np.mean(assym)
        self.initial_values[2] = np.mean(res)
        self.initial_values[3] = np.mean(gamma)
        self.initial_values[4] = np.mean(off)

        if self.reference_data is None:
            self.reference_data = [amp,
                                   assym,
                                   res,
                                   gamma,
                                   off,
                                   FWHM]

    def get_inital_values(self):
        """ Getter function used to return initial_values to the caller

            Return:
            initial_values (list): List of intial calculation values
        """
        self.logger.debug(f'...get_inital_values({self})')
        print(self.initial_values)
        return self.initial_values

    def get_save_data(self):
        """ Return the results as a formattted list ready for saving.
        """
        # first = True
        # output = []

        # # for d in self.resonance_data:
        # #     text_keys = ['ROI_x1', 'ROI_y1', 'ROI_x2', 'ROI_y2', 'Angle']
        # #     text_vals = [self.roi[0][0], self.roi[0][1],
        # #                  self.roi[1][1], self.angle]
        # #     for k, v in d.items():
        # #         text_keys.append(k)
        # #         text_vals.append(v)
        # #     if first:
        # #         output.append(text_keys)
        # #         first = False
        # #     output.append(text_vals)

        # for d in self.resonance_data:
        #     text_keys = ['ROI_x1', 'ROI_y1', 'ROI_x2', 'ROI_y2', 'Angle']
        #     text_vals = [self.roi[0][0], self.roi[0][1], self.roi[1][0],
        #                  self.roi[1][1], self.angle]
        #     for k, v in d.items():
        #         if isinstance(v, list):
        #             for idx, data in enumerate(v):
        #                 text_keys.append(f'{k}_{idx}')
        #                 text_vals.append(data)
        #         else:
        #             text_keys.append(f'{k}')
        #             text_vals.append(v)
        #     if first:
        #         output.append(text_keys)
        #         first = False
        #     output.append(text_vals)

        # return output
        return self.resonance_data

    def get_resonance_data(self, disp=False):
        """ Getter method for returning resonance_data list of dictionaries to
            the caller.
        """
        self.logger.debug(f'...get_resonance_data({self}, disp={disp})')
        if disp:
            print(self.resonance_data)
        return self.resonance_data

    def get_reference_data(self, disp=False):
        """ Getter method for returning reference_data list of values to
            the caller.
        """
        self.logger.debug(f'...get_reference_data({self}, disp={disp})')
        if disp:
            print(self.reference_data)
        return self.reference_data

    def get_roi(self):
        """ Getter method for returning ROI details ((x1, y1), (x2, y2)) to the
            caller
        """
        self.logger.debug(f'...get_roi({self})')
        return self.roi

    def get_im(self):
        """ Getter method to return rotated/cropped image to caller
        """
        self.logger.debug(f'...get_im({self})')
        return self.im

    def get_num_subROIs(self):
        """ Getter method to return the number of subROIs to caller
        """
        self.logger.debug(f'...get_num_subROIs({self})')
        return self.subROIs

    def was_last_call(self):
        return self.last_call_status
