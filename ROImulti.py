import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.optimize import curve_fit
from os.path import join
from ImageProcessor import ImageProcessor


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

    def __init__(self, angle=None, roi=None, res=None):
        logging.info(f'__init__({self}, angle={angle}, roi={roi}, res={res})')
        self.im = None
        self.first = True
        self.roi = roi
        self.angle = angle
        self.res = res
        self.initial_values = [0, 0, 0, 0, 0]
        if res is not None:
            self.initial_values = [0, 0, res[0], res[1], 0]
        self.bounds = ([-1000, -100,  0,   0,    0],
                       [1000, 100,  500, 500,  500])
        self.resonance_data = []
        self.fig = None
        self.ax = None

        self.subROIs = 5

    def fano(self, x, amp, assym, res, gamma, off):
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

    def set_initial_ROI(self, im):
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
        logging.debug('...set_initial_ROI({self}, {im})')
        self.im = im

        if self.first:
            if self.angle is None:
                figure = plt.subplots(1, 2, figsize=(12, 6))
                image_rotator = ImageProcessor('rotate', figure, self.im)
                self.angle = image_rotator.get_angle()
                logging.debug('Angle correction')

            if self.roi is None:
                figure = plt.subplots(1, 2, figsize=(12, 6),
                                      gridspec_kw={'width_ratios': [3, 1]})
                temp = self.im.rotate(self.angle)
                image_cropper = ImageProcessor('crop', figure, temp)
                self.roi = tuple(image_cropper.get_coords())
                logging.debug('ROI selection')

            self.im = self.im.crop((self.roi[0][0], self.roi[0][1],
                                    self.roi[1][0], self.roi[1][1]))
            if self.res is None:
                figure = plt.subplots(1, 2, figsize=(12, 6))
                p0 = ImageProcessor('resonance', figure, self.im)
                # self.set_initial_values(amp=0.2, assym=-100,
                #                         res=None, gamma=None, off=40)
                self.initial_values[2] = p0.get_initial_values()[0]
                self.initial_values[3] = p0.get_initial_values()[1]
                logging.debug('Resonance/Gamma selection')
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

        logging.debug(f'...create_ROI_data({self}, {im}, plot={plot})')
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

    #     logging.debug(f'...determine_assym({self}, plot={plot})')
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

    def process_ROI(self, plot=False):
        """ Processes ROI from initial image, first collapsing image into 1D,
            then applies curve_fit algorithm - updating resonance_data with
            the results
        """
        logging.debug(f'...process_ROI({self}, plot={plot})')

        """
        get ROI dimensions (shape of im)
        integer-divide dimension by number of subROIs
        transpose image array
        slice image array im[0:20]
        transpose slice
        """
        ROI_x_size = np.shape(self.im)[1]
        subROI_size = ROI_x_size // self.subROIs
        transposed_im = np.transpose(self.im)

        popt_dict = {'amp': [],  # A * b**2,
                     'assym': [],
                     'res': [],
                     'gamma': [],
                     'off': [],
                     'FWHM': [],
                     'r2': []}

        for i in range(self.subROIs):
            subROI = np.transpose(
                    transposed_im[(i*subROI_size):((i+1)*subROI_size)])

            collapsed_im = subROI.mean(axis=1)
            xdata = np.arange(0, collapsed_im.shape[0], 1)
            initial = np.array(self.initial_values)
            param_scale = [10, 1, self.initial_values[2],
                           self.initial_values[3], np.amin(collapsed_im)]
            # amp, assym, res, gamma, off

            try:
                # start = timeit.timeit()
                popt, pcov = curve_fit(self.fano, xdata, collapsed_im,
                                       p0=initial, bounds=self.bounds,
                                       x_scale=param_scale)
                # end = timeit.timeit()
                # print(f'Curve fit took: {(end - start):.4} s')
                """amp, assym, gamma, res, off = popt
                print(f'Amp: {amp:.3f}, Assym: {assym:.3f} '
                    f'Gamma: {gamma:.3f}, Resonance: {res:.3f} '
                    f'Offset: {off:.3f}')
                print('--------------------------------------')"""

                A, b, c, d, e = popt
                # fano(self, x, amp, assym, res, gamma, off)
                popt_dict['amp'].append(A)  # A * b**2,
                popt_dict['assym'].append(b)
                popt_dict['res'].append(c)
                popt_dict['gamma'].append(d)
                popt_dict['off'].append(e)

                FWHM = ((2 * np.sqrt(4 * ((b*d)**2) * (b**2 + 2))) /
                        ((2 * b**2) - 4))
                popt_dict['FWHM'].append(FWHM)

                y_bar = np.mean(collapsed_im)
                ss_res = np.sum((self.fano(xdata, *popt) - y_bar)**2)
                ss_tot = np.sum((collapsed_im - y_bar)**2)
                popt_dict['r2'].append(ss_res / ss_tot)

                self.set_initial_values(**popt_dict)

                if plot:
                    fig, ax = plt.subplots(1, figsize=(12, 6))
                    ax.plot(xdata, collapsed_im, 'b')
                    ax.plot(xdata, self.fano(xdata, *popt), 'r')
                    plt.show()

            except RuntimeError as e:
                print(f'Curve fitting did not converge: {e}')
                logging.info(f'Curve fitting did not converge: {e}')
                popt_dict['amp'].append(0)  # A * b**2,
                popt_dict['assym'].append(0)
                popt_dict['res'].append(0)
                popt_dict['gamma'].append(0)
                popt_dict['off'].append(0)
                popt_dict['FWHM'].append(0)
                popt_dict['r2'].append(0)
            except ValueError as e:
                print(f'Curve fitting failed: {e}')
                logging.info(f'Curve fitting failed: {e}')
                popt_dict['amp'].append(0)  # A * b**2,
                popt_dict['assym'].append(0)
                popt_dict['res'].append(0)
                popt_dict['gamma'].append(0)
                popt_dict['off'].append(0)
                popt_dict['FWHM'].append(0)
                popt_dict['r2'].append(0)

        self.resonance_data.append(popt_dict)

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
        logging.debug(f"""set_initial_values({self}, amp={amp}, assym={assym},
                         res={res}, gamma={gamma}, off={off}, FWHM={FWHM},
                         r2={r2})""")
        self.initial_values[0] = np.mean(amp)
        self.initial_values[1] = np.mean(assym)
        self.initial_values[2] = np.mean(res)
        self.initial_values[3] = np.mean(gamma)
        self.initial_values[4] = np.mean(off)

    def get_inital_values(self):
        """ Getter function used to return initial_values to the caller

            Return:
            initial_values (list): List of intial calculation values
        """
        logging.debug(f'...get_inital_values({self})')
        print(self.initial_values)
        return self.initial_values

    def save_data(self, path, name):
        """ Method for saving results to a file, saves the resonance_data list
            of dictionaries as a CSV table

            Attributes:
            path (str): Path to save folder
            name (str): Name of results file
        """
        logging.debug(f'...save_data({self}, {path}, {name})')
        first = True
        output = ''
        with open(join(path, f'{name}.csv'), 'w') as f:
            for d in self.resonance_data:
                text_keys = ''
                text_vals = ''
                for k, v in d.items():
                    text_keys = text_keys + str(k) + ','
                    text_vals = text_vals + str(v) + ','
                if first:
                    output = (f'ROI_x1,ROI_y1,ROI_x2,ROI_y2,Angle,'
                              f'{text_keys[:-1]}\n')
                    first = False
                output = (f'{output}{self.roi[0][0]},{self.roi[0][1]},'
                          f'{self.roi[1][0]},{self.roi[1][1]},{self.angle},'
                          f'{text_vals[:-1]}\n')
            f.write(output[:-1])

        # with open(join(path, f'{name}.json'), 'w') as f:
        #     f.write(json.dumps(params, indent=4, separators=(',', ': ')))

    def get_resonance_data(self, disp=False):
        """ Getter method for returning resonance_data list of dictionaries to
            the caller.
        """
        logging.debug(f'...get_resonance_data({self}, disp={disp})')
        if disp:
            print(self.resonance_data)
        return self.resonance_data

    def get_roi(self):
        """ Getter method for returning ROI details ((x1, y1), (x2, y2)) to the
            caller
        """
        logging.debug(f'...get_roi({self})')
        return self.roi

    def get_im(self):
        """ Getter method to return rotated/cropped image to caller
        """
        logging.debug(f'...get_im({self})')
        return self.im
