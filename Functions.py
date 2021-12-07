import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from os.path import join
from PIL import Image
# import json
# import timeit


class ROI:
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
    def __init__(self):
        print('Initialising ROI')
        self.im = None
        self.first = True
        self.roi = ()
        self.angle = None
        self.initial_values = [0, 0, 0, 0, 0]
        self.bounds = ([-1000, -100,  0,   0,    0],
                       [1000, 100,  500, 500,  500])
        self.resonance_data = []
        self.fig = None
        self.ax = None

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
        self.im = im
        self.first = False

        figure = plt.subplots(1, 2, figsize=(12, 6))
        image_rotator = ImageProcessor('rotate', figure, self.im)
        self.angle = image_rotator.get_angle()
        temp = image_rotator.get_rotated_image()

        figure = plt.subplots(1, 2, figsize=(12, 6),
                              gridspec_kw={'width_ratios': [3, 1]})
        image_cropper = ImageProcessor('crop', figure, temp)
        self.roi = tuple(image_cropper.get_coords())
        self.im = image_cropper.get_cropped_image()

        figure = plt.subplots(1, 2, figsize=(12, 6))
        p0 = ImageProcessor('resonance', figure, self.im)
        # self.set_initial_values(amp=0.2, assym=-100,
        #                         res=None, gamma=None, off=40)
        self.initial_values[2] = p0.get_initial_values()[0]
        self.initial_values[3] = p0.get_initial_values()[1]

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

        self.im = im
        temp = self.im.rotate(self.angle)
        (x1, y1), (x2, y2) = self.roi
        self.im = np.array(temp.crop((x1, y1, x2, y2)))
        # self.process_ROI(plot=plot)

    def determine_assym(self, plot=False):
        """ CURRENTLY NOT REQUIRED BY SETTING INITIAL VALUE TO 0
            Determine positive or negative assymetry value from initial image.
            Curve-fit algorithm seems to requires a positive or negative
            initial guess for the fitting of the Fano function, if the guess
            is the wrong sign then curve-fit will converge to a "bad" fit
        """

        self.initial_values[1] = 100
        self.process_ROI(plot=plot)
        try:
            r2pos = self.resonance_data[-1]['r2']
        except IndexError:
            r2pos = 0
        self.initial_values[1] = -100
        self.process_ROI(plot=plot)
        try:
            r2neg = self.resonance_data[-1]['r2']
        except IndexError:
            r2neg = 0
        print(f'Positive r2 -> {r2pos}  Negative r2 -> {r2neg}')
        if r2pos > r2neg:
            self.initial_values[1] = 100
        self.resonance_data = []

    def process_ROI(self, plot=False):
        """ Processes ROI from initial image, first collapsing image into 1D,
            then applies curve_fit algorithm - updating resonance_data with
            the results
        """
        collapsed_im = self.im.mean(axis=1)
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
            popt_dict = {'amp': A,  # A * b**2,
                         'assym': b,
                         'res': c,
                         'gamma': d,
                         'off': e}

            FWHM = ((2 * np.sqrt(4 * ((b*d)**2) * (b**2 + 2))) /
                    ((2 * b**2) - 4))
            popt_dict['FWHM'] = FWHM

            y_bar = np.mean(collapsed_im)
            ss_res = np.sum((self.fano(xdata, *popt) - y_bar)**2)
            ss_tot = np.sum((collapsed_im - y_bar)**2)
            popt_dict['r2'] = ss_res / ss_tot

            self.resonance_data.append(popt_dict)

            # self.set_initial_values(amp=popt[0], assym=popt[1],
            #                         res=popt[2], gamma=popt[3], off=popt[4])
            self.set_initial_values(**popt_dict)

            if plot:
                fig, ax = plt.subplots(1, figsize=(12, 6))
                ax.plot(xdata, collapsed_im, 'b')
                ax.plot(xdata, self.fano(xdata, *popt), 'r')
                plt.show()

        except RuntimeError as e:
            print(f'Curve fitting did not converge: {e}')
            popt_dict = {'amp': 0,
                         'assym': 0,
                         'res': 0,
                         'gamma': 0,
                         'off': 0,
                         'FWHM': 0,
                         'r2': 0}
        except ValueError as e:
            print(f'Curve fitting failed: {e}')
            popt_dict = {'amp': 0,
                         'assym': 0,
                         'res': 0,
                         'gamma': 0,
                         'off': 0,
                         'FWHM': 0,
                         'r2': 0}

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
        self.initial_values[0] = amp
        self.initial_values[1] = assym
        self.initial_values[2] = res
        self.initial_values[3] = gamma
        self.initial_values[4] = off

    def get_inital_values(self):
        """ Getter function used to return initial_values to the caller

            Return:
            initial_values (list): List of intial calculation values
        """
        print(self.initial_values)
        return self.initial_values

    def save_data(self, path, name):
        """ Method for saving results to a file, saves the resonance_data list
            of dictionaries as a CSV table

            Attributes:
            path (str): Path to save folder
            name (str): Name of results file
        """
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
        if disp:
            print(self.resonance_data)
        return self.resonance_data

    def get_roi(self):
        """ Getter method for returning ROI details ((x1, y1), (x2, y2)) to the
            caller
        """
        return self.roi

    def get_im(self):
        """ Getter method to return rotated/cropped image to caller
        """
        return self.im


class ImageProcessor:
    """ Image processor class - processes images passed to it in one of three
        ways.  1) Rotate - asks user for two points on a known horizontal line,
        then stores this angle for use later. 2) Crop - asks user for two
        points one at the top-left one at the bottom-right to define a
        rectangle, this data is then stored. 3) Resonance - asks user for two
        points (as in Crop) defining a rectangle around the resonance position,
        these values are then stored.

        Attributes:
        mode (str):             Either 'rotate', 'crop', or 'resonance', this
                                then determines which procedure is performed
        figure (figure handle): Used to hold figure data for plotting
        im (PIL image):         Used to hold axis data for plotting
    """
    def __init__(self, mode, figure, im):
        self.mode = mode.lower()
        self.fig, (self.ax1, self.ax2) = figure
        (self.line, ) = self.ax1.plot([0], [0], 'r', linewidth=2)
        self.line.figure = self.fig
        self.im = im
        self.xs = []
        self.ys = []
        self.angle = 0
        self.ax1.imshow(self.im)
        self.ax2.axis('off')
        if mode == 'rotate':
            self.fig.suptitle(
                'Rotate: Select two horizontal points \n '
                'Press <Enter> when finished', fontsize='xx-large')
        elif mode == 'crop':
            self.fig.suptitle(
                'Crop: Rectangle select (i.e. select the top-left and '
                'bottom-right points) around ROI \n '
                'Press <Enter> when finished', fontsize='xx-large')
        elif mode == 'resonance':
            self.fig.suptitle(
                'Resonance: Rectangle select (i.e. select the top-left '
                'and bottom-right points) resonance \n '
                'Press <Enter> when finished', fontsize='xx-large')
        # match mode:
        #     case 'rotate':
        #         self.fig.suptitle('Rotate: Select two horizontal points \n '
        #                           'Press <Enter> when finished',
        #                           fontsize='xx-large')
        #     case 'crop':
        #         self.fig.suptitle(
        #             'Crop: Rectangle select (i.e. select the top-left and '
        #             'bottom-right points) around ROI \n '
        #             'Press <Enter> when finished', fontsize='xx-large')
        #     case 'resonance':
        #         self.fig.suptitle(
        #             'Resonance: Rectangle select (i.e. select the top-left '
        #             'and bottom-right points) resonance \n '
        #             'Press <Enter> when finished', fontsize='xx-large')
        self.connect()
        plt.show()

    def mouseClick(self, event):
        """ Callback function for Mouse click event, receives x- and y-position
            of mouse position when clicked.  Uses this information to update
            the class instance attributes accordingly.  Also forces re-draw of
            figure.
        """
        if event.inaxes != self.line.axes:
            return
        self.update_fig(event.xdata, event.ydata)
        self.add_data(event.xdata, event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

    def keyPress(self, event):
        """ Callback function for "Enter" key press event, at which point closes
            figure
        """
        if event.key == 'enter':
            self.line.figure.canvas.mpl_disconnect(self.cidclick)
            self.line.figure.canvas.mpl_disconnect(self.cidkey)
            self.fig.canvas.mpl_disconnect(self.cidclose)
            plt.close()

    def onClose(self, event):
        """ Callback function for figure close event event, at which point
            closes figure
        """
        self.line.figure.canvas.mpl_disconnect(self.cidclick)
        self.line.figure.canvas.mpl_disconnect(self.cidkey)
        self.fig.canvas.mpl_disconnect(self.cidclose)
        plt.close()

    def connect(self):
        """ Connects mouseClick() method to mouse click event of line object.
            Also connects keyPress() method to key press event of line object.
        """
        self.cidclick = (self.line
                         .figure
                         .canvas
                         .mpl_connect('button_press_event', self.mouseClick))
        self.cidkey = (self.line
                       .figure
                       .canvas
                       .mpl_connect('key_press_event', self.keyPress))
        self.cidclose = (self.fig
                         .canvas
                         .mpl_connect('close_event', self.onClose))

    def add_data(self, x, y):
        """ Adds x- and y-data to the xs and ys attributes, if xs and ys is
            already populated with 2 values (i.e. is complete) then resets the
            two lists.

            Usage:
            Needs to be run after update_fig(), if run first then clearing of
            the xs and ys lists not easy as such run into issues where the xs
            and ys list have three elements when only a maximum of two required

            Attributes:
            x (int): x-position of mouse click event
            y (int): y-position of mouse click event
        """
        if len(self.xs) == 2:
            self.xs = []
            self.ys = []
        self.xs.append(x)
        self.ys.append(y)

    def update_fig(self, xdata, ydata):
        """ Updates figure, by either showing a: rotated version of the image;
            cropped version of the image; or adds two blue horizontal lines and
            one red line to the image indicating both the position and FWH of
            the resonance as selected by the user

            Usage:
            Should be run before add_data() even though add_data() would
            populate the xs and ys list which in is all that this method
            requires. However, if add_data() is ran first it will reset the xs
            and ys list meaning that this method would have no data to work on.

            Attributes:
            xdata (int): x-position of mouse click event
            ydata (int): y-position of mouse click event
        """
        if len(self.xs) == 1:
            if self.mode == 'rotate':
                self.angle = np.degrees(np.tan((ydata-self.ys[0]) /
                                               (xdata-self.xs[0])))
                temp = self.im.rotate(self.angle)
                self.ax2.imshow(temp)
            elif self.mode == 'crop':
                temp = self.im.crop((self.xs[0], self.ys[0], xdata, ydata))
                self.ax2.imshow(temp)
            elif self.mode == 'resonance':
                self.ax2.hlines(y=ydata, xmin=self.xs[0], xmax=xdata,
                                linewidth=2, color='b')
                self.ax2.hlines(y=self.ys[0], xmin=self.xs[0], xmax=xdata,
                                linewidth=2, color='b')
                self.ax2.hlines(y=(self.ys[0] + ydata)/2, xmin=self.xs[0],
                                xmax=xdata, linewidth=2, color='r')
                self.ax2.imshow(self.im)

    def get_angle(self):
        """ Getter method to return rotation angle to caller
        """
        return self.angle

    def get_coords(self):
        """ Getter method to return crop coordinates to the caller, if only
            one element in xs list then returns (0, 0) as first point, this
            avoids errors however is not intended use
        """
        if len(self.xs) < 2:
            return (0, 0), (self.im.size[0], self.im.size[1])
        return zip(self.xs, self.ys)

    def get_initial_values(self):
        """ Getter method to return initial values to caller, these values come
            from the user selecting the resonance position
        """
        try:
            return ((self.ys[0]+self.ys[1])/2,
                    np.absolute(self.ys[0]-self.ys[1]))
        except IndexError:
            _, height = self.get_cropped_image().size
            return (height/2, height/20)

    def get_rotated_image(self):
        """ Getter method to return rotated image to caller, if this fails
            original image
        """
        try:
            return self.im.rotate(self.angle)
        except ValueError:
            return self.im

    def get_cropped_image(self):
        """ Getter method to return cropped image to caller, if this fails
            original image
        """
        try:
            return self.im.crop((self.xs[0], self.ys[0],
                                self.xs[1], self.ys[1]))
        except IndexError:
            return self.im


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


def convert_CSV_to_Image(path_to_csv):
    data = np.asarray(np.genfromtxt(path_to_csv, delimiter=','))
    return Image.fromarray((data).astype(np.float64))
