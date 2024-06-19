import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging


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

    def __init__(self, log_data, details):
        self.log_data = log_data
        self.mode = details['mode'].lower()
        self.fig, (self.ax1, self.ax2) = details['figure']
        (self.line, ) = self.ax1.plot([0], [0], 'r', linewidth=2)
        self.line.figure = self.fig
        self.im = details['im']
        self.xs = []
        self.ys = []
        self.angle = 0
        self.ax1.imshow(self.im)
        self.ax2.axis('off')
        self.last_call_status = False
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_data[1])
        fh = logging.FileHandler(filename=log_data[0], encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s : %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.debug(f'...__init__({self}, {self.mode}, {self.fig}, {self.im})')

        if 'patches' in details:
            for rect_coord in details['patches']:
                (x1, y1), (x2, y2) = rect_coord
                self.ax1.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   edgecolor='red', facecolor='none'))

        if self.mode == 'rotate':
            self.logger.info('Image rotate')
            self.fig.suptitle(
                'Rotate: Select two horizontal points \n '
                'Press <Enter> when finished', fontsize='xx-large')
        elif self.mode == 'crop':
            self.logger.info('Image crop')
            self.fig.suptitle(
                'Select ROI (top-left then bottom-right points) \n '
                f'Press <Enter> to select next ROI   {details['message']}',
                fontsize='xx-large')
        elif self.mode == 'resonance':
            self.logger.info('Image resonance/gamma')
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
        self.logger.debug(f'...mouseClick({self}, {event})')
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
        self.logger.debug(f'...keyPress({self}, key: {event.key})')
        if event.key == 'enter':
            self.close_all()
        if event.key == 'q' or event.key == 'Q':
            self.last_call_status = True
            self.close_all()

    def onClose(self, event):
        """ Callback function for figure close event event, at which point
            closes figure
        """
        self.logger.debug(f'...onClose({self}, {event})')
        self.close_all()

    def close_all(self):
        self.line.figure.canvas.mpl_disconnect(self.cidclick)
        self.line.figure.canvas.mpl_disconnect(self.cidkey)
        self.fig.canvas.mpl_disconnect(self.cidclose)
        plt.close('all')

    def onEnterEvent(self, event):
        """ Callback function mouse pointer entering figure, to force
            focus so that key_press events are registered
        """
        self.logger.debug(f'...onEnter({self}, {event})')
        try:
            self.fig.canvas.setFocus()
        except AttributeError:
            pass  # MacOS does not have a 'setFocus()' option

    def connect(self):
        """ Connects mouseClick() method to mouse click event of line object.
            Also connects keyPress() method to key press event of line object.
        """
        self.cid_enter = (self.fig
                          .canvas
                          .mpl_connect('axes_enter_event', self.onEnterEvent))
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
        self.logger.debug(
            f'...connect - click({self.cidclick}) - key({self.cidkey})'
            f'- close(s{self.cidclose})')

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
        self.logger.debug(f'...add_data({self}, {x}, {y})')
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
        self.logger.debug(f'...update_fig({self}, {xdata}, {ydata})')
        if len(self.xs) == 1:

            x0 = self.xs[0]
            x1 = self.xs[0]
            if xdata > self.xs[0]:
                x1 = xdata
            else:
                x0 = xdata
            y0 = self.ys[0]
            y1 = self.ys[0]
            if ydata > self.ys[0]:
                y1 = ydata
            else:
                y0 = ydata

            if self.mode == 'rotate':
                self.angle = np.degrees(np.tan((ydata-self.ys[0]) /
                                               (xdata-self.xs[0])))
                temp = self.im.rotate(self.angle)
                self.ax2.imshow(temp)
            elif self.mode == 'crop':
                temp = self.im.crop((x0, y0, x1, y1))
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
        self.logger.debug(f'...get_angle({self})')
        return self.angle

    def get_coords(self):
        """ Getter method to return crop coordinates to the caller, if only
            one element in xs list then returns (0, 0) as first point, this
            avoids errors however is not intended use
        """
        self.logger.debug(f'...get_coords({self})')
        if len(self.xs) < 2:
            self.logger.info('Returning (0,0) as first ROI coordinate')
            return (0, 0), (self.im.size[0], self.im.size[1])
        return zip(self.xs, self.ys)

    def get_initial_values(self):
        """ Getter method to return initial values to caller, these values come
            from the user selecting the resonance position
        """
        self.logger.debug(f'...get_initial_values({self})')
        try:
            return ((self.ys[0]+self.ys[1])/2,
                    np.absolute(self.ys[0]-self.ys[1]))
        except IndexError:
            self.logger.info('No ROI coords, returning image data'
                             'for res/gamma')
            _, height = self.get_cropped_image().size
            return (height/2, height/20)

    def get_rotated_image(self):
        """ Getter method to return rotated image to caller, if this fails
            original image
        """
        self.logger.debug(f'...get_rotated_image({self})')
        try:
            return self.im.rotate(self.angle)
        except ValueError:
            self.logger.info(
                'No angle data, returning 0')
            return self.im

    def get_cropped_image(self):
        """ Getter method to return cropped image to caller, if this fails
            original image
        """
        self.logger.debug(f'...get_cropped_image({self})')
        try:
            return self.im.crop((self.xs[0], self.ys[0],
                                self.xs[1], self.ys[1]))
        except IndexError:
            self.logger.info('No crop data, returning full image')
            return self.im

    def was_last_call(self):
        return self.last_call_status