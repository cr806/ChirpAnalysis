from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import timeit
from Functions import ImageProcessor, fano


im = Image.open('./RawImages/0002_rawimage.png')

figure = plt.subplots(1, 2, figsize=(12, 6))
image_rotator = ImageProcessor('rotate', figure, im)
im_rot = image_rotator.get_rotated_image()

figure = plt.subplots(1, 2, figsize=(12, 6),
                      gridspec_kw={'width_ratios': [3, 1]})
image_cropper = ImageProcessor('crop', figure, im_rot)
im_crop = image_cropper.get_cropped_image()

figure = plt.subplots(1, 2, figsize=(12, 6))
initial_values = ImageProcessor('resonance', figure, im_crop)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.imshow(im)

(x1, y1), (x2, y2) = image_cropper.get_coords()
loc = (x1, y1)
width = x2-x1
height = y2-y1
roi = patches.Rectangle(loc, width, height,
                        linewidth=2,
                        edgecolor='r',
                        facecolor='none')
ax.add_patch(roi)
plt.text(loc[0]-10, loc[1]-10, 'ROI 1', ha='left', color='r')
plt.show()

im_array = np.array(im_crop)
collapsed_im = im_array.mean(axis=1)

xdata = np.arange(0, collapsed_im.shape[0], 1)

amp_0 = 0.2
assym_0 = -100
res_0, gamma_0 = initial_values.get_initial_values()
off_0 = 40

initial = np.array([amp_0, assym_0, res_0, gamma_0, off_0])
print(initial)
bounds = ([-np.inf, -100, 0, 0, 0], [np.inf, 100, 500, 100, 100])
try:
    start = timeit.timeit()
    popt, pcov = curve_fit(fano, xdata, collapsed_im,
                           p0=initial, bounds=bounds)
    end = timeit.timeit()
    print(f'Curve fit took: {(end - start):.4} s')
    print('--------------------------------------')

    perr = np.sqrt(np.diag(pcov))

    amp, assym, res, gamma, off = popt
    print(f'Amp: {amp:.3f}, Assym: {assym:.3f} '
          f'Gamma: {gamma:.3f}, Resonance: {res:.3f} '
          f'Offset: {off:.3f}')
    print('--------------------------------------')

    ampStd, assymStd, resStd, gammaStd, offStd = perr
    print(f'AmpStd: {ampStd:.3f}, AssymStd: {assymStd:.3f} '
          f'GammaStd: {gammaStd:.3f}, ResonanceStd: {resStd:.3f} '
          f'OffsetStd: {offStd:.3f}')
    print('--------------------------------------')

    fig, ax = plt.subplots()
    ax.plot(xdata, collapsed_im, 'b')
    ax.plot(xdata, fano(xdata, *popt), 'r')
    plt.show()
except RuntimeError as e:
    print(f'Curve fitting did not converge: {e}')
except ValueError as e:
    print(f'Curve fitting failed: {e}')
