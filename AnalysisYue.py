from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import timeit
import json


def fano(x, amp, assym, res, gamma, off):
    num = ((assym * gamma) + (x - res)) * ((assym * gamma) + (x - res))
    den = (gamma * gamma) + ((x - res)*(x - res))
    return (amp * (num / den)) + off


def get_filenames(path):
    return sorted([f for f in listdir(path)
                   if isfile(join(path, f))])


def filter_filenames(filenames, prefix):
    return sorted([f for f in filenames
                   if f.split('_')[0].lower() == prefix.lower()])


def filter_filenames_list(filenames, prefix):
    outlist = []
    prefix = [p.lower() for p in prefix]
    for f in filenames:
        check = [a.lower() for a in f.split('_')[:len(prefix)]]
        if prefix == check:
            outlist.append(f)
    return sorted(outlist)


def get_params(filenames, prefix):
    # Store params from filenames
    params = []
    for f in filenames:
        parts = f[:-4].split('_')
        if parts[:len(prefix)] == prefix:
            parts = parts[len(prefix):]
            params.append({
                'period': int(parts[0][1:]),
                'dose': float(parts[1][1:])/100,
                'int': float(parts[2][1:])/1000
            })
    return params


def load_data(path, filename, delimiter=','):
    # Load data
    data = []
    path = join(path, filename)
    with open(path) as f:
        for line in f:
            num1, num2 = line.split(delimiter)
            data.append([float(num1), float(num2)])
    data = np.transpose(data)
    return data


def find_max_min_index(data, min, max):
    # Find index of value closest to target
    abs_value_array = np.abs(data - min)
    closest_min_idx = abs_value_array.argmin()
    closest_min_value = data[closest_min_idx]

    abs_value_array = np.abs(data - max)
    closest_max_idx = abs_value_array.argmin()
    closest_max_value = data[closest_max_idx]

    print(f'MIN -> Index: {closest_min_idx}, Value: {closest_min_value}')
    print(f'MAX -> Index: {closest_max_idx}, Value: {closest_max_value}')

    return (closest_min_idx, closest_max_idx)


def trim_data_array(data, min_idx, max_idx):
    temp = data[0][min_idx:max_idx]
    temp2 = data[1][min_idx:max_idx]
    return np.array([temp, temp2])


def load_mirror_data(path, prefix):
    filenames = get_filenames(path)
    filename = filter_filenames_list(filenames, prefix)
    return load_data(path, filename[0], delimiter='\t')


def save_out_referenced_data(path, prefix, mirror_prefix, ref_int):
    data_filenames = get_filenames(path)
    data_filenames = filter_filenames(data_filenames, prefix)
    params = get_params(data_filenames, prefix)
    ref_data = load_mirror_data(path, mirror_prefix)

    for p, f in zip(params, data_filenames):
        new_filename = 'REF_' + f[:-4] + '.csv'
        scale = p['int'] / ref_int
        data = load_data(path, f, delimiter='\t')
        new_data = data[1] / (ref_data[1] * scale)
        data[1] = new_data
        np.savetxt(new_filename, np.transpose(data), delimiter=',')


def run_fit(data, initial, bounds):
    try:
        start = timeit.timeit()
        popt, pcov = curve_fit(fano, data[0], data[1],
                               p0=initial, bounds=bounds)
        end = timeit.timeit()
        print(f'Curve fit took: {(end - start):.4} s')
        amp, assym, res, gamma, off = popt
        print(f'Amp: {amp:.3f}, Assym: {assym:.3f} '
              f'Resonance: {res:.3f}, Gamma: {gamma:.3f} '
              f'Offset: {off:.3f}')
        print('--------------------------------------')

        _, ax = plt.subplots()
        ax.plot(data[0], data[1], 'b')
        ax.plot(data[0], fano(data[0], *popt), 'r')
        plt.show()

        return popt, pcov

    except RuntimeError as e:
        print(f'Curve fitting did not converge: {e}')
    except ValueError as e:
        print(f'Curve fitting failed: {e}')


raw_data_path = './ChristinaData/ALL/'
ref_int = 0.2
prefix = '2D'
mirror_prefix = ['Mirror', 'with1ND', 'withPol']

# # save_out_referenced_data(raw_data_path, [prefix], mirror_prefix, ref_int)

ref_filenames = get_filenames(raw_data_path)
ref_filenames = filter_filenames_list(ref_filenames, ['REF', prefix])
params = get_params(ref_filenames, ['REF', prefix])
print(params)

# _, ax = plt.subplots(1, 1, figsize=(12, 6))
# for f in ref_filenames:
#     data = load_data(raw_data_path, f)
#     ax.plot(data[0], data[1])
# plt.show()

for p, f in zip(params, ref_filenames):
    data = load_data(raw_data_path, f)

    min_idx, max_idx = find_max_min_index(data[0], 550, 750)
    data = trim_data_array(data, min_idx, max_idx)

    #          fano(x,  amp, assym, res, gamma, off)
    initial = np.array([0,   0,     550, 10,    0.1])
    bounds = ([0,      -100, 500, 0,   -0.5],
              [np.inf,  100, 900, 100,  0.5])

    popt, pcov = run_fit(data, initial, bounds)
    # amp, assym, res, gamma, offset
    A, b, c, d, e = popt
    p['resonance'] = c
    FWHM = (2 * np.sqrt(4 * ((b*d)**2) * (b**2 + 2))) / ((2 * b**2) - 4)
    p['FWHM'] = FWHM
    p['amplitude'] = A * b**2

    ss_res = np.sum((data[1] - fano(data[0], *popt))**2)
    ss_tot = np.sum((data[1] - np.mean(data[1]))**2)
    p['r2'] = 1 - (ss_res / ss_tot)

with open(join(raw_data_path, 'resultsTE.json'), 'w') as f:
    f.write(json.dumps(params, indent=4, separators=(',', ': ')))

first = True
with open(join(raw_data_path, 'resultsArray.csv'), 'a') as f:
    for p in params:
        text_keys = ''
        text_vals = ''
        for k, v in p.items():
            text_keys = text_keys + str(k) + ','
            text_vals = text_vals + str(v) + ','
        if first:
            f.write(text_keys[:-1] + '\n')
            first = False
        f.write(text_vals[:-1] + '\n')
