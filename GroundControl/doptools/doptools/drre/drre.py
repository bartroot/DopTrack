import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.optimize as optimize
import scipy.ndimage.filters as filters
import itertools
from sklearn.cluster import DBSCAN
from ..io import read_meta
from ..model_doptrack import Spectrogram
from .masks import area_mask
from .fitting import tanh, fit_tanh, fit_residual


def main(dataid):
    spectrogram = Spectrogram.load(dataid)
    image = spectrogram.image

    mask1 = area_mask(image, ori='freq', dt=spectrogram.dt)
    mask2 = area_mask(image, ori='time', dt=spectrogram.dt)
    mask = np.logical_and(mask1, mask2)

#    plt.figure()
#    plt.imshow(image, clim=(0, 0.1), cmap='viridis', aspect='auto')
#    plt.show()


    # Apply mask and change zeros to the mean value of the data
    image_masked = image * mask + np.logical_not(mask) * image.mean()


#    plt.figure()
#    plt.imshow(image_masked, clim=(0, 0.1), cmap='viridis', aspect='auto')
#    plt.show()



#    plt.figure()
#    plt.imshow(image * mask, clim=(0, 0.1), cmap='viridis', aspect='auto')
#    plt.show()



    xs, ys, vals = filter_datapoints(image_masked, spectrogram.dt)


#    fit_coeffs = fit_tanh(xs, ys, spectrogram.dt)
#    fit = tanh(xs, *fit_coeffs)
#    res = ys - fit

#    plt.figure()
#    plt.imshow(image_masked, clim=(0, 0.1), cmap='viridis', aspect='auto')
#    plt.scatter(ys, xs)
#    plt.show()


    sideband = 600

    for m in [2.5, 1.5, 0.5]:
        fit_coeffs = fit_tanh(xs, ys, spectrogram.dt)
        fit = tanh(xs, *fit_coeffs)
        xs, ys, vals = remove_outliers(xs, ys, vals, fit, m*sideband)
    fit = tanh(xs, *fit_coeffs)


#    plt.figure()
#    plt.scatter(xs, ys)
#    plt.plot(xs, fit, 'r')
#    plt.show()


    res = ys - fit


    xs_orig = xs
    ys_orig = ys
    res_orig = res


    data = np.dstack((xs, res))[0]
    db = DBSCAN(eps=50, min_samples=5).fit(data)
    data = np.column_stack(data)
    plt.figure()
    print(set(db.labels_))
    for i in set(db.labels_):
        plt.scatter(data[0][db.labels_ == i], data[1][db.labels_ == i], label=i)
    plt.legend()
    plt.show()

    from collections import Counter
    cnt = Counter(db.labels_)
    j = cnt.most_common(1)[0][0]
    xs = data[0][db.labels_ == j]
    res = data[1][db.labels_ == j]




#    dif = np.diff(res)
##    for i in range(5):
#    m = 50
#    while max(dif) > m:
#        dif = np.diff(res)
#        res = res[:-1]
#        res = res[abs(dif) < m]
#        xs = xs[:-1]
#        xs = xs[abs(dif) < m]
#        ys = ys[:-1]
#        ys = ys[abs(dif) < m]
#
#
    func, coeffs = fit_residual(xs, res)
    res_fit = func(xs, *coeffs)

#    func, coeffs = fit_residual(xs_orig, res_orig)
#    res_fit_orig = func(xs_orig, *coeffs)



#    tresh = vals.mean() - 0.5 * vals.std()
#    plt.figure()
#    plt.scatter(xs, vals)
#    plt.plot(xs, tresh*np.ones(len(xs)))
#    plt.show()

    plt.figure()
    plt.scatter(xs, res, color='r')
    plt.plot(xs, res_fit)
    plt.show()

##    res_filtered = filters.minimum_filter(res_orig, size=20)
#    res_filtered = filters.median_filter(res_orig, size=25)
##    res_filtered3 = filters.maximum_filter(res_orig, size=20)
##    res_filtered4 = filters.prewitt(res_orig)
##    res_filtered5 = filters.sobel(res_orig)
#    func, coeffs = fit_residual(xs_orig, res_filtered)
#    res_fit_filtered = func(xs_orig, *coeffs)
#
#    plt.figure()
#    plt.scatter(xs_orig, res_filtered, label='median')
#    plt.scatter(xs_orig, res_orig, s=5, color='r')
#    plt.plot(xs_orig, res_fit_orig, label='orig')
#    plt.plot(xs_orig, res_fit_filtered, label='filt')
#    plt.legend()
#    plt.show()


#    plt.figure()
#    plt.imshow(image_masked, clim=(0, 0.1), cmap='viridis', aspect='auto')
#    plt.plot(ys, xs, '.r')
#    plt.show()


def calc_sideband(residual):
    width = residual.max() - residual.min()
    bins = int(width / 50)

    hist = np.histogram(residual, bins)


    for i in itertools.count():
        peaks_ix, _ = signal.find_peaks(hist[0], height=i)
        if len(peaks_ix) == 3:
            print(f'Peaks found at treshold={i}.')
            break
        elif len(peaks_ix) < 3:
            raise RuntimeError('Less than 3 peaks found.')
        elif i == 1000:
            raise RuntimeError('Iteration limit reached. Peaks not found.')
    peaks = hist[1][:-1][peaks_ix]
    sideband = np.diff(peaks).mean()

#        dp1 = peaks[1] - peaks[0]
#        dp2 = peaks[-1] - peaks[1]
#        if dp1 - dp2 > 0.1 * sideband:
#            continue
#        else:
#            break

    plt.figure()
    plt.plot(hist[1][:-1], hist[0])
    plt.plot(hist[1][:-1][peaks_ix], hist[0][peaks_ix], '.r')
    plt.show()

    return sideband


def filter_datapoints(data, dt):

    noisefilter = signal.gaussian(int(14/dt)+1, 2.5)
    noisefilter = noisefilter / np.sum(noisefilter)

    mean = data.mean()

    vals = np.zeros(len(data))
    ys = np.zeros(len(data))

#    rdata = np.zeros(data.shape)

    for i, row in enumerate(data):

        row_norm = row - mean
        # Set all values less than 5 std from mean to zero
        row_norm[row_norm < 6 * np.std(row_norm)] = 0
        # Apply gaussian filter
        row_norm = np.convolve(row_norm, noisefilter, mode='same')

        # If this value is not used then func satSignal is also never used for anything
#        p = np.convolve(row_norm, _filter,mode='same')
#        rdata[i][:] = p

        vals[i] = np.max(row_norm)
        ys[i] = np.argmax(row_norm)

#        # TODO check if this 'if'  is necessary. Is peak_value ever zero?
#        if peak_value:
#            peak_values[i] = peak_value
#            peak_locations[i] = peak_location

    xs = np.arange(len(ys))

    return xs, ys, vals


def remove_outliers(xs, ys, vals, mean, band):
    ys_ = ys[(ys > mean - band) & (ys < mean + band)]
    xs_ = xs[(ys > mean - band) & (ys < mean + band)]
    vals_ = vals[(ys > mean - band) & (ys < mean + band)]
    return xs_, ys_, vals_






