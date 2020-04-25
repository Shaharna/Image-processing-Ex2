#------------- Imports -----------
import numpy as np
import scipy.io.wavfile as wav
from skimage.color import rgb2gray
from scipy.misc import imread
import scipy.signal
# ------------ Constants ---------
DERIVATIVES_CONVOLUTION_ARRAY = np.array([0.5, 0, -0.5])
GRAY_MAX_VALUE = 255

#------------- ex2_helper ---------

import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


#------------- Functions ---------------


def read_image(filename, representation):
    """
    3.1 Reading an image into a given representation.
    :param filename: read_image(filename, representation).
    :param representation: representation code, either 1 or 2 defining
    whether the output should be a grayscale image (1) or an RGB image (2).
    If the input image is grayscale, we won’t call it with representation = 2.
    :return: This function returns an image, make sure the output image
    is represented by a matrix of type np.float64 with intensities
    (either grayscale or RGB channel intensities)
    normalized to the range [0, 1].
    """
    im = (imread(filename).astype(np.float64)) / GRAY_MAX_VALUE

    if representation == 1:

        return rgb2gray(im)

    elif representation == 2:

        return (im)

def DFT(signal):
    """
    This function transform a 1D discrete signal to its Fourier representation
    :param signal: An array of dtype float64 with shape (N,1)
    :return: The complex Fourier signal
    """
    N = signal.shape[0]
    coefficient = calculate_coefficient_matrix(N, -1)
    return np.dot(coefficient, signal)


def calculate_coefficient_matrix(N, sign):
    """
    This function calculates the DFT coefficient matrix
    :param signal: The input signal - An array of dtype float64 with shape(N,1)
    :return: The coefficient matrix from the DFT equation.
    """
    u_array, x_array = np.meshgrid(np.arange(0, N), np.arange(0, N))
    mul_matrix = u_array * x_array
    power = sign * (2 * np.pi * 1j * mul_matrix ) / N
    coefficient_matrix = np.ones((N, N), np.complex128) * np.e
    coefficient_matrix = coefficient_matrix ** power
    return coefficient_matrix

def IDFT(fourier_signal):
    """
    This function calculates the IDFT of Fourier representation
    :param fourier_signal: an array of dtype complex128 with the (N, 1) shape
    :return: A signal from the Fourier representation.
    """
    N = fourier_signal.shape[0]
    coefficient = calculate_coefficient_matrix(N, 1)
    return (1 / N) *(np.dot(coefficient, fourier_signal))

def DFT2(image):
    """
    This functions that convert a 2D discrete signal to its Fourier
     representation.
    :param image: a grayscale image of dtype float64.
    :return:
    """
    im_dft = DFT(image.T)
    return DFT(im_dft.T)

def IDFT2(fourier_image):
    """
    This function calculates the IDFT 2D of Fourier representation
    :param fourier_image: real image transformed with DFT2.
    :return: The image
    """
    return IDFT(IDFT(fourier_image.T).T)

def change_rate(filename, ratio):
    """
    function that changes the duration of an audio file by keeping the same
    samples, but changing the sample rate written in the file header.
    When the audio player uses the same samples as if they were taken in a
    higher sample rate, a “fast forward” effect is created. Given a WAV file,
    this function saves the audio in a new file called change_rate.wav.
    :param filename: is a string representing the path to a WAV file
    :param ratio: is a positive float64 representing the duration change
    :return: The function should not return anything.
    """
    sample_rate, data = wav.read(filename)
    wav.write("change_rate.wav", int(sample_rate * ratio), data)

def change_samples(filename, ratio):
    """
    fast forward function that changes the duration of an audio file
    by reducing the number of samples using Fourier.
    :param filename: a string representing the path to a WAV file
    :param ratio: a positive float64 repre- senting the duration change.
    :return:
    """
    sample_rate, data = wav.read(filename)
    resize_data  = resize(data, ratio)
    wav.write("change_samples.wav", int(sample_rate), resize_data.astype(data.dtype))

def resize(data, ratio):
    """

    :param data: a 1D ndarray of dtype float64 or complex128(*)
    representing the original sample points.
    :param ratio: a positive float64 repre- senting the duration change.
    :return: The returned value of resize is a 1D ndarray of the dtype
     of data representing the new sam- ple points.
    """

    size = data.shape[0]
    dft_data = DFT(data)
    dft_data = np.fft.fftshift(dft_data)

    if ratio < 1:
        pad_length = int(((size / ratio) - size )/2)
        if int(((size / ratio) - size )) % 2 == 0:
            # pad with ((size / ratio) - size) /2 zeros
            dft_data = np.pad(dft_data, (pad_length, pad_length),
                             'constant', constant_values=(0))
        else:
            pad_length = int(np.floor(pad_length))
            dft_data = np.pad(dft_data, (pad_length, pad_length+1),
                              'constant', constant_values=(0))

    else:
        division = (size / ratio)
        start = int(np.floor((len(dft_data) - division)/2))
        end = int(np.ceil(len(dft_data) - start))
        dft_data = dft_data[start + 1: end]

    dft_data = np.fft.ifftshift(dft_data)
    return IDFT(dft_data)

def resize_spectrogram(data, ratio):
    """
    This function speeds up a WAV file, without changing the pitch,
    using spectrogram scaling. This is done by computing the spectrogram,
    changing the number of spectrogram columns, and creating back the audio.
    :param data: data is a 1D ndarray of dtype float64 representing the
    original sample points.
    :param ratio: is a positive float64 representing the rate change of the
    WAV file.
    :return:
    """
    spec = stft(data)

    spec = np.apply_along_axis(resize, 1,spec, ratio)

    resize_spec = istft(spec, spec.shape[0])

    return np.real(resize_spec).astype(data.dtype)

def resize_vocoder(data, ratio):
    """
    This function that speedups a WAV file by phase vocoding its spectrogram.
    :param data: 1D ndarray of dtype float64 representing the original sample
     points.
    :param ratio:positive float64 representing the rate change of the WAV file.
    :return:
    """
    spec = stft(data)
    vocoder_spec = istft(phase_vocoder(spec, ratio))
    return np.real(vocoder_spec).astype(data.dtype)

def conv_der(im):
    """
    This function computes the magnitude of image derivatives.
    :param im:  float64 grayscale image.
    :return:
    """
    dx = scipy.signal.convolve2d(im, DERIVATIVES_CONVOLUTION_ARRAY.reshape(1,3), mode="same")
    dy = scipy.signal.convolve2d(im, DERIVATIVES_CONVOLUTION_ARRAY.reshape(3,1), mode="same")

    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude

def fourier_der(im):
    """
    This function computes the magnitude of image derivatives using Fourier
     transform.
    :param im:  float64 grayscale image.
    :return:
    """
    N, M = im.shape

    x_coefficient = der_axis_coefficient(N)
    y_coefficient = der_axis_coefficient(M)

    dft_im = DFT2(im)
    centered_dft = np.fft.fftshift(dft_im)

    Dx = np.transpose(np.transpose(centered_dft) * x_coefficient)
    dx = IDFT2(Dx)

    Dy = centered_dft * y_coefficient
    dy = IDFT2(Dy)

    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


def der_axis_coefficient(N):
    """
    This function return the x/y axis derivative coefficient
    :param N: The size
    :return: The derivative coefficient
    """
    coefficient_matrix =np.arange(-(np.floor(N / 2)), np.ceil(N / 2))
    return coefficient_matrix * ((2 * np.pi * 1j)/N)





