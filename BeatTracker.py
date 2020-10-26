import numpy as np
import librosa
from scipy.signal import find_peaks


def beatTracker(fileName):
    """Detects the beats in the audio file using a Spectral-Flux-based onset detection function.
        Parameters:
            fileName: path to the audio file
        Returns:
            A vector containing the estimated beat positions in seconds"""

    # Loading the file
    snd, rate = librosa.load(fileName)
    # Calculating the onset positions and the estimated beat period
    odf, interBeatPeriod = onsets(snd, rate, 2048, 512)
    # Initialising the list of beats
    beats = [odf[0]]
    # Controls how strongly the algorithm reacts to tempo changes
    reactiveness = 0.7
    # Initialising the guess for the next beat
    estimatedNextBeat = beats[0] + interBeatPeriod

    i = 0
    maxOnset = max(odf)
    while estimatedNextBeat <= maxOnset:
        # Finding the onset in the odf that is closest to the estimated next beat
        k = np.argmin(np.abs(estimatedNextBeat - odf))
        # If a new onset is found, add it to the list and update the beat period
        if odf[k] > beats[i]:
            beats.append(odf[k])
            interBeatPeriod = reactiveness * (beats[i + 1] - beats[i]) \
                              + (1 - reactiveness) * interBeatPeriod
        # If there is a long pause in the onsets, skip ahead by one period and
        # try again until a new onset is found
        else:
            while odf[k] <= beats[i]:
                estimatedNextBeat += interBeatPeriod
                k = np.argmin(np.abs(estimatedNextBeat - odf))
            beats.append(odf[k])

        # Update the guess for the next beat
        estimatedNextBeat = beats[i + 1] + interBeatPeriod
        i += 1

    # Converting beat positions to seconds
    beatPos = np.asarray(beats[1:]) / rate
    return beatPos


def onsets(audio, rate, windowSize, hopSize):
    """Computes a baseline onset detection function using Spectral Flux
    Parameters:
        audio: array containing the audio samples
        rate: sample rate in Hz
        windowSize: window size for analysis (in samples)
        hopSize: hop size (in samples)
    Returns:
        peaks: A vector containing the position of the onsets in the audio (in samples) with a
        resolution of hopSize samples
        estimatedBeatPeriod: An estimate of the beat period in samples"""

    # Round up window size to next power of 2
    wlen = int(2 ** np.ceil(np.log2(windowSize)))
    # Create a hamming window of size wlen
    window = np.hamming(wlen)
    # Centre frames: first frame at t=0
    audio = np.concatenate([np.zeros(wlen // 2), audio, np.zeros(wlen // 2)])
    # Calculating the total number of frames
    frameCount = int(np.floor((len(audio) - wlen) / hopSize + 1))
    # Initializing spectral Flux measurements as an array of zeros
    spectralFlux = np.zeros(frameCount)
    # Initialising previous frame as an array of zeros
    previousFrame = np.zeros(wlen)

    # loop going through each frame
    for i in range(frameCount):
        # Start of the frame
        start = i * hopSize
        # Calculating the windowed DFT of the frame
        frame = np.fft.fft(audio[start: start + wlen] * window)
        if i > 0:  # Ignoring the first frame
            # Calculate mean Spectral Flux
            x = np.abs(frame) - np.abs(previousFrame)
            spectralFlux[i] = np.mean((x + np.abs(x)) / 2)
        # Updating the previous frame
        previousFrame = frame

    mx = max(spectralFlux)
    if mx > 0:
        spectralFlux /= mx
    # Calculating an estimate of the beat period
    estimatedBeatPeriod = rate * 60 / (librosa.beat.tempo(y=audio, sr=rate, onset_envelope=spectralFlux,
                                                          hop_length=512, start_bpm=120.0))
    # Finding the positions of the peaks in the SF
    peaks = hopSize * find_peaks(spectralFlux, height=[0.3, 1])[0]
    if peaks == []:
        peaks = [0, 0]  # Avoids an error if there is no peak detected

    return peaks[1:], estimatedBeatPeriod
