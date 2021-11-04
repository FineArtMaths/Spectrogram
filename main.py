import numpy as np
import librosa
import scipy
import matplotlib.pyplot as plt

"""

This is a deliberately naive attempt to extract a coarse spectrogram
from a sound. The idea is to use these values directly to drive an
additive synthesizer to approximate the sound. I did it this way
(rather than just using librosa) because I wanted to be able to control
all the steps in the process.

Use numFrequencies to determine how many frequencies to return.
More frequencies should give a better approximation but will cost more later.
(We don't care about the cost of THIS code, though, since it's only run once.)
The top numFrequencies are those that are most prominent across all
time slices. This could be improved by using a psychoacoustic measure of 
loudness / salience rather than raw amplitude.

Use frameLength to determine how frequently the spectrum is sampled in the
time domain. The resynthesis will need to interpolate bewteen these.

spectrum[t] conatins a 2D ndarray. Each row of this array indicates one of
the top frequencies and its amplitude over that time window.

TODO: Output this in a form Faust can use.

TODO: We often get frequencies clustered close together.
        Example: if n = 4, we get 6780 and 6800, which are "the same harmonic".
      It would be better to take more frequencies than we want,
      then average out any that are very close together (less than a semitone?).
      After that we might have to discard the smaller ones to get the
      number we actually need of course.

TODO: Sometimes we seem to get a frequency that ramps up to full volume and
      stays there. Maybe this is a windowing artifact. We probably just
      want to filter those out as in the previous issue.

TODO: Consider adjusting the frequencies to the nearest note
      in a specific tuning.

"""

numFrequencies = 4 # How many top frequencies we select
frameLength = 0.2   # Length of each time window in seconds

np.set_printoptions(suppress=True)

########################################################################
# FUNCTIONS
########################################################################

def get_fft(y, sr):
  yf = scipy.fft.fft(y)

  # We want a 2-column list where col 0 is frequence, col 1 is amplitude

  # Grab the amplitudes
  n = len(y)
  sizes = 2.0/n * np.abs(yf[:n//2])
  # Normalize to a range 0 to 1
  sizes = sizes / np.max(sizes)
  # Match the sizes with their corresponding frequencies.
  # Maybe there's a neater way to do this, I don't know...
  tick = (2.0*sr) / (n/2)
  fftResult = np.vstack((np.array(range(0, len(sizes))) * tick,sizes))
  fftResult = np.transpose(fftResult)

  return fftResult

# Being lazy -- this recalculates the FFT but we only use it for checking / debugging
def show_fft(y, sr):
  yf = scipy.fft.fft(y)
  n = len(y)
  sizes = 2.0/n * np.abs(yf[:n//2])
  sizes = sizes / np.max(sizes)
  tick = (2.0*sr) / (n/2)
  fftResult = np.vstack((np.array(range(0, len(sizes))) * tick,sizes))
  fftResult = np.transpose(fftResult)

  xf = np.linspace(0.0, 2.0*sr, num=int(n/2))
  fig, ax = plt.subplots()
  ax.plot(xf, sizes)
  plt.grid()
  plt.xlabel("Frequency")
  plt.ylabel("Magnitude")
  return plt.show()

def getTopFrequencies(spec, topN):
  # Calculate total amplitude of each frequency across all time slices.
  # Those with the highest total value are assumed to be the most important.
  totals = {}
  for s in spectrum:
    for f in s:
        if str(f[0]) not in totals.keys():
          totals[str(f[0])] = 0
        totals[str(f[0])] = totals[str(f[0])] + f[1]

  # This is clumsy but totals is kind of complicated so I quickly build an ndarray
  # just containing the total amplitudes.
  # This could be tidied up!
  final = np.zeros((1, 1), dtype=np.float64)
  for t in totals.keys():
    totalLevel = float(totals.get(t))
    arr = [totalLevel]
    row =  np.array(arr)
    final = np.append(final,row)
  idx = np.argsort(final)

  retVal = []
  for s in range(0, len(spec) - 1):
    retVal.append(sorted(spec[s][idx[-1 * topN:]].tolist(),key=lambda x: x[0]))
  return retVal

def printSpectrumInTimeDomain():
  t = 0
  for x in spectrograph:
    print("Time:", t, "seconds")
    for f in x:
      print("   ", int(f[0]), round(f[1], 3))
    t = round(t + frameLength, 2)
  print("Total duration:", len(spectrum) * frameLength)

def printSpectrumInFrequencyDomain():
  for f in range(0, numFrequencies):
    print("Frequency:", spectrograph[0][f][0])
    t = 0
    for x in spectrograph:
      print("     t =",t, " : ", x[f][1])
      t = t + frameLength

########################################################################
# MAIN CODE
########################################################################

# Load the sample and get the FFT for each chunk of time:
y, sr = librosa.load("A-sample-0.wav")
spectrum = []
samplesPerFrame = int(sr * frameLength)
startSample = 0
while startSample < len(y):
  f = get_fft(y[startSample:startSample + samplesPerFrame], sr)
  spectrum.append(f)
  startSample = startSample + samplesPerFrame

spectrograph = getTopFrequencies(spectrum, numFrequencies)
#printSpectrumInTimeDomain()
printSpectrumInFrequencyDomain()

# Print the graph of the FFT for the WHOLE sound.
# The top frequencies we chose for the spectrum ought to match
# the most prominent frequencies in this graph.
show_fft(y, sr)