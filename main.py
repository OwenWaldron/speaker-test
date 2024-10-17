import kagglehub
import numpy as np
from numpy.linalg import norm
from speechbrain.inference.speaker import EncoderClassifier
import matplotlib.pyplot as plt
import torchaudio
from scipy.optimize import curve_fit

# Download latest version
# This will cache itself after first download
kagglePath = kagglehub.dataset_download("mozillaorg/common-voice")

# Create generator and set seed
randgen = np.random.default_rng(seed=80085)

# Select 100 random samples (out of the available 4068)
sample_numbers = randgen.permutation(4067)[:100]

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

# Create embeddings
def create_embedding(number):
    "Returns a normalized embedding of the "

    file_path = kagglePath + '/cv-valid-dev/cv-valid-dev/sample-{:0>6}.mp3'.format(number)

    signal, _ = torchaudio.load(file_path)
    out = classifier.encode_batch(signal).numpy()[0][0]
    out = out / norm(out)
    return out

# Create embeddings for sample
samples = np.array([create_embedding(num) for num in sample_numbers])

# Compute cosine similarities
scores = np.inner(samples, samples)
scores_flat = scores.flatten()

print(max(scores_flat))
print(min(scores_flat))

# Show a matrix heatmap
plt.imshow(scores, cmap='viridis', interpolation='none')
plt.colorbar()
plt.show()

# Show a histogram of scores
plt.hist(scores_flat, bins=20)
plt.show()

# Show the E.C.D.F
ecdf_x = np.sort(scores_flat)
ecdf_y = np.linspace(0, 1, len(scores_flat))
plt.plot(ecdf_x, ecdf_y, endpoint=False)
plt.show()

# Fit a model to the E.C.D.F
counts, bins = np.histogram(scores, bins=10)
midbins = (bins[1:] + bins[:-1]) / 2
SQRT_TWO_PI = np.sqrt(2 * np.pi)
def gaussian(x, A, B):
    y = np.exp(-0.5 * ((x-B)/A)**2) / (A * SQRT_TWO_PI)
    return y

def cubic(x, A, B, C, D):
    y = A + B * (x) + C * (x**2) + D * (x**3)
    return y

def quartic(x, A, B, C, D, E):
    y = A + B * (x) + C * (x**2) + D * (x**3) + E * (x**4)
    return y

model = gaussian

parameters, covariance = curve_fit(model, midbins, counts)
print(f'Parameters: {parameters}')
print(f'Covariance: {covariance}')

# Overlay gaussian curve fit
plt.stairs(counts, bins)
x_samp = np.linspace(min(midbins), max(midbins), 1000)
y_samp = model(x_samp, *parameters)
plt.plot(x_samp, y_samp)
plt.show()

