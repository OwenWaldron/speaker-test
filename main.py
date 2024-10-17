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
ecdf_y = np.linspace(0, 1, len(scores_flat), endpoint=False)
plt.plot(ecdf_x, ecdf_y)
plt.show()

# Find quantiles
quantiles = [0.6, 0.7, 0.75, 0.8, 0.9] # ascending order
for quant in quantiles:
    print(f'q({quant}) = {np.quantile(ecdf_x, quant)}')

# Fit a model to the E.C.D.F
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

def invexp(x, A, B):
    y = 1 / (1 + np.exp(- A * (x - B)))
    return y

def invexp2(x, A, B, C):
    y = 1 / (1 + C * np.exp(- A * (x - B)))
    return y

model = invexp

parameters, covariance = curve_fit(model, ecdf_x, ecdf_y)
print(f'Parameters: {parameters}')
print(f'Covariance: {covariance}')

# Overlay gaussian curve fit
plt.plot(ecdf_x, ecdf_y)
x_samp = np.linspace(min(ecdf_x), max(ecdf_x), 1000)
y_samp = model(x_samp, *parameters)
plt.plot(x_samp, y_samp)
plt.show()

