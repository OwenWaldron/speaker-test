import kagglehub
import numpy as np
from numpy.linalg import norm
from speechbrain.inference.speaker import EncoderClassifier
import matplotlib.pyplot as plt
import torchaudio
from ecdf_models import invexp
from scipy.optimize import curve_fit

################## Config settings ##################

# Parameters
MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_SIZE = 100
SEED = 80085
ECDF_MODEL = invexp # Model to fit ECDF curve (if enabled)

# Outputs
PRINT_BOUNDS = True
SHOW_MATRIX = True
SHOW_HIST = True
SHOW_ECDF = True
PRINT_QUANTILES = True
FIT_MODEL_ECDF = True


#####################################################



# Download latest version
# This will cache itself after first download
kagglePath = kagglehub.dataset_download("mozillaorg/common-voice")

# Create generator and set seed
randgen = np.random.default_rng(seed=SEED)

# Select 100 random samples (out of the available 4068)
sample_numbers = randgen.permutation(4067)[:SAMPLE_SIZE]

classifier = EncoderClassifier.from_hparams(source=MODEL, savedir=f"pretrained_models/{MODEL}")

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

if PRINT_BOUNDS:
    print(f'Max: {max(scores_flat)}')
    print(f'Min: {min(scores_flat)}')

# Show a matrix heatmap
if SHOW_MATRIX:
    plt.imshow(scores, cmap='viridis', interpolation='none')
    plt.colorbar(label='Cosine similarity')
    plt.title(f'Similarity scores for {MODEL}')
    plt.show()

# Show a histogram of scores
if SHOW_HIST:
    plt.hist(scores_flat, bins=20)
    plt.title(f'Similarity score distribution for {MODEL}')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Number of occurences')
    plt.show()

# Show the E.C.D.F
if SHOW_ECDF:
    ecdf_x = np.sort(scores_flat)
    ecdf_y = np.linspace(0, 1, len(scores_flat), endpoint=False)
    plt.plot(ecdf_x, ecdf_y)
    plt.title(f'E.c.d.f of similarity scores for {MODEL}')
    plt.xlabel('Cosine similarity')
    plt.ylabel('F(n)')
    plt.show()

# Find quantiles
if PRINT_QUANTILES:
    quantiles = [0.6, 0.7, 0.75, 0.8, 0.9] # ascending order
    for quant in quantiles:
        print(f'q({quant}) = {np.quantile(ecdf_x, quant)}')

# Fit a model to the E.C.D.F
if FIT_MODEL_ECDF and ECDF_MODEL:
    parameters, covariance = curve_fit(ECDF_MODEL, ecdf_x, ecdf_y)
    print(f'Parameters: {parameters}')
    print(f'Covariance: {covariance}')

    # Overlay gaussian curve fit
    plt.plot(ecdf_x, ecdf_y)
    x_samp = np.linspace(min(ecdf_x), max(ecdf_x), 1000)
    y_samp = ECDF_MODEL(x_samp, *parameters)
    plt.plot(x_samp, y_samp)
    plt.title(f'E.c.d.f of similarity scores for {MODEL}')
    plt.xlabel('Cosine similarity')
    plt.ylabel('F(n)')
    plt.show()

