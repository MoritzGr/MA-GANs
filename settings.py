# Path to training dataset
ORIGINAL_IMAGES_PATH = 'C:/Master Thesis (GAN Art)/Data/images full/images'
# Path to store thumbnails of training dataset
RESIZED_IMAGES_PATH = './datasets/'
# Emoji list path
EMOJI_LIST_PATH = "C:/Master Thesis (GAN Art)/Data/Emoji/Full Emoji List, v14.0.html"
# Emoji dataset path
EMOJI_IMAGES_PATH = 'C:/Master Thesis (GAN Art)/Data/Emoji/images'
# Size of Images (number of pixels on one side of the square)
IMG_SIZE = 64
# Number of color channels of the images. 3: RGB 1: greyscale
IMG_CHANNELS = 3
# image shape
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
# Size of the latent space
LATENT_DIM = 128
# Number of iterations to pretrain the discriminator for
PREITERS = 2000
# Batch size
BATCH_SIZE = 64
# Number of Epochs
EPOCHS = 175
# number of total images trained on: (16.8 million in GANgogh)
NUM_IMAGES_TOTAL = 16800000
# whether to use epochs or total number of images for training length
USE_EPOCHS = False
# Number of iterations to train the critic per generator iteration
N_CRITIC = 5
# Number of Classes
N_CLASSES = 14
# Weight for the gradient penalty of the wasserstein loss in the ACWGAN
GP_WEIGHT = 10.0
# Learning rate of the Adam optimizer for both discriminator and critic
LEARNING_RATE = 0.0002
# Beta value 1 of the Adam optimizer for both discriminator and critic
BETA_1 = 0.5
# Beta value 2 of the Adam optimizer for both discriminator and critic
BETA_2 = 0.9