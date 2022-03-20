# example of loading the generator model and generating images
import os
import random
from math import sqrt

import mpl_toolkits.axes_grid1
import numpy.random
from numpy import asarray
from numpy.random import randn
from matplotlib import pyplot
import tensorflow as tf
import ACWGAN_GP
import settings


def generate_images(n_samples, target_class, gen, disc, times_called):
    latent_points, labels = generate_latent_points(settings.LATENT_DIM, n_samples, target_class, times_called)
    # generate images
    X = gen.predict([latent_points, labels])
    # scale from [-1,1] to [0,1]
    Y = disc.predict(X)
    X = (X + 1) / 2.0

    return (X, Y)


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, target_class, seed=0):
    numpy.random.seed(seed)
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = asarray([target_class for _ in range(n_samples)])
    numpy.random.seed()
    return [z_input, labels]


# create and save a plot of generated images
def save_plot(examples, n_examples, filename):
    fig = pyplot.figure(figsize=(int(sqrt(n_examples)), int(sqrt(n_examples))))
    fig.tight_layout()
    pyplot.subplots_adjust(wspace=0., hspace=0., top=1, bottom=0, right=1, left=0)
    grid = mpl_toolkits.axes_grid1.ImageGrid(fig, 111,
                                             nrows_ncols=(int(sqrt(n_examples)), int(sqrt(n_examples))),
                                             axes_pad=0)

    for ax, im in zip(grid, examples):
        ax.imshow(im)
        ax.axis('off')

    if filename == None:
        pyplot.show()
    else:
        pyplot.savefig(filename)

    pyplot.close(fig)



def load_models(checkpoint_dir):
    noise_dim = settings.LATENT_DIM

    g_model = ACWGAN_GP.get_generator_model()

    d_model = ACWGAN_GP.get_discriminator_model()

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=settings.LEARNING_RATE, beta_1=settings.BETA_1, beta_2=settings.BETA_2
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=settings.LEARNING_RATE, beta_1=settings.BETA_1, beta_2=settings.BETA_2
    )

    # Get the wgan model
    wgan = ACWGAN_GP.WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=noise_dim,
    )

    # Compile the wgan model
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=ACWGAN_GP.generator_loss,
        d_loss_fn=ACWGAN_GP.discriminator_loss,
    )
    # set up checkpoints

    checkpoint = tf.train.Checkpoint(wgan=wgan)

    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    checkpoint.restore(ckpt_manager.latest_checkpoint)

    return g_model, d_model


# plots the first 25 images of seed 0
def plot_images(n_examples, target_class, gen, disc, filename=None):
    goodims = []
    while not len(goodims) >= n_examples:
        (X, Y) = generate_images(n_examples, target_class, gen, disc, 0)
        for sample in X:
            goodims.append(sample)

    goodims = asarray(goodims)
    # plot the result
    save_plot(goodims, n_examples, filename)


# plots the first 25 images of seeds 0,1,2,... that fit class
def plot_good_images(n_examples, target_class, gen, disc, filename=None):
    goodims = []
    j = 0
    while not len(goodims) >= n_examples:
        (X, Y) = generate_images(n_examples, target_class, gen, disc, j)
        i = 0
        for sample in X:
            if (1 - Y[1][i][target_class]) < 0.1:
                goodims.append(sample)
            i = i + 1
        j = j + 1
        if not len(goodims) >= j:
            goodims.append(find_best_image(X, Y, target_class))

    goodims = asarray(goodims)
    # plot the result
    save_plot(goodims, n_examples, filename)


# plots the top image of seeds 0,1,2,...,n_examples out of x images each
def find_best_image(X, Y, n_class):
    target = numpy.median(Y[0])
    previous_best = -999999999
    output = None
    found_class_match = False
    for i, sample in enumerate(X):
        if found_class_match:
            if (1 - Y[1][i][n_class]) < 0.1 and Y[0][i] > previous_best:
                previous_best = Y[0][i]
                output = sample
                # print("found new best: ", previous_best)
        else:
            if Y[0][i] > previous_best:
                previous_best = Y[0][i]
                output = sample
                found_class_match = (1 - Y[1][i][n_class]) < 0.1
                # print("found new best: ", previous_best)

    return output


# runs 2 means clustering algorithm and returns the closest example of the "good" cluster to the mean
def find_best_image_clustered(X, Y, n_class):
    output = None
    clusters_changed = True
    cluster_1 = []
    cluster_center_1 = numpy.min(Y[0])
    cluster_2 = []
    cluster_center_2 = numpy.max(Y[0])

    while clusters_changed:
        values_1 = []
        values_2 = []
        for index, value in enumerate(Y[0]):
            if numpy.abs(value - cluster_center_1) < numpy.abs(value - cluster_center_2):
                cluster_1.append(index)
                values_1.append(value)
            else:
                cluster_2.append(index)
                values_2.append(value)
        mean_1 = numpy.mean(values_1)
        mean_2 = numpy.mean(values_2)
        if not mean_1 == cluster_center_1:
            cluster_center_1 = mean_1
            cluster_center_2 = mean_2
            cluster_1 = []
            cluster_2 = []
        else:
            clusters_changed = False

    cluster_1_class_hits = 0
    for index in cluster_1:
        if (1 - Y[1][index][n_class]) < 0.1:
            cluster_1_class_hits += 1

    cluster_2_class_hits = 0
    for index in cluster_2:
        if (1 - Y[1][index][n_class]) < 0.1:
            cluster_2_class_hits += 1

    # print(f"cluster 1 hit rate: {cluster_1_class_hits / len(cluster_1)}")
    # print(f"cluster 2 hit rate: {cluster_2_class_hits / len(cluster_2)}")

    if cluster_1_class_hits / len(cluster_1) > cluster_2_class_hits / len(cluster_2):
        good_cluster = cluster_1
        good_mean = cluster_center_1
    else:
        good_cluster = cluster_2
        good_mean = cluster_center_2

    previous_best = 9999999999
    found_class_match = False
    for index in good_cluster:
        if found_class_match:
            if (1 - Y[1][index][n_class]) < 0.1 and numpy.abs(Y[0][index] - good_mean) < previous_best:
                previous_best = Y[0][index]
                output = X[index]
                # print("found new best: ", previous_best)
        else:
            if numpy.abs(Y[0][index] - good_mean) < previous_best:
                previous_best = Y[0][index]
                output = X[index]
                found_class_match = (1 - Y[1][index][n_class]) < 0.1
                # print("found new best: ", previous_best)

    return output


def plot_very_good_images(n_examples, target_class, gen, disc, filename=None):
    x = 100  # best 1 out of x images
    goodims = []
    j = 0
    while not len(goodims) >= n_examples:
        (X, Y) = generate_images(x, target_class, gen, disc, j)
        goodims.append(find_best_image_clustered(X, Y, target_class))
        j = j + 1

    goodims = asarray(goodims)
    # plot the result
    save_plot(goodims, n_examples, filename)


def plot_random_good_images(n_examples, gen, disc, filename=None):
    x = 100  # best 1 out of x images
    goodims = []
    j = 0
    while not len(goodims) >= n_examples:
        random.seed(j)
        random_class = random.randint(0, settings.N_CLASSES - 1)
        random.seed()
        (X, Y) = generate_images(x, random_class, gen, disc, j)
        goodims.append(find_best_image_clustered(X, Y, random_class))
        j = j + 1

    goodims = asarray(goodims)
    # plot the result
    save_plot(goodims, n_examples, filename)

def save_all_ims(gen, disc, n_examples, dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    plot_random_good_images(n_examples, gen, disc, dir + "random_good_images")
    for target_class in range(settings.N_CLASSES):
        plot_images(n_examples, target_class, gen, disc, dir + "images_class_" + str(target_class))
        plot_good_images(n_examples, target_class, gen, disc, dir + "good_images_class_" + str(target_class))
        plot_very_good_images(n_examples, target_class, gen, disc, dir + "very_good_images_class_" + str(target_class))


def main():
    # load model
    checkpoint_dir = "C:\Master Thesis (GAN Art)\Results\ACWGAN_generated_emoji_bycat_facesonly\generated\checkpoints"
    gen, disc = load_models(checkpoint_dir)

    n_examples = 25  # must be a square

    save_all_ims(gen, disc, n_examples, checkpoint_dir[:-len("generated\checkpoints")] + 'result_ims/')


if __name__ == '__main__':
    main()
