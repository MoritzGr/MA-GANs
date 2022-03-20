"""
Title: WGAN-GP overriding `Model.train_step`
Author: [A_K_Nain](https://twitter.com/A_K_Nain)
Date created: 2020/05/9
Last modified: 2020/05/9
Description: Implementation of Wasserstein GAN with Gradient Penalty.
"""
import os

import custom_dataset
import settings

"""
## Wasserstein GAN (WGAN) with Gradient Penalty (GP)
The original [Wasserstein GAN](https://arxiv.org/abs/1701.07875) leverages the
Wasserstein distance to produce a value function that has better theoretical
properties than the value function used in the original GAN paper. WGAN requires
that the discriminator (aka the critic) lie within the space of 1-Lipschitz
functions. The authors proposed the idea of weight clipping to achieve this
constraint. Though weight clipping works, it can be a problematic way to enforce
1-Lipschitz constraint and can cause undesirable behavior, e.g. a very deep WGAN
discriminator (critic) often fails to converge.
The [WGAN-GP](https://arxiv.org/pdf/1704.00028.pdf) method proposes an
alternative to weight clipping to ensure smooth training. Instead of clipping
the weights, the authors proposed a "gradient penalty" by adding a loss term
that keeps the L2 norm of the discriminator gradients close to 1.
"""

"""
## Setup
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
## Prepare the Fashion-MNIST data
To demonstrate how to train WGAN-GP, we will be using the
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Each
sample in this dataset is a 28x28 grayscale image associated with a label from
10 classes (e.g. trouser, pullover, sneaker, etc.)
"""
"""
## Create the discriminator (the critic in the original WGAN)
The samples in the dataset have a (28, 28, 1) shape. Because we will be
using strided convolutions, this can result in a shape with odd dimensions.
For example,
`(28, 28) -> Conv_s2 -> (14, 14) -> Conv_s2 -> (7, 7) -> Conv_s2 ->(3, 3)`.
While peforming upsampling in the generator part of the network, we won't get 
the same input shape as the original images if we aren't careful. To avoid this,
we will do something much simpler:
- In the discriminator: "zero pad" the input to change the shape to `(32, 32, 1)`
for each sample; and
- Ihe generator: crop the final output to match the shape with input shape.
"""


def conv_block(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        use_bn=False,
        use_dropout=False,
        drop_value=0.5,
):
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_discriminator_model():
    img_input = layers.Input(shape=settings.IMG_SHAPE)

    x = conv_block(
        img_input,
        64,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(0.2),
        use_dropout=False,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        128,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        256,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        512,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
    )

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    out1 = layers.Dense(1)(x)
    out2 = layers.Dense(settings.N_CLASSES, activation='softmax')(x)

    d_model = keras.models.Model(img_input, [out1, out2], name="discriminator")
    return d_model


"""
## Create the generator
"""


def upsample_block(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        up_size=(2, 2),
        padding="same",
        use_bn=False,
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
):
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_generator_model():
    quarter_size = int(settings.IMG_SIZE / 4)

    in_label = layers.Input(shape=(1,))
    li = layers.Embedding(settings.N_CLASSES, 100)(in_label)

    n_nodes = quarter_size * quarter_size * settings.IMG_CHANNELS

    li = layers.Dense(n_nodes)(li)
    li = layers.Reshape((quarter_size, quarter_size, settings.IMG_CHANNELS))(li)

    noise = layers.Input(shape=(settings.LATENT_DIM,))

    n_nodes = 256 * quarter_size * quarter_size
    x = layers.Dense(n_nodes, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((quarter_size, quarter_size, 256))(x)

    merge = layers.Concatenate()([x, li])

    x = upsample_block(
        merge,
        128,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        64,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        merge,
        32,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x, settings.IMG_CHANNELS, layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
    )

    g_model = keras.models.Model([noise, in_label], x, name="generator")
    return g_model


"""
## Create the WGAN-GP model
Now that we have defined our generator and discriminator, it's time to implement
the WGAN-GP model. We will also override the `train_step` for training.
"""


class WGAN(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
            discriminator_extra_steps=settings.N_CRITIC,
            gp_weight=settings.GP_WEIGHT,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_input):
        if isinstance(real_input, tuple):
            real_input = real_input[0]

        (real_images, real_labels) = real_input

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            random_image_labels = tf.random.uniform(
                shape=(batch_size, 1,), maxval=settings.N_CLASSES - 1, dtype=tf.dtypes.int32
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, random_image_labels], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_logits, fake_logits, real_labels)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_image_labels = tf.random.uniform(shape=(batch_size, 1,), maxval=settings.N_CLASSES - 1,
                                                dtype=tf.dtypes.int32)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, random_image_labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits, random_image_labels)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_logits, fake_logits, real_labels_true):
    (real_img, real_labels_pred) = real_logits
    (fake_img, fake_labels_pred) = fake_logits
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    label_loss = scce(real_labels_true, real_labels_pred)
    return fake_loss - real_loss + label_loss


# Define the loss functions for the generator.
def generator_loss(fake_logits, fake_labels_true):
    (fake_img, fake_labels_pred) = fake_logits
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    label_loss = scce(fake_labels_true, fake_labels_pred)
    return -tf.reduce_mean(fake_img) + label_loss


"""
## Create a Keras callback that periodically saves generated images
"""


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=settings.N_CLASSES, latent_dim=128, ckpt_manager=None):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.ckpt_manager = ckpt_manager

    def on_epoch_end(self, epoch, logs=None):
        tf.random.set_seed(0)
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim), seed=0)
        class_labels_vectors = tf.range(0, min(self.num_img, settings.N_CLASSES))
        generated_images = self.model.generator([random_latent_vectors, class_labels_vectors])
        generated_images = (generated_images * 127.5) + 127.5

        if not os.path.isdir("./generated/"):
            os.mkdir("./generated/")

        if not self.ckpt_manager == None:
            self.ckpt_manager.save(checkpoint_number=epoch)

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("./generated/Class_{i}_Epoch_{epoch}.png".format(i=i, epoch=epoch))


def main():
    BATCH_SIZE = settings.BATCH_SIZE

    # Size of the noise vector
    noise_dim = settings.LATENT_DIM

    (train_images, train_labels) = custom_dataset.get_wikiart_dataset()
    print(f"Number of examples: {len(train_images)}")
    print(f"Shape of the images in the dataset: {train_images.shape[1:]}")

    # Reshape each sample and normalize the pixel values in the [-1, 1] range
    train_images = train_images.reshape(train_images.shape[0], *settings.IMG_SHAPE).astype("float32")
    train_images = (train_images - 127.5) / 127.5

    g_model = get_generator_model()
    g_model.summary()

    d_model = get_discriminator_model()
    d_model.summary()

    """
    ## Train the end-to-end model
    """

    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=settings.LEARNING_RATE, beta_1=settings.BETA_1, beta_2=settings.BETA_2
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=settings.LEARNING_RATE, beta_1=settings.BETA_1, beta_2=settings.BETA_2
    )

    # Set the number of epochs for training.
    if settings.USE_EPOCHS:
        epochs = settings.EPOCHS
    else:
        epochs = int(settings.NUM_IMAGES_TOTAL / len(train_images))

    # Get the wgan model
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=noise_dim,
    )

    # Compile the wgan model
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )

    checkpoint_dir = './generated/checkpoints'
    checkpoint = tf.train.Checkpoint(wgan=wgan)

    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        start_epoch = int(ckpt_manager.latest_checkpoint[len('./generated/checkpoints\\ckpt-'):])
        start_epoch = start_epoch + 1
        print('Latest checkpoint restored at epoch: ' + str(start_epoch))

    # Instantiate the customer `GANMonitor` Keras callback.
    cbk = GANMonitor(num_img=settings.N_CLASSES, latent_dim=noise_dim, ckpt_manager=ckpt_manager)
    csv_logger = tf.keras.callbacks.CSVLogger("./generated/log.csv", append=True, separator=',')
    tensorboard_logger = tf.keras.callbacks.TensorBoard(log_dir='./generated/logs', histogram_freq=1, write_graph=True,
                                                        write_images=False)

    # Start training
    input = (train_images, train_labels)
    wgan.fit(input, batch_size=BATCH_SIZE, initial_epoch=start_epoch, epochs=epochs,
             callbacks=[cbk, csv_logger, tensorboard_logger])


if __name__ == "__main__":
    main()
