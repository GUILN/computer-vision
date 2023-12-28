import datetime
import logging
import os
import time
from typing import Tuple
import tensorflow as tf

from gan_network.discriminator import Discriminator, discriminator_loss
from gan_network.generator import Generator, generator_loss


class GanModel:
    def __init__(
        self,
        checkpoint_dir: str,
        save_image_dir: str,
        log_dir: str = "logs/",
        input_size: Tuple[int, int] = (256, 256),
    ):
        self._generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self._discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self._generator = Generator()
        self._discriminator = Discriminator()
        self._log_dir = log_dir
        self._save_image_dir = save_image_dir
        self._summary_writer = tf.summary.create_file_writer(
            log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self._checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self._checkpoint = tf.train.Checkpoint(
            generator_optimizer=self._generator_optimizer,
            discriminator_optimizer=self._discriminator_optimizer,
        )

    def generate_images(
        self,
        test_input: tf.Tensor,
        target: tf.Tensor,
        image_name: str = "generated_image.png",
    ) -> None:
        """
        Generates images.
        :param test_input: the test input
        :param target: the target
        """
        prediction = self._generator(test_input, training=True)
        # save test_input, tar and prediction

        tf.keras.preprocessing.image.save_img(
            os.path.join(self._save_image_dir, "prediction_" + image_name),
            prediction[0].numpy(),
        )
        tf.keras.preprocessing.image.save_img(
            os.path.join(self._save_image_dir, "target_" + image_name),
            target[0].numpy(),
        )
        tf.keras.preprocessing.image.save_img(
            os.path.join(self._save_image_dir, "test_input_" + image_name),
            test_input[0].numpy(),
        )

    def fit(self, train_ds, test_ds, steps: int = 40000):
        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step + 1) % 1000 == 0:
                if step != 0:
                    logging.info(
                        f"Time taken for 1000 steps: {time.time()-start:.2f} sec\n"
                    )
                start = time.time()

                self.generate_images(
                    example_input, example_target, "image_at_step_" + str(step)
                )

            self._train_step(input_image, target, step)
            # Training step
            if (step + 1) % 100 == 0:
                logging.info(f"Step: {step+1}")

            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 5000 == 0:
                self._checkpoint.save(file_prefix=self._checkpoint_prefix)

    @tf.function
    def _train_step(self, input_image: tf.Tensor, target: tf.Tensor, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape as disc_tape:
            gen_output = self._generator(input_image, training=True)
            disc_real_output = self._discriminator([input_image, target], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output=disc_real_output,
                gen_output=gen_output,
                target=target,
            )
            disc_loss = discriminator_loss(
                disc_real_output=disc_real_output,
                disc_generated_output=disc_generated_output,
            )
        generator_gradients = gen_tape.gradient(
            gen_total_loss, self._generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self._discriminator.trainable_variables
        )

        self._generator_optimizer.apply_gradients(
            zip(generator_gradients, self._generator.trainable_variables)
        )
        self._discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self._discriminator.trainable_variables)
        )

        with self._summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step)
            tf.summary.scalar("disc_loss", disc_loss, step=step)
