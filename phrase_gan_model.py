import numpy as np
import tensorflow as tf
import os, time, yaml
from abc import ABC, abstractmethod
from tensorflow.keras import layers

import phrases_lib

TRAINING_CHECKPOINTS_DIR = './training_checkpoints/ckpt'
SAMPLES_DIR = './samples'
EPOCHS = 50000
BATCH_SIZE = 256
EXAMPLES_TO_GENERATE = 16
MAX_RANGE = 32 # FIXME: Derive this from instrument data


class InvalidMaxStepsError(Exception):
  pass

class InvalidMaxRangeError(Exception):
  pass


class PhraseGanModel(ABC):

  def __init__(self,
               max_steps: int,
               instrument_count: int,
               input_width = 100,
               learning_rate: float = 1e-3):
    self.max_steps = max_steps
    self.instrument_count = instrument_count
    self.input_width = input_width
    self.learning_rate = learning_rate
    self.generator = self.make_generator_model()
    self.discriminator = self.make_discriminator_model()
    self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                          discriminator_optimizer=self.discriminator_optimizer,
                                          generator=self.generator,
                                          discriminator=self.discriminator)


  @abstractmethod
  def make_generator_model(self):
    pass


  @abstractmethod
  def make_discriminator_model(self):
    pass


  @abstractmethod
  def phrase_to_predictions(self, phrase):
    pass
  

  @abstractmethod
  def predictions_to_phrase(self, predictions):
    pass


  # Real music should produce all 1s, fake music all 0s.
  def discriminator_loss(self, real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


  # The generator ideally will make the discriminator produce all 1s.
  def generator_loss(self, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


  # @tf.function causes the function to be "compiled".
  @tf.function
  def train_step(self, phrase_batch, batch_size):
    noise = tf.random.normal([batch_size, self.input_width])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_phrases = self.generator(noise, training=True)

      real_output = self.discriminator(phrase_batch, training=True)
      fake_output = self.discriminator(generated_phrases, training=True)

      gen_loss = self.generator_loss(fake_output)
      disc_loss = self.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(
        zip(gradients_of_generator, self.generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    
    return (gen_loss, disc_loss)


  def train(self,
            phrases,
            instruments_data,
            checkpoint_prefix=TRAINING_CHECKPOINTS_DIR,
            samples_dir=SAMPLES_DIR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            samples_to_generate=EXAMPLES_TO_GENERATE,
            checkpoints_every=500,
            generate_every=500):

    phrase_arrays = [self.phrase_to_predictions(phrase) for phrase in phrases]
    dataset = tf.data.Dataset.from_tensor_slices(phrase_arrays).shuffle(len(phrases)).batch(batch_size)
    print("Number of batches is %d" % len(dataset))
    
    # We will reuse this seed to generate progressive examples as the generator
    # is trained.
    seed = tf.random.normal(shape=[samples_to_generate, self.input_width],
                            mean=0.0,
                            stddev=1.0,
                            dtype=tf.dtypes.float32)
    self.generate_and_save_phrases(samples_dir=samples_dir,
                                   instruments_data=instruments_data,
                                   epoch=0,
                                   test_input=seed)

    for epoch in range(epochs):
      start = time.time()

      for phrase_batch in dataset:
        gen_loss, disc_loss = self.train_step(phrase_batch=phrase_batch, batch_size=batch_size)
      
      # Save the model and generate phrases every so many epochs.
      if (epoch + 1) % checkpoints_every == 0:
        self.checkpoint.save(file_prefix=checkpoint_prefix)

      if (epoch + 1) % generate_every == 0:
        self.generate_and_save_phrases(samples_dir=samples_dir,
                                       instruments_data=instruments_data,
                                       epoch=epoch + 1,
                                       test_input=seed)

      print('Time for epoch %d is %0.3fs; last batch gen loss = %f, disc loss = %f' %
          (epoch + 1, time.time() - start, gen_loss.numpy(), disc_loss.numpy()))

    # Generate phrases after the final epoch -- the grand finale! :)
    self.generate_and_save_phrases(samples_dir=samples_dir,
                                   instruments_data=instruments_data,
                                   epoch=epochs,
                                   test_input=seed)


  def generate_and_save_phrases(self, samples_dir, instruments_data, epoch, test_input):
    os.makedirs(samples_dir, exist_ok=True)

    # "training" is set to False so that all layers run in inference mode
    # (batchnorm).
    predictions = self.generator(test_input, training=False)
    for (phrase_idx, phrase_array) in enumerate(predictions):
      phrase = self.predictions_to_phrase(phrase_array)
      phrase.write_to_midi_file(
          output_file='%s/epoch_%05d_phrase_%02d.midi' % (samples_dir, epoch, phrase_idx),
          instruments_data=instruments_data,
          qpm=80)


class PhraseConv1DGanModel(PhraseGanModel):

  def make_generator_model(self):
    if self.max_steps % 16 != 0:
      raise InvalidMaxStepsError("Max steps must be a multiple of 16, currently %d" %
          self.max_steps)
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(self.max_steps // 16 * 64, use_bias=False, input_shape=(self.input_width,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((self.max_steps // 16, 64)))
    assert model.output_shape == (None, self.max_steps // 16, 64) # None is the batch size

    model.add(layers.Conv1DTranspose(64, 5, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, self.max_steps // 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(64, 5, strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, self.max_steps // 8, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(128, 5, strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, self.max_steps // 4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(256, 5, strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, self.max_steps // 2, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(self.instrument_count, 5,
                                    strides=2,
                                    padding='same',
                                    use_bias=False,
                                    activation='tanh'))
    assert model.output_shape == (None, self.max_steps, self.instrument_count)

    return model


  def make_discriminator_model(self):
    if self.max_steps % 16 != 0:
      raise InvalidMaxStepsError("Max steps must be a multiple of 16, currently %d" %
          self.max_steps)
    
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(256, 5,
                            strides=2,
                            padding='same',
                            input_shape=[self.max_steps, self.instrument_count]))
    assert model.output_shape == (None, self.max_steps // 2, 256)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, 5, strides=2, padding='same'))
    assert model.output_shape == (None, self.max_steps // 4, 128)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(64, 5, strides=2, padding='same'))
    assert model.output_shape == (None, self.max_steps // 8, 64)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(64, 5, strides=2, padding='same'))
    assert model.output_shape == (None, self.max_steps // 16, 64)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    assert model.output_shape == (None, 1)

    return model


  def phrase_to_predictions(self, phrase):
    # Our max instrument range is about 29 pitches, plus the -1 for NO_PITCH, so we
    # subtract 14 and divide by 15 to distribute it evenly in the [-1, 1] range.
    return (phrase.notes_array.transpose().astype('float32') - 14) / 15
  

  def predictions_to_phrase(self, predictions):
    # Reverse the transformation done in phrases_to_dataset.
    notes_array = np.round(np.array(predictions).transpose() * 15 + 14).astype('int')
    return phrases_lib.Phrase(notes_array = notes_array)


class PhraseConv2DGanModel(PhraseGanModel):

  def make_generator_model(self):
    if self.max_steps % 16 != 0:
      raise InvalidMaxStepsError("Max steps must be a multiple of 16, currently %d" %
          self.max_steps)
    
    if MAX_RANGE % 8 != 0:
      raise InvalidMaxRangeError("Max instrument range must be a multiple of 8, currently %d" %
          MAX_RANGE)
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(32 * self.max_steps // 16 * MAX_RANGE // 8,
                           use_bias=False,
                           input_shape=(self.input_width,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((32, self.max_steps // 16, MAX_RANGE // 8)))
    assert model.output_shape == (None, 32, self.max_steps // 16, MAX_RANGE // 8) # None is the batch size

    model.add(layers.Conv2DTranspose(32, (4, 4),
                                     strides=(1, 1),
                                     padding='same',
                                     use_bias=False,
                                     data_format='channels_first'))
    assert model.output_shape == (None, 32, self.max_steps // 16, MAX_RANGE // 8)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (4, 4),
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False,
                                     data_format='channels_first'))
    assert model.output_shape == (None, 32, self.max_steps // 8, MAX_RANGE // 4)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (4, 4),
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False,
                                     data_format='channels_first'))
    assert model.output_shape == (None, 64, self.max_steps // 4, MAX_RANGE // 2)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (4, 4),
                                     strides=(2, 2),
                                     padding='same',
                                     data_format='channels_first'))
    assert model.output_shape == (None, 128, self.max_steps // 2, MAX_RANGE)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(self.instrument_count, (4, MAX_RANGE),
                                     strides=(2, 1),
                                     padding='same',
                                     use_bias=False,
                                     activation='sigmoid',
                                     data_format='channels_first'))
    assert model.output_shape == (None, self.instrument_count, self.max_steps, MAX_RANGE)

    return model


  def make_discriminator_model(self):
    if self.max_steps % 16 != 0:
      raise InvalidMaxStepsError("Max steps must be a multiple of 16, currently %d" %
          self.max_steps)
    
    if MAX_RANGE % 8 != 0:
      raise InvalidMaxRangeError("Max instrument range must be a multiple of 8, currently %d" %
          MAX_RANGE)

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (4, MAX_RANGE),
                            strides=(2, 1),
                            padding='same',
                            data_format='channels_first',
                            input_shape=(self.instrument_count, self.max_steps, MAX_RANGE)))
    assert model.output_shape == (None, 128, self.max_steps // 2, MAX_RANGE)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', data_format='channels_first'))
    assert model.output_shape == (None, 64, self.max_steps // 4, MAX_RANGE // 2)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', data_format='channels_first'))
    assert model.output_shape == (None, 32, self.max_steps // 8, MAX_RANGE // 4)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', data_format='channels_first'))
    assert model.output_shape == (None, 32, self.max_steps // 16, MAX_RANGE // 8)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    assert model.output_shape == (None, 1)

    return model


  def phrase_to_predictions(self, phrase):
    # Transform each pitch into a one-hot vector.
    return tf.one_hot(phrase.notes_array + 1, MAX_RANGE)
  

  def predictions_to_phrase(self, predictions):
    # Take the pitch with the max output value as the single pitch for that
    # instrument for that step.
    notes_array = tf.math.argmax(predictions, 2).numpy() - 1
    return phrases_lib.Phrase(notes_array = notes_array)


class PhraseDenseGanModel(PhraseGanModel):

  def make_generator_model(self):
    model = tf.keras.Sequential()
    model.add(layers.Dense(200, input_shape=(self.input_width,)))
    assert model.output_shape == (None, 200) # None is the batch size
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(200))
    assert model.output_shape == (None, 200)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(self.max_steps * self.instrument_count, activation='tanh'))
    model.add(layers.Reshape((self.max_steps, self.instrument_count)))
    assert model.output_shape == (None, self.max_steps, self.instrument_count)

    return model


  def make_discriminator_model(self):
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(self.max_steps, self.instrument_count)))
    assert model.output_shape == (None, self.max_steps * self.instrument_count)

    model.add(layers.Dense(200))
    assert model.output_shape == (None, 200)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(200))
    assert model.output_shape == (None, 200)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1))
    assert model.output_shape == (None, 1)

    return model


  def phrase_to_predictions(self, phrase):
    # Our max instrument range is about 29 pitches, plus the -1 for NO_PITCH, so we
    # subtract 14 and divide by 15 to distribute it evenly in the [-1, 1] range.
    return (phrase.notes_array.transpose().astype('float32') - 14) / 15
  

  def predictions_to_phrase(self, predictions):
    # Reverse the transformation done in phrases_to_dataset.
    notes_array = np.round(np.array(predictions).transpose() * 15 + 14).astype('int')
    return phrases_lib.Phrase(notes_array = notes_array)
