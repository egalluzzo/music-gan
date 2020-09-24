import numpy as np
import tensorflow as tf
import os, random, time, yaml
from abc import ABC, abstractmethod
from tensorflow.keras import layers, Input, Model

import phrases_lib

TRAINING_CHECKPOINTS_DIR = './training_checkpoints'
SAMPLES_DIR = './samples'
EPOCHS = 50000
BATCH_SIZE = 256
EXAMPLES_TO_GENERATE = 16
MAX_RANGE = 32 # FIXME: Derive this from instrument data
KEY_EMBEDDING_SIZE = 20

class InvalidMaxStepsError(Exception):
  pass

class InvalidMaxRangeError(Exception):
  pass


class PhraseCganModel(ABC):

  def __init__(self,
               max_steps: int,
               instrument_count: int,
               input_width = 100,
               learning_rate: float = 1e-3):
    self.max_steps = max_steps
    self.instrument_count = instrument_count
    self.input_width = input_width
    self.learning_rate = learning_rate

    self.discriminator = self.make_discriminator_model()
    self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    self.discriminator.compile(loss='binary_crossentropy',
                               optimizer=self.discriminator_optimizer,
                               metrics=['accuracy'])
    
    self.generator = self.make_generator_model()
    # Make discriminator weights not trainable while training the generator.
    self.discriminator.trainable = False
    generator_latent_input, generator_keys_input = self.generator.input
    generator_output = self.generator.output

    # Make a combined model where we feed the generator's output through the
    # discriminator.
    combined_output = self.discriminator([generator_output, generator_keys_input])
    self.combined_model = Model([generator_latent_input, generator_keys_input], combined_output)
    self.combined_optimizer = tf.keras.optimizers.Adam(learning_rate)
    self.combined_model.compile(loss='binary_crossentropy', optimizer=self.combined_optimizer)


  def make_generator_model(self):
    # We could also use the time signature as a label, but we're not yet.

    # We want start and end key to use the same embedding, so start key is
    # the first element and end key is the second element.
    keys_input = Input(shape=(2,), dtype='int32')
    keys_input_layer = layers.Embedding(24, KEY_EMBEDDING_SIZE)(keys_input)
    keys_input_layer = layers.Flatten()(keys_input_layer)
    keys_input_layer = layers.Dense(self.max_steps // 16)(keys_input_layer)
    # Reshape to an additional feature map.
    keys_input_layer = layers.Reshape((self.max_steps // 16, 1))(keys_input_layer)

    # Define the latent input.
    latent_input = Input(shape=(self.input_width,))

    # Build a dense network turning the latent input into a stack of feature maps.
    latent_input_layer = layers.Dense(self.max_steps // 16 * 64)(latent_input)
    latent_input_layer = layers.LeakyReLU()(latent_input_layer)
    latent_input_layer = layers.Dropout(0.3)(latent_input_layer)
    latent_input_layer = layers.Reshape((self.max_steps // 16, 64))(latent_input_layer)

    # Concatenate the two input layers.
    merged_layer = layers.Concatenate()([keys_input_layer, latent_input_layer])

    # Now deconvolve several times until we get to a phrase.
    feature_layer = layers.Conv1DTranspose(64, 4, strides=2, padding='same', use_bias=False)(merged_layer)
    feature_layer = layers.BatchNormalization()(feature_layer)
    feature_layer = layers.LeakyReLU()(feature_layer)

    feature_layer = layers.Conv1DTranspose(128, 4, strides=2, padding='same', use_bias=False)(feature_layer)
    feature_layer = layers.BatchNormalization()(feature_layer)
    feature_layer = layers.LeakyReLU()(feature_layer)

    feature_layer = layers.Conv1DTranspose(256, 4, strides=2, padding='same', use_bias=False)(feature_layer)
    feature_layer = layers.BatchNormalization()(feature_layer)
    feature_layer = layers.LeakyReLU()(feature_layer)

    output_layer = layers.Conv1DTranspose(self.instrument_count, 4,
                                          strides=2,
                                          padding='same',
                                          use_bias=False,
                                          activation='tanh')(feature_layer)

    return Model([latent_input, keys_input], output_layer)


  def make_discriminator_model(self):
    # We could also use the time signature as a label, but we're not yet.

    # We want start and end key to use the same embedding, so start key is
    # the first element and end key is the second element.
    keys_input = Input(shape=(2,), dtype='int32')
    keys_input_layer = layers.Embedding(24, KEY_EMBEDDING_SIZE)(keys_input)
    keys_input_layer = layers.Flatten()(keys_input_layer)
    keys_input_layer = layers.Dense(self.max_steps)(keys_input_layer)
    # Reshape to an additional channel.
    keys_input_layer = layers.Reshape((self.max_steps, 1))(keys_input_layer)

    # Define the phrase input.
    phrase_input = Input(shape=(self.max_steps, self.instrument_count))

    # Concatenate the key signature layer as a new instrument.
    merged_layer = layers.Concatenate()([keys_input_layer, phrase_input])

    # Now we use a standard 1D convolutional network, downsampling a few times.
    feature_layer = layers.Conv1D(256, 4, strides=2, padding='same')(merged_layer)
    feature_layer = layers.LeakyReLU()(feature_layer)
    feature_layer = layers.Dropout(0.3)(feature_layer)

    feature_layer = layers.Conv1D(128, 4, strides=2, padding='same')(feature_layer)
    feature_layer = layers.LeakyReLU()(feature_layer)
    feature_layer = layers.Dropout(0.3)(feature_layer)

    feature_layer = layers.Conv1D(64, 4, strides=2, padding='same')(feature_layer)
    feature_layer = layers.LeakyReLU()(feature_layer)
    feature_layer = layers.Dropout(0.3)(feature_layer)

    feature_layer = layers.Conv1D(64, 4, strides=2, padding='same')(feature_layer)
    feature_layer = layers.LeakyReLU()(feature_layer)
    feature_layer = layers.Dropout(0.3)(feature_layer)

    feature_layer = layers.Flatten()(feature_layer)

    output_layer = layers.Dense(1, activation='sigmoid')(feature_layer)

    return Model([phrase_input, keys_input], output_layer)


  def phrase_to_predictions(self, phrase):
    # Add 1 so that the "no pitch" is 0, then scale to the range -1..1.
    return (phrase.notes_array.transpose().astype('float32') - (MAX_RANGE / 2) + 1) / (MAX_RANGE / 2)
  

  def predictions_to_phrase(self, predictions):
    # Reverse the transformation done in phrases_to_dataset.
    notes_array = np.round(np.array(predictions).transpose() * (MAX_RANGE / 2) + (MAX_RANGE / 2) - 1).astype('int')
    return phrases_lib.Phrase(notes_array = notes_array)


  def generate_latent_input_and_keys(self, batch_size):
    noise = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, self.input_width))
    random_keys = np.array([[random.randrange(0, 24), random.randrange(0, 24)] for i in range(0, batch_size)], dtype='int')
    return noise, random_keys


  def train_step(self, phrase_batch):
    train_phrases, train_keys = phrase_batch
    this_batch_size = len(train_phrases)
    ones = np.ones((this_batch_size, 1))
    zeros = np.zeros((this_batch_size, 1))
    disc_loss_real, disc_acc_real = self.discriminator.train_on_batch([train_phrases, train_keys], ones)

    fake_latent_input, fake_keys = self.generate_latent_input_and_keys(this_batch_size)
    fake_phrases = self.generator.predict([fake_latent_input, fake_keys])
    disc_loss_fake, disc_acc_fake = self.discriminator.train_on_batch([fake_phrases, fake_keys], zeros)

    # Train the generator twice since we trained the discriminator twice.
    # Fair is fair! :)
    for _ in range(2):
      fake_latent_input, fake_keys = self.generate_latent_input_and_keys(this_batch_size)
      gen_loss = self.combined_model.train_on_batch([fake_latent_input, fake_keys], ones)
    
    return gen_loss, (disc_loss_real + disc_loss_fake) / 2, (disc_acc_real + disc_acc_fake) / 2


  def train(self,
            phrases,
            instruments_data,
            checkpoints_dir=TRAINING_CHECKPOINTS_DIR,
            samples_dir=SAMPLES_DIR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            samples_to_generate=EXAMPLES_TO_GENERATE,
            checkpoints_every=500,
            generate_every=500):
    
    os.makedirs(checkpoints_dir, exist_ok=True)

    phrase_arrays = tf.data.Dataset.from_tensor_slices([self.phrase_to_predictions(phrase) for phrase in phrases])
    key_arrays = tf.data.Dataset.from_tensor_slices([[phrase.start_key.key_index, phrase.end_key.key_index] for phrase in phrases])
    dataset = tf.data.Dataset.zip((phrase_arrays, key_arrays)).shuffle(len(phrases)).batch(batch_size)
    print("Number of batches is %d" % len(dataset))
    
    # We will reuse this seed and these keys to generate progressive examples
    # as the generator is trained.  We'll also output the keys that the samples
    # are supposed to start and end in.  (Whether they actually do is another
    # story...)
    seed, keys = self.generate_latent_input_and_keys(samples_to_generate)
    self.write_keys_to_file('%s/sample_keys.txt' % samples_dir, keys)
    self.generate_and_save_phrases(samples_dir=samples_dir,
                                   instruments_data=instruments_data,
                                   epoch=0,
                                   test_input=[seed, keys])

    for epoch in range(epochs):
      start = time.time()

      for phrase_batch in dataset:
        gen_loss, disc_loss, disc_acc = self.train_step(phrase_batch)

      
      # Save the model and generate phrases every so many epochs.
      if (epoch + 1) % checkpoints_every == 0:
        self.discriminator.save("%s/%s-discriminator-%d.h5" % (checkpoints_dir, "ckpt", (epoch + 1)))
        self.combined_model.save("%s/%s-combined-%d.h5" % (checkpoints_dir, "ckpt", (epoch + 1)))

      if (epoch + 1) % generate_every == 0:
        self.generate_and_save_phrases(samples_dir=samples_dir,
                                       instruments_data=instruments_data,
                                       epoch=epoch + 1,
                                       test_input=[seed, keys])

      print('Time for epoch %d is %0.3fs; last batch gen loss = %f, disc loss = %f, disc acc = %f' %
          (epoch + 1, time.time() - start, gen_loss, disc_loss, disc_acc))

    # Generate phrases after the final epoch -- the grand finale! :)
    self.generate_and_save_phrases(samples_dir=samples_dir,
                                   instruments_data=instruments_data,
                                   epoch=epochs,
                                   test_input=[seed, keys])


  def write_keys_to_file(self, filename, keys):
    with open(filename, 'w') as f:
      for sample_idx, key_indices in enumerate(keys):
        f.write("Phrase %d: %s -> %s\n" %
            (sample_idx, phrases_lib.KEY_NAMES[key_indices[0]], phrases_lib.KEY_NAMES[key_indices[1]]))


  def generate_and_save_phrases(self, samples_dir, instruments_data, epoch, test_input):
    os.makedirs(samples_dir, exist_ok=True)

    predictions = self.generator.predict(test_input)
    for (phrase_idx, phrase_array) in enumerate(predictions):
      phrase = self.predictions_to_phrase(phrase_array)
      phrase.write_to_midi_file(
          output_file='%s/epoch_%05d_phrase_%02d.midi' % (samples_dir, epoch, phrase_idx),
          instruments_data=instruments_data,
          qpm=80)
