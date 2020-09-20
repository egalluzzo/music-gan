from phrase_gan_model import PhraseDenseGanModel, PhraseConv1DGanModel, PhraseConv2DGanModel
import os, sys, yaml
import phrases_lib


DEFAULT_METADATA_FILE = './chorale-phrases.yaml'
MAX_QUARTERS = 6 * 4  # 6 bars of 4/4, or 8 bars of 3/4
STEPS_PER_QUARTER = 4
MAX_STEPS = MAX_QUARTERS * STEPS_PER_QUARTER


if len(sys.argv) != 2:
  print("Expected a single command-line argument, the directory containing all the MIDI files")
  exit(1)

midi_dir = sys.argv[1]
if not os.path.isdir(midi_dir):
  print("Directory %s does not exist" % midi_dir)
  exit(1)

# FIXME: Shouldn't be loading this here and in phrase_lib.
data = {}
with open(DEFAULT_METADATA_FILE) as metadata_file:
  data = yaml.safe_load(metadata_file)
instruments_data = data['instruments']

# phrases = phrases_lib.read_all_files_in_directory(midi_dir=midi_dir,
#                                                   instruments_data=instruments_data,
#                                                   steps_per_quarter=STEPS_PER_QUARTER,
#                                                   max_quarters=MAX_QUARTERS)

phrases = phrases_lib.read_midi_files(midi_dir=midi_dir,
                                      metadata_filename=DEFAULT_METADATA_FILE,
                                      steps_per_quarter=STEPS_PER_QUARTER,
                                      max_quarters=MAX_QUARTERS)
print("Read %d phrases" % len(phrases))

# PhraseConv1DGanModel
model = PhraseConv1DGanModel(MAX_STEPS, instrument_count=4, learning_rate=1e-4)
test = model.predictions_to_phrase(model.phrase_to_predictions(phrases[0]))
test.write_to_midi_file('./test.midi', instruments_data=instruments_data, qpm=80)
# model.train(phrases=phrases,
#             instruments_data=instruments_data,
#             batch_size=BATCH_SIZE,
#             epochs=EPOCHS,
#             checkpoints_every=250,
#             generate_every=100)
model.train(phrases=phrases,
            instruments_data=instruments_data,
            batch_size=50,
            epochs=10000,
            checkpoints_every=250,
            generate_every=100)

# PhraseConv2DGanModel
# model = PhraseConv2DGanModel(MAX_STEPS, instrument_count=4, learning_rate=1e-3)
# test = model.predictions_to_phrase(model.phrase_to_predictions(phrases[0]))
# test.write_to_midi_file('./test.midi', instruments_data=instruments_data, qpm=80)
# model.train(phrases=phrases,
#             instruments_data=instruments_data,
#             batch_size=BATCH_SIZE,
#             epochs=5000,
#             checkpoints_every=250,
#             generate_every=250)
