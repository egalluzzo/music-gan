import os, re, sys, traceback, yaml
import numpy as np
from fractions import Fraction
from note_seq import constants, midi_io, sequences_lib
from note_seq.protobuf import music_pb2


NO_PITCH = -1

DEFAULT_MAX_QUARTERS = 24 # 6 bars of 4/4
DEFAULT_STEPS_PER_QUARTER = 4 # 16th notes
DEFAULT_QUARTERS_PER_MINUTE = 120
DEFAULT_TIME_SIGNATURE_FRACTION = Fraction('4/4')

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
MODE_NAMES = ['M', 'm']
KEY_NAMES = [
  "CM", "DbM", "DM", "EbM", "EM", "FM", "GbM", "GM", "AbM", "AM", "BbM", "BM", # Major keys (mode = 0)
  "Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm"  # Minor keys (mode = 1)
]

MODE_INDICES = {
  "M": 0,
  "Maj": 0,
  "maj": 0,
  "m": 1,
  "min": 1,
  "Min": 1
}

NOTE_INDICES = {
  "Cb": 11,
  "C": 0,
  "C#": 1,
  "Db": 1,
  "D": 2,
  "D#": 3,
  "Eb": 3,
  "E": 4,
  "E#": 5,
  "Fb": 4,
  "F": 5,
  "F#": 6,
  "Gb": 6,
  "G": 7,
  "G#": 8,
  "Ab": 8,
  "A": 9,
  "A#": 10,
  "Bb": 10,
  "B": 11,
  "B#": 0
}


class NoTimeSignaturesError(Exception):
  """Exception indicating that a NoteSequence has no time signatures.
  """
  pass

class InvalidMetadataError(Exception):
  """Exception indicating that the metadata for a particular MIDI file contains
  no 'phrases' element, which is required to parse phrases.
  """
  pass

class InvalidMixedNumberError(Exception):
  """Exception indicating that the given string was not a mixed number.  Mixed
  numbers are of one of the following forms:

  * Mixed number (12 7/8)
  * Fraction (7/8)
  * Integer (12)
  """
  pass

class NoEndOfPhraseError(Exception):
  """Exception indicating that the end of a phrase in the file metadata was not
  given.  Each phrase must have an 'end' element with a bar offset of the end
  of the phrase, as a mixed number.
  """
  pass

class PhraseBoundaryNotOnQuantizedStepError(Exception):
  """Exception indicating that the start or end of a phrase was not on a
  quantized step boundary.  For example, if the start of the phrase was given
  as "1 2/3" in 3/4 time, that would raise this exception.
  """
  pass

class InvalidInstrumentCountError(Exception):
  """Exception indicating that a NoteSequence has an invalid number of
  instruments.
  """
  pass

class IncompatibleRangeError(Exception):
  """Exception indicating that two ranges cannot be operated on together, for
  example, because they have different step values.
  """
  pass

class InconsistentNumberOfInstrumentsError(Exception):
  """Exception indicating that the number of instruments in a phrase are
  different from the number of instruments in the supplied metadata.
  """
  pass

class InvalidKeySignatureNameError(Exception):
  """Exception indicating that a key signature name is not valid.  Valid key
  signatures are of the form "F#m" (F# minor) or "EM" (E major).  If no "M" or
  "m" is given, major is assumed.  Modes are not supported at this time.
  """
  pass


class Key:
  def __init__(self, key_name, transposition = 0):
    key_match = re.match(r"^([A-Ga-g][b#]?)(M|m|maj|min)?$", key_name.strip())
    if not key_match:
      raise InvalidKeySignatureNameError(
          "Could not parse key signature %s, must be of the form C or F#m or EbM" % key_name)

    tonic = key_match.group(1)
    mode_name = key_match.group(2)

    if not mode_name:
      mode = 0
    else:
      if mode_name not in MODE_INDICES:
        raise InvalidKeySignatureNameError(
            "Unknown mode %s, must be M, m, maj or min" % mode_name)
      mode = MODE_INDICES[mode_name]
    
    if tonic not in NOTE_INDICES:
      raise InvalidKeySignatureNameError(
          "Unknown tonic %s, must be a standard note name like A or Bb or C#" % tonic)
    
    self.tonic = (NOTE_INDICES[tonic] + transposition) % 12
    self.mode = mode
    self.key_index = self.mode * 12 + self.tonic
    self.key_name = KEY_NAMES[self.key_index]


class Phrase:
  """A phrase, consisting of a matrix of pitches for each voice and an index.

  Arguments:
    notes_array: An np.array where rows represent instruments and columns represent time
    instruments_data: Metadata about the instruments, including name, min and max MIDI pitches
    phrase_idx: The index of the phrase within the track (starting at 0)
    transposition: The transposition from the original notes
  """
  def __init__(self,
               notes_array: np.ndarray,
               phrase_idx: int = 0,
               transposition: int = 0,
               start_key: Key = Key("C"),
               end_key: Key = Key("C")):
    self.notes_array = notes_array
    self.phrase_idx = phrase_idx
    self.transposition = transposition
    self.start_key = start_key
    self.end_key = end_key
  
  
  def to_sequence(self,
                  instruments_data: dict,
                  qpm: int = DEFAULT_QUARTERS_PER_MINUTE,
                  steps_per_quarter: int = DEFAULT_STEPS_PER_QUARTER,
                  time_signature_fraction: Fraction = DEFAULT_TIME_SIGNATURE_FRACTION):
    sequence = music_pb2.NoteSequence()
    sequence.ticks_per_quarter = constants.STANDARD_PPQ

    sequence.tempos.add().qpm = qpm

    # time_signature = sequence.time_signatures.add()
    # time_signature.numerator = time_signature_fraction.numerator
    # time_signature.denominator = time_signature_fraction.denominator

    if len(self.notes_array) != len(instruments_data):
      raise InconsistentNumberOfInstrumentsError(
          "%d instruments in notes array, %d in instrument metadata" %
              (len(self.notes_array), len(instruments_data)))

    for instrument_idx, pitches in enumerate(self.notes_array):
      # Tack an extra NO_PITCH onto the end so that we write out the last note
      # properly.
      pitches = list(pitches)
      pitches.append(NO_PITCH)
      instrument_data = instruments_data[instrument_idx]
      previous_pitch = NO_PITCH
      pitch_start_step = 0
      for pitch_step, pitch in enumerate(pitches):
        if previous_pitch != pitch:
          if previous_pitch != NO_PITCH:
            # Write out the previous pitch, converting steps to seconds.
            note = sequence.notes.add()
            note.start_time = pitch_start_step / steps_per_quarter * 60 / qpm
            note.end_time = pitch_step / steps_per_quarter * 60 / qpm
            note.pitch = previous_pitch + instrument_data['minPitch'] - self.transposition
            note.velocity = 95 # 0..127, we'll just pick something
            note.instrument = instrument_idx
            note.program = instrument_data['program']
          
          previous_pitch = pitch
          pitch_start_step = pitch_step
    
    return sequence
  
  def write_to_midi_file(self,
                         output_file,
                         instruments_data: dict,
                         qpm: int = DEFAULT_QUARTERS_PER_MINUTE,
                         steps_per_quarter: int = DEFAULT_STEPS_PER_QUARTER,
                         time_signature_fraction: Fraction = DEFAULT_TIME_SIGNATURE_FRACTION):
    sequence = self.to_sequence(instruments_data=instruments_data,
                                qpm=qpm,
                                steps_per_quarter=steps_per_quarter,
                                time_signature_fraction=time_signature_fraction)
    with open(output_file, 'wb') as f:
      midi_io.sequence_proto_to_pretty_midi(sequence).write(f)


def read_midi_files(midi_dir: str,
                    metadata_filename: str,
                    steps_per_quarter: int = DEFAULT_STEPS_PER_QUARTER,
                    max_quarters: int = DEFAULT_MAX_QUARTERS):
  """Read a directory of MIDI files and translate them into Phrase objects.

  The given metadata file must have entries for each expected instrument in
  the files as a whole, as well as data for each file that should be
  translated.  The file format is as follows:

      instruments:    # Section containing the exact list of required
                      # instruments in each MIDI file
      - name: Soprano
        min_pitch: 60 # C4
        max_pitch: 81 # A5
      - name: Alto
        min_pitch: 55 # G3
        max_pitch: 77 # F5
      ...

      files:
        foo.mid:      # Indicates that foo.mid (relative to MIDI directory)
                      # should be included in the dataset
          key: Ebm    # Overall key of the MIDI file (E flat minor)
          time: 3/4   # Time signature
          pickup: 1/4 # Length of pickup in first measure (defaults to 0)
          phrases:
          - start: 1 3/4   # Bar offset of the start of this phrase
                           # The beginning of the first bar is "1"
                           # Optional on anything but the first phrase
                           # Defaults to the end of the previous phrase
            end: 5 3/4     # Bar offset of the end of this phrase (required)
            startKey: Ebm  # Key at the start of the phrase (E flat minor)
            endKey: F#M    # Key at the end of the phrase (F# major)
          - end: 10        # Next phrase starts at 5 3/4 and ends at 10
            endKey: BM     # Phrase is assumed to start in F#M and ends in BM
          ...
        
        bar.mid:
          ...
  
  Args:
    midi_dir: Directory containing MIDI files
    metadata_filename: Name of file containing metadata for each MIDI file
    steps_per_quarter: Quantized steps per quarter note
    max_quarters: The maximum length of a phrase, in quarter notes
  """
  with open(metadata_filename) as metadata_file:
    data = yaml.safe_load(metadata_file)

    if 'files' not in data:
      raise InvalidMetadataError("No 'files' element in chorale metadata")
    if not isinstance(data['files'], dict):
      raise InvalidMetadataError(
          "'files' element must be a dictionary of file data")
    files_data = data['files']
    
    if 'instruments' not in data:
      raise InvalidMetadataError("No 'instruments' element in chorale metadata")
    if not isinstance(data['instruments'], list):
      raise InvalidMetadataError(
          "'instruments' element must be a list of data for each instrument")
    instruments_data = data['instruments']

    phrases = []
    for midi_file in os.listdir(midi_dir):
      if midi_file in files_data:
        print(midi_file)
        try:
          phrases.extend(
            read_midi_file(filename=os.path.join(midi_dir, midi_file),
                           file_data=files_data[midi_file],
                           instruments_data=instruments_data,
                           steps_per_quarter=steps_per_quarter,
                           max_quarters=max_quarters))
        except Exception as e:
          print('Could not read file %s: %s' % (midi_file, e))
          traceback.print_exc(file=sys.stdout)
      else:
        # print('No data for %s, skipping' % midi_file)
        pass
  
  return phrases


def read_midi_file(filename: str,
                   file_data: dict,
                   instruments_data: list,
                   steps_per_quarter: int = DEFAULT_STEPS_PER_QUARTER,
                   max_quarters: int = DEFAULT_MAX_QUARTERS):
  if 'phrases' not in file_data:
    raise InvalidMetadataError(
        "No 'phrases' element in data for file %s" % filename)

  if 'key' not in file_data:
    raise InvalidMetadataError(
        "No 'key' element in data for file %s" % filename)

  sequence = midi_io.midi_file_to_sequence_proto(filename)
  if sequence.time_signatures:
    time_signature = Fraction(sequence.time_signatures[0].numerator,
                              sequence.time_signatures[0].denominator)
  else:
    raise NoTimeSignaturesError("No time signatures in file %s" % filename)

  instrument_count = len(sequence.instrument_infos)
  if instrument_count != len(instruments_data):
    raise InvalidInstrumentCountError(
        "There were %d instruments (must be %d)" %
            (instrument_count, len(instruments_data)))

  quantized_sequence = sequences_lib.quantize_note_sequence(
      sequence, steps_per_quarter=steps_per_quarter)
  
  phrase_arrays = _read_phrases_from_quantized_sequence(
      quantized_sequence=quantized_sequence,
      file_data=file_data,
      instruments_data=instruments_data,
      time_signature=time_signature,
      steps_per_quarter=steps_per_quarter,
      max_quarters=max_quarters)
  
  return phrase_arrays

def read_all_files_in_directory(midi_dir: str,
                                instruments_data: list,
                                steps_per_quarter: int = DEFAULT_STEPS_PER_QUARTER,
                                max_quarters: int = DEFAULT_MAX_QUARTERS):
  phrases = []
  for midi_file in os.listdir(midi_dir):
    try:
      sequence = midi_io.midi_file_to_sequence_proto(os.path.join(midi_dir, midi_file))

      if len(sequence.instrument_infos) != len(instruments_data):
        print("File %s has %d instruments (wanted %d), skipping" %
            (midi_file, len(sequence.instrument_infos), len(instruments_data)))
        continue

      if sequence.time_signatures:
        time_signature = Fraction(sequence.time_signatures[0].numerator,
                                  sequence.time_signatures[0].denominator)
      else:
        print("File %s has no time signatures, skipping" % midi_file)
        continue

      quantized_sequence = sequences_lib.quantize_note_sequence(
          sequence, steps_per_quarter=steps_per_quarter)
      print("Processing file %s..." % midi_file)
      phrases.extend(
        _read_all_phrases_from_quantized_sequence(quantized_sequence=quantized_sequence,
                                                  instruments_data=instruments_data,
                                                  time_signature=time_signature,
                                                  steps_per_quarter=steps_per_quarter,
                                                  max_quarters=max_quarters))
    except Exception as e:
      print('Could not read file %s: %s' % (midi_file, e))
  
  return phrases


def _read_phrases_from_quantized_sequence(quantized_sequence,
                                          file_data,
                                          instruments_data,
                                          time_signature,
                                          steps_per_quarter,
                                          max_quarters):
  phrases = []
  max_steps = steps_per_quarter * max_quarters
  steps_in_bar = int(time_signature * 4 * steps_per_quarter)
  previous_end_step = 0
  previous_end_key = Key(file_data['key'])
  for phrase_num, phrase in enumerate(file_data['phrases'], start=1):
    if 'start' in phrase:
      start_step = _bar_number_to_step_number(phrase['start'],
                                              time_signature,
                                              steps_per_quarter)
    else:
      # If there is no explicit start of the phrase, start at the end of the
      # previous phrase.
      start_step = previous_end_step
    
    if 'end' not in phrase:
      raise NoEndOfPhraseError(
          "No 'end' for phrase #%d" % phrase_num)
    else:
      end_step = _bar_number_to_step_number(phrase['end'],
                                            time_signature,
                                            steps_per_quarter)
    
    previous_end_step = end_step
    
    if end_step <= start_step:
      print("Phrase #%d ends before it starts, skipping" % phrase_num)
      continue
    
    if 'startKey' in phrase:
      start_key = Key(phrase['startKey'])
    else:
      # Assume it hasn't changed keys.
      start_key = previous_end_key
    
    if 'endKey' in phrase:
      end_key = Key(phrase['endKey'])
    else:
      # Assume it hasn't changed keys.
      end_key = start_key
    
    previous_end_key = end_key
    
    #print("Phrase boundary: steps %d-%d" % (start_step, end_step))
    step_offset = start_step // steps_in_bar * steps_in_bar
    if end_step - step_offset > max_steps:
      print("Phrase #%d ends at step %d, which greater than %d steps, skipping" %
          (phrase_num, end_step - step_offset, max_steps))
      continue
    phrases.extend(_make_all_valid_transposed_phrases(quantized_sequence=quantized_sequence,
                                                      instruments_data=instruments_data,
                                                      max_steps=max_steps,
                                                      start_step=start_step,
                                                      end_step=end_step,
                                                      phrase_num=phrase_num,
                                                      step_offset=step_offset,
                                                      start_key=start_key,
                                                      end_key=end_key))
  
  return phrases


def _read_all_phrases_from_quantized_sequence(quantized_sequence,
                                              instruments_data,
                                              time_signature,
                                              steps_per_quarter,
                                              max_quarters):
  phrases = []
  max_steps = steps_per_quarter * max_quarters
  steps_per_bar = time_signature * 4 * steps_per_quarter
  if steps_per_bar.denominator != 1:
    print("Time signature %s is not an integral number of steps, skipping" % time_signature)
    return []
  
  steps_per_bar = int(steps_per_bar)
  total_bars = (quantized_sequence.total_quantized_steps + steps_per_bar - 1) // steps_per_bar
  bars_per_phrase = max_steps // steps_per_bar

  for start_bar in range(0, max(1, total_bars - bars_per_phrase)):
    start_step = start_bar * steps_per_bar
    end_step = (start_bar + bars_per_phrase) * steps_per_bar
    phrases.extend(_make_all_valid_transposed_phrases(quantized_sequence=quantized_sequence,
                                                      instruments_data=instruments_data,
                                                      max_steps=max_steps,
                                                      start_step=start_step,
                                                      end_step=end_step,
                                                      phrase_num=start_bar + 1,
                                                      step_offset=start_step))
  
  return phrases


def _make_all_valid_transposed_phrases(quantized_sequence,
                                       instruments_data,
                                       max_steps,
                                       start_step,
                                       end_step,
                                       phrase_num,
                                       step_offset,
                                       start_key = Key('C'),
                                       end_key = Key('C')):

  phrase_arrays = []

  possible_transposition_range = _find_possible_transpositions(
      quantized_sequence=quantized_sequence,
      instruments_data=instruments_data,
      phrase_num=phrase_num,
      start_step=start_step,
      end_step=end_step)

  # print("Possible transposition range for phrase #%d is %s" %
  #     (phrase_num, possible_transposition_range))
  if len(possible_transposition_range) == 0:
    print("Phrase #%d has no possible transpositions that will stay in range of all instruments, skipping" %
        phrase_num)
    return []

  if 0 not in possible_transposition_range:
    print("Warning: Original notes are not in range of their instruments, using transposition range: %s" %
        possible_transposition_range)

  for transposition in possible_transposition_range:
    phrase_array = _build_phrase_array(quantized_sequence=quantized_sequence,
                                       instruments_data=instruments_data,
                                       max_steps=max_steps,
                                       step_offset=step_offset,
                                       start_step=start_step,
                                       end_step=end_step,
                                       transposition=transposition)
    phrase_arrays.append(Phrase(notes_array=phrase_array,
                                phrase_idx=phrase_num - 1,
                                transposition=transposition,
                                start_key=Key(start_key.key_name, transposition=transposition),
                                end_key=Key(end_key.key_name, transposition=transposition)))
  
  return phrase_arrays


def _build_phrase_array(quantized_sequence,
                        instruments_data,
                        max_steps,
                        step_offset,
                        start_step,
                        end_step,
                        transposition):
  
  phrase_array = np.full((len(instruments_data), max_steps),
                          NO_PITCH,
                          dtype=np.int16)
  for instrument_idx, instrument_data in enumerate(instruments_data):
    notes = sorted([n for n in quantized_sequence.notes
                  if n.instrument == instrument_idx and
                  (start_step <= n.quantized_start_step < end_step or
                      start_step < n.quantized_end_step <= end_step)],
                  key=lambda note: note.quantized_start_step)
    min_pitch = instrument_data['minPitch']
    max_pitch = instrument_data['maxPitch']
    for note in notes:
      if note.velocity == 0:
        continue

      if note.pitch + transposition < min_pitch or note.pitch + transposition > max_pitch:
        print("Note has pitch %d which is out of the instrument's range (%d-%d), skipping" %
            (note.pitch, min_pitch, max_pitch))

      # Truncate the note to the (start_step, end_step) boundaries.
      note_start_step = max(note.quantized_start_step, start_step) - step_offset
      note_end_step = min(note.quantized_end_step, end_step) - step_offset
      phrase_array[instrument_idx][note_start_step:note_end_step] = note.pitch - min_pitch + transposition
  
  return phrase_array


def _find_possible_transpositions(quantized_sequence,
                                  instruments_data,
                                  phrase_num,
                                  start_step,
                                  end_step):
    possible_transposition_range = range(-127, 127)
    for instrument_idx, instrument_data in enumerate(instruments_data):
      notes = [n for n in quantized_sequence.notes
               if n.instrument == instrument_idx and
               (start_step <= n.quantized_start_step < end_step or
                   start_step < n.quantized_end_step <= end_step)]
      min_pitch = instrument_data['minPitch']
      max_pitch = instrument_data['maxPitch']
      notes_min_pitch = min([n.pitch for n in notes if n.velocity != 0])
      notes_max_pitch = max([n.pitch for n in notes if n.velocity != 0])
      new_transposition_range = range(min_pitch - notes_min_pitch,
                                      max_pitch - notes_max_pitch + 1)
      # print("Phrase #%d, instrument %d: Note ranges %d-%d, instrument range (%d-%d), possible transposition is %s" %
      #     (phrase_num, instrument_idx, notes_min_pitch, notes_max_pitch, min_pitch, max_pitch, new_transposition_range))
      possible_transposition_range = _range_intersection(possible_transposition_range,
                                                         new_transposition_range)
    
    return possible_transposition_range


def _range_intersection(range1, range2):
  if range1.step != range2.step:
    raise IncompatibleRangeError("Ranges have different step values (%d and %d)" %
        (range1.step, range2.step))
  return range(max(range1.start, range2.start), min(range1.stop, range2.stop), range1.step)


def _bar_number_to_step_number(bar, time_signature, steps_per_quarter):
  bar_mixed_number = _parse_mixed_number(bar)
  step_fraction = (bar_mixed_number - 1) * 4 * steps_per_quarter * time_signature
  if step_fraction.numerator == 0:
    return 0
  if step_fraction.denominator != 1:
    raise PhraseBoundaryNotOnQuantizedStepError(
        "Phrase boundary (bar %s, step %s, time_signature %s) not on quantized step (%s steps per quarter)" %
            (bar, step_fraction, time_signature, steps_per_quarter))
  return step_fraction.numerator


def _parse_mixed_number(mixed_number):
  if isinstance(mixed_number, Fraction):
    return mixed_number
  
  if isinstance(mixed_number, int):
    return Fraction(mixed_number)
  
  mixed_number_match = re.match(r"^\s*(\d+)\s+(\d+\s*/\s*\d+)\s*$", mixed_number)
  if mixed_number_match:
    whole_number = mixed_number_match.group(1)
    fraction = mixed_number_match.group(2)
    return Fraction(fraction) + int(whole_number)
  else:
    fraction_match = re.match(r"^\s*(\d+/\d+)\s*$", mixed_number)
    if fraction_match:
      return Fraction(fraction_match.group(1))
    else:
      int_match = re.match(r"^\s*(\d+)\s*", mixed_number)
      if int_match:
        return Fraction(int(int_match.group(1)))
      else:
        raise InvalidMixedNumberError("Invalid mixed number: %s" % mixed_number)
