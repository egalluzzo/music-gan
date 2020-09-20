# Music GAN

This project shows how to write a simple GAN using TensorFlow to generate music.  It is a companion repository to my _Working in Bars_ presentation.

## Prerequisites

You will need:

* [Python](https://www.python.org/) 3.4 or later
* [TensorFlow](https://www.tensorflow.org/) 2.3.0 or later
* [NumPy](https://numpy.org/), which TensorFlow also requires
* [Magenta](https://magenta.tensorflow.org/)
* [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
* The [MIDI files](https://github.com/jamesrobertlloyd/infinite-bach/tree/master/data/chorales/midi) from [Infinite Bach](https://github.com/jamesrobertlloyd/infinite-bach/)

All except the last can be installed in a Python environment via:

    pip install tensorflow magenta pyyaml

However, you may want to look into different options for installing TensorFlow, for example, to enable GPU support.

## Running

    python main.py "/path/to/infinite-bach/data/chorales/midi"

This should generate some samples every so many epochs in the `samples` directory.  Training checkpoints are saved in the `training_checkpoints` directory.
