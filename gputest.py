import midi2audio
import fluidsynth
from midi2audio import FluidSynth
fs = FluidSynth()
fs.play_midi('Allegro-Music-Transformer-Music-Composition_1')
fs.midi_to_audio('Allegro-Music-Transformer-Music-Composition_1', 'Allegro-Music-Transformer/audio1.flac')
