#@title Import modules


print('=' * 70)
print('Loading core Allegro Music Transformer modules...')

import os
import pickle
import random
import secrets
import statistics
import tqdm
print(os.getcwd())
print('=' * 70)
print('Loading main Allegro Music Transformer modules...')
import torch

import TMIDIX
from x_transformer import *

print('=' * 70)
print('Loading aux Allegro Music Transformer modules...')

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

from midi2audio import FluidSynth
from IPython.display import Audio, display

from huggingface_hub import hf_hub_download


print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)

#@title Load Allegro Music Transformer Small Model (BEST)

#@markdown Fast model, 32 layers, 225k MIDIs training corpus

full_path_to_model_checkpoint = "/Desktop/music_test/Allegro-Music-Transformer/Models/Small/Allegro_Music_Transformer_Small_Trained_Model_56000_steps_0.9399_loss_0.7374_acc.pth" #@param {type:"string"}

#@markdown Model precision option

model_precision = "float16" # @param ["bfloat16", "float16", "float32"]

#@markdown bfloat16 == Third precision/triple speed (if supported, otherwise the model will default to float16)

#@markdown float16 == Half precision/double speed

#@markdown float32 == Full precision/normal speed

plot_tokens_embeddings = True # @param {type:"boolean"}

print('=' * 70)
print('Loading Allegro Music Transformer Small Pre-Trained Model...')
print('Please wait...')
print('=' * 70)

if os.path.isfile(full_path_to_model_checkpoint):
  print('Model already exists...')

else:
  hf_hub_download(repo_id='asigalov61/Allegro-Music-Transformer',
                  filename='Allegro_Music_Transformer_Small_Trained_Model_56000_steps_0.9399_loss_0.7374_acc.pth',
                  local_dir='/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/Models/Small/',
                  local_dir_use_symlinks=False)
print('=' * 70)
print('Instantiating model...')

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cpu'

if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
  dtype = 'bfloat16'
else:
  dtype = 'float16'

if model_precision == 'float16':
  dtype = 'float16'

if model_precision == 'float32':
  dtype = 'float32'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

SEQ_LEN = 2048

# instantiate the model

model = TransformerWrapper(
    num_tokens = 3088,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 1024, depth = 32, heads = 8, attn_flash=True)
)

model = AutoregressiveWrapper(model)

model = torch.nn.DataParallel(model)


print('=' * 70)

print('Loading model checkpoint...')

model.load_state_dict(torch.load(full_path_to_model_checkpoint))
print('=' * 70)

model.eval()

print('Done!')
print('=' * 70)

print('Model will use', dtype, 'precision...')
print('=' * 70)

# Model stats
print('Model summary...')
summary(model)

# Plot Token Embeddings

if plot_tokens_embeddings:

  tok_emb = model.module.net.token_emb.emb.weight.detach().cpu().tolist()

  cos_sim = metrics.pairwise_distances(
    tok_emb, metric='cosine'
  )
  plt.figure(figsize=(7, 7))
  plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")
  im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
  plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
  plt.xlabel("Position")
  plt.ylabel("Position")
  plt.tight_layout()
  plt.plot()
  plt.savefig("/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/Models/Small/Allegro-Music-Transformer-Small-Tokens-Embeddings-Plot.png", bbox_inches="tight")

  # @title Load Seed MIDI

  # @markdown Press play button to to upload your own seed MIDI or to load one of the provided sample seed MIDIs from the dropdown list below

  select_seed_MIDI = "Upload your own custom MIDI"  # @param ["Upload your own custom MIDI", "Allegro-Music-Transformer-Piano-Seed-1", "Allegro-Music-Transformer-Piano-Seed-2", "Allegro-Music-Transformer-Piano-Seed-3", "Allegro-Music-Transformer-Piano-Seed-4", "Allegro-Music-Transformer-Piano-Seed-5", "Allegro-Music-Transformer-MI-Seed-1", "Allegro-Music-Transformer-MI-Seed-2", "Allegro-Music-Transformer-MI-Seed-3", "Allegro-Music-Transformer-MI-Seed-4", "Allegro-Music-Transformer-MI-Seed-5"]
  number_of_prime_tokens = 300  # @param {type:"slider", min:128, max:5000, step:16}
  render_MIDI_to_audio = False  # @param {type:"boolean"}

  print('=' * 70)
  print('Allegro Music Transformer Seed MIDI Loader')
  print('=' * 70)

  f = ''

  if select_seed_MIDI != "Upload your own custom MIDI":
      print('Loading seed MIDI...')
      f = '/content/Allegro-Music-Transformer/Seeds/' + select_seed_MIDI + '.mid'
      score = TMIDIX.midi2single_track_ms_score(open(f, 'rb').read(), recalculate_channels=False)

  else:
      print('Upload your own custom MIDI...')
      print('=' * 70)
      uploaded_MIDI = files.upload()
      if list(uploaded_MIDI.keys()):
          score = TMIDIX.midi2single_track_ms_score(list(uploaded_MIDI.values())[0], recalculate_channels=False)
          f = list(uploaded_MIDI.keys())[0]

  if f != '':

      print('=' * 70)
      print('File:', f)
      print('=' * 70)

      # =======================================================
      # START PROCESSING

      melody_chords_f = []

      # INSTRUMENTS CONVERSION CYCLE
      events_matrix = []
      itrack = 1
      patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

      patch_map = [
          [0, 1, 2, 3, 4, 5, 6, 7],  # Piano
          [24, 25, 26, 27, 28, 29, 30],  # Guitar
          [32, 33, 34, 35, 36, 37, 38, 39],  # Bass
          [40, 41],  # Violin
          [42, 43],  # Cello
          [46],  # Harp
          [56, 57, 58, 59, 60],  # Trumpet
          [64, 65, 66, 67, 68, 69, 70, 71],  # Sax
          [72, 73, 74, 75, 76, 77, 78],  # Flute
          [-1],  # Drums
          [52, 53],  # Choir
          [16, 17, 18, 19, 20]  # Organ
      ]

      while itrack < len(score):
          for event in score[itrack]:
              if event[0] == 'note' or event[0] == 'patch_change':
                  events_matrix.append(event)
          itrack += 1

      events_matrix.sort(key=lambda x: x[1])

      events_matrix1 = []

      for event in events_matrix:
          if event[0] == 'patch_change':
              patches[event[2]] = event[3]

          if event[0] == 'note':
              event.extend([patches[event[3]]])
              once = False

              for p in patch_map:
                  if event[6] in p and event[3] != 9:  # Except the drums
                      event[3] = patch_map.index(p)
                      once = True

              if not once and event[3] != 9:  # Except the drums
                  event[3] = 15  # All other instruments/patches channel
                  event[5] = max(80, event[5])

              if event[3] < 12:  # We won't write chans 12-16 for now...
                  events_matrix1.append(event)

      if len(events_matrix1) > 0:

          # =======================================================
          # PRE-PROCESSING

          # checking number of instruments in a composition
          instruments_list_without_drums = list(set([y[3] for y in events_matrix1 if y[3] != 9]))

          if len(events_matrix1) > 0 and len(instruments_list_without_drums) > 0:

              # recalculating timings
              for e in events_matrix1:
                  e[1] = int(e[1] / 8)  # Max 2 seconds for start-times
                  e[2] = int(e[2] / 32)  # Max 4 seconds for durations

              # Sorting by pitch, then by start-time
              events_matrix1.sort(key=lambda x: x[4], reverse=True)
              events_matrix1.sort(key=lambda x: x[1])

              # =======================================================
              # FINAL PRE-PROCESSING

              melody_chords = []

              pe = events_matrix1[0]

              for e in events_matrix1:
                  # Cliping all values...
                  time = max(0, min(255, e[1] - pe[1]))
                  dur = max(1, min(127, e[2]))
                  cha = max(0, min(11, e[3]))
                  ptc = max(1, min(127, e[4]))

                  # Calculating octo-velocity
                  vel = max(8, min(127, e[5]))
                  velocity = round(vel / 15) - 1

                  # Writing final note
                  melody_chords.append([time, dur, cha, ptc, velocity])

                  pe = e

              times = [y[0] for y in melody_chords[12:]]
              avg_time = sum(times) / len(times)

              times_list = list(set(times))

              mode_dur = statistics.mode([y[1] for y in melody_chords if y[2] != 9])

              instruments_list = list(set([y[2] for y in melody_chords]))
              num_instr = len(instruments_list)

              # =======================================================

              # TOTAL DICTIONARY SIZE 3087+1=3088

              # =======================================================
              # MAIN PROCESSING CYCLE
              # =======================================================

              chords_count = 0

              melody_chords_f.extend([2816])  # Zero chords count

              if melody_chords[0][0] == 0:
                  melody_chords_f.extend([0])  # Zero time, if present

              notes_counter = 0
              chords_counter = 0

              for m in melody_chords:

                  time = m[0]

                  # Chords counter token
                  if chords_count % 50 == 0 and chords_count != 0 and time != 0:
                      melody_chords_f.extend([2816 + min(255, ((chords_count // 50)))])

                  if time != 0:
                      chords_count += 1

                  # WRITING EACH NOTE HERE
                  dur_vel = (m[1] * 8) + m[4]
                  cha_ptc = (m[2] * 128) + m[3]

                  if time != 0:
                      melody_chords_f.extend([time, dur_vel + 256, cha_ptc + 1280])
                      chords_counter += 1

                  else:
                      melody_chords_f.extend([dur_vel + 256, cha_ptc + 1280])

                  notes_counter += 1

      melody_chords_f = melody_chords_f[:number_of_prime_tokens]

      # =======================================================

      song = melody_chords_f

      song_f = []

      time = 0
      dur = 0
      vel = 90
      pitch = 0
      channel = 0

      for ss in song:

          if ss > 0 and ss < 256:
              time += ss * 8

          if ss >= 256 and ss < 1280:
              dur = ((ss - 256) // 8) * 32
              vel = (((ss - 256) % 8) + 1) * 15

          if ss >= 1280 and ss < 2816:
              channel = (ss - 1280) // 128
              pitch = (ss - 1280) % 128

              song_f.append(['note', time, dur, channel, pitch, vel])

      detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature='Allegro Music Transformer',
                                                                output_file_name='/content/Allegro-Music-Transformer-Seed-Composition',
                                                                track_name='Project Los Angeles',
                                                                list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 65, 73,
                                                                                      0, 53, 19, 0, 0, 0, 0]
                                                                )

      # =======================================================

      print('=' * 70)
      print('Composition stats:')
      print('Composition has', notes_counter, 'notes')
      print('Composition has', chords_counter, 'chords')
      print('Composition has', len(melody_chords_f), 'tokens')
      print('=' * 70)

      fname = '/content/Allegro-Music-Transformer-Seed-Composition'

      x = []
      y = []
      c = []

      colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

      block_lines = [(song_f[-1][1] / 1000)]
      block_tokens = [min(len(melody_chords_f), number_of_prime_tokens)]

      for s in song_f:
          x.append(s[1] / 1000)
          y.append(s[4])
          c.append(colors[s[3]])

      if render_MIDI_to_audio:
          FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'),
                                                                                  str(fname + '.wav'))
          display(Audio(str(fname + '.wav'), rate=16000))

      plt.figure(figsize=(14, 5))
      ax = plt.axes(title=fname)
      ax.set_facecolor('black')

      plt.scatter(x, y, c=c)
      plt.xlabel("Time")
      plt.ylabel("Pitch")
      plt.show()

  else:
      print('=' * 70)