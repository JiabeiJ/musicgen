import flask
from flask import Flask, render_template,send_file,session,jsonify,send_from_directory
import sys
import eel
import fluidsynth
from midi2audio import FluidSynth
import subprocess


from musefunc_1 import instantiate, full_path_to_model_checkpoint, model_precision,load_seed_MIDI,preprocess_events_matrix,compose_melody_chords
from musefunc_1 import melody,compose
print('=' * 70)
print('Loading core Allegro Music Transformer modules...')

from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pickle
import torch
from flask import request
import torchsummary

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
from IPython.display import Audio, display
from huggingface_hub import hf_hub_download
full_path_to_model_checkpoint = "/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/Models/Small/Allegro_Music_Transformer_Small_Trained_Model_56000_steps_0.9399_loss_0.7374_acc.pth" #@param {type:"string"}


model_precision = "float16"
print(full_path_to_model_checkpoint)
import TMIDIX
# Create Flask app
app = Flask(__name__)
app.secret_key = 'jjb'
model,_=instantiate(full_path_to_model_checkpoint, model_precision)
# select_seed_MIDI = "Upload your own custom MIDI"
# Define a route
@app.route('/', methods=['GET', 'POST'])
@app.route('/compose', methods=['GET', 'POST'])
def compose_1():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            import torch
            number_of_prime_tokens = 300
            model, ctx = instantiate(full_path_to_model_checkpoint, model_precision)
            _,melody_chords_f = melody(file_path,number_of_prime_tokens)
            number_of_tokens_to_generate = 50  # @param {type:"slider", min:33, max:1023, step:3}
            number_of_batches_to_generate = 4  # @param {type:"slider", min:1, max:16, step:1}
            preview_length_in_tokens = 120  # @param {type:"slider", min:33, max:240, step:3}
            number_of_memory_tokens = 2048  # @param {type:"slider", min:402, max:2048, step:2}
            temperature = 1  # @param {type:"slider", min:0.1, max:1, step:0.1}
            # @markdown Other settings
            render_MIDI_to_audio = True  # @param {type:"boolean"}
            device = torch.device('mps')
            preview = melody_chords_f[-preview_length_in_tokens:]
            inp = [melody_chords_f[-number_of_memory_tokens:]] * number_of_batches_to_generate
            inp = torch.LongTensor(inp).to(device)
            print(inp)
            model.to(device)

            with ctx:
                out = model.module.generate(inp,
                                            number_of_tokens_to_generate,
                                            temperature=temperature,
                                            return_prime=False,
                                            verbose=True)

            out0 = out.tolist()

            # ======================================================================

            for i in range(number_of_batches_to_generate):
                out1 = out0[i]
                if len(out) != 0:
                    song = preview + out1
                    song_f = []
                    time = 0
                    dur = 0
                    vel = 0
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
                                                                              output_file_name='/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/Allegro-Music-Transformer-Music-Composition_' + str(
                                                                                  i),
                                                                              track_name='Project Los Angeles',
                                                                              list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56,
                                                                                                    65, 73, 0, 53, 19, 0, 0, 0,
                                                                                                    0]
                                                                              )
                    #fs = FluidSynth()
                    #fs.midi_to_audio('/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/Allegro-Music-Transformer-Seed-Composition.mid', '/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/output.wav')


    return render_template("musicgenerate.html")

LOCAL_FILE_DIRECTORY = '/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer'
@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(LOCAL_FILE_DIRECTORY, filename)

@app.route('/process_selection', methods=['GET', 'POST'])
def process_selection():
    select_seed_MIDI = "Upload your own custom MIDI"
    score,f = load_seed_MIDI(select_seed_MIDI)
    return score,f
@app.route('/seed', methods=['GET', 'POST'])
def seed_compo():
    number_of_prime_tokens = 300
    song_f,_ = melody(number_of_prime_tokens)
    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                              output_signature='Allegro Music Transformer',
                                                              output_file_name='/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/Allegro-Music-Transformer-Seed-Composition',
                                                              track_name='Project Los Angeles',
                                                              list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 65,
                                                                                    73,
                                                                                    0, 53, 19, 0, 0, 0, 0]
                                                              )
    # Now send the generated file to the user for download
    return send_file('/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/Allegro-Music-Transformer-Seed-Composition.mid', as_attachment=True)




ALLOWED_EXTENSIONS = {'mid'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = '/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/Uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import TMIDIX

@app.route('/generate_midi')
def generate_midi():
    # Code to generate MIDI file
    midi_file_path = '/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/templates/Allegro-Music-Transformer-Music-Composition_1.mid'
    return send_file('/Users/jiabei/Desktop/music_test/Allegro-Music-Transformer/templates/Allegro-Music-Transformer-Music-Composition_1.mid', mimetype='audio/midi')


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
