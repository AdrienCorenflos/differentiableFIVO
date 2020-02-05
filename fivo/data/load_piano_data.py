import pickle
import numpy as np
import os,sys
import tensorflow as tf
import json
import deepdish as dd


piano_directory = '/data/hylia/thornton/diff_particle_filter/data'
file_names = ['musedata.pkl', 'jsb.pkl', 'nottingham.pkl', 'piano-midi.de.pkl']
for file_name in file_names:
    name = os.path.splitext(file_name)[0]
    h5_name = "{0}.h5".format(name)

    file_path = os.path.join(piano_directory, file_name)
    hd5_file_path = os.path.join(piano_directory, h5_name)

    pickle_file_exists = file_name in os.listdir(piano_directory)
    h5_file_exists = h5_name in os.listdir(piano_directory)

    if (pickle_file_exists and not h5_file_exists):
        print("Load file: {0}".format(file_path))

        with tf.gfile.Open(file_path, 'rb') as f:
            pianorolls = pickle.load(f)

        print("Save file: {0}\n".format(hd5_file_path))
        hd5_file_path = os.path.join(piano_directory, "{0}.h5".format(name))
        dd.io.save(hd5_file_path, pianorolls)
