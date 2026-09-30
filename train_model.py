# MelodyForge AI
# AI Music Generation using LSTM Neural Network

import os
import pickle
import glob
import numpy as np
import music21

from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ---------------- CONFIGURATION ----------------

DATASET_PATH = "dataset"
OUTPUT_PATH = "output"

SEQUENCE_LENGTH = 40
EPOCHS = 30
BATCH_SIZE = 64

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ---------------- EXTRACT NOTES FROM MIDI ----------------

def extract_notes(file_path):

    print(f"Reading MIDI: {file_path}")

    try:
        midi = converter.parse(file_path)
        notes = []

        parts = instrument.partitionByInstrument(midi)

        if parts:
            music_stream = parts.parts[0].recurse()
        else:
            music_stream = midi.flat.notes

        for element in music_stream:

            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                chord_notes = ".".join(
                    str(pitch) for pitch in element.pitches
                )
                notes.append(chord_notes)

        return notes

    except Exception as error:
        print(f"Could not read {file_path}: {error}")
        return []


# ---------------- LOAD MIDI DATASET ----------------

def load_dataset():

    midi_files = glob.glob(
        os.path.join(DATASET_PATH, "*.mid")
    )

    midi_files += glob.glob(
        os.path.join(DATASET_PATH, "*.midi")
    )

    if not midi_files:
        raise FileNotFoundError(
            "No MIDI files found. Add MIDI files to the dataset folder."
        )

    all_notes = []

    for file_path in midi_files:
        notes = extract_notes(file_path)
        all_notes.extend(notes)

    print("Total notes collected:", len(all_notes))

    if len(all_notes) <= SEQUENCE_LENGTH:
        raise ValueError(
            "Not enough notes. Add more MIDI files to your dataset."
        )

    return all_notes


# ---------------- PREPARE TRAINING DATA ----------------

def prepare_sequences(notes):

    pitchnames = sorted(set(notes))

    note_to_int = {
        name: number
        for number, name in enumerate(pitchnames)
    }

    network_input = []
    network_output = []

    for i in range(len(notes) - SEQUENCE_LENGTH):

        sequence = notes[i:i + SEQUENCE_LENGTH]

        next_note = notes[i + SEQUENCE_LENGTH]

        network_input.append(
            [note_to_int[n] for n in sequence]
        )

        network_output.append(
            note_to_int[next_note]
        )

    n_patterns = len(network_input)
    n_vocab = len(pitchnames)

    network_input = np.reshape(
        network_input,
        (n_patterns, SEQUENCE_LENGTH, 1)
    )

    network_input = network_input / float(n_vocab)

    network_output = to_categorical(
        network_output,
        num_classes=n_vocab
    )

    return (
        network_input,
        network_output,
        note_to_int,
        pitchnames
    )


# ---------------- BUILD LSTM MODEL ----------------

def build_model(n_vocab):

    model = Sequential([
        LSTM(
            128,
            input_shape=(SEQUENCE_LENGTH, 1),
            return_sequences=True
        ),

        Dropout(0.3),

        LSTM(128),

        Dense(128, activation="relu"),

        Dropout(0.3),

        Dense(n_vocab, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam"
    )

    model.summary()

    return model


# ---------------- TRAIN MODEL ----------------

def train_model():

    print("\n🎵 MelodyForge AI Training Started")

    notes = load_dataset()

    (
        network_input,
        network_output,
        note_to_int,
        pitchnames
    ) = prepare_sequences(notes)

    print("Training patterns:", len(network_input))
    print("Unique musical tokens:", len(pitchnames))

    model = build_model(len(pitchnames))

    model.fit(
        network_input,
        network_output,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Save trained model
    model.save(
        os.path.join(OUTPUT_PATH, "melody_model.keras")
    )

    # Save note mapping
    with open(
        os.path.join(OUTPUT_PATH, "note_mapping.pkl"),
        "wb"
    ) as file:
        pickle.dump(pitchnames, file)

    print("\n✅ Training completed!")
    print("Model saved in output/melody_model.keras")
    print("Note mapping saved in output/note_mapping.pkl")


# ---------------- MAIN ----------------

if __name__ == "__main__":
    train_model()