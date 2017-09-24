import math
def m2f(note):
    """Convert MIDI note number to frequency in Hertz.
    See https://en.wikipedia.org/wiki/MIDI_Tuning_Standard.
    """
    return (2 ** ((note - 69) / 12)) * 440
print(m2f(64))

