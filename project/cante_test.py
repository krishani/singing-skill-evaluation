import cante
file_name='ayanna_kiyanna.wav'

cante.transcribe(file_name, acc=True, f0_file=False, recursive=False)

base,ext=file_name.split(".")
new_part='.notes.csv'
new_fname=base+new_part

