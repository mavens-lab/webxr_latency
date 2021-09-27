Dependancies: Python 3 version 3.7+, numpy, numpy-quaternion https://quaternion.readthedocs.io/en/latest/, matplotlib

Current version of the script works on most gltf 2.0 files that have been separated into .gltf and .bin files of the same name.
Script requires edits to the __main__ method to set camera initial position, orientation, ect.

Some debug functions as well as the VIA-Image functionality has been commented out but may be re-enabled.

command to run: python VINS.py <gltf file name>

resultant files will be writen to the same directory in the order of increasing priority (<gltf file name>0..n<.gltf or .bin>)

small edits to website javascript may be necesssary for multiple file requests instead of single file requests.
