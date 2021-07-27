import os

def load_text_file(path_to_file):
    f  = open(path_to_file, "r")
    print (f.read())