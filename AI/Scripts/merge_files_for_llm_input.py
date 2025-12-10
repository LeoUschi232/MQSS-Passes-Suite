from os import getcwd, walk
from os.path import join

dir_list = getcwd().split("/")
while True:
    if dir_list[-1] == "MQSS-Passes-Suite":
        break
    else:
        dir_list = dir_list[:-1]
include_dir = "/".join(dir_list + ["AI", "include"])
src_dir = "/".join(dir_list + ["AI", "src"])
output_filepath = "/".join(dir_list + ["AI", "Scripts", "llm_input.txt"])
prefix = """
Consider the following files of a C++ project:
"""
suffix = """
I want to create an image of a single residual block as in this implementation.
For that I would like for you to implement as minmal as possible implementation of a neural network using pytorch with the same architecture as the C++ code provides
such that using some function like e.g. rochviz, I can then create a concept image for a research paper of what one residual block looks like.
For the weightnorm i had to implement a custom weightnorm in libtorch BUT when writing the code in pytorch use the weight-norm provided by pytorch, do NOT implement your own weight-norm.
"""
files_to_join = [
    "layers_and_wrappers.hpp",
    "layers_and_wrappers.cpp",
    "tcn_network.hpp",
    "tcn_network.cpp",
    "single_output_lstm.hpp",
    "single_output_lstm.cpp",
    "agent_architectures.hpp",
    "agent_architectures.cpp",
]
filepaths = []
for root, _, files in walk(include_dir):
    for file in files:
        if file in files_to_join:
            filepaths.append(join(root, file))
for root, _, files in walk(src_dir):
    for file in files:
        if file in files_to_join:
            filepaths.append(join(root, file))
with open(output_filepath, "w") as output_file:
    output_file.write(prefix)
    output_file.write("\n```\n")
    for filepath in filepaths:
        output_file.write(f"\n{filepath}:\n\n")
        with open(filepath) as input_file:
            output_file.write(input_file.read())
            output_file.write("\n")
    output_file.write("\n```\n")
    output_file.write(suffix)
print("Done.")
