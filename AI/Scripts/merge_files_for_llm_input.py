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
prefix = """Consider the following subset of files of the project.
Assume actors output softmax, not logits, the input is unbatched, so no extra [B,...] dimension and all includes are implicitly correct.
The code builds and runs.
The only problems could be with logic e.g. .detach() present when inappropriate, absent when appropriate, wrong/bad tensor arithmetic.
Try to find errors, report about errors, if any.
"""
files_to_join = [
    "layers_and_wrappers.cpp",
    "base_actor_critic.hpp",
    "base_actor_critic.cpp",
    "base_acer_agent.hpp",
    "base_acer_agent.cpp",
    "acer_trainer.hpp",
    "acer_trainer.cpp"
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
    for filepath in filepaths:
        output_file.write(f"\n{filepath}:\n\n")
        with open(filepath) as input_file:
            output_file.write(input_file.read())
            output_file.write("\n")
print("Done.")
