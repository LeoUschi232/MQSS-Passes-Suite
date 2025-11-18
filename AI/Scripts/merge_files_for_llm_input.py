from os import getcwd

dir_list = getcwd().split("/")
while True:
    if dir_list[-1] == "MQSS-Passes-Suite":
        break
    else:
        dir_list = dir_list[:-1]
include_dir = "/".join(dir_list + ["AI", "include"])
src_dir = "/".join(dir_list + ["AI", "src"])
output_filepath = "/".join(dir_list + ["AI", "Scripts", "llm_input.txt"])
final_text = """
Consider the following subset of files of the project.
Assume actors output softmax, not logits, the input is unbatched, so no extra [B,...] dimension and all includes are implicitly correct.
The code builds and runs.
The only problems could be with logic e.g. .detach() present when inappropriate, absent when appropriate, wrong/bad tensor arithmetic.
"""
files_to_join = [
    "base_actor_critic.hpp",
    "base_actor_critic.cpp",
    "base_sdsac_agent.hpp",
    "base_sdsac_agent.cpp",
    "sdsac_trainer.hpp",
    "sdsac_trainer.cpp",
]
# TODO: Recusrively go through all files in includes and src and check for matching filesnames.
# For file in files -> add full filepath to final_text, then ":\n\n", then file content.
with open(output_filepath, "w") as output_file:
    output_file.write(final_text)
print("Done.")