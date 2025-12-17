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
YOU ARE AN EXPERT MACHINE LEARNING ENGINEER AND THESIS WRITER.
Consider the following latex code:
```

```

Write the TODO in the section on PPO according to my writing style above.
Use the following code as reference:
"""
suffix = """"""
files_to_join = [
    "abstract_agent.hpp",
    "abstract_agent.cpp",
    "base_actor_critic.hpp",
    "base_actor_critic.cpp",
    "base_ppo_agent.hpp",
    "base_ppo_agent.cpp",
    "ppo_trainer.hpp",
    "ppo_trainer.cpp"
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
