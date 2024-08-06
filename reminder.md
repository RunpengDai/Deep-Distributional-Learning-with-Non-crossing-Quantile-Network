## Installation
1. python 3.7 or less
2. Installation of atari check for https://github.com/openai/atari-py?tab=readme-ov-file

## Configuration
1. Original memory_size is 1000000, which is 4M frames, which cost 13G of memory, which slow down the sampling process. So, I just change it to .5M frames.

Trying to find out why the whole process is so slow, when tasks are not many, the sampling usually take 12s and the bottle neck is step time, which depends on the gym env cannot be optimized. When tasks are many, sampling time increases to 100 which because the to gpu is slow when the batch size is large.
![alt text](image.png)