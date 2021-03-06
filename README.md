# Keystroke-Inference-Attack-Synthetic-Dataset-Generator
Synthetic Dataset Generator for "Revisiting the Threat Space for Vision-Based Keystroke Inference Attacks" (ECCV 2020 Workshops)


This is the code for our synthetic dataset generator for vision-based keystroke inference attacks presented in "Revisiting the Threat Space for Vision-Based Keystroke Inference Attacks" at ECCV (CV-COPS Workshop) 2020

The link to the paper is in `paper/` or you can access it [here](https://arxiv.org/abs/2009.05796)!


### To run: 
### Step 0:
Install docker and nvidia-docker using `bash Docker_scripts/install_docker.sh`
Run our `Dockerfile` using `bash Docker_scripts/docker_4_p3d.sh`
you will need to change the volumes (-v flag) in `Docker_scripts/docker_4_p3d.sh` 


### Step 1a: 
To generate single key press images run `python single_key_press.py`

### Step 2a:
To align these images run `python render_single_key_press.py`

### Step 1b: 
To generate videos (i.e. full text messages or pins) run `python videos.py`
Note that you will most likely have to make adjustments to your own `parse_word_list()` function depending on the text you want to generate.

### Step 2b:
To align these videos run `python render_videos.py`

For each of the above steps, you will need to go into the args and change the data files. If you run into any problem or need clarification, please create an issue and I will get it to as soon as I can 
