# RoboVerse Pouring Simulation

### Installation

First, follow RoboVerse instructions to install the environment.

- Install the RoboVerse - IsaacLab 1.4 environment
  - Follow the instructions from https://roboverse.wiki/metasim/get_started/installation

- Install CuRobo from RoboVerse
  - Follow the instructions from https://roboverse.wiki/metasim/get_started/advanced_installation/curobo


- Prepare assets (scene and material assets for high quality rendering)
  - Follow the instructions from https://huggingface.co/datasets/RoboVerseOrg/roboverse_data



### Run the demo
```
# export DISPLAY=:1 for server headless mode


cd sim
python collect_pouring_data.py
```

