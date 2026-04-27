# DS_Health_UWM_Project
This is the repo will all our stuff for the UWM Project submission
1) Go to classroom machine
2) set up environment with uv
3) git clone https://github.com/William-Ohonba/DS_Health_UWM_Project/
4) pip install -r requirements.txt
5) Enable tmux with tmux attach -t (name_you_want_to_use)
6) go to code folder
7) python3 train.py --n_slices 3    - for training on 3 slices
8) python3 train.py --n_slices 5    - for training on 5 slices
9) python3 train.py --n_slices 3 --upscale - upscaled on 3 slices
10) If you want to activate tensorboard
11) open a new window ssh to classroom machines and activate your virtual environment with uv then use tmux command again
12) when in tmux environment press ctrl + b then pause and press c
13) go to train.py
14) run tensorboard --logdir runs/
15) on local machine use 
16) ssh -L (domain_number):localhost:(domain_number) (username)@pamd.sc.fsu.edu -t ssh -L (domain_number):localhost:(domain_number) (username)@(classmachine_number).sc.fsu.edu
17) then on your browser paste http://localhost:(domain_number)/
