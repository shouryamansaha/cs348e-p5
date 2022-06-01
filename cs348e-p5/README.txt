We recommend Ubuntu or MacOS as the operating systems. However, the following installation instructions "should" work for Windows as well.

1. Go to (https://www.anaconda.com/download/) and install the Python 3 version of Anaconda.

2. Open a new terminal and run the following commands to create a new conda environment (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

conda create -n drl-bullet python=3.6

3. Activate & enter the new environment you just creared:

conda activate drl-bullet

4. Inside the new environment (you should see "(drl-bullet)" preceeding your ternimal prompt), and inside the project directory:

pip install -r requirements.txt

5. Install pytorch (should work with or without cuda):

conda install pytorch 	(check pytorch website for your favorite version)

6. See if this runs without problem (You might see EndOfFile Error, that is normal when the program exits correctly): *See handout for the real training command once you finish coding*

python -m a2c_ppo_acktr.train_policy --env-name "HumanoidSwimmerEnv-v1" --num-steps 1000  --num-processes 2  --lr 3e-4 --entropy-coef 0 --ppo-epoch 10 --num-mini-batch 32 --num-env-steps 2000 --use-linear-lr-decay  --clip-param 0.2 --save-dir trained_models_quad_0 --seed 20062

7. After finishing the project, to deactivate an active environment, use:

conda deactivate
