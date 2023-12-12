REINFORCEMENT LEARNING FOR TRIPLE-WELL MODEL

This file is to show how to use the codes for Reinforcement Learning for a triple-well potential (of the state s), by employing Q-table.
According to [I], the Q-table Q(s,T,a), where s, T are states and a is action, is trained with N_epoch epochs, each epoch contains N_step steps of applying action a (changing T), in each step of N_step, the triple-well is called for a certain time and the last value of s is used for the next step.   

If you use the code in your work, please cite the following reference
   (I)  Uyen Tu Lieu and Natsuhiko Yoshinaga, "Dynamic control of self-assembly of quasicrystalline structures through reinforcement learning", submitted, https://arxiv.org/abs/2309.06869


### 1-STRUCTURE OF THE FILES ### 
   (1) 	RL_3well.m   	main file (MATLAB)
  

### 2-TESTED ENVIRONMENT ###
We tested the codes in the following environments:
   - Windows 10 ver22H2
   - MATLAB R2018b
   		 
 	

### 3-HOW TO USE ###
*INPUT:
   - For RL_3well.m (1), the input for RL is in line #4-14, input for the triple-well potential is in line #15-23
 	
*OUTPUT: 
After each epoch i, the output data is
   - Q-table after each epoch (train_Q_epoch*.dat)       
   - training data of s,T each epoch (traindata_epoch*.dat)
   - file saving the input of RL (00input.dat)	
After each epoch i, the output figures are
   - the policy after each epoch, determined by argmax_a Q(s, T, a). The blue, grey, red elements on the sigma, T plane corresponds to action decrease T, keep T constant, increase T. (fig_epoch*_policy.fig)
   - the trajectories of s and T in each epoch (fig_epoch*_sT.fig)

  	