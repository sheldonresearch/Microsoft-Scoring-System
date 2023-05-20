

./Bank: this is the source for Bank environment

./Could: this is the source for could environment


Each fold contains:

-  data_analysis: the unit processing the real data. We provide the source data of Bank. The dataset of cloud computing platform user activities is an internal dataset containing business secrets.
Therefore we can not open this dataset to public. If the paper is accepted, we will try to do the data desensitization on it

- distance: this is the module for soft-dtw

- gym_hybrid: this is the implemented environment for Bank and Could. The environment is built on OpenAI Gym.

- src: this is the executive program of the main body. 

 