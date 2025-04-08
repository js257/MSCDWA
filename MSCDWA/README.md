# Note: 
Results may vary across different runtime environments. If any discrepancies occur, please consider adjusting the weight of the diss loss.
# Prepare
## Dataset
Download the MOSI pkl file (https://drive.google.com/drive/folders/1_u1Vt0_4g0RLoQbdslBwAdMslEdW1avI?usp=sharing). Put it under the "./dataset" directory.

## Pre-trained language model
Download the SentiLARE language model files (https://drive.google.com/file/d/1onz0ds0CchBRFcSc_AkTLH_AZX_iNTjO/view?usp=share_link), and then put them into the "./pretrained-model/sentilare_model" directory.

# Run
'''
python train.py
'''

Note: the scale of MOSI dataset is small, so the training process is not stable. To get results close to those in our paper, you can set the seed in args to 6758. The experimental results of this paper are obtained on the Linux system.

