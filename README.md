# NeurIPS_Two_Stage_Predict-Optimize

This repository is the official implementation of the paper: Two-Stage Predict+Optimize for Mixed Integer Linear Programs with Unknown Parameters in Constraints.

Download and extract the [datasets](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155136882_link_cuhk_edu_hk/Eme0llGwtFJBg6aQRoEB53UBwxwKl65PpJmmRTib2GS8kQ?e=YCpS0j).

To run the three benchmarks: the alloy production problem, the 0-1 knapsack problem, and the nurse scheduling problem:
1.	Enter the corresponding folder
2.	Move the dataset into the “data” folder and unzip all the data files
3.	Run “python3 train.py”
For example, to run the brass experiment in the alloy production problem:
1.	Enter “./code/Alloy production” folder
2.	Move all the files in the“./data/Alloy production/brass” folder into the “./code/Alloy production/data” folder and unzip all the data files
3.	Run “python3 train.py”
