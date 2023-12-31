# Learning to Learn: How to Continuously Teach Humans and Machines (ICCV 2023)

Authors: Parantak Singh, You Li, Ankur Sikarwar, Weixian Lei, Daniel Gao, Morgan Bruce Talbot, Ying Sun, Mike Zheng Shou, Gabriel Kreiman, Mengmi Zhang\
This work has been accepted to the **Internation Conference on Computer Vision (ICCV) 2023**.

Access to our unofficial manuscript [HERE](https://arxiv.org/abs/2211.15470), supplementary material [HERE](https://arxiv.org/abs/2211.15470), poster [HERE](https://docs.google.com/presentation/d/1zobdGlkLEgpPxwI1P53SCrHRMI1mxsT-/edit?usp=sharing&ouid=114037434904273947350&rtpof=true&sd=true) and presentation video [HERE](https://youtu.be/xz1TSRAQCN4).

Our education system comprises a series of curricula. For example, when we learn mathematics at school, we learn in order from addition to multiplication, and later to integration.  Delineating a curriculum for teaching either a human or a machine shares the underlying goal of maximizing the positive knowledge transfer from early to later tasks and minimizing forgetting of the early tasks. Here, we exhaustively surveyed the effect of curricula on existing continual learning algorithms in the class-incremental setting, where algorithms must learn classes one at a time from a single pass of the data stream. We observed that across a breadth of possible class orders (curricula), curricula influence the retention of information and that this effect is not just a product of stochasticity. Further, as a primary effort toward automated curriculum design, we proposed a method capable of designing and ranking effective curricula based on inter-class feature similarities. We compared the predicted curricula against empirically determined effective curricula and observed significant overlaps between the two. To support the study of a curriculum designer, we conducted a series of human psychophysics experiments and contributed a new continual learning benchmark in object recognition. We assessed the degree of agreement in effective curricula between humans and machines. Our curriculum designer predicts a reasonable set of curricula that is effective for human learning. There are many considerations in curriculum designs, such as timely student feedback and learning with multiple modalities. Our study is the first attempt to set a standard framework for the community to tackle the problem of teaching humans and machines to learn to learn continuously.

## Novel Object Dataset (NOD)

Download the NOD dataset from [HERE](https://drive.google.com/drive/folders/1SPA8TIZr20VZodPs7feFk8DYPiCOPXbE?usp=sharing) \
Extract the dataset to ```Learning2Learn/continual baselines/data/```

PyTorch `Dataset` for each of our custom tasks can be found in ```Learning2Learn/continual baselines/utils/custom_datasets.py```

### Dependencies

- tested with python 3.8 and cuda 11.3
- dependencies can be installed using `/requirements.txt`

## Training and Testing
After extracting the NOD dataset, run the following command from ```Learning2Learn/continual baselines``` directory to extract logs on dataset 'x' with strategy 'y'\
Do not set the -logging_only argument if you want to train from scratch
- dataset - (FashionMNIST/MNIST/CIFAR10)\
	strategy - (naive/ewc/lwf)\
	`python paradigm-I.py --num_subset_classes 5 --num_runs 3 --logging_only True --dataset x --strategy y`
- dataset - (NOD)\
  strategy - (naive/ewc/lwf)\
	`python paradigm-I.py --num_experiences 4 --num_runs 3 --logging_only True --dataset NovelNet --strategy y`
	
	*to generate agreement between curricula\
	set the follwing args with logging: `--strategy_comparison True --no_avg_strategy True`

Refer to ```Learning2Learn/continual baselines/help.md``` for possible arguments.

## Psychophysics Experiments
To launch the psychophysics experiments highlighted in the paper, refer to the following repository:
[NOD-Experiment](https://github.com/ZhangLab-DeepNeuroCogLab/nod-experiment)

## Schematic Illustration of the Problem Setting

<br>
<p align="center"><img align="center"  src="./images/Parantak_intro-cropped.png" alt="..." width="550">
</p>
