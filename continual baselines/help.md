usage: paradigm-I.py [-h] [--cuda CUDA] [--num_runs NUM_RUNS]
                     [--train_batch_size TRAIN_BATCH_SIZE]
                     [--eval_batch_size EVAL_BATCH_SIZE] [--epochs EPOCHS]
                     [--lr LR] [--data_path DATA_PATH] [--log_dir LOG_DIR]
                     [--save_path SAVE_PATH] [--num_workers NUM_WORKERS]
                     [--model {SqueezeNet,TestNet,SimpleCNN,ResNet}]
                     [--pretrained PRETRAINED]
                     [--dataset {ImageNetTiny,StyleNet,ParadigmDataset,CIFAR10,MNIST,FashionMNIST,NovelNet,ImageNet2012}]
                     [--class_order CLASS_ORDER [CLASS_ORDER ...]]
                     [--logging_only LOGGING_ONLY] [--map_type {slope,delta}]
                     [--load_weights LOAD_WEIGHTS]
                     [--strategy {ewc,naive,lwf}]
                     [--ewc_lambda EWC_LAMBDA] [--snet_layer {3,6,9,11,12}]
                     [--num_experiences NUM_EXPERIENCES]
                     [--num_permutations NUM_PERMUTATIONS] [--world_size N]
                     [--schedule SCHEDULE]
                     [--milestone MILESTONE [MILESTONE ...]] [--n_bins N_BINS]
                     [--percentile PERCENTILE]
                     [--strategy_comparison STRATEGY_COMPARISON]
                     [--lwf_alpha LWF_ALPHA [LWF_ALPHA ...]]
                     [--lwf_temperature LWF_TEMPERATURE]
                     [--initialization {uniform,xavier,gaussian}]
                     [--num_subset_classes NUM_SUBSET_CLASSES]
                     [--no_avg_strategy NO_AVG_STRATEGY]

optional arguments:
  -h, --help            show this help message and exit
  --cuda CUDA           select zero-indexed cuda device. -1 to use CPU.
  --num_runs NUM_RUNS   choose number of complete code run-throughs
  --train_batch_size TRAIN_BATCH_SIZE
                        train batch size (default: 64)
  --eval_batch_size EVAL_BATCH_SIZE
                        eval batch size (default: 64)
  --epochs EPOCHS       train epochs (default: 1)
  --lr LR               learning rate (default: 0.001)
  --data_path DATA_PATH
                        dataset root path (default: ./dataset)
  --log_dir LOG_DIR     tensorboard log dir (default: ./log)
  --save_path SAVE_PATH
                        model save loc (default: ./weights)
  --num_workers NUM_WORKERS
                        choose number of workers
  --model {SqueezeNet,TestNet,SimpleCNN,ResNet}
                        choose between different available models
  --pretrained PRETRAINED
                        if true, download model pretrained on full-ImageNet
  --dataset {ImageNetTiny,StyleNet,ParadigmDataset,CIFAR10,MNIST,FashionMNIST,NovelNet,ImageNet2012}
                        choose between different available datasets
  --class_order CLASS_ORDER [CLASS_ORDER ...]
                        choose a specific curriculum for the model
  --logging_only LOGGING_ONLY
                        if true, run through logs only (default: False)
  --map_type {slope,delta}
                        choose between slope or delta for graph-mapping
  --load_weights LOAD_WEIGHTS
                        if true, loads weights for SqueezeNet trained on
                        ImageNet100 (default: True)
  --strategy {ewc,naive,lwf}
                        selects a strategy
  --ewc_lambda EWC_LAMBDA
                        penalty hyperparameter for ewc
  --snet_layer {3,6,9,11,12}
                        selects which layer to probe in SqueezeNet
  --num_experiences NUM_EXPERIENCES
                        selects the number of tasks
  --num_permutations NUM_PERMUTATIONS
                        randomly select n permutations to run
  --world_size N        num GPUs
  --schedule SCHEDULE   if True, enables LR Step Scheduling
  --milestone MILESTONE [MILESTONE ...]
                        mention milestone epochs for LR schedule
  --n_bins N_BINS       number of bins
  --percentile PERCENTILE
                        percentile threshold
  --strategy_comparison STRATEGY_COMPARISON
                        runs a string based similarity comparison between strategies, 
                        runs in logging only mode
  --lwf_alpha LWF_ALPHA [LWF_ALPHA ...]
                        Penalty hyperparameter for LwF
  --lwf_temperature LWF_TEMPERATURE
                        temperature for softmax used in distillation
  --initialization {uniform,xavier,gaussian}
                        select weight initialization method
  --num_subset_classes NUM_SUBSET_CLASSES
                        select a subset of the original number of classes
  --no_avg_strategy NO_AVG_STRATEGY
                        if False, treats each run independently for strategy
                        comparison, runs in logging only and strategy comparison mode
