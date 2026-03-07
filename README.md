# DA6401 Assignment 1

This project follows the required submission structure.

## Run training
```bash
python3 src/train.py -d mnist -e 5 -b 64 -l cross_entropy -o rmsprop -lr 0.001 -wd 0.0 -nhl 2 -sz 128 128 -a relu -w_i xavier
```

## Run inference
```bash
python3 src/inference.py --model_path src/best_model.npy --config_path src/config.json -d mnist
```

Replace the GitHub link and W&B link below before submission.

- GitHub link: ADD_YOUR_PUBLIC_GITHUB_LINK
- W&B report link: ADD_YOUR_PUBLIC_WANDB_LINK
