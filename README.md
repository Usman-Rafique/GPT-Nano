# GPT-Nano
Building a GPT-style transformer network for text generation. Following the video lecture by [Andrej karpathy](https://github.com/karpathy), I am building GPT-Nano from scratch, mostly :)

## Usage

Currently, all the settings are in the `train.py` file. There are two options for `model_type` in `train.py`: `small` and `big`. You should be able to train the `small` model on a CPU machine/laptop. For the `big` model, you probably need a GPU machine -- this model takes around 4.3 G memory on GPU.

## Modifications

While I am closely following the video lecture to code GPT-Nano, these are some of the changes that I have made in this repo so far:

- I have broken down the code to two files: `train.py` and `gpt.py`. I am splitting the model and training code to (1) separate out the GPT-Nano model and (2) to make sure all the hyperparameters are explicitly passed to the model
- I am saving a trained model checkpoint. This should be helpful to reuse the trained model later on



## ToDo

The planned next steps are

- [x] Update expected train and val losses (for both light and big models)
- [ ] Add a `generate` script
- [ ] Train on another (fun) dataset, suggestions welcome :)
- [ ] Add installation / virtual environment instructions
- [ ] Add command line arguments?
- [ ] Try other encoding, such as BPE

# Acknowledgement

This repository is built following the [video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [github repo](https://github.com/karpathy/ng-video-lecture) by [Andrej karpathy](https://github.com/karpathy).