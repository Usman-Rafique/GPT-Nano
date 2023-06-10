# GPT-Nano: Light-weight GPT from scratch
Building a GPT-style transformer network from scratch. for text generation. Following the video lecture by [Andrej Karpathy](https://github.com/karpathy), I am building GPT-Nano from scratch, mostly :)

Following the GPT protocol, the model is trained to predict the next character in the sequence. While modern models use words (or subwords), a character-level model is a strong starting point.

Once trained, we can use the model to generate as much text as we want. We can give it any random character, have the network generate the next character, and then feed both characters to the network. Using this, we can generate as much text as we want :)

But since we are only modeling characters, and working with limited data (and work with limited sequence length), we should not expect the model to generate semantically coherent text.

## Usage

Currently, all the settings are in the `train.py` file. There are two options for `model_type` in `train.py`: `small` and `big`. You should be able to train the `small` model on a CPU machine/laptop. For the `big` model, you probably need a GPU machine -- this model takes around 4.3 G memory on GPU.

## Datasets

We start with a dataset of William Shakespeare. I have compiled a dataset of Sherlock Holmes as well. (attribution and links in the acknowledgement section below)

## Sample Output

After training the `large` GPT-Nano model, the network trained on Shakespeare generates this text

```
LUCIO:
Here comes the maid: I shall do thee my nurse.
If I be gone, as you can read it, I have stopp'd;
You must have paid against my hands. What comfort here,
That I shall be slandering on a side
And dangerous can you nake, if I love.

TRANIO:
Such I am no soldier,
Lest I should shake you foes of my fellow sweet?

BENVOLIO:
I opprison him in these affairs.

ROMEO:
He was infamed to be fitted with Bianca,
And with the entertainment of the garden,
Which weaks me, and in the execution like the villain;
I'll gladly on coronation: the good kind's time,
That retire my grave I meabour'd your honours.
```

The model trained on Sherlock Holmes generate an output

```
“Oh, man!” said he, “I didn't fancy that I didn't know not now. A Greek deal purpose yesterday! That's neither o'clock before we go across? It was a large night when I ring it, shranting it in the doorway. It was becoming one, so now a projective directrict. But here comes and a lonely marker, with her and a great bulk quick femish in the observation of a town-to-day. She was deliciously paper, and a half-firckened, respected chokler, gave me indoing the foul play, which so distinguished to her, I could earn anything, but she was hard on the night in an overnage so terrible an assault. ‘Is I known place,’ she cried, ‘that with me on her singular spot tinge-dress, and we have only the merest through with Mrs. Hudson entered the room, but that fellow was a paralyer who had just heart of her evidence. Has she engaged in some way gone and her husband fe? or not, and would expect that she had handed justice.”

The young hunted back hung broughly open her arms as had been in the room, but a large and trapped took up me from the heart of the stair, locked in my 'ollow village towards us.

“I have had an and two times,” said she. “I had seen the contents of the private case. I therefore I knew her heart of her since I was engaged.”

“My eyes?” said Holmes, lighting a bicy from his sofa, as he laid his hand upon his exorting-carpet and picking up his way.

“What say you dictate this back?” asked Lestrade.

“I suppose,” said Adelbert. “Appreciate of your son? I heard his heavy stick success that I ever failed.
```

These don't make much sense, but it is cool that the model can generate text which matches the style of the training dataset.

## Modifications

While I am closely following the video lecture to code GPT-Nano, these are some of the changes that I have made in this repo so far:

- I have broken down the code to two files: `train.py` and `gpt.py`. I am splitting the model and training code to (1) separate out the GPT-Nano model and (2) to make sure all the hyperparameters are explicitly passed to the model
- I am saving a trained model checkpoint. This should be helpful to reuse the trained model later on

## ToDo

The planned next steps are

- [x] Update expected train and val losses (for both light and big models)
- [ ] Add a `generate` script
- [x] Train on another (fun) dataset, suggestions welcome :)
- [ ] Upload trained network checkpoints
- [ ] Add installation / virtual environment instructions
- [ ] Add command line arguments?
- [ ] Try some other encoding, such as BPE

# Acknowledgements

- This repository is built following the [video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [github repo](https://github.com/karpathy/ng-video-lecture) by Karpathy.
- The transformer architecture was proposed in [Attention is All You Need paper](https://arxiv.org/abs/1706.03762).
- The Shakespeare dataset is provided by [Karpathy at this public link](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
- I have compiled the Sherlock Holmes dataset from this website: https://sherlock-holm.es/stories/html/cano.html

