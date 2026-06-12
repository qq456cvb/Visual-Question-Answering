# Visual-Question-Answering

A PyTorch implementation of the **Spatial Memory Network (SMN)** from [Ask, Attend and Answer: Exploring Question-Guided Spatial Attention for Visual Question Answering (ECCV 2016)](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_28) by Xu and Saenko, in a single notebook (`VQA.ipynb`).

## Model

The SMN treats the spatial grid of CNN features as a memory that the question attends over:

- **Image memory** — a frozen, pretrained **ResNet-152** truncated before the last pooling stage yields a 14×14 grid of visual features (L = 196 memory slots).
- **Question encoding** — word embeddings fed through both an LSTM (for per-word vectors used in attention) and a learned bag-of-words pooling (for the sentence vector).
- **Word-guided attention (hop 1)** — each word is correlated against every memory slot; the per-slot maximum over words is softmaxed into a spatial attention map, which pools an "evidence" embedding of the image. The attended visual feature plus the question vector gives the first-hop output.
- **Second hop** — the hop-1 output queries the memory again to refine the attention before the final softmax classifier over the answer vocabulary. Both one-hop and two-hop variants are implemented (set `hop` when constructing `SMN`).

Answering is framed as classification over single-word answers with a negative log-likelihood loss.

## Data

The notebook supports two datasets, selected by the `dataset` flag in the configuration cell:

- **DAQUAR** — expects the train/val question-answer text files and images under `./datasets/DAQUAR/`.
- **VQA (v1, MS-COCO)** — expects the official annotation/question JSONs and images under `./datasets/VQA/`; the answer vocabulary is restricted to the top 1000 training answers, as in the paper.

Set `preprocess = True` on the first run to build the question/answer tensors and vocabulary mappings.

## Running

Open `VQA.ipynb` and run the cells top to bottom. Configuration (CUDA, dataset, epochs, checkpoint resume) lives in one flag cell near the top; checkpoints are written to `./model/` and per-epoch losses/accuracy to `log_<dataset>.txt`. The last cells visualize a validation sample together with its learned attention map.

Note this is an early-PyTorch, Python 2 era notebook (`print` statements, `xrange`, `autograd.Variable`); expect to modernize it for current PyTorch versions.

## Reference

```bibtex
@inproceedings{xu2016ask,
  title={Ask, Attend and Answer: Exploring Question-Guided Spatial Attention for Visual Question Answering},
  author={Xu, Huijuan and Saenko, Kate},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2016}
}
```
