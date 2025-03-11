---
license: mit
---

# Foundation Tactile (FoTa) - a multi-sensor multi-task large dataset for tactile sensing

This repository stores the FoTa dataset and the pretrained checkpoints of Transferable Tactile Transformers (T3).

<a href="https://arxiv.org/abs/2406.13640" class="btn btn-light" role="button" aria-pressed="true">
    <svg class="btn-content" style="height: 1.5rem" aria-hidden="true" focusable="false" data-prefix="fas" data-icon="file-pdf" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512" data-fa-i2svg=""><path fill="currentColor" d="M181.9 256.1c-5-16-4.9-46.9-2-46.9 8.4 0 7.6 36.9 2 46.9zm-1.7 47.2c-7.7 20.2-17.3 43.3-28.4 62.7 18.3-7 39-17.2 62.9-21.9-12.7-9.6-24.9-23.4-34.5-40.8zM86.1 428.1c0 .8 13.2-5.4 34.9-40.2-6.7 6.3-29.1 24.5-34.9 40.2zM248 160h136v328c0 13.3-10.7 24-24 24H24c-13.3 0-24-10.7-24-24V24C0 10.7 10.7 0 24 0h200v136c0 13.2 10.8 24 24 24zm-8 171.8c-20-12.2-33.3-29-42.7-53.8 4.5-18.5 11.6-46.6 6.2-64.2-4.7-29.4-42.4-26.5-47.8-6.8-5 18.3-.4 44.1 8.1 77-11.6 27.6-28.7 64.6-40.8 85.8-.1 0-.1.1-.2.1-27.1 13.9-73.6 44.5-54.5 68 5.6 6.9 16 10 21.5 10 17.9 0 35.7-18 61.1-61.8 25.8-8.5 54.1-19.1 79-23.2 21.7 11.8 47.1 19.5 64 19.5 29.2 0 31.2-32 19.7-43.4-13.9-13.6-54.3-9.7-73.6-7.2zM377 105L279 7c-4.5-4.5-10.6-7-17-7h-6v128h128v-6.1c0-6.3-2.5-12.4-7-16.9zm-74.1 255.3c4.1-2.7-2.5-11.9-42.8-9 37.1 15.8 42.8 9 42.8 9z"></path></svg>
    <span class="btn-content">Paper</span>
</a>
<a href="https://github.com/alanzjl/t3" class="btn btn-light" role="button" aria-pressed="true">
    <svg class="btn-content" style="height: 1.5rem" aria-hidden="true" focusable="false" data-prefix="fab" data-icon="github" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512" data-fa-i2svg=""><path fill="currentColor" d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"></path></svg>
    <span class="btn-content">Code</span>
</a>
<a href="https://colab.research.google.com/drive/1MmO9w1y59Gy6ds0iKlW04olszGko56Vf?usp=sharing" class="btn btn-light" role="button" aria-pressed="true">
    <svg class="btn-content" style="height: 1.5rem" viewBox="0 0 24 24"><!--?lit$3829301306$--><g id="colab-logo"><path d="M4.54,9.46,2.19,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36A3.59,3.59,0,0,1,4.54,9.46Z" style="fill:var(--colab-logo-dark)"></path><path d="M2.19,7.1,4.54,9.46a3.59,3.59,0,0,1,5.08,0l1.71-2.93h0l-.1-.08h0A6.93,6.93,0,0,0,2.19,7.1Z" style="fill:var(--colab-logo-light)"></path><path d="M11.34,17.46h0L9.62,14.54a3.59,3.59,0,0,1-5.08,0L2.19,16.9a6.93,6.93,0,0,0,9,.65l.11-.09" style="fill:var(--colab-logo-light)"></path>
        <path d="M12,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36a3.59,3.59,0,1,1,5.08-5.08L21.81,7.1A6.93,6.93,0,0,0,12,7.1Z" style="fill:var(--colab-logo-light)"></path>
        <path d="M21.81,7.1,19.46,9.46a3.59,3.59,0,0,1-5.08,5.08L12,16.9A6.93,6.93,0,0,0,21.81,7.1Z" style="fill:var(--colab-logo-dark)"></path>
      </g></svg>
    <span class="btn-content">Colab</span>
</a>

[[Project Website]](https://t3.alanz.info/)

[Jialiang (Alan) Zhao](https://alanz.info/), 
[Yuxiang Ma](https://yuxiang-ma.github.io/), 
[Lirui Wang](https://liruiw.github.io/), and 
[Edward H. Adelson](https://persci.mit.edu/people/adelson/)

MIT CSAIL


<img src="https://t3.alanz.info/imgs/dataset.png" width="80%" />

## Overview
FoTa was released with Transferable Tactile Transformers (T3) as a large dataset for tactile representation learning.
It aggregates some of the largest open-source tactile datasets, and it is released in a unified [WebDataset](https://webdataset.github.io/webdataset/) format.

Fota contains over 3 million tactile images collected from 13 camera-based tactile sensors and 11 tasks.

## File structure

After downloading and unzipping, the file structure of FoTa looks like:

```
dataset_1
    |---- train
        |---- count.txt
        |---- data_000000.tar
        |---- data_000001.tar
        |---- ...
    |---- val
        |---- count.txt
        |---- data_000000.tar
        |---- ...
dataset_2
:
dataset_n
```

Each `.tar` file is one sharded dataset. At runtime, wds (WebDataset) api automatically loads, shuffles, and unpacks all shards on demand.
The nicest part of having a `.tar` file, instead of saving all raw data into matrices (e.g. `.npz` for zarr), is that `.tar` is easy to visualize without the need of any code.
Simply double click on any `.tar` file to check its content.

Although you will never need to unpack a `.tar` manually (wds does that automatically), it helps to understand the logic and file structure.

```
data_000000.tar
    |---- file_name_1.jpg
    |---- file_name_1.json
    :
    |---- file_name_n.jpg
    |---- file_name_n.json
```
The `.jpg` files are tactile images, and the `.json` files store task-specific labels.

For more details on operations of the paper, checkout our GitHub repository and Colab tutorial.

<img src="https://t3.alanz.info/imgs/data_vis.jpg" width="80%" />

## Getting started

Checkout our [Colab](https://colab.research.google.com/drive/1MmO9w1y59Gy6ds0iKlW04olszGko56Vf?usp=sharing) for a step-by-step tutorial!

## Download and unpack

Download either with the web interface or using the python interface:
```sh
pip install huggingface_hub
```

then inside a python script or in ipython, run the following:
```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="alanz-mit/FoundationTactile", repo_type="dataset", local_dir=".", local_dir_use_symlinks=False)
```

To unpack the dataset which has been split into many `.zip` files:
```sh
cd dataset
zip -s 0 FoTa_dataset.zip --out unsplit_FoTa_dataset.zip
unzip unsplit_FoTa_dataset.zip
```

## Citation

```
@article{zhao2024transferable,
      title={Transferable Tactile Transformers for Representation Learning Across Diverse Sensors and Tasks}, 
      author={Jialiang Zhao and Yuxiang Ma and Lirui Wang and Edward H. Adelson},
      year={2024},
      eprint={2406.13640},
      archivePrefix={arXiv},
}
```

MIT License.