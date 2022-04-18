# Visual Keyword Spotting with Attention

This is the official implementation of the [Transpotter paper](https://arxiv.org/pdf/2110.15957.pdf). The code has been tested with Python version 3.6.8. Pre-trained checkpoints are also released. 

## Setup
- `pip install -r requirements.txt`
- Download the necessary checkpoints and the test set pickle files.
  - `cd checkpoints/`
  - `sh download_models.sh`

## Feature extraction

Please follow the steps in [this repository](https://github.com/prajwalkr/vtp) to extract the features for the LRS2, LRS3 test set. Please use the model trained on LRS2 + LRS3 for the feature extraction. The provided code and pre-trained models work with these features.  

## Computing the scores on LRS2 and LRS3 test sets

The following command is used to compute the scores mentioned in the last row of Table 1 of the [paper](https://arxiv.org/pdf/2110.15957.pdf)

```bash

# LRS3
python test_and_score.py --data_root /path/to/lrs3/test/ --test_pkl_file checkpoints/lrs3_test.pkl --ckpt_path checkpoints/ft_lrs3.pth --localization

# LRS2
python test_and_score.py --data_root /path/to/lrs2/vid/ --test_pkl_file checkpoints/lrs2_test.pkl --ckpt_path checkpoints/ft_lrs2.pth --localization
```

##### Note:
- `--localization` flag is only used to compute $mAP^{loc}$. The other metrics can be computed by not using this flag. 
- LRS3 test scores are off by 0.2 points than the ones mentioned in the paper, because of missing files. 


## Citation

Please cite the following paper if you find our work useful:
```
@inproceedings{prajwal2021visual,
  title={Visual Keyword Spotting with Attention},
  author={Prajwal, KR and Momeni, Liliane and Afouras, Triantafyllos and Zisserman, Andrew},
  booktitle={BMVC},
  year={2021}
}
```

## Acknowledgements

We thank the author of [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) for the Transformer implementation. 
