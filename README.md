# A-MAL - Automatic Motion Assessment Learning

This repository contains Python code for the algorithm described in the paper [**A-MAL: Automatic Motion Assessment Learning from Properly Performed Motions in 3D Skeleton Videos**](https://arxiv.org/pdf/1907.10004.pdf), by Tal Hakim and Ilan Shimshoni.

### Citation
Please cite the paper if you have found A-MAL helpful for your research, as:
```
@inproceedings{hakim2019mal,
  title={A-MAL: Automatic Motion Assessment Learning from Properly Performed Motions in 3D Skeleton Videos},
  author={Hakim, Tal and Shimshoni, Ilan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}
```

**Note**: We are checking the legality options for making the dataset demonstrated in the paper public. For now, the code is available for usage on user data.

## Prerequisite
* Python 3 
* Numpy and Scipy
* [SKVP-Python](https://github.com/skvp-owner/SKVP-Python)
* [simpledtw](https://github.com/talcs/simpledtw) (optional)

The solution uses SKVP formatted skeleton 3D video files. The SKVP format is detailed [here](https://github.com/skvp-owner/SKVP-Python).

## Usage
### Training a Model
The training algorithm requires a directory of SKVP formatted videos of **properly-performed** motions of the same type. Given such a directory, the training command is:
```
python run_amal.py train <input directory of videos> <output learned model file> [warping method] [ablations] [reference video output path]
```
Example: `python run_amal.py train /path/to/my_training_videos /path/to/trained_model.txt`

* The warping method default value is "our", as described in the paper. It can be overridden by "none" and "dtw". Using the "dtw" option will require installation of the [simpledtw](https://github.com/talcs/simpledtw) module.

* The ablations default value is "none". It can be overridden using one or more ablation types, separated by commas. The possible ablations are: `none, active, segmentation, time, diminish`. For example, a possible ablations argumemt value can be `active,time`.

* The reference video output path is only useful when warping with "dtw" method, as the assessment flow will need a reference video to be aligned to (while our default warping method does not need one, as it relies on pre-computed points of interest).

### Assessing Motions
The assessment algorithm requires a pretrained model file and an SKVP formatted video of a motion of the same type as used for training the model. Given these two inputs, the assessment command is:
```
python run_amal.py test <input model file> <input video file> [warping method] [ablations] [input reference video file]
```
Example: `python run_amal.py test /path/to/trained_model.txt /path/to/a_video_i_want_to_assess.skvp`

#### Sample output:
```
FEEDBACK: The angle between the right shoulder and the right wrist, through the right elbow, is too acute, from 0.00s to 1.87s
FEEDBACK: The right hand tip is too left, from 0.60s to 1.80s
FEEDBACK: The right hand is too left, from 0.67s to 1.73s
FEEDBACK: The right wrist is too left, from 0.67s to 1.73s
Scores: Active: 0.784891, NonActive: 0.757188, Time: 1.000000
Score: 0.838114
```
According to the above output, the final score is 83.8%. The score components are listed as well, which refer to the active-joint related, inactive-joint related and time-related parameters. In addition, 4 feedback items are printed.
