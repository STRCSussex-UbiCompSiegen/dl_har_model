# DL-HAR - Model Submodule

This is the model submodule repository of the [dl_har_public repository](https://github.com/STRCSussex-UbiCompSiegen/dl_har_public).

## Contributing to this repository

If you want to contribute to this repository **make sure to fork and clone the main repository [dl_har_public repository](https://github.com/STRCSussex-UbiCompSiegen/dl_har_public) with all its submodules**. To do so please run:

```
git clone --recurse-submodules -j8 git@github.com:STRCSussex-UbiCompSiegen/dl_har_public.git
```
If you want to have your modification be merged into the repository, please issue a **pull request**. If you don't know how to do so, please check out [this guide](https://jarv.is/notes/how-to-pull-request-fork-github/).

## Repository structure
In the following each of the main components will be briefly summarised. Output of the training process is a 
a dataframe containing the train, validation and (if applicable) test results as well as the predictions itself per subject and per random seed. 

### ```model``` directory
Contains python scripts of implemented deep learning architectures within the area of HAR. Each model needs to inherit from the ```BaseModel``` class to have all the necessary functionalities to work with the other repositories.

We currently support three model architectures namely the DeepConvLSTM, a shallow version of the DeepConvLSTM and the Attend and Discriminate model.

### ```train.py```
Contains all relevant methods to train the model architectures. 

Currently two validation methods are supported namely Leave-One-Subject-Out (LOSO) cross-validation and a predefined train-valid-test split. The latter can be defined/ altered via the YAML files as described in the [dataloader submodule](https://github.com/STRCSussex-UbiCompSiegen/dl_har_dataloader).

### ```eval.py```
Contains all relevant methods to evaluate a trained model architecture.