# DL-HAR - Model Submodule

This is the submodule repository of the [dl_har_public repository](https://github.com/STRCSussex-UbiCompSiegen/dl_har_public).

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