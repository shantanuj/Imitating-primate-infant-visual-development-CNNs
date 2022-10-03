## Model and experiments code for paper: What do CNNs gain by imitating the visual development of primate infants?

- PyTorch code for paper "What do CNNs gain by imitating the visual development of primate infants" (BMVC 2020) [Paper](https://www.bmvc2020-conference.com/assets/papers/0196.pdf)

- This repository primarily conmprises:
1. Model code for "growing" models (custom variants of VGG and ResNet)
    - [`custom_VGG.py`](custom_VGG.py) for VGG-growth (for ImageNet dataset)
    - [`custom_VGG_cifar.py`](custom_VGG_cifar.py) for VGG-growth for cifar dataset
    - [`filter_growth_resnet.py`](filter_growth_resnet.py) for ResNet with filter receptive field growth
    
2. Custom data transform layer for stage-wise color (saturation and contrast) and resolution variations
    - See [`custom_dataloader_and_utils.py`](custom_dataloader_and_utils.py) for data-loader and transforms code for stage-wise refinement of saturation, contrast and resolution of input images.
    
3. Sample training and evaluation scripts ImageNet and CIFAR object classification in a stage-wise manner
    - [`main_cifar100_train_with_gradual_growth.py`](main_cifar100_train_with_gradual_growth.py) for training and evaluating on CIFAR100 (all code utils in single script for cifar-specific settings; might need cleanup)
    - [`main_train_imnet_gradualgrowth_model.py`](main_train_imnet_gradualgrowth_model.py) for training and evaluating on ImageNet
    
4. Data visualization and ImageNet hiearchical subset creation
    - See jupyter notebook [`Scripts_for_ImgNet_Hierarchical_Subset_and_Custom_DataTransforms.ipynb`](Scripts_for_ImgNet_Hierarchical_Subset_and_Custom_DataTransforms.ipynb)  

## Citation
If you find this codebase or this project useful for your work, please consider citing:

    @inproceedings{jaiswal2020cnns,
      title={What do CNNs gain by imitating the visual development of primate infants?},
      author={Jaiswal, Shantanu and Choi, Dongkyu and Fernando, Basura},
      booktitle={BMVC},
      year={2020}
    }

## Acknowledgement
The codebase utilizes the [torchvision](https://github.com/pytorch/vision) libraries and built on [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100) repository for CIFAR100 models.

## License
This project's codebase is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Contact Information
In case of any suggestions or questions, please leave a message here or contact me directly at jaiswals@ihpc.a-star.edu.sg, thanks!
