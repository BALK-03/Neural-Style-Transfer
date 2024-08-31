This project uses techniques from the paper [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). This README will explain the basics from the paper and show you how to set up and use the project.



Project Organization
------------

    ├── test_environment.py         <- Script to ensure the development environment is set up correctly
    │
    ├── requirements.txt            <- File listing the requirements for reproducing the analysis environment,
    │                                   e.g., generated with `pip freeze > requirements.txt`
    │
    ├── images                     <- Folder for test images
    │   └── <test_images>           <- Test images used for evaluating the project
    │
    ├── saved_images               <- Folder for storing output images
    │   └── <output_images>         <- Output images generated from the test images
    │
    └── src                         <- Source code for the project
        │
        ├── data                    <- Scripts for data preprocessing
        │   └── preprocess_data.py
        │
        └── models                  <- Scripts for training models and making predictions
            └── make_prediction.py