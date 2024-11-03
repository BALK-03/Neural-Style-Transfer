![zlafa](https://github.com/user-attachments/assets/796ce373-03dc-46d5-aa67-f682d080c72e)
![+ (1)](https://github.com/user-attachments/assets/d9d3fbcc-6577-433c-a85c-8052e287329f)



Project Structure
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








How to Run the Project
---------------------
1. *Open cmd*<br>
2. *Navigate to your desired directory*<br>
`cd C:\Users\user\Desktop`
3. *Clone the project repository from GitHub*<br>
`git clone https://github.com/BALK-03/Neural-Style-Transfer.git`
4. *Navigate to the project directory*<br>
`cd Neural-Style-Transfer`
5. *Set up a virtual environment, python 3.10 required for package compatibility*<br>
`py -3.10 -m venv .venv`
6. *Activate the virtual environment*<br>
`.venv\scripts\activate`
7. *Install project dependencies*<br>
`pip install -r requirements.txt`
8. *Run the model to generate an image*<br>
The first argument is the path to the content image, the second is the path to the style image, and the third is the path where the generated image will be saved. You can also specify optional parameters such as learning rate, content weight, style weight, and epochs. By default, the learning rate is set to 5, the style weight to 10, the content weight to 1000, and the number of epochs to 3000.
- To use default parameters:<br>
`python src/models/make_prediction.py "PATH/TO/CONTENT/IMAGE.jpg" "PATH/TO/STYLE/IMAGE.jpg" "PATH/TO/SAVE/IMAGE.jpg"`
- To customize parameters:<br>
`python src/models/make_prediction.py "PATH/TO/CONTENT/IMAGE.jpg" "PATH/TO/STYLE/IMAGE.jpg" "PATH/TO/SAVE/IMAGE.jpg" learningRate contentWeight styleWeight epochs`









References
---------------------
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)
<br>
[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
