# About this project

This is a very well-documented project that utilizes a Facial Recognition Software. <br>
For this project, we are going to use FaceNet which is a pre-trained model by Google which has capabilities to detect face features 
extracted from an image.

One interesting note about this repository, we are going to fetch all images from a MySQL database. In this case, an example will be 
about tenant records. 

Once we fetch images, we need to encode and write them as .png files to an output folder `out/train/images`

Then, we will use SVM to embed our parameters to create our <b>Facial Recognition Model</b> to which will be outputted as a `svm_model_160x160.pkl` file. Let us not forget our X, Y embeddings which is compressed into `faces_embeddings_done_4classes.npz`

# Dependencies 

- MTCNN
- Tensorflow
- OpenCV
- Pandas
- NumPy
- MatPlotlib
- dotenv
- MySQL Client (Python)

# Data information

The images we are fetching from a MySQL Database includes the following facts about a tenant record. 
<br>
A tenant record is defined under the entity: 
<br>

> FACE_IMG (Entity)
>> Face ID: <br> 
>> Tenant ID: <br>
>> Tenant Full Name: <br> 
>> Status <br>
>> Face IMG data: <br> 

<p>
    Where, <br>
    FACE_IMG is a single entity represents a record for a single individual who is a tenant.
    `FACE ID` is our primary key to identify a single record which is an information about their face capture <br>
    `Tenant ID`s represents a foreign key that holds an entire biodata about a tenant's identity <br>
    `Tenant Full Name` , a tenant's full name which is used to identify their face captures <br>
    `Status`, for recognized faces, it is valued as <i>Authorized</i>. Otherwise, <i>Unauthorized</i> <br>
    `Face IMG Data` which is a LONGBLOB type that contains raw data on how to encode the image to .png
    
</p>


<hr>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)


![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

<hr>