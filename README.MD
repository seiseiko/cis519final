@@ -1,24 +1 @@

# **Machine Learning for COVID-19 Diagnosis**

---
Team: Lanqing Bao, Yuqi Zhang, Zeyuan Xu

### Environment
Python == 3.10  
Tensorflow == 2.10.0

### Dataset
1. COVID-19 RADIOGRAPHY DATABASE https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2. COVID-QU-Ex Dataset https://www.kaggle.com/datasets/anasmohammedtahir/covidqu

### Usage 

To train the model, pull up one of the following:
1. XCeption.py
2. MobileNet.py
3. VGG19.py
4. Inception.py
5. InceptionResnet.py
6. EfficientNet.py

To pre-process the dataset, run ***process.py***
To output the metrics on dataset, run ***test.py***
To output figure, run ***plot.py***

### Implemented transfer learning using the following network:

1. XCeption 
2. MobileNet
3. VGG19
4. Inception
5. InceptionResnetV2
6. EfficientNet
