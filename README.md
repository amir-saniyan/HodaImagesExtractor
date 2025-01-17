In the name of God

# Hoda images extractor
This repository extracts [Hoda](http://farsiocr.ir/) dataset images.

# Extract images
To extract Hoda dataset images, type the following command at the command prompt:
```
python3 ./hoda_images_extractor.py
```

# Hoda Farsi Digit Dataset
Hoda dataset is the first dataset of handwritten Farsi digits that has been developed during an MSc. project in Tarbiat
Modarres University entitled: Recognizing Farsi Digits and Characters in SANJESH Registration Forms. This project has
been carried out in cooperation with Hoda System Corporation. It was finished in summer 2005 under supervision of Prof.
Ehsanollah Kabir.
Samples of the dataset are handwritten characters extracted from about 12000 registration forms of university entrance
examination in Iran. The dataset specifications is as follows:

* Resolution of samples: 200 dpi
* Total samples: 102,352 samples
* Training samples: 60,000 samples
* Test samples: 20,000 samples
* Remaining samples: 22,352 samples

Number of samples per each class:
* 0: 10070
* 1: 10330
* 2: 9923
* 3: 10334
* 4: 10333
* 5: 10110
* 6: 10254
* 7: 10363
* 8: 10264
* 9: 10371

For more information please refer to the paper: [Introducing a very large dataset of handwritten Farsi digits and a
study on their varieties](http://farsiocr.ir/Archive/dataset_PRL.pdf)

**This dataset is free of charge for research purposes and non commercial uses only.**

Dataset website: [http://farsiocr.ir/](http://farsiocr.ir/مجموعه-داده/مجموعه-ارقام-دستنویس-هدی)

# Dataset Samples

Samples with different writing styles in the dataset:

![Samples with different writing styles in the dataset](Farsi_Digits_Sample_1.gif)

Samples with different qualities in the dataset:

![Samples with different qualities in the dataset](Farsi_Digits_Sample_2.gif)

# Dependencies
* Python 3
* numpy
* imageio

# Links
* http://farsiocr.ir/مجموعه-داده/مجموعه-ارقام-دستنویس-هدی
* http://dadegan.ir/catalog/hoda
* https://github.com/amir-saniyan/HodaImagesExtractor
