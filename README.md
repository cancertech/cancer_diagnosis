# A Unified Software Package for Cancer Diagnosis

This research is supported by the National Cancer Institute grant U01 CA231782, PI: Linda G. Shapiro, co-PI: Joann G. Elmore

The long-term goal of this project is to develop a unified software package for sharing image analysis and machine learning tools to improve the accuracy and efficiency of cancer diagnosis, thus aiding in improving the quality of both cancer research and clinical practice. Our specific aims are as follows: 1. Regions of Interest: Produce a ROI-finder classifier and associated tools for use by researchers or pathologists for automatic identification of potential ROIs on whole slide images of breast biopsy slides; 2. Diagnosis: Produce a diagnostic classifier and associated tools that can not only suggest the potential diagnosis of a whole slide image, but can also produce the reasons for the diagnosis in terms of regions on the image, their color, their texture, and their structure; 3. Dissemination: Develop a unified software package containing this suite of tools, so they can be easily shared and provided (standalone and through the existing Pathology Image Informatics Platform (PIIP)) to both cancer researchers and clinical pathologists. 

For more detail information, please visit [our project website](cancertech.cs.washington.edu). 

# Installation

## Files

- `./YNet`: source code for [YNet](https://arxiv.org/abs/1806.01313)
- `./data`: contains a sample test image
- `./models`: contains pre-trained models
- `./output`: output files for all modules (contains pre-computed features for sample image)
- `./utils`: other supporting source code
- `ROIWindowClassifier.py`: Python script of a ROIWindowClassifier that detects diagnostically relevant regions of interest in whole slide images
- `ROISegmentation.py`: Python script of ROISegmentation module that performs semantic segmentatin on detected regions-of interest. 
- `Diagnosis.py`: Python script of Diagnoser that makes suggested diagnosis based on ROISegmentation result. 
- environment.yml: conda environment files



## Install Anaconda

You need to install Python and dependencies required to run the provided package. We use Anaconda to manage Python dependencies, and you can download the latest version of Anaconda with Python 3.6 or 3.7 from [here](https://www.anaconda.com/distribution/).

After installing Anaconda, you can install all the required packages by opening Anaconda Prompt. 

First, navigate to the folder of `cancer_diagnosis` . 

Then, find out the installation path of Anaconda using the command `where conda` in Anaconda Prompt. 

Open `environment.yml` in `cancer_diagnosis`. Scroll to the last line. Make sure the path to Anaconda (in the above example `C:\Anaconda` ) is the right prefix. Otherwise, replace the last line in the following format: 

```visual basic
prefix: [path_to_anaconda]\envs\cancer_env
```

Now, copy the following command into Anaconda Prompt:

```bash
conda env create -f requirements.yml
```

The installation may take around 10-20 minutes. After installation, you can proceed to Step 1. 



## Install Cuda (Optional)

In the semantic segmentation part, we will use Convolutional Neural Networks to analyze the input ROI images, and this slow process can be accelerated by using Nvidia GPUs.
If you have an Nvidia GPU in your computer, you can [download](https://developer.nvidia.com/cuda-downloads) and install CUDA 10.2 before running our programs.



# Step 1: Get ROI from Whole Slide Image

In Anaconda prompt, type in the following command:

```bash
conda activate cancer_env
python ROIWIndowClassifier.py
```



Then, you will see the following interface where you can provide the path to a pretrained model, breast biopsy image and output directory.

<img src="./images/ROI_main.jpg" alt="Image description" style="zoom:50%;" />

First, Click on "Select Pre-trained Model Path" to select `cancer_diagnosis/models`, which contains `kmeans.pkl`,  `hcluster.pkl` and `clf.pkl`. Then, Click on "Select Input Image" to select the image you want to identify regions-of-interest from (default is `cancer_diagnosis/data/1180_crop.jpg`). You can select one image at a time. Note: This demo is only designed to handle images with a size of fewer than 2^64 pixels. 

### Example for ROI Identification

Here, we show the results for a cropped whole-slide image (`1180_crop.jpg`). 

<img src="./images/1180_crop.jpg" style="zoom:50%;" />

The ROIWIndowClassifier can finish processing this provided sample from scratch in about 20 minutes depending on the memory and CPU capacity. To speed up the testing process, the pre-computed feature file `1180_crop_feat.pkl` is included in the package. The progress bar in the interface also shows the progress of feature computation.

In the end, the main window would display the ROI identification results where the regions-of-interest are marked in red boxes.

<img src="./images/ROI_result.jpg" style="zoom:50%;" />

##### Output Files

In the end, the following files will be generated in the output folder provided, which can later be used for ROI segmentation and diagnosis prediction. 

<img src="./images/ROI_files.JPG" style="zoom:100%;" />

The files are:

- **<u>`1180_crop_0.jpg`</u>**: the resultant ROI image (input to ROI Segmentation)
- 1180_crop_feat.pkl: the saved feature from the input image (for future use)
- 1180_crop_marked.jpg: regions-of-interest visualization with identified regions marked in red boxes



# Step 2: ROI Segmentations

After you have the resultant ROI image from Step 1, we can now move to ROI segmentations. The following figures illustrate how to use this tool to get segmentation for ROIs. The segmentation results will be used as features for diagnosis prediction. 

In Anaconda prompt, type in the following command:

```bash
python ROISegmentation.py
```

Then you will see the following interface where you can select the model, ROI image, and output directory.

<img src="./images/ynet_page1.JPG" width="70%" align="middle"/>



You should choose one or more ROI images at a time as shown below. Hold the "Control" key if you want to select multiple files, which is the standard multi-file selection in Windows OS. 
Similarly, you can choose the desired output folder by clicking the "Select Output Directory" button and choose a folder.



You can also change the parameter for "Batch Size" by using the slider, where the batch size is a term used in machine learning and refers to the number of samples processed in one iteration. When the computer has lots of memory or a large GPU, you can use a larger batch size. Usually, large batch size can make the CNN runs faster, but a large batch size would require lots of memory. We recommend to you the default setting unless your computer memory is too low or too high. If you saw any kind of memory error printed by the GUI, then restart the process with smaller batch size.

At last, click on the "Begin Segmentation" button to process these ROI images. Depends on your computer hardware (memory, GPU, etc) and the size of ROI, it usually takes 2 to 20 minutes to process each ROI on a GPU. 
CPUs are usually more than 10x slower than GPUs for deep learning, and we do not recommend users to use CPU for this step (i.e. ROI segmentation).

<!--
If the ROI contains more than 1 billion pixels, then it might take over an hour to process the ROI without GPU acceleration. Note that an 8GB GPU usually runs 10x faster than an i7 CPU.
-->

The 8 semantic segmentation classes  are:

1. Backgrounds
2. Benign Epithelium
3. malignant Epithelium
4. Normal Stroma
5. Desmoplastic Stroma
6. Secretion
7. Blood
8. Necrosis


## Examples for ROI Segmentation

Here, we show the results for the a ROI, which are extracted from the first step. 

### Input ROIs

This ROI is chosen by the "Select Images".

<img src="./images/1180_crop_0.jpg" width="40%" align="middle"/>

### Output Files

The CNN can finish processing these two ROIs after a 1-hour process on CPU or 10-minute process on GPU (Nvidia GTX 1080 Ti). Note that there are over 0.8 billion pixels in these two ROIs.
The following files are generated in the output folder, which can be used for visualization and diagnosis prediction.

<img src="./images/segmentation_output.JPG" width="40%" align="middle"/>

The files are:

* `_seg_label.png` (segmentation label), where the masks are stored.
* `_seg_viz.png` (segmentation visualization)
* `csv` (features for all tiles) files.

### Visualize Segmentation Mask

Here we show the segmentation visualization images. 

<img src="./images/1180_crop_0_seg_viz.png" width="40%" align="middle"/>



# Step 3: Diagnosis based on Segmentation Results

The ROI finder (in Step 1 above) usually selects more than one ROI for each subject, because there are often several interesting locations for analysis. The diagnosis prediction is given for each ROI, and the final diagnosis prediction for the subject is the maximum of diagnosis of all ROIs.

In Anaconda prompt, type in the following command:

```bash
python Diagnosis.py
```

Then you will see the following interface, where you can select the CSV files generated from the previous step (i.e. ROI segmentation).
Similar to the previous step, you should select all CSV files in the same window by holding the "Control" key.

<img src="./images/dx_page1.JPG" width="70%" align="middle"/>

Then, you can click on the "Begin Classification" button, and the program will produce the result in another window.
Note that this step is super fast, which takes less than 10 seconds for each CSV file.

<img src="./images/dx_page2.JPG" width="70%" align="middle"/>

Here, the red font shows the prediction for the input CSV files.
Again, we will choose the largest value (worst diagnosis prediction) as the final prediction for the subject if multiple ROIs are extracted from Step 1.

<img src="./images/dx_page3.JPG" width="70%" align="middle"/>