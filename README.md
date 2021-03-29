# A Unified Software Package for Cancer Diagnosis

This research is supported by the National Cancer Institute grant U01 CA231782, PI: Linda G. Shapiro, co-PI: Joann G. Elmore

The long-term goal of this project is to develop a unified software package for sharing image analysis and machine learning tools to improve the accuracy and efficiency of cancer diagnosis, thus aiding in improving the quality of both cancer research and clinical practice. Our specific aims are as follows: 1. Regions of Interest: Produce a ROI-finder classifier and associated tools for use by researchers or pathologists for automatic identification of potential ROIs on whole slide images of breast biopsy slides; 2. Diagnosis: Produce a diagnostic classifier and associated tools that can not only suggest the potential diagnosis of a whole slide image, but can also produce the reasons for the diagnosis in terms of regions on the image, their color, their texture, and their structure; 3. Dissemination: Develop a unified software package containing this suite of tools, so they can be easily shared and provided (standalone and through the existing Pathology Image Informatics Platform (PIIP)) to both cancer researchers and clinical pathologists. 

For more detail information, please visit <a href="http://cancertech.cs.washington.edu" target="_blank">our project website</a>.



## **Installation**

The installation instructions are shown in Windows operation system. MacOS or Linux are not fully supported yet.

### **Sedeen Viewer**
To use the plugins, you need to download [Sedeen Viewer](https://pathcore.com/sedeen) first.

### **Plugins and supporting files**

Download our tools from [Github page](https://github.com/cancertech/cancer_diagnosis) by clicking on the "Clone or download" button first and then clicking on the "Download ZIP" button.

<img src="docs/img_sedeen/download_repo.PNG" width="40%" align="middle"/>

If you have already installed Anaconda and set up the conda environment, please skip to [Create environment variable](#create-environment-variable).

Unzip `cancer_diagnosis-master.zip`, you will see the following folders:

- YNet: source code for <a href="https://arxiv.org/abs/1806.01313" target="_blank">YNet</a> 
- data: contains a sample test image
- models: contains pre-trained models
- output: output files for all modules (contains pre-computed features for sample image)
- utils: other supporting source code

The unzipped folder can either be named as "cancer_diagnosis" or "cancer_diagnosis-master", which will not affect how the program runs.

### **Install Anaconda**

You need to install Python and dependencies required to run the provided package. We use Anaconda to manage Python dependencies, and you can download the latest version of Anaconda with Python 3.7 or 3.8 from 
<a href="https://www.anaconda.com/distribution/" target="_blank">here</a> .

You should follow the instructions as shown in the screenshots below. **Pay attention to the buttons marked with red ink.
Installing Anaconda for all users to the "C:/ProgramData/Anaconda3/" path can make the program running smoothly.**

<!-- <img src="docs/tutorial_img/anaconda_1.JPG" width="50%" align="middle"/> 	 -->
<img src="docs/tutorial_img/anaconda_2.JPG" width="50%" align="middle"/>
<img src="docs/tutorial_img/anaconda_3.JPG" width="50%" align="middle"/>
<img src="docs/tutorial_img/anaconda_4.JPG" width="50%" align="middle"/>
<img src="docs/tutorial_img/anaconda_5.JPG" width="50%" align="middle"/>

<br><br>

### **Install Dependencies**

After installing Anaconda, you can install all the required packages by double clicking on the `0_install_dependencies.bat` file, as shown below.

<img src="docs/img_sedeen/install_packages.JPG" width="50%" align="middle"/>

The installation may take around 10-20 minutes. After installation, you can proceed to tutorial.


If you see a "Windows protected your PC" window as below. You can first click on the "More Info" button and then "Run anyway" button to allow our program to run. 

<img src="docs/tutorial_img/windows_protect.png" width="100%" align="middle"/>

When the installation is done, you can see a similar message as shown below.

<img src="docs/tutorial_img/package_install_done.JPG" width="50%" align="middle"/>


<br><br>

### **(Optional) Install CUDA for Nvidia GPU Only**

In the semantic segmentation part, we will use Convolutional Neural Networks to analyze the input ROI images, and this slow process can be accelerated by using Nvidia GPUs.
If you have an Nvidia GPU in your computer, you can 
<a href="https://developer.nvidia.com/cuda-downloads" target="_blank">download</a>
and install CUDA 10.2 before running our programs.

### **Supported Image Formats**

We support PNG, JPG, TIFF, SVS, and many other image formats for MacOS and Linux. Unfortunately, we do not support SVS for Windows machine at the current stage, but you can easily convert your SVS files to JPG files by using
<a href="https://www.reaconverter.com/convert/svs_to_jpg.html" target="_blank">this converter</a>.

### **Create environment variable**

1. Right-click the **Computer** icon and choose **Properties**, or in Windows Control Panel, choose **System**.
2. Choose **Advanced system settings**.

<img src="docs/img_sedeen/advanced_system_settings.JPG" width="50%" align="middle"/>

3. On the Advanced tab, click **Environment Variables**.

<img src="docs/img_sedeen/environment_variables.JPG" width="50%" align="middle"/>

4. Click **New** under the **User variables** to create a new user variable.

```
Variable name: SedeenPythonHome
Variable value: "Full path of cancer_env"
```

Note that the path of **cancer_env** is usually *C:\Users\%Username%\.conda\envs\cancer_env*. If you're not sure where the conda env is, you can launch the command lines by *Win+R*, open **cmd**, and run 

```
conda env list
```

The path listed beside **cancer_env** is the value you should put in user variable.

<img src="docs/img_sedeen/cmd.JPG" width="50%" align="middle"/>
<img src="docs/img_sedeen/env_list.JPG" width="50%" align="middle"/>

<br><br>
5. Apply this change and reboot your computer.

### **Install Sedeen Plugins**

Copy the folder [ITCR](./ITCR) to `%Sedeen Viewer Folder%\plugins\cpp`.

`%Sedeen Viewer Folder%` is where Sedeen is installed.

<img src="docs/img_sedeen/plugins.JPG" width="50%" align="middle"/>
<br><br>

**Now You can start using our plugins.**

<br><br>
## **Tutorial**

### **Open Sedeen Viewer**
To use our plugins in `Sedeen Viewer`, you need to launch Sedeen in the conda environment `cancer_env`. **To make sure the plugins are running successfully, it's highly recommended that you launch a new `Sedeen Viewer` everytime you run a plugin.**

1. Launch command lines with *Win+R*, open **cmd**.

<img src="docs/img_sedeen/cmd.JPG" width="50%" align="middle"/>

2. Change directory to where Sedeen Viewer is installed by

```
cd C:\Program Files\Sedeen Viewer
```

Note that the directory could change, depending on where you install it.

3. Switch to the environment by

```
conda activate cancer_env
```
<img src="docs/img_sedeen/launch_sedeen.JPG" width="50%" align="middle"/>

4. Open Sedeen Viewer with

```
sedeen.exe
```

### **ROI Window Classifier**

1. Open the image you want to process with File->Open.

<img src="docs/img_sedeen/Open_image.JPG" width="70%" align="middle"/>

2. Select `ROI Window Classifier` from the algorithm option box in `Analysis Manager` window. If the window is not showing, you can go to View->Windows->Analysis Manager to get it displayed.

3. Choose an output path from the `Output Path` box.

<img src="docs/img_sedeen/alg.JPG" width="70%" align="middle"/>

4. Click `Run` and select [Sedeen_ROIWindowClassifier.py](<./Sedeen Scripts/Sedeen_ROIWindowClassifier.py>)

<img src="docs/img_sedeen/ROIWindowClassifier.JPG" width="70%" align="middle"/>

5. If it's the first time you run this image, it may take more than 10 minutes, depending on the memory and CPU capacity.


After running the program, Sedeen Viewer would display the ROI identification results where the regions-of-interest are marked in green boxes.

<img src="docs/img_sedeen/ROIWindowClassifier_result.JPG" width="70%" align="middle"/>

The following files will be generated in the selected folder, which can later be used for ROI segmentation and diagnosis prediction.

<img src="docs/img_sedeen/ROIWindowClassifier_output.JPG" width="70%" align="middle"/>

### **ROI Segmentation**

1. Open an image in Sedeen Viewer and select `ROI Semantic Segmentation` from the algorithm option box. The image could be random as it's only for enabling the algorithm part.

2. Select one or more ROI images. Hold the "Control" key if you want to select multiple files, which is the standard multi-file selection in Windows OS. 
    
    You can also change the parameter for "Batch Size" by using the slider, where the batch size is a term used in machine learning and refers to the number of samples processed in one iteration. When the computer has lots of memory or a large GPU, you can use a larger batch size. Usually, large batch size can make the CNN runs faster, but a large batch size would require lots of memory. We recommend to you the default setting unless your computer memory is too low or too high. If you saw any kind of memory error printed by the GUI, then restart the process with smaller batch size.

3. Click `Run` and select [Sedeen_ROISegmentation.py](<./Sedeen Scripts/Sedeen_ROISegmentation.py>).

    Depends on your computer hardware (memory, GPU, etc) and the size of ROI, it usually takes 2 to 20 minutes to process each ROI on a GPU. CPUs are usually more than 10x slower than GPUs for deep learning, and we do not recommend users to use CPU for this step (i.e. ROI segmentation).

    The 8 semantic segmentation classes are:

    <img src="docs/tutorial_img/seg_color_map.png" width="50%" align="middle"/>

4. After running the program, a result panel will pop up, showing the segmentation results and the results in superpixels. You can click "Previous" or "Next" to view the corresponding segmentations of the selected images.

### **Diagnosis**

1. Open an image in Sedeen Viewer and select `ROI Diagnosis` from the algorithm option box. The image could be random as it's only for enabling the algorithm part.

2. Select the CSV files generated from the previous step (i.e. ROI Segmentation).

3. Click `Run` and select [Diagnosis.py](<./Sedeen Scripts/Sedeen_Diagnosis.py>).

    Note that this step is super fast, and you should see the the diagnosis prediction in the result panel in a short time.

<img src="docs/img_sedeen/diagnosis_result.jpg" width="50%" align="middle"/>

## **User Support**
If you have any questions, you can visit the [Github issue page]https://github.com/cancertech/cancer_diagnosis/issues) and submit an issue via the "New issue" button shown below.

<img src="docs/tutorial_img/user_issue.jpg" width="50%" align="middle"/>

