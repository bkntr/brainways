# Getting Started

## Requirements

* Installed [python](https://www.python.org/downloads/) 3.9-3.12.

!!! tip
    We recommend installing brainways in a separated virtual environment, for example using [anaconda](https://docs.anaconda.com/free/anaconda/install/).

!!! tip
    If you intend to use Cell Detection features in Brainways, it is highly recommended to install StarDist with GPU support. To do this, first install TensorFlow with GPU support before installing brainways (brainways install StarDist as a dependency). Please refer to the [TensorFlow](https://www.tensorflow.org/install/pip) website for detailed instructions, then proceed with brainways installation.

## 1. Install Brainways

Run the following command (if you created a virtual environment, run this command inside the virtual environment):

```bash
pip install brainways
```

This will install the brainways GUI along with all of its dependencies. After this command successfully finished, you are ready to use brainways.

## 2. Launch Brainways

Launch brainways using the following command (if you installed brainways in a virtual environment, execute this command inside the virtual environment):

```bash
brainways ui
```

The first time Brainways is launched, it will automatically download the following dependencies:

* [QuPath](https://qupath.github.io/), used in Brainways for its amazing image reading capabilities.
* Rat/mice 3d atlases (via [bg-atlasapi](https://github.com/brainglobe/bg-atlasapi)).
* Brainways automatic registration model weights.

## 3. Your First Brainways Project

If you have images waiting to be registered, you can create a [new brainways project](#creating-a-new-brainways-project). If you don't have your own images, or just want to jump right in, brainways comes with two [sample projects](#loading-a-sample-project).

### Loading a Sample Project

Brainways comes with two sample projects:

1. Sample project: contains sample images.
1. Annotated sample projects: contains fully-annotated and quantified sample images to quickly showcase all of the features that are enabled after fully annotating a project.

The sample projects can be loaded by using the following menu items: `File -> Open Sample -> Brainways -> Sample project / Annotated sample project`

![Loading sample project](./assets/load-demo.jpg){: width='70%' }

### Creating a New Brainways Project

1. In the right hand panel, click the `Project -> New` button.
1. In the file menu dialog, create your project in a **new empty directory**.
1. The "New Brainways Project" dialog will open with the following options:
    * **Atlas:** Choose the atlas you'd like to register to (*whs_sd_rat_39um* comes with an automatic registration model, but other atlases can be used for manual registration).
    * **Condition types:** the condition types in your experiment. Each subject can be assigned a condition for each of the given condition types, for example "age" and "sex". If you have more than one condition type, they should be separated by a comma, for example: `age;sex`

#### Adding Subjects

![Create subject dialog](./assets/new_subject.jpg)

1. Click the `Add Subject` button and insert a Subject ID.
1. If condition types were given, fill in the experimental conditions (for example `age=juvenile`, `sex=male`).
1. Click the "Add Image(s)" button and choose one or more slice image files to add to the current subject (add all slice images that belong to a single brain. We do not recommend mixing different brains in a single subject).
1. For multi-channel images, choose the channel with the fluorescent marker to be quantified.
1. To finish creating a subject, click the "Create" button.
1. Follow the previous steps again to add all subjects in your experiment.

## 4. Working with Brainways

### Slice registration to the 3D atlas

Once a new project is created, register all the slices to the 3D atlas by following the registration steps for each slice image.

!!! tip
    All registration sliders can be quickly adjusted using keyboard shortcuts. To see a full list of the keyboard shortcuts for each step, click "?" on your keyboard.

#### Atlas registration

In the "Atlas registration" panel, you will see the slice image on the right and an atlas slice image on the left. Adjust the sliders on the right panel until your slice matches the atlas slice. If only one of the hemispheres is visible, choose the visible hemisphere in the "Hem" box.

If an auomatic registration algorithm is available for the current atlas, it will be automatically run on every image upon image opening.

![Atlas registration](./assets/atlas-reg.gif)

#### Rigid registration

In the "Rigid registration" panel, adjust the sliders until the slice image roughly matches the atlas annotated that is overlayed on the image.

![Rigid registration](./assets/affine-reg.gif)

#### Non-rigid registration

To accomodate subtle differences between different brains, the images can be elastically deformed to perfectly match the atlas slice. To do that, move the blue dots until the underlying structures matches the atlas annotation overlay. The "Elastix" button can be used to perform automatic elastic registration using the Elastix algorithm.

![Non-rigid registration](./assets/non-rigid-reg.gif)

### Cell detection

!!! tip
    To get satisfactory cell detection results, it is highly recommended to train a custom StarDist model on data from your experiment. Instructions detailing how to do it can be [found here](03_cell_detection.md).

!!! tip
    If you intend to use Cell Detection features in Brainways, it is highly recommended to install brainways with GPU support. To do this, first install TensorFlow with GPU support before installing brainways. Please refer to the [TensorFlow](https://www.tensorflow.org/install/pip) website for detailed instructions, then proceed with brainways installation.

Active cells can be detected for each image using the [StarDist](https://github.com/stardist/stardist) cell detection algorithm. Because cell detection is performed on the full resolution images and can take a significant amount of time, detections can be run on a small preview crop to make sure that it works properly. To run cell detection on a preview crop, double-click on the relevant area in your image, and then click on the "Run on preview" button. If the detections are not satisfactory, it is recommended to train your own model following [these instructions](03_cell_detection.md). Image normalization parameters can be adjusted to try and achieve better results with the default StarDist pretrained model, but in our experience, it is very difficult to achieve good results without training a custom model on our data.

The "unique" check box can be used to determine whether the adjusted normalization parameters will be used only for this image or for the whole project:
* If the "unique" checkbox is ticked, the parameters will only be used for this image.
* If the "unique" checkbox is not ticked, the parameters will be used for the whole project.

After verifying the cell detection algorithm works well, run the cell detector for the entire project by clicking the "Run Cell Detector" button. Note that it may take a few hours for a big project.

![Cell detection](./assets/cell-detection.jpg)

### Outputs

#### Export Excel with Cell Density per Region

Brainways allows you to export cell density data for each brain region and animal to an Excel file. Follow the steps below:

1. **Open the Calculate Results Dialog:**
   - Navigate to the `Analysis` section and click the `Calculate results` button.

2. **Configure Calculation Options:**
   - In the calculation dialog, you will need to specify the following options:
     - **Min Structure Square Area (μm):** Minimum area (in µm²) for regions to be included in the excel.
     - **Cells Per Square Area (μm):** Excel will output cell density per square area (raw cell counts will also be provided).
     - **Min Cell Area (μm):** Filter out detected cells with area smaller than this value.
     - **Max Cell Area (μm):** Filter out detected cells with area larger than this value.
     - **Excel Mode:** Select the mode for the Excel output. Options include:
       - `Row per Subject`: One row per subject.
       - `Row per Slice`: One row per slice image.

3. **Calculate Results:**
   - After configuring the options, click the `OK` button to start the calculation process. A progress bar will indicate the status of the calculation.

4. **Verify Exported Files:**
   - Once the calculation is complete, navigate to the specified output directory to find the exported Excel file. The file will contain cell density data for each brain region and animal.

#### Export Registered Annotation Masks

The Export Registered Annotation Masks feature allows you to export the registered annotations of your brain slices to various file formats. Follow the steps below to use this feature:

1. **Open the Export Dialog:**
   - Navigate to the `Analysis` section and click the `Export Registered Annotation Masks` button.

2. **Configure Export Options:**
   - In the export dialog, you will need to specify the following options:
     - **Output Directory:** The directory where the registered annotation masks will be saved.
     - **Slice Selection:** Choose which slices to export. Options include:
       - `Current Slice`: Export the currently selected slice.
       - `Current Subject`: Export all slices of the currently selected subject.
       - `All Subjects`: Export all slices of all subjects in the project.
     - **File Format:** Select the file format for the exported masks. Supported formats include:
       - `CSV`
       - `NPZ`
       - `MAT`

3. **Export the Masks:**
   - After configuring the options, click the `OK` button to start the export process. A progress bar will indicate the status of the export.

4. **Verify Exported Files:**
   - Once the export is complete, navigate to the specified output directory to find the exported files. Each file will be named according to the slice image name and saved in the chosen format.

### Analysis

Once all images are annotated and cell detection is run on all images, statistical analysis can be performed to find neural patterns which differentiate between experimental conditions.

Several kinds of analyses are supported:

1. ANOVA contrast analysis
1. PLS (Partial Least Squares) contrast analysis
1. Network graph analysis

#### ANOVA contrast analysis

ANOVA contrast analysis can be performed through Brainways to identify and visualize the brain regions that contributed to the contrast between different experimental conditions. ANOVA is performed on each brain region separately, followed by FDR correction for multiple comparisons. Other multiple comparison methods can be used (see full list of multiple comparison methods [here](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html)). Regions that significantly contributed to the contrast are subjected to post hoc analysis to identify the particular differences between condition pairs. To display a posthoc statistical analysis between two conditions, click the "Show Posthoc" button. All ANOVA and posthoc values are saved to an excel file in the project directory under:

```
<PROJECT_DIR>/contrast-<CONDITION>-cells.xlsx
```

![Analysis](./assets/analysis.jpg)

#### PLS (Partial Least Squares) contrast analysis

Task PLS is a multivariate statistical technique that is used to identify optimal neural activity patterns that differentiate between experimental conditions. PLS produces a set of mutually orthogonal pairs of latent variables (LV). One element of the LV depicts the contrast, which reflects a commonality or difference between conditions. The other element of the LV, the relative contribution of each brain region (termed here ‘salience’), identifies brain regions that show the activation profile across tasks, indicating which brain areas are maximally expressed in a particular LV. Statistical assessment of PLS is performed using permutation testing for LVs and bootstrap estimation of standard error for the brain region saliences. The significance of latent variables is assessed by permutation testing. The reliability of the salience of the brain region is assessed using bootstrap estimation of standard error. Brain regions with a bootstrap ratio >2.57 (roughly corresponding to a confidence interval of 99%) are considered to be reliably contributing to the pattern. Missing values are interpolated by the average for the test condition. Brainways uses the [pyls](https://github.com/rmarkello/pyls) Python package to perform the PLS analysis.

The results of the analysis are exported to an Excel file and to PNG files containing the salience plot, the LV p values and the LV contrast direction. The files can be found in the project directory under:

```
<PROJECT_DIR>/__outputs__/PLS/
```

#### Network graph analysis

To examine the functional connectivity between the different brain regions, Brainways can be used to create a network graph based on inter-region positive cell count correlation matrices. Network nodes consist of the different brain regions as defined in the atlas, and the edges of the network consist of significant correlations between the regions (the significance threshold can be adjusted by the user, by default p<0.05), based on Pearson’s pairwise correlation. The values are FDR corrected for multiple comparisons. The network graph is exported to a graphml file, and can be used with any graph analysis tools and algorithms for further analysis. The file can be found in the project directory under:

```
<PROJECT_DIR>/__outputs__/network_graph/
```
