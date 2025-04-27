# Cell Detection Instructions

Welcome to the cell detection guide! This document will walk you through the steps needed to detect cells using the StarDist tool.

## Prerequisites

Before you begin, ensure you have the following:

-   A computer with GPU support (recommended for faster processing).
-   Optional: The Brainways software [installed](02_getting_started.md) (which includes StarDist).

## Step-by-Step Instructions

### 1. Install StarDist

StarDist is a powerful tool for cell detection. You can find the installation instructions [here](https://github.com/stardist/stardist?tab=readme-ov-file#installation). If you have already installed Brainways with GPU support, you do not need to install StarDist separately.

### 2. Annotate Your Images

To train the StarDist model, you need to annotate your slice images. Follow [these instructions](https://github.com/stardist/stardist?tab=readme-ov-file#annotating-images) from StarDist. Typically, annotating 20 crops with a few dozen cells/nuclei in each crop yields good results.

### 3. Train Your StarDist Model

Use the provided [Jupyter Notebooks](https://github.com/stardist/stardist/tree/main/examples/2D) to train your own StarDist model. These notebooks will guide you through the training process step-by-step.

### 4. Create a New Brainways Project

When setting up a new Brainways project, select the directory containing your custom-trained StarDist model in the "Custom StarDist Model" option. If a Brainways project was already created, you can edit the project settings instead by clicking the "Edit" button under the "Project" section.

### 5. Test Your Model

1.  **Run on Preview**:

    -   Go to the "Cell Detection" step in Brainways.
    -   Click the "Run on preview" button. This will run the model on a small crop to verify that the model is working as expected.

2.  **Test on Multiple Crops**:

    -   Run your project on several crops to ensure the model works well on diverse scenarios relevant to your project.

3.  **Identify and Address Issues**:
    -   If you identify any issues, create more crops that are similar to the problematic scenario.
    -   Retrain the model using these new crops.

### 6. Run the Cell Detector on the Entire Project

Once you have verified that the model works well on a few images, you can run the cell detector on the entire project. Click the "Run Cell Detector" button to start the process.

## Tips for Success

-   **Annotation Quality**: The quality of your annotations significantly impacts the model's performance. Take your time to annotate accurately.
-   **Model Training**: Training the model might take some time, especially on large datasets. If you have a GPU, make sure you installed StarDist with GPU support, this will very significantly reduce model training and inference times.
-   **Validation**: Always validate your model on a few images before running it on the entire project to ensure it performs as expected. If you identify any issues, create more crops that are similar to the problematic scenario and retrain the model.

## Troubleshooting

If you encounter any issues, consider the following:

-   **Installation Problems**: Ensure all dependencies are installed correctly. Refer to the StarDist installation guide for troubleshooting tips.
-   **Model Performance**: If the model is not performing well in some scenarios, consider annotating more images and retrain your model.
