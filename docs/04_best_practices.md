# Best Practices for Using Brainways

Following these best practices will help you get the most accurate and reliable results from Brainways. These tips are designed for all users, even those without a technical background.

## 1. Carefully Check Slice Registrations

Accurate registration of your brain slices to the 3D atlas is crucial for reliable results.

*   **Visually Inspect Each Slice:** After Brainways performs the automatic registration steps (Atlas, Rigid, Non-rigid), take the time to visually inspect each slice.
*   **Use the Sliders:** Look at the overlay of the atlas on your slice image in the "Atlas registration", "Rigid registration", and "Non-rigid registration" steps. Use the sliders and control points provided in each step to make fine adjustments until the alignment looks correct to your eye. Pay attention to key anatomical landmarks.
*   **Don't Rush:** It might seem tedious, but ensuring good alignment at this stage prevents errors later on.

## 2. Train Your Own Cell Detection Model

While Brainways comes with a default cell detection model, every experiment and imaging setup is slightly different. For the best cell detection results, you should train a model specifically on your own data.

*   **Why Train Your Own Model?** Your imaging conditions (e.g., microscope, staining, tissue quality) might differ from the data used for the default model. A custom model learns the specific appearance of cells in *your* images, leading to more accurate detections.
*   **Follow the Guide:** We have a detailed guide on how to annotate your images and train a custom StarDist model. Please refer to the [Cell Detection Instructions](03_cell_detection.md) for step-by-step guidance.
*   **Invest the Time:** Training a custom model requires some initial effort to annotate images, but it significantly improves the quality of your cell counts.

## 3. Verify Results After Analysis

Statistical analysis might highlight specific brain regions as being particularly relevant to your experimental conditions. However, statistics don't tell the whole story. It's important to go back and visually verify these findings.

*   **Export Annotated Regions:** After running ANOVA or PLS analysis and identifying potentially relevant regions, use the `Analysis -> Export Annotated Region Images` feature. This allows you to select specific brain regions and export images of those regions, overlaid with the atlas boundaries and detected cells for each relevant slice.
*   **Manual Inspection:** Carefully examine these exported images.
    *   Check Registration: Does the atlas overlay still look correctly aligned in these specific regions of interest? Sometimes, small misalignments that weren't obvious before become apparent when focusing on a specific area.
    *   Check Cell Detection: Are the detected cells (often shown as dots or outlines) accurately marking the real cells in the image? Are there many false positives (detections where there's no cell) or false negatives (missed cells)?
*   **Iterate if Necessary:**
    *   If you find registration issues in these key regions, go back to the registration steps for the affected slices and correct the alignment.
    *   If you find problems with cell detection (e.g., the model consistently misses cells in a particular region or condition), consider annotating more image crops from these problematic areas and retraining your custom StarDist model, as described in the [Cell Detection Instructions](03_cell_detection.md).

By following these steps, you can be more confident in the biological conclusions drawn from your Brainways analysis.
