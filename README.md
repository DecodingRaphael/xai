# Raphael Painting Analysis with Explainable AI

This project uses explainable AI techniques to understand what makes a Raphael painting distinctively a "Raphael." By applying advanced visualization techniques to a deep learning model, we can literally see what aspects of paintings the model focuses on when making classification decisions.

## Background

What makes a Raphael painting a Raphael? This question is central to art authentication and attribution, but traditionally relies heavily on expert connoisseurship. Recent advances in deep learning have shown promising results in automated art classification, but these models act as "black boxes" - they make decisions without revealing their reasoning.

This project extends research from [Ugail et al. (2023)](https://www.nature.com/articles/s40494-023-01094-0), which used ResNet50 and Support Vector Machines to classify Raphael paintings with 98% accuracy [Associated Github](https://github.com/ugail/RaphaelHeritageSciencePaper). While their model was effective, it couldn't explain *why* it identified a painting as a Raphael.

By applying explainable AI techniques, we can now visualize which aspects of paintings - from composition to brushwork details - influence the model's decision. This helps answer the fundamental question: what distinctive features characterize Raphael's work according to AI?

## Setup

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/raphael-xai.git
   cd raphael-xai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies:
   - tensorflow
   - numpy
   - pandas
   - dianna
   - opencv-python
   - scikit-image
   - matplotlib
   - joblib
   - diskcache

3. Data organization (go to [Github](https://github.com/ugail/RaphaelHeritageSciencePaper) for download options):
   - Place Raphael paintings in `data/Raphael/`
   - Place non-Raphael paintings in `data/Not Raphael/`
   - Example paintings are included in the `data/` directory

4. Pre-trained models (go to [Github](https://github.com/ugail/RaphaelHeritageSciencePaper) for donwload options):
   - The repository includes pre-trained models in the `models/` directory:
     - `resnet50_model.h5`: The ResNet50 model for feature extraction
     - `28_09_2023_svm_final_model.pkl`: The SVM classifier

### Hardware Requirements

- 8GB RAM minimum (16GB recommended)
- GPU recommended but not required
- Expect longer processing times without GPU acceleration
- Processing a single painting with 50 masks takes approximately 2-5 minutes on a standard CPU

### Usage

Run the main analysis script:
```bash
python rise_imagenet.py
```

This will:
1. Process the example painting (`data/0_Edinburgh_Nat_Gallery.jpg`)
2. Generate visualizations using 50 masks, 0.3 keep ratio, and feature resolution of 6. These parameters can be modified in the script for increasing accuracy.
3. Run the analysis 3 times to ensure stability
4. Integrate results from the 3 runs
5. Save all outputs to the `output/` directory

You can modify the parameters in the script to analyze different paintings or adjust the analysis settings.

## XAI with DIANNA

This project uses the [DIANNA](https://dianna.readthedocs.io/en/latest/) explainable AI package to implement the RISE (Random Input Sampling for Explanation) technique. DIANNA provides tools for visualizing which parts of an input image influence a model's decision.

### How RISE Works

1. **Masking**: RISE randomly masks portions of the input image
2. **Model Prediction**: The model makes predictions on each masked version
3. **Aggregation**: By correlating masks with model outputs, we generate heatmaps showing which regions influence decisions
4. **Integration**: Running multiple times and aggregating results provides more stable explanations

The masking approach reveals which elements of Raphael's paintings are most distinctive according to the model - potentially identifying unique brushwork patterns, composition elements, or color choices that characterize his style.

## Structure and Contents

### model.py

This file implements the classification model:
- Loads pre-trained ResNet50 for feature extraction
- Implements the Support Vector Machine (SVM) classifier
- Contains functions for edge detection and feature calculation (Canny, Sobel, Laplacian, and Scharr)
- Handles different image formats and preprocessing
- Includes a caching mechanism to improve performance

### rise_imagenet.py

This file implements the explanation generation:
- `custom_rise()`: Custom implementation of RISE for our specific model
- `explain_painting()`: Processes a painting and generates explanations
- `integrate_results()`: Combines multiple runs for more stable explanations
- `visualize_edge_heatmap_overlay()`: Creates edge-enhanced visualizations
- `calculate_clarity_metrics()`: Quantifies the quality of explanations

The workflow is as follows:
1. Load and preprocess the image
2. Apply random masks to the image
3. Get model predictions for each masked version
4. Correlate masks with predictions to create relevance maps
5. Generate various visualizations
6. Calculate metrics to quantify explanation quality
7. Integrate results across multiple runs

## Results Interpretation

The project generates several types of visualizations that help interpret what the model has learned about Raphael's distinctive style.

### Standard Heatmaps

The basic heatmaps show which regions influence classification:
- **Raphael Heatmap**: Red/yellow areas strongly indicate Raphael's style
- **Non-Raphael Heatmap**: Red/yellow areas strongly indicate non-Raphael features

For example, in the Edinburgh National Gallery painting analysis, the model focuses on facial features and hand positions when identifying Raphael's style.

#### Example: Raphael Heatmap
![Raphael Heatmap](output/integrated/visualizations/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_mean_Raphael.png)

#### Example: Non-Raphael Heatmap
![Non-Raphael Heatmap](output/integrated/visualizations/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_mean_Non-Raphael.png)

### Edge-Enhanced Visualizations

These visualizations highlight brushwork patterns within important regions:
- White edges show brushstrokes the model finds significant
- Brighter edges indicate more influential brushwork patterns
- High edge density shows areas with complex brushwork that influence decisions

The edge-enhanced visualizations reveal that the model identifies Raphael's distinctive brushwork in areas such as fabric folds, facial details, and background elements.

#### Example: Edge-Enhanced Raphael Features
![Edge-Enhanced Raphael](output/integrated/visualizations/edge_analysis/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_Raphael_combined_edges_combined.png)

#### Example: Edge-Enhanced Non-Raphael Features
![Edge-Enhanced Non-Raphael](output/integrated/visualizations/edge_analysis/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_Non-Raphael_combined_edges_combined.png)

### Difference Maps

These show regions that distinguish Raphael from non-Raphael paintings:
- Red areas are distinctively characteristic of Raphael
- Blue areas are more characteristic of non-Raphael works
- White/neutral areas have minimal influence on classification

Difference maps help isolate the most discriminative features between Raphael and non-Raphael styles.

#### Example: Difference Map
![Difference Map](output/integrated/visualizations/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_difference_map.png)

#### Example: Edge-Enhanced Difference Map
![Edge-Enhanced Difference Map](output/integrated/visualizations/edge_analysis/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_difference_combined_edges_combined.png)

### Confidence Maps

These show stable features across multiple runs:
- Bright red/yellow indicates high confidence features
- Medium orange shows moderate confidence
- Dark blue/green represents low confidence or high variation

Confidence maps help identify which features the model consistently uses across multiple runs, suggesting these are reliable indicators of Raphael's style.

#### Example: Raphael Confidence Map
![Raphael Confidence Map](output/integrated/visualizations/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_confidence_Raphael.png)

#### Example: Non-Raphael Confidence Map
![Non-Raphael Confidence Map](output/integrated/visualizations/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_confidence_Non-Raphael.png)

### Clarity Metrics

The analysis generates several metrics to quantify the quality of explanations:
- **Contrast Ratio**: Higher values indicate clearer distinction between relevant and non-relevant areas
- **Overlap IoU**: Measures how much overlap exists between Raphael and non-Raphael heatmaps (lower is better)
- **Entropy**: Measures how focused or diffused the attention is (lower means more focused)
- **Map Correlation**: Correlation between Raphael and non-Raphael maps (lower means better differentiation)
- **Clarity Score**: Combined score indicating overall explanation quality

For detailed interpretation guidance, see the [Color Interpretation Guide](docs/Color%20Interpretation%20Guide%20for%20Raphael.md).

## Example Results

The analysis of "Edinburgh National Gallery" painting (0_Edinburgh_Nat_Gallery.jpg) produced the following key findings:

1. **Brushstroke Analysis**: The edge-enhanced visualizations reveal that the model identifies distinctive brushwork patterns in the drapery and facial features as characteristic of Raphael.

2. **Feature Importance**: The standard heatmaps show that the model focuses strongly on the Madonna's face, the Christ child, and the interaction between them - suggesting that Raphael's handling of these elements is distinctive.

3. **Confidence Analysis**: The confidence maps indicate that the model consistently identifies certain areas (like the Madonna's face) across multiple runs, suggesting these are reliable indicators of Raphael's style.

4. **Metrics**: The clarity metrics show moderate contrast (0.31 for Raphael features) and high overlap (0.98 IoU), indicating that while the model can identify Raphael features, there's significant overlap with non-Raphael features in this painting.

### Visual Interpretation Guide

When interpreting the visualizations, look for:

1. **Areas of High Attention**: Bright red/yellow regions in heatmaps indicate areas the model focuses on strongly.
   
   ![Standard RISE Map](output/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_0_Raphael.png)

2. **Brushwork Patterns**: White edges in edge-enhanced visualizations show brushstrokes the model finds distinctive.
   
   ![Edge-Enhanced Visualization](output/integrated/visualizations/edge_analysis/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_Raphael_sobel_edges_combined.png)

3. **Distinctive Features**: Red areas in difference maps highlight features characteristic of Raphael.
   
   ![Difference Map](output/integrated/visualizations/edge_analysis/0_Edinburgh_Nat_Gallery.jpg_nmasks_50_pkeep_0.3_res_6_difference_sobel_edges_combined.png)

4. **Consistency**: Compare visualizations across multiple runs to identify stable patterns.

## Future Work

Potential directions for extending this research:

1. **Expanded Dataset**: Apply this analysis to a larger collection of Raphael and non-Raphael paintings to identify more general patterns.

2. **Feature Comparison**: Compare the features identified by the model with art historians' understanding of Raphael's style.

3. **Sequential Analysis**: Apply the technique to paintings from different periods of Raphael's career to track stylistic evolution.

4. **Model Comparison**: Compare multiple models to see if they focus on the same aspects of Raphael's style.

5. **Interactive Tool**: Develop an interactive tool allowing art historians to explore the visualizations and draw their own conclusions.

## Acknowledgments

This project builds on research by [Ugail et al. (2023)](https://www.nature.com/articles/s40494-023-01094-0) and uses the DIANNA explainable AI package for visualization.
