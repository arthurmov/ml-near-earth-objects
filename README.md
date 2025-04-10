# ml-near-earth-objects
### NASA Near-Earth Object Classifier

This project uses machine learning algorithms to classify whether a near-Earth object is hazardous based on physical and orbital characteristics. The dataset comes from a [Kaggle dataset](https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024) containing historical asteroid observations.

Developed as part of a CISC 3440 Machine Learning course.

---

### Dataset Features

The dataset (`train.csv`) contains the following columns:

| Column Name             | Description |
|--------------------------|-------------|
| `absolute_magnitude`     | Brightness of the object |
| `estimated_diameter_min` | Minimum estimated diameter |
| `estimated_diameter_max` | Maximum estimated diameter |
| `relative_velocity`      | Velocity relative to Earth |
| `miss_distance`          | Distance the object will miss Earth |
| `is_hazardous`           | Target label (1 if hazardous, 0 otherwise) |

---

### Models Implemented

#### Part 1: Perceptron Classifier
- Implemented from scratch in NumPy
- Trains on the feature set to classify objects as hazardous or not
- Plots classification results and weight progression

#### Part 2: Linear Regression
- Trains a linear regression model to fit the same classification task
- Compares prediction accuracy to the Perceptron model

---

### Requirements

Install required libraries using:

```bash
pip install numpy matplotlib pandas
