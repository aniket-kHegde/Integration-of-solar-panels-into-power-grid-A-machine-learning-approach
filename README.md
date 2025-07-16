# Efficient Integration of Solar Energy

## Project Overview
This project aims to optimize the integration of solar energy into the power grid by developing a Physics-Informed Neural Network (PINN) model for accurate solar power prediction. The system accounts for shadow effects, weather conditions, and grid demand to reduce energy wastage and improve grid stability. The solution includes a web-based platform for real-time visualization and decision-making.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
   - [Data Collection and Processing](#data-collection-and-processing)
   - [Model Selection and Architecture](#model-selection-and-architecture)
   - [Shadow Calculation](#shadow-calculation)
   - [Solar Power Prediction with Shading](#solar-power-prediction-with-shading)
   - [Load Curve Generation](#load-curve-generation)
4. [Tools and Technologies](#tools-and-technologies)
5. [Results](#results)



## Objectives
1. **Accurate Solar Power Prediction**: Develop a PINN model to predict solar power generation using weather, geography, and building data.
2. **Shadow-Aware Forecasting**: Incorporate shadow effects from surrounding buildings to improve prediction accuracy.
3. **Real-Time Grid Integration**: Create a web platform to visualize solar-adjusted load curves for grid operators.
4. **Interactive Visualization**: Provide dashboards for monitoring and decision-making.

---

## Methodology

### Data Collection and Processing
- **Datasets Used**: Solar power generation and weather data from Gandikota, Andhra Pradesh (Kaggle).
- **Data Cleaning**:
  - Timestamp alignment for power and weather datasets.
  - Handling missing values via interpolation or removal.
  - Outlier detection (e.g., negative power readings).
- **Feature Selection**:
  - Strong correlations: Irradiation (0.99), module temperature (0.95), ambient temperature (0.72).

### Model Selection and Architecture
- **Selected Model**: Physics-Informed Bidirectional LSTM.
- **Architecture**:
  - Three Bidirectional LSTM layers (64, 48, 32 units) with dropout (0.1–0.2).
  - Dense layers (128, 96, 64, 32 units) with ReLU activation and L1/L2 regularization.
  - Physics-informed output layer enforcing non-negative power.
- **Performance Metrics**:
  - R²: 0.969
  - RMSE: 717.75 kW over 136,227 m².

### Shadow Calculation
- **Data Collection**:
  - Automated screenshot capture using Selenium (5 AM–7 PM).
  - Randomized sampling to minimize bias.
- **Shadow Detection**:
  - Image segmentation and pixel analysis to classify shadowed areas.
  - Shadow percentage formula:
    $\text{Shadow Percentage} = \left( \frac{A_{\text{shadow}}}{A_{\text{total}}} \right) \times 100\%$
<img width="847" height="575" alt="Screenshot 2025-07-16 at 10 51 29 PM" src="https://github.com/user-attachments/assets/ce196b7f-202e-488b-8148-2ba660d9641b" />

<img width="802" height="367" alt="Screenshot 2025-07-16 at 10 49 26 PM" src="https://github.com/user-attachments/assets/07e0d2b7-6980-41c4-9b57-b68843e33ed4" />

### Solar Power Prediction with Shading
1. **Theoretical Power (Pₜ)**:
   - Predicted by the PINN model using irradiation, module temperature, and ambient temperature.
2. **Shaded Power (Pₛ)**:
   - Panels in shaded regions
   - Efficiency factor interpolation (0.2 to 0.8):
      Shaded power: $P_s = \frac{1}{N_{\text{shaded}}} \int_{0.2}^{0.8} P_t \cdot e_f \, d(e_f)$  
- Unshaded power: $P_{\text{unshaded}} = P_t \cdot A_{\text{unshaded}}$
4. **Total Power**:
 $P_{\text{total}} = P_{\text{unshaded}} + P_s$
<img width="846" height="585" alt="Screenshot 2025-07-16 at 10 25 16 PM" src="https://github.com/user-attachments/assets/51232d05-d944-4967-a331-c1a05a459e44" />


### Load Curve Generation
1. **Original Load Curve**:
   - Derived from historical peak load data (SLDC Odisha).
   - Normalized per square meter:
     $\text{Load per m²} = \frac{\text{Monthly Peak Load (kW)}}{\text{Total Roof Area (m²)}}$
2. **Net Load Curve**:
   The solar-adjusted net load is calculated as $P_{\text{net}}(t) = P_{\text{load}}(t) - P_{\text{solar}}(t)$.
<img width="845" height="600" alt="Screenshot 2025-07-16 at 10 49 40 PM" src="https://github.com/user-attachments/assets/5425fbfa-0adb-49bc-bf01-c0ae5a7547b8" />

---

## Tools and Technologies
- **Machine Learning**: TensorFlow, Keras, Python (Jupyter Notebook, Pandas, NumPy).
- **Web Development**: Flask, Leaflet.js, OpenStreetMap API.
- **Data Visualization**: Matplotlib, Chart.js.
- **APIs**: Open-Meteo (weather data), SLDC Odisha (demand data).
- **Automation**: Selenium (shadow data collection).

---

## Results
- **Model Accuracy**: R² = 0.966, RMSE = 717.75 kW.
- **Shadow Integration**: Improved prediction realism by accounting for partial shading.
- **Web Platform**: Interactive dashboards for real-time solar and load data visualization.


