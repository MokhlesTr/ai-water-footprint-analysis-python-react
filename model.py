import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# 1. GENERATE SYNTHETIC DATASET
# ============================================

# Data center regions (based on Table III from your article)
regions = ['North Virginia, USA', 'Dublin, Ireland', 'Singapore', 
           'Santiago, Chile', 'Amsterdam, NL', 'Arizona, USA', 
           'Central Chile', 'Middle East', 'São Paulo, Brazil',
           'Mumbai, India', 'Beijing, China', 'Frankfurt, Germany']

# Water stress levels (1-5, where 5 is extremely high)
water_stress = {
    'North Virginia, USA': 4, 'Dublin, Ireland': 3, 'Singapore': 5,
    'Santiago, Chile': 4, 'Amsterdam, NL': 2, 'Arizona, USA': 4,
    'Central Chile': 4, 'Middle East': 5, 'São Paulo, Brazil': 3,
    'Mumbai, India': 4, 'Beijing, China': 4, 'Frankfurt, Germany': 2
}

# Typical WUE values (L/kWh) by region
wue_values = {
    'North Virginia, USA': 1.8, 'Dublin, Ireland': 1.5, 'Singapore': 2.1,
    'Santiago, Chile': 1.9, 'Amsterdam, NL': 1.1, 'Arizona, USA': 2.0,
    'Central Chile': 1.8, 'Middle East': 2.3, 'São Paulo, Brazil': 1.6,
    'Mumbai, India': 1.9, 'Beijing, China': 1.7, 'Frankfurt, Germany': 1.2
}

# Cooling technologies
cooling_types = ['Evaporative Cooling', 'Air Cooling', 'Liquid Immersion', 
                 'Hybrid Cooling', 'Free Cooling']

# Model types
model_types = ['GPT-3', 'GPT-4', 'LLaMA-2', 'Gemini', 'Claude', 
               'Stable Diffusion', 'DALL-E', 'BERT-Large']

# Generate dataset with 500 records
n_records = 500

data = {
    'region': np.random.choice(regions, n_records),
    'model_type': np.random.choice(model_types, n_records),
    'cooling_type': np.random.choice(cooling_types, n_records),
    'training_compute_hours': np.random.gamma(2, 500, n_records).astype(int),
    'inference_queries_per_day': np.random.exponential(10000, n_records).astype(int),
    'data_capacity_mw': np.random.uniform(1, 500, n_records).round(1),
}

df = pd.DataFrame(data)

# Add derived features based on region
df['water_stress_level'] = df['region'].map(water_stress)
df['base_wue'] = df['region'].map(wue_values)

# Add cooling type efficiency factors
cooling_factor = {
    'Evaporative Cooling': 1.0,
    'Air Cooling': 0.6,
    'Liquid Immersion': 0.3,
    'Hybrid Cooling': 0.5,
    'Free Cooling': 0.4
}
df['cooling_efficiency_factor'] = df['cooling_type'].map(cooling_factor)

# Calculate actual WUE
df['wue_actual'] = df['base_wue'] * df['cooling_efficiency_factor']

# Calculate direct water consumption (liters)
# Training water: training hours * avg power (100 kW) * WUE
df['training_water_liters'] = (df['training_compute_hours'] * 100 * 
                                df['wue_actual']).astype(int)

# Inference water: daily queries * 0.01 L per query (based on your article: 20-50 queries = 500ml)
df['inference_water_liters_per_day'] = (df['inference_queries_per_day'] * 0.015).astype(int)

# Calculate total annual water footprint (liters)
df['total_annual_water_liters'] = (df['training_water_liters'] + 
                                    df['inference_water_liters_per_day'] * 365)

# Calculate indirect water from electricity (electricity mix factor)
electricity_water_factor = np.random.uniform(0.5, 3.0, n_records)  # L/kWh
df['indirect_water_liters'] = (df['total_annual_water_liters'] * 
                               np.random.uniform(0.3, 0.6, n_records))

# One Health impact scores (0-100, higher = worse)
# Human health impact: based on water stress and water consumption
df['human_health_impact'] = ((df['water_stress_level'] / 5) * 60 + 
                              (df['total_annual_water_liters'] / df['total_annual_water_liters'].max()) * 40).round(1)

# Animal health impact: based on water stress and cooling type
animal_base = df['water_stress_level'] * 12
animal_cooling_penalty = df['cooling_type'].map({
    'Evaporative Cooling': 20, 'Air Cooling': 10, 
    'Liquid Immersion': 5, 'Hybrid Cooling': 8, 'Free Cooling': 6
})
df['animal_health_impact'] = (animal_base + animal_cooling_penalty + 
                               np.random.normal(0, 5, n_records)).clip(0, 100).round(1)

# Environmental health impact: based on water consumption and stress
df['environmental_health_impact'] = ((df['total_annual_water_liters'] / df['total_annual_water_liters'].max()) * 50 +
                                      (df['water_stress_level'] / 5) * 50).round(1)

# Calculate total One Health Impact Score (composite)
df['one_health_score'] = ((df['human_health_impact'] * 0.4 +
                           df['animal_health_impact'] * 0.3 +
                           df['environmental_health_impact'] * 0.3)).round(1)

# Add a binary target: High Risk (1) if One Health Score > median
threshold = df['one_health_score'].median()
df['high_risk'] = (df['one_health_score'] > threshold).astype(int)

print("=" * 70)
print("DATASET GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nBasic statistics:")
print(df.describe())