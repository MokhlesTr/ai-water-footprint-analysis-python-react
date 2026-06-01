# 🌍 AI Water Footprint Analysis (Python + React)

![AI Water Footprint Dashboard Banner](./dashboard/public/comprehensive_ai_water_analysis.png)
*(Note: You can replace this image link with an actual screenshot of your React dashboard!)*

## 📖 Overview
This project is a **Full-Stack Data Science Application** designed to simulate, analyze, and visualize the severe water footprint of Artificial Intelligence models (like GPT-4 and LLaMA) globally. 

Because exact server-level water consumption data is rarely disclosed by tech giants, this project uses **Python** to engineer a scientifically accurate synthetic dataset based on real-world parameters (Water Usage Effectiveness, regional water stress levels, and cooling technologies). It then uses a **React + Vite** frontend to visualize this data in an interactive dashboard, evaluating the impacts through the **One Health Framework** (Human, Animal, and Environmental Health).

---

## 🚀 Features
- **Synthetic Data Generation (Python):** Uses probability distributions to simulate realistic compute hours, queries per day, and water footprint footprints.
- **Machine Learning (Scikit-Learn):** Uses a `RandomForestRegressor` to predict health impact scores and a `LogisticRegression` classifier to identify high-risk data centers based on location and cooling systems.
- **Dynamic React Dashboard:** Consumes the Python-generated JSON to display real-time analytics.
- **Custom Mathematical Visualizations:** Features a custom JavaScript Pearson Correlation Matrix (Heatmap) and dynamic Recharts components.

---

## 🛠️ Tech Stack
- **Data Engineering & ML:** Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Frontend Framework:** React 18, Vite
- **UI & Styling:** Vanilla CSS (Glassmorphism, Dark Mode), Recharts (Interactive Graphs), Lucide-React (Icons)

---

## ⚙️ Installation & Setup

### Step 1: Run the Python Backend
First, generate the synthetic dataset and run the Machine Learning models.

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-water-footprint-analysis-python-react.git
cd ai-water-footprint-analysis-python-react

# Create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required Python libraries
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the data analysis script
python ai_water_footprint_analysis.py
```
*Running this script will output terminal predictions, generate two PNG graphs, and successfully export a `data.json` file straight into the React frontend folder.*

### Step 2: Run the React Frontend
Once `data.json` is generated, you can launch the visualization dashboard.

```bash
# Navigate to the dashboard directory
cd dashboard

# Install Node.js dependencies
npm install

# Start the Vite development server
npm run dev
```
*Open your browser and navigate to `http://localhost:5173` to view the interactive dashboard.*

---

## 🧠 Project Architecture: How it works
1. **`ai_water_footprint_analysis.py`**: Simulates the dataset based on real parameters (e.g., Liquid Immersion cooling provides a 70% reduction in water use). 
2. **`data.json`**: The bridge between backend and frontend. The Python script cleans the simulated data and dumps the 500 final records here.
3. **`App.jsx`**: The React application fetches `data.json` on load. It parses the data and calculates dynamic aggregations (like Pearson correlations) entirely on the client side to map out interactive visuals.

---

## 📜 References
This project and its mathematical parameters are heavily inspired by research regarding the global water crisis and the hidden environmental costs of Artificial Intelligence computation.

---
*Created for academic and research presentation purposes.*
