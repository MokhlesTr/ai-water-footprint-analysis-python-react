// Generates the synthetic dataset for AI Water Footprint Analysis

const regions = ['North Virginia, USA', 'Dublin, Ireland', 'Singapore', 
                 'Santiago, Chile', 'Amsterdam, NL', 'Arizona, USA', 
                 'Central Chile', 'Middle East', 'São Paulo, Brazil',
                 'Mumbai, India', 'Beijing, China', 'Frankfurt, Germany'];

const waterStress = {
    'North Virginia, USA': 4, 'Dublin, Ireland': 3, 'Singapore': 5,
    'Santiago, Chile': 4, 'Amsterdam, NL': 2, 'Arizona, USA': 4,
    'Central Chile': 4, 'Middle East': 5, 'São Paulo, Brazil': 3,
    'Mumbai, India': 4, 'Beijing, China': 4, 'Frankfurt, Germany': 2
};

const wueValues = {
    'North Virginia, USA': 1.8, 'Dublin, Ireland': 1.5, 'Singapore': 2.1,
    'Santiago, Chile': 1.9, 'Amsterdam, NL': 1.1, 'Arizona, USA': 2.0,
    'Central Chile': 1.8, 'Middle East': 2.3, 'São Paulo, Brazil': 1.6,
    'Mumbai, India': 1.9, 'Beijing, China': 1.7, 'Frankfurt, Germany': 1.2
};

const coolingTypes = ['Evaporative Cooling', 'Air Cooling', 'Liquid Immersion', 'Hybrid Cooling', 'Free Cooling'];
const modelTypes = ['GPT-3', 'GPT-4', 'LLaMA-2', 'Gemini', 'Claude', 'Stable Diffusion', 'DALL-E', 'BERT-Large'];

const coolingFactor = {
    'Evaporative Cooling': 1.0,
    'Air Cooling': 0.6,
    'Liquid Immersion': 0.3,
    'Hybrid Cooling': 0.5,
    'Free Cooling': 0.4
};

const animalCoolingPenalty = {
    'Evaporative Cooling': 20, 'Air Cooling': 10, 
    'Liquid Immersion': 5, 'Hybrid Cooling': 8, 'Free Cooling': 6
};

// Helper function to get random item from array
const randomItem = (arr) => arr[Math.floor(Math.random() * arr.length)];

// Helper function for random number in range
const randomUniform = (min, max) => Math.random() * (max - min) + min;

// Simple normal distribution generator using Box-Muller transform
const randomNormal = (mean = 0, std = 1) => {
    let u1 = Math.random();
    let u2 = Math.random();
    let z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0 * std + mean;
};

// Simple gamma approximation
const randomGamma = (shape, scale) => {
    // simplified approximation for integer shape
    let d = shape - 1/3;
    let c = 1 / Math.sqrt(9 * d);
    while (true) {
        let x = randomNormal(0, 1);
        let v = 1 + c * x;
        while (v <= 0) {
            x = randomNormal(0, 1);
            v = 1 + c * x;
        }
        v = v * v * v;
        let u = Math.random();
        let x2 = x * x;
        if (u < 1 - 0.0331 * x2 * x2 || Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) {
            return scale * d * v;
        }
    }
};

// Exponential approximation
const randomExponential = (lambda) => -Math.log(1 - Math.random()) * lambda;

export function generateData(nRecords = 500) {
    const data = [];
    let maxTotalAnnualWater = 0;
    
    // First pass to generate base data and find max water for normalization
    const intermediateData = [];
    
    for (let i = 0; i < nRecords; i++) {
        const region = randomItem(regions);
        const cooling_type = randomItem(coolingTypes);
        
        const training_compute_hours = Math.floor(randomGamma(2, 500));
        const inference_queries_per_day = Math.floor(randomExponential(10000));
        
        const water_stress_level = waterStress[region];
        const base_wue = wueValues[region];
        const cooling_efficiency_factor = coolingFactor[cooling_type];
        const wue_actual = base_wue * cooling_efficiency_factor;
        
        const training_water_liters = Math.floor(training_compute_hours * 100 * wue_actual);
        const inference_water_liters_per_day = Math.floor(inference_queries_per_day * 0.015);
        const total_annual_water_liters = training_water_liters + (inference_water_liters_per_day * 365);
        
        if (total_annual_water_liters > maxTotalAnnualWater) {
            maxTotalAnnualWater = total_annual_water_liters;
        }
        
        intermediateData.push({
            id: i,
            region,
            model_type: randomItem(modelTypes),
            cooling_type,
            training_compute_hours,
            inference_queries_per_day,
            data_capacity_mw: parseFloat(randomUniform(1, 500).toFixed(1)),
            water_stress_level,
            base_wue,
            cooling_efficiency_factor,
            wue_actual,
            training_water_liters,
            inference_water_liters_per_day,
            total_annual_water_liters
        });
    }
    
    // Second pass to calculate health scores
    let healthScores = [];
    
    for (let i = 0; i < nRecords; i++) {
        const d = intermediateData[i];
        
        const human_health_impact = Number(((d.water_stress_level / 5) * 60 + 
                                   (d.total_annual_water_liters / maxTotalAnnualWater) * 40).toFixed(1));
                                   
        const animal_base = d.water_stress_level * 12;
        let animal_health_impact = animal_base + animalCoolingPenalty[d.cooling_type] + randomNormal(0, 5);
        animal_health_impact = Math.max(0, Math.min(100, animal_health_impact));
        animal_health_impact = Number(animal_health_impact.toFixed(1));
        
        const environmental_health_impact = Number(((d.total_annual_water_liters / maxTotalAnnualWater) * 50 +
                                            (d.water_stress_level / 5) * 50).toFixed(1));
                                            
        const one_health_score = Number((human_health_impact * 0.4 + 
                                 animal_health_impact * 0.3 + 
                                 environmental_health_impact * 0.3).toFixed(1));
                                 
        healthScores.push(one_health_score);
                                 
        data.push({
            ...d,
            human_health_impact,
            animal_health_impact,
            environmental_health_impact,
            one_health_score
        });
    }
    
    // Calculate median for high risk flag
    healthScores.sort((a, b) => a - b);
    const medianScore = healthScores[Math.floor(healthScores.length / 2)];
    
    return data.map(d => ({
        ...d,
        high_risk: d.one_health_score > medianScore ? 1 : 0
    }));
}
