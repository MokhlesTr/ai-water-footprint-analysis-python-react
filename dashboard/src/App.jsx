import React, { useState, useEffect, useMemo } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, LineChart, Line, Cell
} from 'recharts';
import { Droplet, Activity, Server, RefreshCw, FileText, Presentation } from 'lucide-react';
import { generateData } from './data';
import './index.css';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip">
        <p className="label">{label}</p>
        {payload.map((entry, index) => (
          <p key={`item-${index}`} className="desc" style={{ color: entry.color }}>
            {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

function getPearsonCorrelation(x, y) {
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
  const minLength = Math.min(x.length, y.length);
  for (let i = 0; i < minLength; i++) {
    const xi = x[i], yi = y[i];
    sumX += xi; sumY += yi; sumXY += xi * yi;
    sumX2 += xi * xi; sumY2 += yi * yi;
  }
  const step1 = (minLength * sumXY) - (sumX * sumY);
  const step2 = (minLength * sumX2) - (sumX * sumX);
  const step3 = (minLength * sumY2) - (sumY * sumY);
  const step4 = Math.sqrt(step2 * step3);
  const answer = step1 / step4;
  return isNaN(answer) ? 0 : answer;
}

// Function to get a color between blue (0) and red (1) for heatmap
function getHeatmapColor(value) {
  // Map roughly from value [0, 1] to a color.
  // Actually Pearson is [-1, 1], but in this specific dataset correlations are mostly positive.
  // We will map 0 -> white, 1 -> dark red, -1 -> dark blue
  if (value > 0) {
    const intensity = Math.floor(value * 255);
    return `rgba(139, 0, 0, ${value})`; // Dark red with opacity
  } else {
    const intensity = Math.floor(Math.abs(value) * 255);
    return `rgba(8, 48, 107, ${Math.abs(value)})`; // Dark blue with opacity
  }
}


function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  const refreshData = async () => {
    setLoading(true);
    try {
      const response = await fetch('/data.json');
      const jsonData = await response.json();
      setData(jsonData);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshData();
  }, []);

  // Aggregations
  const regionData = useMemo(() => {
    const map = new Map();
    data.forEach(d => {
      if (!map.has(d.region)) map.set(d.region, { sum: 0, count: 0 });
      map.get(d.region).sum += d.total_annual_water_liters;
      map.get(d.region).count++;
    });
    return Array.from(map, ([region, val]) => ({
      region,
      water: Math.round(val.sum / val.count)
    })).sort((a, b) => b.water - a.water);
  }, [data]);

  const coolingWueData = useMemo(() => {
    const map = new Map();
    data.forEach(d => {
      if (!map.has(d.cooling_type)) map.set(d.cooling_type, { sum: 0, count: 0 });
      map.get(d.cooling_type).sum += d.wue_actual;
      map.get(d.cooling_type).count++;
    });
    return Array.from(map, ([cooling_type, val]) => ({
      cooling_type,
      wue: Number((val.sum / val.count).toFixed(2))
    })).sort((a, b) => a.wue - b.wue);
  }, [data]);

  const modelWaterData = useMemo(() => {
    const map = new Map();
    data.forEach(d => {
      if (!map.has(d.model_type)) map.set(d.model_type, { sum: 0, count: 0 });
      map.get(d.model_type).sum += d.total_annual_water_liters;
      map.get(d.model_type).count++;
    });
    return Array.from(map, ([model_type, val]) => ({
      model_type,
      water: Math.round(val.sum / val.count)
    })).sort((a, b) => a.water - b.water);
  }, [data]);

  const componentMeans = useMemo(() => {
    if (data.length === 0) return [];
    let human = 0, animal = 0, env = 0;
    data.forEach(d => {
      human += d.human_health_impact;
      animal += d.animal_health_impact;
      env += d.environmental_health_impact;
    });
    return [
      { name: 'Human Health', score: Number((human / data.length).toFixed(1)), fill: '#2ecc71' },
      { name: 'Animal Health', score: Number((animal / data.length).toFixed(1)), fill: '#3498db' },
      { name: 'Environmental', score: Number((env / data.length).toFixed(1)), fill: '#e74c3c' }
    ];
  }, [data]);

  const stressHealthData = useMemo(() => {
    const map = new Map();
    data.forEach(d => {
      if (!map.has(d.water_stress_level)) map.set(d.water_stress_level, { sum: 0, count: 0 });
      map.get(d.water_stress_level).sum += d.one_health_score;
      map.get(d.water_stress_level).count++;
    });
    return Array.from(map, ([level, val]) => ({
      level,
      score: Number((val.sum / val.count).toFixed(1))
    })).sort((a, b) => a.level - b.level);
  }, [data]);

  const correlationMatrix = useMemo(() => {
    if (data.length === 0) return { fields: [], matrix: [] };
    const fields = [
      'water_stress_level', 'wue_actual', 'total_annual_water_liters', 
      'human_health_impact', 'animal_health_impact', 'environmental_health_impact', 'one_health_score'
    ];
    
    // Extract series
    const series = fields.map(f => data.map(d => d[f]));
    
    // Compute matrix
    const matrix = [];
    for (let i = 0; i < fields.length; i++) {
      const row = [];
      for (let j = 0; j < fields.length; j++) {
        row.push(getPearsonCorrelation(series[i], series[j]));
      }
      matrix.push(row);
    }
    
    return { fields, matrix };
  }, [data]);

  // KPIs
  const totalWater = data.reduce((acc, curr) => acc + curr.total_annual_water_liters, 0);
  const avgWue = data.length ? data.reduce((acc, curr) => acc + curr.wue_actual, 0) / data.length : 0;
  const avgHealth = data.length ? data.reduce((acc, curr) => acc + curr.one_health_score, 0) / data.length : 0;

  return (
    <div className="dashboard-container">
      <header className="header">
        <div className="header-flex">
          <div>
            <h1>AI Water Footprint Dashboard</h1>
            <p>One Health Framework Analysis</p>
          </div>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <a href="https://slides.kimi.link/ppt/share/19e83df9-4d02-83a9-8000-00008ddaebf7" target="_blank" rel="noopener noreferrer" className="button-refresh" style={{ textDecoration: 'none', background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}>
              <Presentation size={18} /> View Presentation
            </a>
            <a href="/article.pdf" target="_blank" className="button-refresh" style={{ textDecoration: 'none', background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}>
              <FileText size={18} /> View Article PDF
            </a>
            <button className="button-refresh" onClick={refreshData}>
              <RefreshCw size={18} /> Regenerate Data
            </button>
          </div>
        </div>
      </header>

      <div className="kpi-grid">
        <div className="kpi-card">
          <Droplet className="kpi-icon" size={28} />
          <div className="kpi-label">Total Water Consumption</div>
          <div className="kpi-value">{(totalWater / 1e6).toFixed(2)}M L</div>
        </div>
        <div className="kpi-card">
          <Server className="kpi-icon" style={{ color: 'var(--accent-2)' }} size={28} />
          <div className="kpi-label">Average WUE</div>
          <div className="kpi-value">{avgWue.toFixed(2)} L/kWh</div>
        </div>
        <div className="kpi-card">
          <Activity className="kpi-icon" style={{ color: 'var(--accent-4)' }} size={28} />
          <div className="kpi-label">Avg One Health Score</div>
          <div className="kpi-value">{avgHealth.toFixed(1)} / 100</div>
        </div>
      </div>

      <div className="grid-container">
        <div className="card">
          <h2 className="card-title">Average Annual Water Consumption by Region</h2>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={regionData} margin={{ top: 10, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="region" stroke="#94a3b8" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 12 }} />
                <YAxis stroke="#94a3b8" tickFormatter={(val) => `${(val/1000).toFixed(0)}k`} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="water" name="Water (Liters)" fill="var(--accent-1)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h2 className="card-title">Water Usage Effectiveness by Cooling Technology</h2>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={coolingWueData} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                <XAxis type="number" stroke="#94a3b8" />
                <YAxis dataKey="cooling_type" type="category" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="wue" name="WUE (L/kWh)" fill="var(--accent-3)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h2 className="card-title">Water Footprint by AI Model Type</h2>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelWaterData} margin={{ top: 10, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="model_type" stroke="#94a3b8" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 12 }} />
                <YAxis stroke="#94a3b8" tickFormatter={(val) => `${(val/1000).toFixed(0)}k`} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="water" name="Water (Liters)" fill="var(--accent-4)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h2 className="card-title">One Health Component Scores</h2>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={componentMeans} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 14 }} />
                <YAxis stroke="#94a3b8" domain={[0, 100]} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="score" name="Impact Score" radius={[4, 4, 0, 0]}>
                  {componentMeans.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h2 className="card-title">One Health Score by Water Stress Level</h2>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stressHealthData} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="level" stroke="#94a3b8" tick={{ fontSize: 14 }} />
                <YAxis stroke="#94a3b8" domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="score" name="One Health Score" stroke="#8b0000" strokeWidth={3} dot={{ r: 6, fill: '#8b0000' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card" style={{ gridColumn: '1 / -1' }}>
          <h2 className="card-title">Correlation Matrix</h2>
          <div style={{ display: 'flex', flexDirection: 'column', overflowX: 'auto', paddingBottom: '1rem' }}>
            <div style={{ display: 'flex', flexDirection: 'column', minWidth: '800px', backgroundColor: 'var(--bg-secondary)', borderRadius: '8px', overflow: 'hidden' }}>
              
              {/* Header row */}
              <div style={{ display: 'flex', borderBottom: '1px solid var(--border-color)' }}>
                <div style={{ width: '150px', padding: '0.5rem', fontWeight: 'bold', fontSize: '0.75rem', color: 'var(--text-secondary)' }}></div>
                {correlationMatrix.fields.map((field, i) => (
                  <div key={i} style={{ flex: 1, padding: '0.5rem', fontWeight: 'bold', fontSize: '0.7rem', color: 'var(--text-secondary)', transform: 'rotate(-45deg)', transformOrigin: 'left bottom', whiteSpace: 'nowrap', height: '100px', display: 'flex', alignItems: 'flex-end' }}>
                    {field}
                  </div>
                ))}
              </div>

              {/* Data rows */}
              {correlationMatrix.matrix.map((row, i) => (
                <div key={i} style={{ display: 'flex', borderBottom: i < correlationMatrix.matrix.length - 1 ? '1px solid var(--border-color)' : 'none' }}>
                  <div style={{ width: '150px', padding: '0.75rem 0.5rem', fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', borderRight: '1px solid var(--border-color)' }}>
                    {correlationMatrix.fields[i]}
                  </div>
                  {row.map((val, j) => (
                    <div key={j} style={{ 
                      flex: 1, 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      backgroundColor: getHeatmapColor(val),
                      padding: '0.75rem 0.5rem',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      color: Math.abs(val) > 0.5 ? '#fff' : 'var(--text-primary)',
                      borderRight: j < row.length - 1 ? '1px solid rgba(255,255,255,0.1)' : 'none'
                    }}>
                      {val.toFixed(2)}
                    </div>
                  ))}
                </div>
              ))}

            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
