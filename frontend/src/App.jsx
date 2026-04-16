import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import { UploadCloud, Image as ImageIcon, Download, Loader2, ArrowRightLeft, Sparkles, SlidersHorizontal, ChevronRight, Aperture, Activity, Info } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [enhancedUrl, setEnhancedUrl] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  
  const [algorithm, setAlgorithm] = useState("auto");
  const [strength, setStrength] = useState(1.0);
  const [isLoading, setIsLoading] = useState(false);
  
  const fileInputRef = useRef(null);
  const workspaceRef = useRef(null);
  const sliderRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setEnhancedUrl(null);
      setTelemetry(null);
      setSliderPos(50);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setEnhancedUrl(null);
      setTelemetry(null);
      setSliderPos(50);
    }
  };

  const scrollToWorkspace = () => {
    workspaceRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const processImage = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("algorithm", algorithm);
    formData.append("strength", strength);

    try {
      const response = await axios.post("http://localhost:8000/process", formData);
      const data = response.data;
      
      if (data.error) {
        alert("Image Error: " + data.error);
        return;
      }
      
      if (!data.metrics) {
        alert("Failed to process image. Incomplete data returned.");
        return;
      }
      
      setEnhancedUrl(data.image_base64);
      setTelemetry({
        metrics: data.metrics,
        histograms: data.histograms
      });
      setSliderPos(50);
    } catch (error) {
      console.error("Error processing image:", error);
      alert("Failed to process image. Make sure the backend API is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSliderMove = useCallback((e) => {
    if (!isDragging || !sliderRef.current) return;
    const rect = sliderRef.current.getBoundingClientRect();
    let x = e.clientX - rect.left;
    const width = rect.width;
    x = Math.max(0, Math.min(x, width)); // Clamp
    setSliderPos((x / width) * 100);
  }, [isDragging]);

  const handleMouseUp = () => setIsDragging(false);

  useEffect(() => {
    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('mousemove', handleSliderMove);
    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('mousemove', handleSliderMove);
    };
  }, [handleSliderMove]);

  const getChartData = () => {
    if (!telemetry || !telemetry.histograms) return [];
    return telemetry.histograms.original.map((val, index) => ({
      intensity: index,
      Original: val,
      Enhanced: telemetry.histograms.enhanced[index]
    }));
  };

  return (
    <div className="app-container">
      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-brand title-gradient">Luminix</div>
        <button onClick={scrollToWorkspace} className="btn-text" style={{ background: 'transparent', border: 'none', color: 'var(--text-main)', cursor: 'pointer', fontWeight: 600 }}>Try Demo</button>
      </nav>

      {/* Hero Section */}
      <section className="hero">
        <h1 className="title-gradient">Reveal the unseen. <br/> Enhance the invisible.</h1>
        <p>Luminix utilizes state-of-the-art Recursive Histogram Equalization (R-ESIHE & RS-ESIHE) to rescue critically low-exposure photography with mathematical precision.</p>
        <button className="hero-btn" onClick={scrollToWorkspace}>
          Start Processing <ChevronRight size={20} />
        </button>
      </section>

      {/* Features Overview */}
      <section className="features-container">
        <div className="feature-card glass-panel">
          <div className="feature-icon"><Sparkles size={24} /></div>
          <h3>Smart Selector</h3>
          <p style={{color: 'var(--text-muted)'}}>Automatically analyzes image complexity and entropy to select the optimal enhancement algorithm (R-ESIHE vs RS-ESIHE) dynamically.</p>
        </div>
        <div className="feature-card glass-panel">
          <div className="feature-icon"><Aperture size={24} /></div>
          <h3>Adaptive Recursion</h3>
          <p style={{color: 'var(--text-muted)'}}>Intelligently determines the exact depth of recursive division needed by isolating flat histograms, preventing noise amplification.</p>
        </div>
        <div className="feature-card glass-panel">
          <div className="feature-icon"><SlidersHorizontal size={24} /></div>
          <h3>Hybrid Blending</h3>
          <p style={{color: 'var(--text-muted)'}}>Provides a custom Hybrid tuning pipeline that merges LAB-space CLAHE with Gamma corrections for vivid color retention.</p>
        </div>
      </section>

      {/* The Application Workspace */}
      <section ref={workspaceRef} className="workspace-wrapper">
        <div className="workspace-header">
          <h2 className="title-gradient">Enhancer Engine</h2>
          <p style={{color: 'var(--text-muted)'}}>Upload an image to start transforming your dark shots into stunning visuals.</p>
        </div>

        <div className="workspace-grid">
          
          {/* Settings Sidebar */}
          <aside className="control-panel glass-panel">
            <div 
              className="upload-zone"
              onClick={() => fileInputRef.current.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input 
                type="file" 
                hidden 
                ref={fileInputRef} 
                accept="image/*"
                onChange={handleFileChange}
              />
              <UploadCloud size={48} color="var(--primary)" />
              <div>
                <h3 style={{margin: '0 0 0.5rem 0'}}>Select Image</h3>
                <span style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>Drag & drop or click</span>
              </div>
            </div>

            <div className="setting-row">
              <label>Processing Algorithm</label>
              <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
                <option value="auto">Smart Auto-Select (Recommended)</option>
                <option value="R-ESIHE">R-ESIHE (Standard Enhancer)</option>
                <option value="RS-ESIHE">RS-ESIHE (Deep Low-Light)</option>
                <option value="Hybrid">Hybrid (CLAHE + Gamma + HE)</option>
              </select>
            </div>

            <div className="setting-row">
              <label>
                <span>Effect Strength</span>
                <span style={{color: 'var(--primary)'}}>{Math.round(strength * 100)}%</span>
              </label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.05" 
                value={strength} 
                onChange={(e) => setStrength(parseFloat(e.target.value))}
              />
            </div>

            <button 
              className="action-btn" 
              onClick={processImage}
              disabled={!selectedFile || isLoading}
              style={{marginTop: '1rem'}}
            >
              {isLoading ? (
                <> <Loader2 className="animate-spin" /> Processing Data... </>
              ) : (
                <> <Sparkles size={20} /> Ignite Enhancement </>
              )}
            </button>
          </aside>

          {/* Visualizer Area */}
          <section className="preview-area">
            {!previewUrl && (
              <div className="glass-panel empty-state">
                <ImageIcon size={64} style={{opacity: 0.3}} />
                <h3>No Image Loaded</h3>
                <p>Upload a file to preview the enhancement output.</p>
              </div>
            )}

            {previewUrl && !enhancedUrl && (
              <div className="glass-panel" style={{height: '600px', display: 'flex', alignItems: 'center', justifyContent:'center', padding: '1rem'}}>
                <img src={previewUrl} alt="Original" style={{maxWidth: '100%', maxHeight: '100%', borderRadius: '12px', objectFit:'contain'}} />
              </div>
            )}

            {previewUrl && enhancedUrl && (
              <div className="results-view">
                <div 
                  className="slider-container"
                  ref={sliderRef}
                  onMouseDown={() => setIsDragging(true)}
                  onTouchStart={() => setIsDragging(true)}
                >
                  <img src={previewUrl} alt="Original Before" className="slider-base-img" />
                  <div className="slider-label label-right">Original</div>
                  
                  <div className="slider-overlay" style={{width: `${sliderPos}%`}}>
                    <img src={enhancedUrl} alt="Enhanced" className="slider-overlay-img" />
                    <div className="slider-label label-left" style={{color: 'var(--primary)'}}>Enhanced</div>
                  </div>

                  <div className="slider-handle" style={{left: `${sliderPos}%`}}>
                    <div className="handle-icon">
                      <ArrowRightLeft size={20} />
                    </div>
                  </div>
                </div>

                <div className="download-bar">
                  <a href={enhancedUrl} download={`luminix_${algorithm}.jpg`} className="download-btn">
                    <Download size={18} /> Export Image
                  </a>
                </div>
              </div>
            )}

            {/* Novelty: Live Analytics UI */}
            {telemetry && (
              <div className="telemetry-dashboard glass-panel">
                <h3 className="title-gradient" style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                  <Activity size={24} /> Recursive Telemetry Data
                </h3>
                
                <div className="telemetry-grid">
                  <div className="telemetry-card">
                    <h4>Algorithm Activated</h4>
                    <p className="large-stat">{telemetry.metrics.algorithm_used}</p>
                  </div>
                  <div className="telemetry-card">
                    <h4>Entropy Extracted (Complexity)</h4>
                    <p className="large-stat">
                      {telemetry.metrics.original_entropy.toFixed(2)} &rarr; <span style={{color: 'var(--primary)'}}>{telemetry.metrics.enhanced_entropy.toFixed(2)}</span>
                    </p>
                  </div>
                  <div className="telemetry-card">
                    <h4>Exposure Gain (Luminance)</h4>
                    <p className="large-stat">
                      {telemetry.metrics.original_exposure.toFixed(2)} &rarr; <span style={{color: 'var(--primary)'}}>{telemetry.metrics.enhanced_exposure.toFixed(2)}</span>
                    </p>
                  </div>
                </div>

                <div className="chart-container">
                  <h4>Intensity Equalization Topology</h4>
                  <div className="chart-wrapper">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={getChartData()} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="intensity" stroke="var(--text-muted)" fontSize={12} />
                        <YAxis stroke="var(--text-muted)" fontSize={12} />
                        <Tooltip contentStyle={{backgroundColor: 'rgba(5,5,5,0.9)', border: '1px solid var(--border-subtle)', borderRadius: '8px'}} />
                        <Legend wrapperStyle={{paddingTop: '20px'}}/>
                        <Line type="monotone" dataKey="Original" stroke="#94a3b8" dot={false} strokeWidth={2} isAnimationActive={true} />
                        <Line type="monotone" dataKey="Enhanced" stroke="var(--primary)" dot={false} strokeWidth={2} isAnimationActive={true} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="chart-helper"><Info size={14}/> The active curve indicates how the algorithm stretched and divided dense, low-light histogram walls into readable contrast zones.</p>
                </div>
              </div>
            )}
          </section>

        </div>
      </section>

      {/* Footer */}
      <footer>
        <p>© 2026 Luminix Enhancer Engine. Built with Advanced Recursive Core.</p>
      </footer>
    </div>
  );
}

export default App;
