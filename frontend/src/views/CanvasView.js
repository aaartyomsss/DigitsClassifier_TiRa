import React, { useState } from 'react';
import Canvas from '../components/Canvas';
import PredictionContainer from '../components/PredictionContainer';
import './assets/CanvasView.css';

const CanvasView = () => {
  const [evaluating, setEvaluating] = useState(false);
  const [prediction, setPrediction] = useState(null);

  return (
    <div className="canvas-view-container">
      <div>
        <Canvas setPrediction={setPrediction} setEvaluating={setEvaluating} />
      </div>
      <div className="prediction-container">
        <PredictionContainer prediction={prediction} evaluating={evaluating} />
      </div>
    </div>
  );
};

export default CanvasView;
