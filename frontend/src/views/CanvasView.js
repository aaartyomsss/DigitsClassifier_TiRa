import React, { useState } from 'react';
import Canvas from '../components/Canvas';
import PredictionContainer from '../components/PredictionContainer';
import { Button, Spinner } from 'react-bootstrap';
import { ENDPOINT_URLS } from '../constants';
import postRequst from '../utils/postFetch';
import './assets/CanvasView.css';

const CanvasView = () => {
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [prediction, setPrediction] = useState(null);

  const trainModel = async () => {
    setLoading(true);
    const res = await postRequst(ENDPOINT_URLS.retrainModel);
    if (res.ok) {
      setLoading(false);
    }
  };

  if (loading) return <Spinner animation="border" />;

  return (
    <div className="canvas-view-container">
      <div>
        <Canvas setPrediction={setPrediction} setEvaluating={setEvaluating} />
        <Button onClick={trainModel}>Train model</Button>
      </div>
      <div className="prediction-container">
        <PredictionContainer prediction={prediction} evaluating={evaluating} />
      </div>
    </div>
  );
};

export default CanvasView;
