import React, { useState, useRef } from 'react';
import { SketchField, Tools } from 'react-sketch';
import postRequst from '../utils/postFetch';
import { ENDPOINT_URLS } from '../constants';
import { Button } from 'react-bootstrap';
import './assets/Canvas.css';

// Main component responsible for user handwriting
const Canvas = ({ setPrediction, setEvaluating }) => {
  const imageCaptureRef = useRef(null);
  // To simplify rerendering of the canvas to clear it from previous state
  // May not be the most efficient solution, but rather clean and simple
  const [rerenderKey, setRerenderKey] = useState(1);

  // Captures the img and sends it to the server in order to make evaluation
  const handleSubmit = async (e) => {
    // Prevents refresh of page
    e.preventDefault();
    setEvaluating(true);
    const img = await imageCaptureRef.current.toDataURL({
      format: 'base64',
      quality: 0.5,
      width: 140,
      height: 140,
    });

    const data = {
      base64_image: img,
    };

    // Sending base64 img to the server and receiving the response
    const prediction = await postRequst(ENDPOINT_URLS.uploadImage, data);
    setEvaluating(false);
    setPrediction(prediction.result);
    setRerenderKey((prevKey) => prevKey + 1);
  };

  return (
    <div>
      <div>
        <SketchField
          width="140px"
          height="140px"
          tool={Tools.Pencil}
          lineColor="white"
          lineWidth={3}
          backgroundColor="black"
          aria-label="sketch-field"
          ref={imageCaptureRef}
          key={rerenderKey}
        />
      </div>
      <Button variant="primary" onClick={handleSubmit}>
        Submit
      </Button>
    </div>
  );
};

export default Canvas;
