import React, { useRef } from 'react';
import { SketchField, Tools } from 'react-sketch';
import postRequst from '../utils/postFetch';
import './assets/Canvas.css';

// Main component responsible for user handwriting
const Canvas = () => {
  const imageCaptureRef = useRef(null);

  // Captures the img and sends it to the server in order to make evaluation
  const handleSubmit = async (e) => {
    // Prevents refresh of page
    e.preventDefault();

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
    const res = await postRequst('upload-image/', data);
    console.log(res);
  };

  return (
    <div>
      <div>
        <SketchField
          width="140px"
          height="140px"
          tool={Tools.Pencil}
          lineColor="black"
          lineWidth={3}
          backgroundColor="white"
          ref={imageCaptureRef}
        />
      </div>
      <button onClick={handleSubmit}>Submit</button>
    </div>
  );
};

export default Canvas;
