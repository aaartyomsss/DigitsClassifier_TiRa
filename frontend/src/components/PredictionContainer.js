import React from 'react';
import { Spinner } from 'react-bootstrap';
import './assets/PredictionContainer.css';

// This component will simply display
// The results of prediciton
// {prediction} in terms of React is a prop
// can be compared to the paramater passed to a function
const ResponseContainer = ({ prediction, evaluating }) => {
  console.log(prediction, '!!!!!!!!!!!!!!!!!!!!');
  if (evaluating) {
    return (
      <div className="prediction-container">
        <p className="helper-text">
          Trying to understand your questionable handwriting. Gimme a second.
        </p>
        <p className="prediction">
          <Spinner animation="border" />
        </p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="prediction-container">
        <p className="helper-text">
          Draw something please! I will try to understand what it is.
        </p>
      </div>
    );
  }

  return (
    <div className="prediction-container">
      <p className="helper-text">
        Based on some simple maths and calculations I would say that what you
        have drawn is:
      </p>
      <p className="prediction">{prediction}</p>
    </div>
  );
};

export default ResponseContainer;
