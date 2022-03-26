import React from 'react';
import CanvasView from './views/CanvasView';
import './App.css';

/* Component resoponsible for bringing everything together
   In React everything consists of components. Those are small building blocks
   that each have their own logic for representation of the UI and mutation 
   of state within it */
const App = () => {
  return (
    <div id="main">
      <CanvasView />
    </div>
  );
};

export default App;
