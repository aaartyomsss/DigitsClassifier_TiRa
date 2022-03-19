import React from 'react';
import CanvasView from './views/CanvasView';

const App = () => {
  console.log('It should re render');
  return (
    <div>
      Hello lol it works
      <CanvasView />
    </div>
  );
};

export default App;
