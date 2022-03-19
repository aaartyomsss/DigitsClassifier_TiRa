import React from 'react';
import { SketchField, Tools } from 'react-sketch';

const Canvas = () => {
  return (
    <SketchField
      width="128px"
      height="128px"
      tool={Tools.Pencil}
      lineColor="black"
      lineWidth={3}
    />
  );
};

export default Canvas;
