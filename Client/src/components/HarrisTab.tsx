import { useState } from "react";
import ImageUploader from "./ImageUploader";
import OutputViewer from "./OutputViewer";

const HarrisTab = () => {
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [outputImage, setOutputImage] = useState<string | null>(null);

  return (
    <div className="flex h-full gap-4">
      <div className="w-1/4 min-w-0">
        <ImageUploader image={inputImage} onImageChange={setInputImage} />
      </div>
      <div className="flex-1 min-w-0">
        <OutputViewer image={outputImage} label="Harris Corner Detection" />
      </div>
    </div>
  );
};

export default HarrisTab;
