// Create new file: src/components/ui/NCCDemo.tsx
import { useState } from 'react';
import { ImageUploader } from './ImageUploader';
import { OutputViewer } from './OutputViewer';
import { useNCCMatching } from '@/hooks/useNCCMatching';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, RotateCcw } from 'lucide-react';

const NCCDemo = () => {
  const [image1, setImage1] = useState<string | null>(null);
  const [image2, setImage2] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.75);
  const { matchFeatures, isLoading, result, reset } = useNCCMatching();

  const handleMatch = async () => {
    if (!image1 || !image2) {
      alert('Please upload both images');
      return;
    }
    
    await matchFeatures(image1, image2, {
      ratio_thresh: threshold,
      patch_size: 15,
      num_keypoints: 100,
    });
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <ImageUploader image={image1} onImageChange={setImage1} label="Image 1" />
        <ImageUploader image={image2} onImageChange={setImage2} label="Image 2" />
      </div>
      
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Match Threshold: {threshold.toFixed(2)}</span>
        </div>
        <Slider
          value={[threshold]}
          onValueChange={([v]) => setThreshold(v)}
          min={0.1}
          max={1.0}
          step={0.05}
        />
      </div>
      
      <div className="flex gap-2">
        <Button 
          onClick={handleMatch} 
          disabled={!image1 || !image2 || isLoading}
          className="flex-1"
        >
          <Play className="w-4 h-4 mr-2" />
          {isLoading ? 'Matching...' : 'Match Features'}
        </Button>
        <Button onClick={reset} variant="outline">
          <RotateCcw className="w-4 h-4" />
        </Button>
      </div>
      
      {result && (
        <OutputViewer 
          image={result.visualization}
          computationTime={result.computationTime * 1000}
          matches={result.matches}
          label="Matching Results"
        />
      )}
    </div>
  );
};

export default NCCDemo;