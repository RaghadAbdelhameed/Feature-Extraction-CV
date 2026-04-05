import { useState } from "react";
import { Play, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import ImageUploader from "./ImageUploader";
import OutputViewer from "./OutputViewer";

type MatchMethod = "ssd" | "ncc";

const MatchingTab = () => {
  const [image1, setImage1] = useState<string | null>(null);
  const [image2, setImage2] = useState<string | null>(null);
  const [outputImage, setOutputImage] = useState<string | null>(null);
  const [method, setMethod] = useState<MatchMethod>("ssd");
  const [matchThreshold, setMatchThreshold] = useState(0.75);

  const handleReset = () => {
    setOutputImage(null);
  };

  return (
    <div className="flex h-full gap-4">
      <div className="w-1/4 flex flex-col gap-2 min-w-0">
        <div className="flex gap-2 h-[80px]">
          <div className="flex-1 min-w-0">
            <ImageUploader image={image1} onImageChange={setImage1} label="Image 1" compact />
          </div>
          <div className="flex-1 min-w-0">
            <ImageUploader image={image2} onImageChange={setImage2} label="Image 2" compact />
          </div>
        </div>

        <div className="flex-1 glass-panel p-3 flex flex-col gap-3">
          <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">Parameters</h3>

          <div className="space-y-1.5">
            <span className="text-xs text-muted-foreground">Method</span>
            <div className="flex gap-1.5">
              <button
                onClick={() => setMethod("ssd")}
                className={`flex-1 text-xs py-1.5 px-2 rounded-md font-medium transition-all ${
                  method === "ssd"
                    ? "bg-primary text-primary-foreground tab-active-glow"
                    : "bg-secondary text-secondary-foreground hover:bg-surface-hover"
                }`}
              >
                SSD
              </button>
              <button
                onClick={() => setMethod("ncc")}
                className={`flex-1 text-xs py-1.5 px-2 rounded-md font-medium transition-all ${
                  method === "ncc"
                    ? "bg-primary text-primary-foreground tab-active-glow"
                    : "bg-secondary text-secondary-foreground hover:bg-surface-hover"
                }`}
              >
                NCC
              </button>
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Match Threshold</span>
              <span className="font-mono text-primary">{matchThreshold.toFixed(2)}</span>
            </div>
            <Slider
              value={[matchThreshold]}
              onValueChange={([v]) => setMatchThreshold(v)}
              min={0.1}
              max={1.0}
              step={0.05}
            />
          </div>

          <div className="flex gap-2 mt-auto">
            <Button disabled={!image1 || !image2} className="flex-1 gap-1.5 text-xs" size="sm">
              <Play className="w-3 h-3" />
              Match
            </Button>
            <Button onClick={handleReset} variant="outline" size="sm" className="text-xs">
              <RotateCcw className="w-3 h-3" />
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 min-w-0">
        <OutputViewer image={outputImage} label="Feature Matching" />
      </div>
    </div>
  );
};

export default MatchingTab;
