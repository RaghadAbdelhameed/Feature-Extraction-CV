import { useState } from "react";
import { Play, RotateCcw, Bug } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import ImageUploader from "./ImageUploader";
import OutputViewer from "./OutputViewer";
import { useToast } from "@/hooks/use-toast";

type MatchMethod = "ssd" | "ncc";

const MatchingTab = () => {
  const [image1, setImage1] = useState<string | null>(null);
  const [image2, setImage2] = useState<string | null>(null);
  const [outputImage, setOutputImage] = useState<string | null>(null);
  const [method, setMethod] = useState<MatchMethod>("ncc");
  const [matchThreshold, setMatchThreshold] = useState(0.65);
  const [isLoading, setIsLoading] = useState(false);
  const [computationTime, setComputationTime] = useState<number | null>(null);
  const [matches, setMatches] = useState<any[]>([]);
  const [debugInfo, setDebugInfo] = useState<any>(null);
  const [patchSize, setPatchSize] = useState(15);
  
  const { toast } = useToast();

  const handleReset = () => {
    setOutputImage(null);
    setComputationTime(null);
    setMatches([]);
    setDebugInfo(null);
  };

  const handleDebug = async () => {
    if (!image1 || !image2) {
      toast({
        title: "Missing Images",
        description: "Please upload both images for debugging",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:5000/api/ncc/debug", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image1: image1,
          image2: image2,
        }),
      });

      const data = await response.json();
      
      if (data.success) {
        setDebugInfo(data.debug_info);
        toast({
          title: "Debug Info Retrieved",
          description: `Found ${data.debug_info.corners1_count} corners in image1, ${data.debug_info.corners2_count} in image2`,
        });
      } else {
        toast({
          title: "Debug Failed",
          description: data.error,
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Debug error:", error);
      toast({
        title: "Connection Error",
        description: "Could not connect to backend",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleMatch = async () => {
    if (!image1 || !image2) {
      toast({
        title: "Missing Images",
        description: "Please upload both images before matching",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setOutputImage(null);
    setComputationTime(null);
    setDebugInfo(null);

    try {
      if (method === "ncc") {
        const response = await fetch("http://localhost:5000/api/ncc/feature-match", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image1: image1,
            image2: image2,
            patch_size: patchSize,
            num_keypoints: 200,
            ratio_thresh: matchThreshold,
            min_ncc_thresh: 0.4,
            use_cross_check: true,
          }),
        });

        const data = await response.json();

        if (data.success) {
          setOutputImage(data.visualization);
          setComputationTime(data.computational_time * 1000);
          
          if (data.num_matches === 0) {
            toast({
              title: "No Matches Found",
              description: "Try lowering the threshold or using different images",
              variant: "default",
            });
          } else {
            toast({
              title: "Match Complete!",
              description: `Found ${data.num_matches} matches between the images`,
              variant: "default",
            });
          }
          
          setMatches(data.matches);
        } else {
          toast({
            title: "Matching Failed",
            description: data.error || "Unknown error occurred",
            variant: "destructive",
          });
        }
      } else if (method === "ssd") {
        toast({
          title: "SSD Coming Soon",
          description: "SSD matching will be implemented in the next update",
          variant: "default",
        });
      }
    } catch (error) {
      console.error("Error during matching:", error);
      toast({
        title: "Connection Error",
        description: "Failed to connect to backend server. Make sure it's running on port 5000",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleMatchThresholdChange = (value: number[]) => {
    setMatchThreshold(value[0]);
  };

  const handlePatchSizeChange = (value: number[]) => {
    setPatchSize(value[0]);
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
              onValueChange={handleMatchThresholdChange}
              min={0.3}
              max={0.9}
              step={0.05}
            />
            <p className="text-[10px] text-muted-foreground">
              Lower = more matches, Higher = more accurate
            </p>
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Patch Size</span>
              <span className="font-mono text-primary">{patchSize}</span>
            </div>
            <Slider
              value={[patchSize]}
              onValueChange={handlePatchSizeChange}
              min={7}
              max={31}
              step={2}
            />
            <p className="text-[10px] text-muted-foreground">
              Larger = more context, Smaller = more precise
            </p>
          </div>

          <div className="flex gap-2 mt-auto">
            <Button 
              disabled={!image1 || !image2 || isLoading} 
              className="flex-1 gap-1.5 text-xs" 
              size="sm"
              onClick={handleMatch}
            >
              <Play className="w-3 h-3" />
              {isLoading ? "Matching..." : "Match"}
            </Button>
                      <Button onClick={handleReset} variant="outline" size="sm" className="text-xs">
              <RotateCcw className="w-3 h-3" />
            </Button>
          </div>

          {/* Debug Info Panel */}
          {debugInfo && (
            <div className="mt-2 p-2 bg-muted/30 rounded text-[10px] space-y-1">
              <p className="font-semibold text-xs">Debug Info:</p>
              <div className="grid grid-cols-2 gap-x-2 gap-y-1">
                <span className="text-muted-foreground">Corners in Image 1:</span>
                <span className={debugInfo.corners1_count === 0 ? "text-red-500" : "text-green-500"}>
                  {debugInfo.corners1_count}
                </span>
                <span className="text-muted-foreground">Corners in Image 2:</span>
                <span className={debugInfo.corners2_count === 0 ? "text-red-500" : "text-green-500"}>
                  {debugInfo.corners2_count}
                </span>
              </div>
              {debugInfo.corners1_count === 0 && (
                <p className="text-yellow-600 mt-1">
                  ⚠️ No corners detected! Try images with more texture or edges.
                </p>
              )}
            </div>
          )}

          {/* Match Statistics */}
          {matches.length > 0 && (
            <div className="mt-2 p-2 bg-green-500/10 rounded text-[10px] space-y-1">
              <p className="font-semibold text-xs text-green-600">Matches: {matches.length}</p>
              <p className="text-muted-foreground">
                Best Score: {Math.max(...matches.map(m => m.ncc_score)).toFixed(3)}
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 min-w-0">
        <OutputViewer 
          image={outputImage} 
          label="Feature Matching" 
          computationTime={computationTime}
          matches={matches}
        />
      </div>
    </div>
  );
};

export default MatchingTab;