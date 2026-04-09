import { useState } from "react";
import ImageUploader from "./ImageUploader";
import OutputViewer from "./OutputViewer";
import { Play, Loader2, Crosshair, Settings2 } from "lucide-react";

const SiftTab = () => {
  // Image states
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [outputImage, setOutputImage] = useState<string | null>(null);
  
  // Processing states
  const [isProcessing, setIsProcessing] = useState(false);
  const [computationTime, setComputationTime] = useState<number | null>(null);
  const [totalKeypoints, setTotalKeypoints] = useState<number | null>(null);

  // Parameter states (Standard default SIFT values)
  const [contrastThr, setContrastThr] = useState<number>(0.04);
  const [edgeThr, setEdgeThr] = useState<number>(10);
  const [sigma, setSigma] = useState<number>(1.6);

  const handleFindKeypoints = async () => {
    if (!inputImage) return;
    
    setIsProcessing(true);
    setComputationTime(null);
    setTotalKeypoints(null);

    try {
      const res = await fetch(inputImage);
      const blob = await res.blob();
      const file = new File([blob], "upload.jpg", { type: "image/jpeg" });

      const formData = new FormData();
      formData.append("image", file);
      formData.append("contrast_thr", contrastThr.toString());
      formData.append("edge_thr", edgeThr.toString());
      formData.append("sigma", sigma.toString());

      // Adjust this URL to match your exact backend route
      const response = await fetch("http://127.0.0.1:5000/api/sift", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();
      
      setOutputImage(data.result_image_base64);
      if (data.computation_time_seconds) {
        setComputationTime(data.computation_time_seconds * 1000); 
      }
      
      // Checking for common key names your backend might use
      if (data.total_points !== undefined) {
        setTotalKeypoints(data.total_points);
      } else if (data.total_keypoints !== undefined) {
        setTotalKeypoints(data.total_keypoints);
      }

    } catch (error) {
      console.error("Error processing image:", error);
      alert("Failed to process image. Make sure your Python backend is running!");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-full gap-5">
      {/* LEFT COLUMN: Fixed Width, Compact, NO Scrolling */}
      <div className="w-[320px] shrink-0 flex flex-col gap-3.5">
        
        {/* Image Uploader */}
        <div className="h-32 shrink-0">
          <ImageUploader image={inputImage} onImageChange={setInputImage} />
        </div>

        {/* Stats Panel Slot */}
        <div className="h-[68px] shrink-0">
          {totalKeypoints !== null ? (
            <div className="h-full px-3 py-2 rounded-xl border border-border/50 bg-secondary/10 flex items-center gap-3 shadow-sm">
               <div className="p-2 rounded-lg bg-primary/20">
                  <Crosshair className="w-5 h-5 text-primary" />
               </div>
               <div className="flex flex-col justify-center">
                 <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                   Keypoints Detected
                 </span>
                 <span className="text-xl font-bold text-foreground leading-none mt-0.5">
                   {totalKeypoints}
                 </span>
               </div>
            </div>
          ) : (
            <div className="h-full rounded-xl border border-dashed border-border/40 bg-secondary/5 flex items-center justify-center">
              <span className="text-xs text-muted-foreground font-medium">Run detection to see stats</span>
            </div>
          )}
        </div>

        {/* Parameters Panel */}
        <div className="flex flex-col gap-3.5 p-4 rounded-xl border border-border/50 bg-secondary/10 shadow-sm shrink-0">
          <div className="flex items-center gap-2 mb-0.5">
            <Settings2 className="w-4 h-4 text-muted-foreground" />
            <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
              Detection Parameters
            </h3>
          </div>

          {/* Contrast Threshold Slider */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between items-center">
              <label className="text-xs font-medium text-foreground">Contrast Thr</label>
              <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                {contrastThr}
              </span>
            </div>
            <input 
              type="range" 
              min="0.01" max="0.1" step="0.01"
              value={contrastThr} 
              onChange={(e) => setContrastThr(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>

          {/* Edge Threshold Slider */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between items-center">
              <label className="text-xs font-medium text-foreground">Edge Thr</label>
              <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                {edgeThr}
              </span>
            </div>
            <input 
              type="range" 
              min="5" max="20" step="1" 
              value={edgeThr} 
              onChange={(e) => setEdgeThr(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>

          {/* Sigma Slider */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between items-center">
              <label className="text-xs font-medium text-foreground">Sigma</label>
              <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                {sigma}
              </span>
            </div>
            <input 
              type="range" 
              min="0.5" max="3.0" step="0.1" 
              value={sigma} 
              onChange={(e) => setSigma(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>

          <button
            onClick={handleFindKeypoints}
            disabled={!inputImage || isProcessing}
            className="mt-1 flex items-center justify-center gap-2 w-full bg-primary text-primary-foreground py-2.5 rounded-lg text-sm font-semibold shadow-sm hover:bg-primary/90 focus:ring-2 focus:ring-primary/50 focus:ring-offset-2 focus:ring-offset-background disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Find Keypoints
              </>
            )}
          </button>
        </div>
      </div>

      {/* RIGHT COLUMN: Output Viewer */}
      <div className="flex-1 min-w-0 h-full">
        <OutputViewer 
          image={outputImage} 
          computationTime={computationTime}
          label="SIFT Feature Descriptors" 
        />
      </div>
    </div>
  );
};

export default SiftTab;