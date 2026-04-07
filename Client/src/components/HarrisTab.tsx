import { useState } from "react";
import ImageUploader from "./ImageUploader";
import OutputViewer from "./OutputViewer";
import { Play, Loader2, Crosshair, Settings2 } from "lucide-react";

const HarrisTab = () => {
  // Image states
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [outputImage, setOutputImage] = useState<string | null>(null);
  
  // Processing states
  const [isProcessing, setIsProcessing] = useState(false);
  const [computationTime, setComputationTime] = useState<number | null>(null);
  const [totalPoints, setTotalPoints] = useState<number | null>(null);

  // Parameter states
  const [method, setMethod] = useState<string>("harris");
  const [blockSize, setBlockSize] = useState<number>(3);
  const [kValue, setKValue] = useState<number>(0.04);
  const [threshold, setThreshold] = useState<number>(0.02);

  const handleRunDetection = async () => {
    if (!inputImage) return;
    
    setIsProcessing(true);
    setComputationTime(null);
    setTotalPoints(null);

    try {
      const res = await fetch(inputImage);
      const blob = await res.blob();
      const file = new File([blob], "upload.jpg", { type: "image/jpeg" });

      const formData = new FormData();
      formData.append("image", file);
      formData.append("method", method); 
      formData.append("block_size", blockSize.toString());
      formData.append("k", kValue.toString());
      formData.append("threshold_ratio", threshold.toString());

      const response = await fetch("http://127.0.0.1:5000/api/harris", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();
      
      setOutputImage(data.result_image_base64);
      setComputationTime(data.computation_time_seconds * 1000); 
      setTotalPoints(data.total_points);

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
        
        {/* Image Uploader (Slightly shorter to guarantee it all fits) */}
        <div className="h-32 shrink-0">
          <ImageUploader image={inputImage} onImageChange={setInputImage} />
        </div>

        {/* Stats Panel Slot (Always takes up space to prevent layout shift!) */}
        <div className="h-[68px] shrink-0">
          {totalPoints !== null ? (
            <div className="h-full px-3 py-2 rounded-xl border border-border/50 bg-secondary/10 flex items-center gap-3 shadow-sm">
               <div className="p-2 rounded-lg bg-primary/20">
                  <Crosshair className="w-5 h-5 text-primary" />
               </div>
               <div className="flex flex-col justify-center">
                 <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                   Corners Detected
                 </span>
                 <span className="text-xl font-bold text-foreground leading-none mt-0.5">
                   {totalPoints}
                 </span>
               </div>
            </div>
          ) : (
            <div className="h-full rounded-xl border border-dashed border-border/40 bg-secondary/5 flex items-center justify-center">
              <span className="text-xs text-muted-foreground font-medium">Run detection to see stats</span>
            </div>
          )}
        </div>

        {/* Parameters Panel (Fixed layout via visibility preservation) */}
        <div className="flex flex-col gap-3.5 p-4 rounded-xl border border-border/50 bg-secondary/10 shadow-sm shrink-0">
          <div className="flex items-center gap-2 mb-0.5">
            <Settings2 className="w-4 h-4 text-muted-foreground" />
            <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
              Detection Parameters
            </h3>
          </div>

          {/* Method Selector */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-foreground">Algorithm</label>
            <select 
              value={method} 
              onChange={(e) => setMethod(e.target.value)}
              className="w-full bg-secondary/30 border border-border/50 rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all cursor-pointer"
            >
              <option value="harris">Harris Corner Detection</option>
              <option value="shi-tomasi">Shi-Tomasi (λ-)</option>
            </select>
          </div>

          {/* Block Size Slider */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between items-center">
              <label className="text-xs font-medium text-foreground">Block Size</label>
              <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                {blockSize} px
              </span>
            </div>
            <input 
              type="range" 
              min="3" max="15" step="2"
              value={blockSize} 
              onChange={(e) => setBlockSize(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>


          {/* Threshold Slider */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between items-center">
              <label className="text-xs font-medium text-foreground">Threshold Ratio</label>
              <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                {threshold}
              </span>
            </div>
            <input 
              type="range" 
              min="0.01" max="0.1" step="0.01" 
              value={threshold} 
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>

          {/* K Value Slider - INVISIBLE WHEN SHI-TOMASI (Keeps panel height exactly the same!) */}
          <div className={`flex flex-col gap-1.5 transition-opacity duration-200 ${method === "harris" ? "opacity-100 visible" : "opacity-0 invisible pointer-events-none"}`}>
            <div className="flex justify-between items-center">
              <label className="text-xs font-medium text-foreground">K Value</label>
              <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                {kValue}
              </span>
            </div>
            <input 
              type="range" 
              min="0.01" max="0.1" step="0.01" 
              value={kValue} 
              onChange={(e) => setKValue(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>

          <button
            onClick={handleRunDetection}
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
                Run Detection
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
          label={method === "harris" ? "Harris Corner Result" : "Shi-Tomasi Result"} 
        />
      </div>
    </div>
  );
};

export default HarrisTab;