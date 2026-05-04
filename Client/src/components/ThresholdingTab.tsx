import { useState } from "react";
import ImageUploader from "./ImageUploader";
import OutputViewer from "./OutputViewer";
import { Play, Loader2, ScanLine, Settings2 } from "lucide-react";

type Method = "optimal" | "otsu" | "spectral" | "local";

const METHOD_LABELS: Record<Method, string> = {
  optimal:  "Optimal (Iterative)",
  otsu:     "Otsu's Method",
  spectral: "Spectral (Multi-Modal)",
  local:    "Local Thresholding",
};

const METHOD_DESCRIPTIONS: Record<Method, string> = {
  optimal:  "Iteratively finds the best global threshold by minimizing intra-class variance.",
  otsu:     "Maximizes between-class variance to find the optimal global threshold.",
  spectral: "Handles multi-modal histograms — splits image into more than 2 classes.",
  local:    "Computes a threshold for each pixel based on its local neighbourhood.",
};

const ThresholdingTab = () => {
  const [inputImage, setInputImage]     = useState<string | null>(null);
  const [outputImage, setOutputImage]   = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [computationTime, setComputationTime] = useState<number | null>(null);
  const [threshold, setThreshold]       = useState<number | null>(null);

  // Parameters
  const [method, setMethod]           = useState<Method>("otsu");
  const [nClasses, setNClasses]       = useState<number>(3);   // spectral
  const [blockSize, setBlockSize]     = useState<number>(35);  // local (must be odd)
  const [offset, setOffset]           = useState<number>(10);  // local

  const handleRun = async () => {
    if (!inputImage) return;
    setIsProcessing(true);
    setComputationTime(null);
    setThreshold(null);

    try {
      const res  = await fetch(inputImage);
      const blob = await res.blob();
      const file = new File([blob], "upload.jpg", { type: "image/jpeg" });

      const formData = new FormData();
      formData.append("image",      file);
      formData.append("method",     method);
      formData.append("n_classes",  nClasses.toString());
      formData.append("block_size", blockSize.toString());
      formData.append("offset",     offset.toString());

      const response = await fetch("http://127.0.0.1:5000/api/thresholding", {
        method: "POST",
        body:   formData,
      });

      if (!response.ok) throw new Error(`Server error ${response.status}`);

      const data = await response.json();
      setOutputImage(`data:image/png;base64,${data.result_image_base64}`);      setComputationTime(data.computation_time_seconds * 1000);
      if (data.threshold_value !== undefined) {
        setThreshold(data.threshold_value);
      }

    } catch (err) {
      console.error(err);
      alert("Failed to process. Make sure the Python backend is running!");
    } finally {
      setIsProcessing(false);
    }
  };

  // Ensure block size is always odd
  const handleBlockSizeChange = (val: number) => {
    setBlockSize(val % 2 === 0 ? val + 1 : val);
  };

  return (
    <div className="flex h-full gap-5">
      {/* LEFT COLUMN */}
      <div className="w-[320px] shrink-0 flex flex-col gap-3.5">

        {/* Image Uploader */}
        <div className="h-32 shrink-0">
          <ImageUploader image={inputImage} onImageChange={setInputImage} />
        </div>

        {/* Stats */}
        <div className="h-[68px] shrink-0">
          {threshold !== null ? (
            <div className="h-full px-3 py-2 rounded-xl border border-border/50 bg-secondary/10 flex items-center gap-3 shadow-sm">
              <div className="p-2 rounded-lg bg-primary/20">
                <ScanLine className="w-5 h-5 text-primary" />
              </div>
              <div className="flex flex-col justify-center">
                <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                  Threshold Value
                </span>
                <span className="text-xl font-bold text-foreground leading-none mt-0.5">
                  {Array.isArray(threshold)
                    ? (threshold as number[]).join(", ")
                    : typeof threshold === "number"
                    ? threshold.toFixed(1)
                    : "—"}
                </span>
              </div>
            </div>
          ) : (
            <div className="h-full rounded-xl border border-dashed border-border/40 bg-secondary/5 flex items-center justify-center">
              <span className="text-xs text-muted-foreground font-medium">Run thresholding to see stats</span>
            </div>
          )}
        </div>

        {/* Parameters */}
        <div className="flex flex-col gap-3.5 p-4 rounded-xl border border-border/50 bg-secondary/10 shadow-sm shrink-0">
          <div className="flex items-center gap-2 mb-0.5">
            <Settings2 className="w-4 h-4 text-muted-foreground" />
            <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
              Thresholding Parameters
            </h3>
          </div>

          {/* Method Selector */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-foreground">Method</label>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value as Method)}
              className="w-full bg-secondary/30 border border-border/50 rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all cursor-pointer"
            >
              {(Object.keys(METHOD_LABELS) as Method[]).map((m) => (
                <option key={m} value={m}>{METHOD_LABELS[m]}</option>
              ))}
            </select>
            <p className="text-[10px] text-muted-foreground bg-secondary/20 rounded-lg px-3 py-2 leading-relaxed">
              {METHOD_DESCRIPTIONS[method]}
            </p>
          </div>

          {/* N Classes — spectral only */}
          {method === "spectral" && (
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between items-center">
                <label className="text-xs font-medium text-foreground">Number of Classes</label>
                <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                  {nClasses}
                </span>
              </div>
              <input
                type="range" min="3" max="8" step="1"
                value={nClasses}
                onChange={(e) => setNClasses(Number(e.target.value))}
                className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
              />
              <p className="text-[10px] text-muted-foreground">Min 3 classes (more than 2 modes)</p>
            </div>
          )}

          {/* Block Size & Offset — local only */}
          {method === "local" && (
            <>
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-medium text-foreground">Block Size</label>
                  <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                    {blockSize}
                  </span>
                </div>
                <input
                  type="range" min="11" max="99" step="2"
                  value={blockSize}
                  onChange={(e) => handleBlockSizeChange(Number(e.target.value))}
                  className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <p className="text-[10px] text-muted-foreground">Must be odd — neighbourhood window size</p>
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-medium text-foreground">Offset</label>
                  <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                    {offset}
                  </span>
                </div>
                <input
                  type="range" min="0" max="30" step="1"
                  value={offset}
                  onChange={(e) => setOffset(Number(e.target.value))}
                  className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
                />
              </div>
            </>
          )}

          <button
            onClick={handleRun}
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
                Run Thresholding
              </>
            )}
          </button>
        </div>
      </div>

      {/* RIGHT COLUMN */}
      <div className="flex-1 min-w-0 h-full">
        <OutputViewer
          image={outputImage}
          computationTime={computationTime}
          label={`${METHOD_LABELS[method]} Result`}
        />
      </div>
    </div>
  );
};

export default ThresholdingTab;
