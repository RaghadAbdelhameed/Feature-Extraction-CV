import { useState } from "react";
import ImageUploader from "./ImageUploader";
import OutputViewer from "./OutputViewer";
import { Play, Loader2, Layers, Settings2 } from "lucide-react";

type Method = "kmeans" | "region_growing" | "agglomerative" | "mean_shift";

const METHOD_LABELS: Record<Method, string> = {
  kmeans:          "K-Means",
  region_growing:  "Region Growing",
  agglomerative:   "Agglomerative",
  mean_shift:      "Mean Shift",
};

const SegmentationTab = () => {
  const [inputImage, setInputImage]     = useState<string | null>(null);
  const [outputImage, setOutputImage]   = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [computationTime, setComputationTime] = useState<number | null>(null);
  const [nSegments, setNSegments]       = useState<number | null>(null);

  // Parameters
  const [method, setMethod]         = useState<Method>("kmeans");
  const [nClusters, setNClusters]   = useState<number>(4);
  const [threshold, setThreshold]   = useState<number>(15);

  const needsClusters = method === "kmeans" || method === "agglomerative";
  const needsThreshold = method === "region_growing";

  const handleRun = async () => {
    if (!inputImage) return;
    setIsProcessing(true);
    setComputationTime(null);
    setNSegments(null);

    try {
      const res  = await fetch(inputImage);
      const blob = await res.blob();
      const file = new File([blob], "upload.jpg", { type: "image/jpeg" });

      const formData = new FormData();
      formData.append("image",      file);
      formData.append("method",     method);
      formData.append("n_clusters", nClusters.toString());
      formData.append("threshold",  threshold.toString());

      const response = await fetch("http://127.0.0.1:5000/api/segmentation", {
        method: "POST",
        body:   formData,
      });

      if (!response.ok) throw new Error(`Server error ${response.status}`);

      const data = await response.json();
      setOutputImage(data.result_image_base64);
      setComputationTime(data.computation_time_seconds * 1000);
      setNSegments(data.n_segments);

    } catch (err) {
      console.error(err);
      alert("Failed to process. Make sure the Python backend is running!");
    } finally {
      setIsProcessing(false);
    }
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
          {nSegments !== null ? (
            <div className="h-full px-3 py-2 rounded-xl border border-border/50 bg-secondary/10 flex items-center gap-3 shadow-sm">
              <div className="p-2 rounded-lg bg-primary/20">
                <Layers className="w-5 h-5 text-primary" />
              </div>
              <div className="flex flex-col justify-center">
                <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                  Segments Found
                </span>
                <span className="text-xl font-bold text-foreground leading-none mt-0.5">
                  {nSegments}
                </span>
              </div>
            </div>
          ) : (
            <div className="h-full rounded-xl border border-dashed border-border/40 bg-secondary/5 flex items-center justify-center">
              <span className="text-xs text-muted-foreground font-medium">Run segmentation to see stats</span>
            </div>
          )}
        </div>

        {/* Parameters */}
        <div className="flex flex-col gap-3.5 p-4 rounded-xl border border-border/50 bg-secondary/10 shadow-sm shrink-0">
          <div className="flex items-center gap-2 mb-0.5">
            <Settings2 className="w-4 h-4 text-muted-foreground" />
            <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
              Segmentation Parameters
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
          </div>

          {/* N Clusters — shown for kmeans & agglomerative */}
          <div className={`flex flex-col gap-1.5 transition-opacity duration-200 ${needsClusters ? "opacity-100 visible" : "opacity-0 invisible pointer-events-none"}`}>
            <div className="flex justify-between items-center">
              <label className="text-xs font-medium text-foreground">Number of Clusters</label>
              <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                {nClusters}
              </span>
            </div>
            <input
              type="range" min="2" max="10" step="1"
              value={nClusters}
              onChange={(e) => setNClusters(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>

          {/* Threshold — shown for region growing */}
          <div className={`flex flex-col gap-1.5 transition-opacity duration-200 ${needsThreshold ? "opacity-100 visible" : "opacity-0 invisible pointer-events-none"}`}>
            <div className="flex justify-between items-center">
              <label className="text-xs font-medium text-foreground">Similarity Threshold</label>
              <span className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded-md">
                {threshold}
              </span>
            </div>
            <input
              type="range" min="5" max="60" step="5"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>

          {/* Mean shift note */}
          {method === "mean_shift" && (
            <p className="text-[10px] text-muted-foreground bg-secondary/20 rounded-lg px-3 py-2">
              Mean Shift auto-detects the number of segments — no parameters needed.
            </p>
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
                Run Segmentation
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

export default SegmentationTab;