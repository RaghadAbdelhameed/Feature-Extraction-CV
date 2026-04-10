import { useState } from "react";
import { Play, RotateCcw, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import ImageUploader from "./ImageUploader";
import OutputViewer from "./OutputViewer";
import { useToast } from "@/hooks/use-toast";

type MatchMethod = "ncc" | "ssd";

const METHOD_LABELS: Record<MatchMethod, string> = {
  ncc: "NCC — Normalised Cross-Correlation",
  ssd: "SSD — Sum of Squared Differences",
};

const MatchingTab = () => {
  const [image1, setImage1]                   = useState<string | null>(null);
  const [image2, setImage2]                   = useState<string | null>(null);
  const [outputImage, setOutputImage]         = useState<string | null>(null);
  const [method, setMethod]                   = useState<MatchMethod>("ncc");
  const [dropdownOpen, setDropdownOpen]       = useState(false);
  const [matchThreshold, setMatchThreshold]   = useState(0.85);
  const [patchSize, setPatchSize]             = useState(21);
  const [numKeypoints, setNumKeypoints]       = useState(200);
  const [isLoading, setIsLoading]             = useState(false);
  const [computationTime, setComputationTime] = useState<number | null>(null);
  const [matches, setMatches]                 = useState<any[]>([]);

  const { toast } = useToast();

  const handleReset = () => {
    setOutputImage(null);
    setComputationTime(null);
    setMatches([]);
  };

  const handleMatch = async () => {
    if (!image1 || !image2) {
      toast({
        title: "Missing Images",
        description: "Upload both images before matching.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setOutputImage(null);
    setComputationTime(null);

    try {
      const res = await fetch("http://localhost:5000/api/ncc/feature-match", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image1,
          image2,
          method,                         // 'ncc' or 'ssd'
          patch_size:    patchSize,
          num_keypoints: numKeypoints,
          ratio_thresh:  matchThreshold,
        }),
      });

      const data = await res.json();

      if (data.success) {
        setOutputImage(data.visualization);
        setComputationTime(data.computational_time * 1000);
        setMatches(data.matches ?? []);

        if (data.num_matches === 0) {
          toast({
            title: "No Matches Found",
            description: "Try lowering the threshold or increasing keypoints.",
          });
        } else {
          toast({
            title: "Match Complete!",
            description: `${data.num_matches} matches · ${(data.computational_time * 1000).toFixed(1)} ms`,
          });
        }
      } else {
        toast({
          title: "Matching Failed",
          description: data.error ?? "Unknown error.",
          variant: "destructive",
        });
      }
    } catch {
      toast({
        title: "Connection Error",
        description: "Cannot reach backend on port 5000.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const bestScore = matches.length
    ? Math.max(...matches.map((m) => m.ncc_score))
    : null;
  const meanScore = matches.length
    ? matches.reduce((s, m) => s + m.ncc_score, 0) / matches.length
    : null;

  return (
    <div className="flex h-full gap-4">
      {/* ── LEFT PANEL ─────────────────────────────────────────────── */}
      <div className="w-1/4 flex flex-col gap-2 min-w-0">

        {/* Image uploaders */}
        <div className="flex gap-2 h-[80px]">
          <div className="flex-1 min-w-0">
            <ImageUploader image={image1} onImageChange={setImage1} label="Image 1" compact />
          </div>
          <div className="flex-1 min-w-0">
            <ImageUploader image={image2} onImageChange={setImage2} label="Image 2" compact />
          </div>
        </div>

        {/* Parameter panel */}
        <div className="flex-1 glass-panel p-3 flex flex-col gap-3 overflow-y-auto">
          <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">
            Matching Parameters
          </h3>

          {/* Method dropdown */}
          <div className="space-y-1.5">
            <span className="text-xs text-muted-foreground">Method</span>
            <div className="relative">
              <button
                onClick={() => setDropdownOpen((o) => !o)}
                className="w-full flex items-center justify-between text-xs py-2 px-3 rounded-md font-medium bg-primary text-primary-foreground tab-active-glow hover:opacity-90 transition-opacity"
              >
                <span>{METHOD_LABELS[method]}</span>
                <ChevronDown
                  className={`w-3 h-3 ml-2 shrink-0 transition-transform ${dropdownOpen ? "rotate-180" : ""}`}
                />
              </button>

              {dropdownOpen && (
                <div className="absolute z-50 top-full mt-1 w-full rounded-md border border-border bg-background shadow-lg overflow-hidden">
                  {(Object.keys(METHOD_LABELS) as MatchMethod[]).map((m) => (
                    <button
                      key={m}
                      onClick={() => { setMethod(m); setDropdownOpen(false); }}
                      className={`w-full text-left text-xs px-3 py-2 transition-colors
                        ${m === method
                          ? "bg-primary/10 text-primary font-semibold"
                          : "hover:bg-secondary text-foreground"
                        }
                        ${m === "ssd" ? "opacity-60" : ""}
                      `}
                    >
                      {METHOD_LABELS[m]}
                      {m === "ssd" && (
                        <span className="ml-2 text-[10px] bg-muted text-muted-foreground rounded px-1">
                        
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Ratio threshold */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">
                {method === "ssd" ? "Ratio Threshold" : "NCC Threshold"}
              </span>
              <span className="font-mono text-primary">{matchThreshold.toFixed(2)}</span>
            </div>
            <Slider
              value={[matchThreshold]}
              onValueChange={([v]) => setMatchThreshold(v)}
              min={0.50}
              max={0.99}
              step={0.01}
            />
            <p className="text-[10px] text-muted-foreground">
              {method === "ssd"
                ? "Lower → stricter (fewer but better)"
                : "Higher → stricter (fewer but better)"}
            </p>
          </div>

          {/* Patch size */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Patch Size</span>
              <span className="font-mono text-primary">{patchSize} × {patchSize}</span>
            </div>
            <Slider
              value={[patchSize]}
              onValueChange={([v]) => setPatchSize(v % 2 === 0 ? v + 1 : v)}
              min={7}
              max={41}
              step={2}
            />
            <p className="text-[10px] text-muted-foreground">
              Odd integers only · larger = more context
            </p>
          </div>

          {/* SIFT keypoints */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">SIFT Keypoints</span>
              <span className="font-mono text-primary">{numKeypoints}</span>
            </div>
            <Slider
              value={[numKeypoints]}
              onValueChange={([v]) => setNumKeypoints(v)}
              min={50}
              max={400}
              step={25}
            />
            <p className="text-[10px] text-muted-foreground">
              Top-N SIFT keypoints used per image
            </p>
          </div>

          {/* Actions */}
          <div className="flex gap-2 mt-auto pt-1">
            <Button
              disabled={!image1 || !image2 || isLoading}
              className="flex-1 gap-1.5 text-xs"
              size="sm"
              onClick={handleMatch}
            >
              <Play className="w-3 h-3" />
              {isLoading ? "Matching…" : "Match"}
            </Button>
            <Button
              onClick={handleReset}
              variant="outline"
              size="sm"
              className="text-xs"
            >
              <RotateCcw className="w-3 h-3" />
            </Button>
          </div>

          {/* Match statistics */}
          {matches.length > 0 && (
            <div className="p-2 bg-green-500/10 rounded text-[10px] space-y-0.5">
              <p className="font-semibold text-xs text-green-600">
                ✓ {matches.length} {method.toUpperCase()} matches
              </p>
              
            </div>
          )}
        </div>
      </div>

      {/* ── RIGHT PANEL ──────────────────────────────────────────── */}
      <div className="flex-1 min-w-0">
        <OutputViewer
          image={outputImage}
          label={`${method.toUpperCase()} Feature Matching`}
          computationTime={computationTime}
          matches={matches}
        />
      </div>
    </div>
  );
};

export default MatchingTab;
