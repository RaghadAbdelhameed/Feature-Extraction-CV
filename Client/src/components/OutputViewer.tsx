import { ImageIcon, Clock, TrendingUp } from "lucide-react";

interface OutputViewerProps {
  image: string | null;
  computationTime?: number | null;
  label?: string;
  matches?: any[];
}

const OutputViewer = ({ image, computationTime, label = "Output", matches = [] }: OutputViewerProps) => {
  return (
    <div className="flex flex-col gap-1.5 h-full">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{label}</span>
        <div className="flex items-center gap-3">
          {matches.length > 0 && (
            <div className="flex items-center gap-1 text-xs font-mono text-green-600">
              <TrendingUp className="w-3 h-3" />
              {matches.length} matches
            </div>
          )}
          {computationTime !== null && computationTime !== undefined && (
            <div className="flex items-center gap-1 text-xs font-mono text-primary">
              <Clock className="w-3 h-3" />
              {computationTime.toFixed(2)}ms
            </div>
          )}
        </div>
      </div>
      <div className="flex-1 rounded-lg border border-border/50 bg-secondary/20 flex items-center justify-center overflow-hidden">
        {image ? (
          <img src={image} alt="Output" className="w-full h-full object-contain" />
        ) : (
          <div className="flex flex-col items-center gap-2 text-muted-foreground/50">
            <ImageIcon className="w-10 h-10" />
            <span className="text-xs">Upload images and click Match</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default OutputViewer;