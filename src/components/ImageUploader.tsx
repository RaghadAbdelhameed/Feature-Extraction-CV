import { Upload, Image as ImageIcon } from "lucide-react";
import { useCallback } from "react";

interface ImageUploaderProps {
  image: string | null;
  onImageChange: (image: string) => void;
  label?: string;
  compact?: boolean;
}

const ImageUploader = ({ image, onImageChange, label = "Input", compact = false }: ImageUploaderProps) => {
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (ev) => onImageChange(ev.target?.result as string);
        reader.readAsDataURL(file);
      }
    },
    [onImageChange]
  );

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => onImageChange(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex flex-col gap-1 h-full">
      <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">{label}</span>
      <label
        className="relative flex-1 flex flex-col items-center justify-center rounded-lg border border-dashed border-border/70 bg-secondary/30 cursor-pointer hover:border-primary/50 hover:bg-secondary/50 transition-all overflow-hidden"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        {image ? (
          <img src={image} alt="Input" className="w-full h-full object-contain" />
        ) : (
          <div className="flex flex-col items-center gap-1 p-2">
            <Upload className={compact ? "w-4 h-4 text-muted-foreground" : "w-6 h-6 text-muted-foreground"} />
            {!compact && <span className="text-xs text-muted-foreground">Drop or click</span>}
          </div>
        )}
        <input type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
      </label>
    </div>
  );
};

export default ImageUploader;
