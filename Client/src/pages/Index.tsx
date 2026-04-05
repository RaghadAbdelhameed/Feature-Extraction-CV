import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Crosshair, Fingerprint, GitCompare } from "lucide-react";
import HarrisTab from "@/components/HarrisTab";
import SiftTab from "@/components/SiftTab";
import MatchingTab from "@/components/MatchingTab";

const Index = () => {
  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-background p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center glow-border">
            <Crosshair className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-foreground tracking-tight">FeatureVision</h1>
            <p className="text-[10px] text-muted-foreground">Feature Extraction & Matching</p>
          </div>
        </div>
      </div>

      {/* Main Tabs */}
      <Tabs defaultValue="harris" className="flex-1 flex flex-col min-h-0">
        <TabsList className="w-fit bg-secondary/50 border border-border/50 p-1 mb-3">
          <TabsTrigger
            value="harris"
            className="gap-1.5 text-xs data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:tab-active-glow"
          >
            <Crosshair className="w-3.5 h-3.5" />
            Harris Corner
          </TabsTrigger>
          <TabsTrigger
            value="sift"
            className="gap-1.5 text-xs data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:tab-active-glow"
          >
            <Fingerprint className="w-3.5 h-3.5" />
            SIFT
          </TabsTrigger>
          <TabsTrigger
            value="matching"
            className="gap-1.5 text-xs data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:tab-active-glow"
          >
            <GitCompare className="w-3.5 h-3.5" />
            Feature Matching
          </TabsTrigger>
        </TabsList>

        <TabsContent value="harris" className="flex-1 min-h-0 mt-0">
          <HarrisTab />
        </TabsContent>
        <TabsContent value="sift" className="flex-1 min-h-0 mt-0">
          <SiftTab />
        </TabsContent>
        <TabsContent value="matching" className="flex-1 min-h-0 mt-0">
          <MatchingTab />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Index;
