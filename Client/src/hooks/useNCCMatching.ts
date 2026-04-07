// Create new file: src/hooks/useNCCMatching.ts
import { useState } from 'react';
import { useToast } from '@/hooks/use-toast';

interface MatchResult {
  visualization: string;
  numMatches: number;
  computationTime: number;
  matches: Array<{
    point1: [number, number];
    point2: [number, number];
    ncc_score: number;
  }>;
}

export function useNCCMatching() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<MatchResult | null>(null);
  const { toast } = useToast();

  const matchFeatures = async (
    image1: string,
    image2: string,
    params: {
      patch_size?: number;
      num_keypoints?: number;
      ratio_thresh?: number;
    } = {}
  ) => {
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/ncc/feature-match', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image1,
          image2,
          patch_size: params.patch_size || 15,
          num_keypoints: params.num_keypoints || 100,
          ratio_thresh: params.ratio_thresh || 0.75,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setResult({
          visualization: data.visualization,
          numMatches: data.num_matches,
          computationTime: data.computational_time,
          matches: data.matches,
        });
        
        toast({
          title: 'Match Complete!',
          description: `Found ${data.num_matches} matches between the images`,
        });
        
        return data;
      } else {
        throw new Error(data.error || 'Matching failed');
      }
    } catch (error) {
      console.error('NCC Matching error:', error);
      toast({
        title: 'Matching Failed',
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: 'destructive',
      });
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
  };

  return {
    matchFeatures,
    reset,
    isLoading,
    result,
  };
}