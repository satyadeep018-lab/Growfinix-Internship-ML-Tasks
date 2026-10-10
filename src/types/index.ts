export type EmotionClass = 'Angry' | 'Happy' | 'Sad';

export interface AcousticAnalysis {
  predictedEmotion: EmotionClass;
  confidence: number;
  probabilities: Record<EmotionClass, number>;
  meanPitchHz: number;
  pitchStdHz: number;
  maxEnergy: number;
  meanEnergy: number;
  spectralCentroidHz: number;
  diagnosticNotes: string[];
  pitchContour: number[];
  energyContour: number[];
  spectrogramMatrix: number[][]; // 2D intensity grid
}

export interface CodeFile {
  name: string;
  category: 'core' | 'model' | 'ui' | 'docs' | 'config';
  description: string;
  code: string;
}
