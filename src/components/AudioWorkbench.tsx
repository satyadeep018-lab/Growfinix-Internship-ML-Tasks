import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Upload, Play, Pause, Volume2 } from 'lucide-react';
import { AcousticAnalysis, EmotionClass } from '../types';
import { generateEmotionalSpeechAudio, analyzeAudioBuffer } from '../utils/audioSynthesis';

export const AudioWorkbench: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [sourceName, setSourceName] = useState<string>('Select an audio source');
  const [analysis, setAnalysis] = useState<AcousticAnalysis | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const spectrogramCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Initialize with a demo sample
  useEffect(() => {
    handleLoadDemoSample('Angry');
  }, []);

  // Spectrogram canvas rendering
  useEffect(() => {
    if (!analysis || !spectrogramCanvasRef.current) return;
    const canvas = spectrogramCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const matrix = analysis.spectrogramMatrix;
    const numRows = matrix.length;
    const numCols = matrix[0].length;
    const cellWidth = canvas.width / numCols;
    const cellHeight = canvas.height / numRows;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < numRows; r++) {
      for (let c = 0; c < numCols; c++) {
        const val = matrix[r][c];
        const norm = Math.max(0, Math.min(1, (val + 80) / 80));
        
        // Heatmap colors
        const red = Math.floor(Math.min(255, norm * 255 * 1.5));
        const green = Math.floor(Math.max(0, (norm - 0.3) * 200));
        const blue = Math.floor(Math.max(0, 180 - norm * 150));
        
        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        ctx.fillRect(c * cellWidth, canvas.height - (r + 1) * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
      }
    }
  }, [analysis]);

  const handleStartRecording = async () => {
    try {
      audioChunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const recordedBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(recordedBlob);
        setAudioUrl(url);
        setSourceName('Live Microphone Recording');
        await runAnalysis(recordedBlob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
      setRecordingSeconds(0);

      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => {
          if (prev >= 5) {
            handleStopRecording();
            return 5;
          }
          return prev + 1;
        });
      }, 1000);
    } catch {
      alert('Microphone access is required to record. You can also upload a WAV file or use the buttons below.');
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    setIsRecording(false);
  };

  const handleLoadDemoSample = async (emotion: EmotionClass) => {
    const { blob, url } = generateEmotionalSpeechAudio(emotion);
    setAudioUrl(url);
    setSourceName(`${emotion} Voice Sample`);
    await runAnalysis(blob);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      setSourceName(file.name);
      await runAnalysis(file);
    }
  };

  const runAnalysis = async (blob: Blob) => {
    try {
      const result = await analyzeAudioBuffer(blob);
      setAnalysis(result);
    } catch (e) {
      console.error(e);
    }
  };

  const togglePlayback = () => {
    if (!audioPlayerRef.current) return;
    if (isPlaying) {
      audioPlayerRef.current.pause();
      setIsPlaying(false);
    } else {
      audioPlayerRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <div className="space-y-5">
      {/* 1. Simple Audio Input Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-sm">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2">
            {!isRecording ? (
              <button
                onClick={handleStartRecording}
                className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-3.5 py-2 text-xs font-semibold text-white hover:bg-sky-500 transition-colors"
              >
                <Mic className="h-4 w-4" />
                <span>Record Voice</span>
              </button>
            ) : (
              <button
                onClick={handleStopRecording}
                className="flex items-center gap-1.5 rounded-lg bg-red-600 px-3.5 py-2 text-xs font-semibold text-white animate-pulse"
              >
                <Square className="h-4 w-4" />
                <span>Stop ({recordingSeconds}s / 5s)</span>
              </button>
            )}

            <label className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition-colors">
              <Upload className="h-3.5 w-3.5" />
              <span>Upload WAV</span>
              <input type="file" accept="audio/wav,audio/*" onChange={handleFileUpload} className="hidden" />
            </label>
          </div>

          {/* Sample quick buttons */}
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-slate-400 mr-1">Test Samples:</span>
            <button
              onClick={() => handleLoadDemoSample('Angry')}
              className="rounded-md border border-red-500/30 bg-red-950/20 px-2.5 py-1 text-xs font-medium text-red-400 hover:bg-red-900/40 transition-colors"
            >
              Angry
            </button>
            <button
              onClick={() => handleLoadDemoSample('Happy')}
              className="rounded-md border border-emerald-500/30 bg-emerald-950/20 px-2.5 py-1 text-xs font-medium text-emerald-400 hover:bg-emerald-900/40 transition-colors"
            >
              Happy
            </button>
            <button
              onClick={() => handleLoadDemoSample('Sad')}
              className="rounded-md border border-sky-500/30 bg-sky-950/20 px-2.5 py-1 text-xs font-medium text-sky-400 hover:bg-sky-900/40 transition-colors"
            >
              Sad
            </button>
          </div>
        </div>

        {/* Audio Player */}
        {audioUrl && (
          <div className="mt-3 flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2">
            <div className="flex items-center gap-2.5">
              <button
                onClick={togglePlayback}
                className="flex h-7 w-7 items-center justify-center rounded-full bg-sky-500/20 text-sky-400 hover:bg-sky-500/30"
              >
                {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5 ml-0.5" />}
              </button>
              <span className="text-xs text-slate-300 font-medium">{sourceName}</span>
            </div>

            <audio
              ref={audioPlayerRef}
              src={audioUrl}
              onEnded={() => setIsPlaying(false)}
              className="hidden"
            />

            <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
              <Volume2 className="h-3.5 w-3.5" />
              <span>Standardized 22,050 Hz</span>
            </div>
          </div>
        )}
      </div>

      {/* 2. Main Layout: Left = Simple Prediction Box, Right = Combined Unified Graphs Box */}
      {analysis && (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
          {/* Left Column: Decreased Simple CNN Box (lg:col-span-4) */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 lg:col-span-4 shadow-sm flex flex-col justify-between">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Prediction
              </p>

              <div className="mt-2 flex items-baseline justify-between">
                <span
                  className={`text-2xl font-bold ${
                    analysis.predictedEmotion === 'Angry'
                      ? 'text-red-400'
                      : analysis.predictedEmotion === 'Happy'
                      ? 'text-emerald-400'
                      : 'text-sky-400'
                  }`}
                >
                  {analysis.predictedEmotion}
                </span>
                <span className="font-mono text-xs text-slate-400 tabular-nums">
                  {(analysis.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>

              {/* Simple 3 Progress Bars */}
              <div className="mt-4 space-y-2">
                {(['Angry', 'Happy', 'Sad'] as EmotionClass[]).map((emo) => {
                  const prob = analysis.probabilities[emo];
                  const isTop = analysis.predictedEmotion === emo;
                  return (
                    <div key={emo} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className={isTop ? 'font-semibold text-white' : 'text-slate-400'}>
                          {emo}
                        </span>
                        <span className="font-mono text-slate-400 tabular-nums text-[11px]">
                          {(prob * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                        <div
                          className={`h-full transition-all duration-300 ${
                            emo === 'Angry'
                              ? 'bg-red-500'
                              : emo === 'Happy'
                              ? 'bg-emerald-500'
                              : 'bg-sky-500'
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Concise Diagnostic Note */}
            <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950/60 p-2.5 text-xs text-slate-300">
              <span className="text-sky-400 font-semibold">Diagnosis: </span>
              <span>{analysis.diagnosticNotes[0] || 'Vocal features match this emotion profile.'}</span>
            </div>
          </div>

          {/* Right Column: Unified Combined Graphs Box (lg:col-span-8) */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 lg:col-span-8 shadow-sm space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <h3 className="text-xs font-semibold text-white">Audio & Acoustic Graphs</h3>
              <div className="flex items-center gap-3 text-[11px]">
                <span className="text-red-400 flex items-center gap-1">● Pitch (F0)</span>
                <span className="text-sky-400 flex items-center gap-1">● Energy (RMS)</span>
              </div>
            </div>

            {/* Spectrogram View */}
            <div>
              <p className="text-[11px] text-slate-400 mb-1">2D Log-Mel Spectrogram (Frequency over Time)</p>
              <div className="overflow-hidden rounded-lg border border-slate-800 bg-black">
                <canvas
                  ref={spectrogramCanvasRef}
                  width={600}
                  height={130}
                  className="h-28 w-full"
                />
              </div>
            </div>

            {/* Combined Acoustic Curve */}
            <div>
              <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                <span>Pitch (Hz) & Energy Dynamics</span>
                <span className="font-mono text-slate-500 text-[10px]">0.0s — 3.0s</span>
              </div>
              <div className="h-24 w-full rounded-lg border border-slate-800 bg-slate-950 p-2">
                <svg className="h-full w-full" viewBox="0 0 100 40" preserveAspectRatio="none">
                  {/* Energy Waveform Curve (Blue) */}
                  <polyline
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    points={analysis.energyContour
                      .map((val, idx) => {
                        const x = (idx / (analysis.energyContour.length - 1)) * 100;
                        const y = 38 - Math.min(36, Math.max(2, (val / (analysis.maxEnergy || 0.1)) * 34));
                        return `${x},${y}`;
                      })
                      .join(' ')}
                  />
                  {/* Pitch Contour Curve (Red) */}
                  <polyline
                    fill="none"
                    stroke="#ef4444"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    points={analysis.pitchContour
                      .map((val, idx) => {
                        const x = (idx / (analysis.pitchContour.length - 1)) * 100;
                        const y = 38 - Math.min(36, Math.max(2, ((val - 80) / 220) * 34));
                        return `${x},${y}`;
                      })
                      .join(' ')}
                  />
                </svg>
              </div>
              <div className="mt-1 flex justify-between text-[10px] text-slate-500 font-mono">
                <span>Avg Pitch: {analysis.meanPitchHz} Hz</span>
                <span>Peak Energy: {analysis.maxEnergy}</span>
                <span>Centroid: {analysis.spectralCentroidHz} Hz</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
