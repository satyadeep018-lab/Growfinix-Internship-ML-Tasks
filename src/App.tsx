import React from 'react';
import { AudioWorkbench } from './components/AudioWorkbench';
import { Waves } from 'lucide-react';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-sky-500 selection:text-white">
      {/* Ultra-Clean Minimal Header - No GitHub or Download clutter */}
      <header className="border-b border-slate-800 bg-slate-950/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sky-500/15 text-sky-400 border border-sky-500/30">
              <Waves className="h-4 w-4" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight text-white">
                Spoken Emotion Recognition
              </h1>
              <p className="text-[11px] text-slate-400">
                Audio Speech Emotion Classifier
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Single-View Workspace */}
      <main className="flex-1 mx-auto w-full max-w-5xl px-4 py-6 sm:px-6">
        <AudioWorkbench />
      </main>

      <footer className="border-t border-slate-900 bg-slate-950 py-3 text-center text-xs text-slate-500">
        <p>Spoken Emotion Recognition · Real-time Audio Classifier</p>
      </footer>
    </div>
  );
}
