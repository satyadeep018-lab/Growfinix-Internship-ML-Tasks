import React, { useState } from 'react';
import { Copy, Check, Download, FileCode, Folder } from 'lucide-react';
import { PROJECT_FILES } from '../data/projectFiles';
import { downloadProjectZip } from '../utils/zipExport';

export const CodeViewer: React.FC = () => {
  const [activeFileName, setActiveFileName] = useState<string>('app.py');
  const [copied, setCopied] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  const activeFile = PROJECT_FILES.find((f) => f.name === activeFileName) || PROJECT_FILES[0];

  const handleCopy = () => {
    navigator.clipboard.writeText(activeFile.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      await downloadProjectZip();
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Header with Download Action */}
      <div className="flex flex-col gap-3 rounded-xl border border-slate-800 bg-slate-900/80 p-5 sm:flex-row sm:items-center sm:justify-between shadow-sm">
        <div>
          <h2 className="text-base font-semibold text-white">Repository Source Files</h2>
          <p className="text-xs text-slate-400">
            Complete, beginner-friendly standalone Python scripts ready to run locally and push to GitHub.
          </p>
        </div>

        <button
          onClick={handleDownload}
          disabled={isDownloading}
          className="flex items-center justify-center gap-2 rounded-lg bg-sky-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-sky-500 transition-colors"
        >
          <Download className="h-4 w-4" />
          <span>{isDownloading ? 'Bundling ZIP...' : 'Download Project ZIP'}</span>
        </button>
      </div>

      {/* Main File Explorer Layout */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        {/* Left Column: File Tree */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 lg:col-span-4 shadow-sm">
          <div className="flex items-center gap-2 pb-3 border-b border-slate-800 text-xs font-semibold text-slate-300">
            <Folder className="h-4 w-4 text-sky-400" />
            <span>task_1_audio_emotion/</span>
          </div>

          <div className="mt-3 space-y-1">
            {PROJECT_FILES.map((file) => {
              const isActive = file.name === activeFileName;
              return (
                <button
                  key={file.name}
                  onClick={() => setActiveFileName(file.name)}
                  className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-xs transition-colors ${
                    isActive
                      ? 'bg-sky-500/15 font-semibold text-sky-300 border border-sky-500/30'
                      : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
                  }`}
                >
                  <div className="flex items-center gap-2 truncate">
                    <FileCode className={`h-3.5 w-3.5 shrink-0 ${isActive ? 'text-sky-400' : 'text-slate-500'}`} />
                    <span className="truncate">{file.name}</span>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500 uppercase">
                    {file.category}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="mt-6 rounded-lg border border-slate-800/80 bg-slate-950/60 p-3 text-[11px] text-slate-400">
            <p className="font-semibold text-slate-300">Directory Structure</p>
            <p className="mt-1">
              Includes pre-configured data directories, model checkpoint paths, and output plots under 8MB to stay within GitHub limits.
            </p>
          </div>
        </div>

        {/* Right Column: Code Display */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/80 lg:col-span-8 overflow-hidden shadow-sm flex flex-col">
          {/* File Tab Header */}
          <div className="flex items-center justify-between border-b border-slate-800 bg-slate-950/80 px-4 py-2.5">
            <div className="flex items-center gap-2">
              <FileCode className="h-4 w-4 text-sky-400" />
              <span className="text-xs font-semibold text-white">{activeFile.name}</span>
            </div>

            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-800/60 px-2.5 py-1 text-[11px] font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition-colors"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5 text-emerald-400" />
                  <span className="text-emerald-400">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5 text-slate-400" />
                  <span>Copy Code</span>
                </>
              )}
            </button>
          </div>

          {/* Description Callout */}
          <div className="border-b border-slate-800 bg-slate-950/40 px-4 py-2 text-xs text-slate-400">
            {activeFile.description}
          </div>

          {/* Code Body */}
          <div className="max-h-[540px] overflow-auto p-4 bg-slate-950">
            <pre className="font-mono text-xs text-slate-300 leading-relaxed">
              <code>{activeFile.code}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
