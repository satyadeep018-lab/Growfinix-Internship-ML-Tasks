import React, { useState } from 'react';
import { Github, Video, Check, Copy, ExternalLink, Terminal, AlertCircle } from 'lucide-react';

export const GitHubGuide: React.FC = () => {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const gitCommands = [
    {
      title: '1. Navigate to your project folder',
      command: 'cd task_1_audio_emotion',
      comment: 'Opens the directory where all task files are stored.',
    },
    {
      title: '2. Initialize Git repository',
      command: 'git init',
      comment: 'Creates a hidden .git tracker in your folder.',
    },
    {
      title: '3. Stage and commit files',
      command: 'git add .\ngit commit -m "Complete Task 1: Spoken Emotion Recognition with Librosa, CNN, and Streamlit"',
      comment: 'Packages your code with a clean commit message.',
    },
    {
      title: '4. Rename default branch to main',
      command: 'git branch -M main',
      comment: 'Ensures the primary branch conforms to GitHub standards.',
    },
    {
      title: '5. Connect to your GitHub repository',
      command: 'git remote add origin https://github.com/<your-username>/Growfinix-Internship-Tasks.git',
      comment: 'Replace <your-username> with your actual GitHub username.',
    },
    {
      title: '6. Push code to GitHub',
      command: 'git push -u origin main',
      comment: 'Uploads all code, README, and scripts to your public GitHub repo.',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-5 shadow-sm">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-800 text-white">
              <Github className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Beginner GitHub Upload Guide</h2>
              <p className="text-xs text-slate-400">
                Official Growfinix repository requirement: name repository containing &ldquo;Growfinix&rdquo; (e.g.{' '}
                <code className="text-sky-300 font-mono">Growfinix-Internship-Tasks</code>).
              </p>
            </div>
          </div>

          <a
            href="https://github.com/new"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-slate-700 transition-colors"
          >
            <span>Create Repo on GitHub</span>
            <ExternalLink className="h-3.5 w-3.5 text-slate-400" />
          </a>
        </div>
      </div>

      {/* Terminal Steps */}
      <div className="space-y-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Terminal className="h-4 w-4 text-sky-400" />
          <span>Step-by-Step Terminal Commands</span>
        </h3>

        <div className="space-y-3">
          {gitCommands.map((step, idx) => (
            <div
              key={idx}
              className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 transition-colors hover:border-slate-700"
            >
              <div className="flex items-center justify-between text-xs font-medium text-slate-300">
                <span>{step.title}</span>
                <span className="text-[11px] text-slate-500">{step.comment}</span>
              </div>

              <div className="mt-2 flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                <pre className="font-mono text-xs text-sky-300 overflow-x-auto whitespace-pre-wrap">
                  <code>{step.command}</code>
                </pre>
                <button
                  onClick={() => copyToClipboard(step.command, idx)}
                  className="ml-3 shrink-0 rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
                  title="Copy command"
                >
                  {copiedIndex === idx ? (
                    <Check className="h-4 w-4 text-emerald-400" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Important Beginner Note */}
      <div className="flex items-start gap-3 rounded-xl border border-amber-500/20 bg-amber-950/15 p-4 text-xs text-amber-300">
        <AlertCircle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
        <div>
          <p className="font-semibold text-amber-200">How to prevent Git &ldquo;File too large (&gt; 100MB)&rdquo; errors:</p>
          <p className="mt-1 text-amber-300/80">
            Never push the raw CREMA-D 2GB folder or Python virtual environment (<code className="font-mono">venv/</code>). The included{' '}
            <code className="font-mono text-amber-200">.gitignore</code> file is already pre-configured to keep your repository under 15MB, making GitHub pushes instant and error-free.
          </p>
        </div>
      </div>

      {/* LinkedIn Video Script */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-5 shadow-sm space-y-4">
        <div className="flex items-center gap-2 text-white">
          <Video className="h-4 w-4 text-sky-400" />
          <h3 className="text-base font-semibold">LinkedIn Video Script (60–90 Seconds)</h3>
        </div>
        <p className="text-xs text-slate-400">
          Growfinix requires a short video demonstration on LinkedIn tagging{' '}
          <strong className="text-slate-200">@GrowfinixTechnology</strong>. Use this timestamped script while recording your screen:
        </p>

        <div className="space-y-3 text-xs">
          <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
            <span className="font-mono font-bold text-sky-400">[0:00 - 0:15] Introduction</span>
            <p className="mt-1 text-slate-300 italic">
              &ldquo;Hello everyone! As part of my Month 2 Machine Learning Internship with Growfinix Technology, I have completed Task 1: Spoken Emotion Recognition from Audio.&rdquo;
            </p>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
            <span className="font-mono font-bold text-sky-400">[0:15 - 0:35] Problem & Tech Stack</span>
            <p className="mt-1 text-slate-300 italic">
              &ldquo;The objective is to classify speech audio into Angry, Happy, and Sad. Using Librosa, raw waveforms were converted into 2D Log-Mel Spectrograms. I then designed a 2D Convolutional Neural Network with TensorFlow/Keras and evaluated it using speaker-independent data splits to avoid data leakage.&rdquo;
            </p>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
            <span className="font-mono font-bold text-sky-400">[0:35 - 1:05] Live UI Demonstration</span>
            <p className="mt-1 text-slate-300 italic">
              *(Show screen of the Streamlit app)* &ldquo;Here is the Streamlit app. We can record live on the spot using the microphone or upload a WAV file. The CNN displays real-time prediction confidence, and the explainability panel graphs pitch contour and RMS energy to show why an emotion was detected.&rdquo;
            </p>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
            <span className="font-mono font-bold text-sky-400">[1:05 - 1:20] Closing & Mentions</span>
            <p className="mt-1 text-slate-300 italic">
              &ldquo;The complete code and documentation are available on my GitHub repository. Thank you to Growfinix Technology for this practical learning experience!&rdquo;
            </p>
          </div>
        </div>

        {/* Copyable Post Caption */}
        <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300 mb-2">
            <span>Ready-to-Post LinkedIn Caption</span>
            <button
              onClick={() =>
                copyToClipboard(
                  `Excited to share my completed Task 1 for the Month 2 Machine Learning Internship with @GrowfinixTechnology! 🚀\n\n🎯 Project: Spoken Emotion Recognition from Audio\n🛠 Tech Stack: Python, Librosa, TensorFlow/Keras (CNN), Streamlit\n📈 Highlights: Log-Mel Spectrograms, 2D CNN with speaker-isolated validation, live mic recording & explainability graphs.\n\nCode on GitHub: [Your Repo URL]\n\n#MachineLearning #DeepLearning #GrowfinixTechnology #Internship #AudioProcessing #TensorFlow`,
                  99
                )
              }
              className="text-sky-400 hover:text-sky-300"
            >
              {copiedIndex === 99 ? 'Copied!' : 'Copy Caption'}
            </button>
          </div>
          <p className="text-[11px] text-slate-400 font-mono whitespace-pre-line leading-relaxed">
            {`Excited to share my completed Task 1 for the Month 2 Machine Learning Internship with @GrowfinixTechnology! 🚀

🎯 Project: Spoken Emotion Recognition from Audio
🛠 Tech Stack: Python, Librosa, TensorFlow/Keras (CNN), Streamlit
📈 Highlights: Log-Mel Spectrograms, 2D CNN with speaker-isolated validation, live mic recording & explainability graphs.

#MachineLearning #DeepLearning #GrowfinixTechnology #Internship #AudioProcessing #TensorFlow`}
          </p>
        </div>
      </div>
    </div>
  );
};
