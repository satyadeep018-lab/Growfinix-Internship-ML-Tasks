import React from 'react';
import { Award, CheckCircle2, Clock, ExternalLink } from 'lucide-react';

export const InternshipRoadmap: React.FC = () => {
  const tasks = [
    {
      id: 1,
      title: 'Task 1: Spoken Emotion Recognition',
      domain: 'Audio & Deep Learning',
      stack: 'Python, Librosa, TensorFlow/Keras CNN, Streamlit',
      status: 'Ready & Implemented',
      description: 'Convert voice into 2D spectrograms and train CNN to predict Anger, Happiness, and Sadness with acoustic explainability.',
    },
    {
      id: 2,
      title: 'Task 2: Spam & Phishing Detection',
      domain: 'Cyber-Security & NLP',
      stack: 'Python, NLTK, PyTorch (LSTMs)',
      status: 'Upcoming Milestone',
      description: 'Analyze text sequences and URL patterns using recurrent deep learning to classify phishing attempts vs genuine communication.',
    },
    {
      id: 3,
      title: 'Task 3: Assembly Line Defect Detection',
      domain: 'Manufacturing & Computer Vision',
      stack: 'Python, OpenCV, PyTorch (ResNet)',
      status: 'Upcoming Milestone',
      description: 'Scan industrial component photos and leverage Transfer Learning to detect Defective vs Perfect products.',
    },
    {
      id: 4,
      title: 'Task 4: Movie Recommender Engine',
      domain: 'Entertainment & Recommender Systems',
      stack: 'Python, Scikit-Surprise, Pandas',
      status: 'Upcoming Milestone',
      description: 'Matrix factorization collaborative filtering system to recommend movies based on historical ratings of similar users.',
    },
    {
      id: 5,
      title: 'Task 5: Automating Machine Learning',
      domain: 'Workflow Automation & AutoML',
      stack: 'Python, PyCaret or H2O.ai',
      status: 'Upcoming Milestone',
      description: 'Train, evaluate, and benchmark multiple ML models in ~10 lines of code, automatically selecting the champion model.',
    },
  ];

  return (
    <div className="space-y-6">
      {/* LOR Criteria Callout */}
      <div className="rounded-xl border border-sky-500/30 bg-sky-950/20 p-5 shadow-sm">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-sky-500/20 text-sky-400">
              <Award className="h-6 w-6" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Letter of Recommendation (LOR) Pathway</h2>
              <p className="text-xs text-slate-300">
                Candidates who complete all 5 tasks and submit LinkedIn videos and GitHub repositories are eligible for a Letter of Recommendation.
              </p>
            </div>
          </div>

          <a
            href="https://www.growfinix.in/task/submit"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-1.5 rounded-lg bg-sky-600 px-4 py-2 text-xs font-semibold text-white hover:bg-sky-500 transition-colors"
          >
            <span>Growfinix Submission Portal</span>
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>

      {/* Task Cards List */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-white">Month 2 Curriculum Breakdown</h3>
        {tasks.map((task) => (
          <div
            key={task.id}
            className={`rounded-xl border p-4 transition-colors ${
              task.id === 1
                ? 'border-sky-500/40 bg-slate-900/90'
                : 'border-slate-800 bg-slate-900/50'
            }`}
          >
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-2.5">
                {task.id === 1 ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
                ) : (
                  <Clock className="h-4 w-4 text-slate-500 shrink-0" />
                )}
                <h4 className="text-sm font-semibold text-white">{task.title}</h4>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono text-slate-400">{task.domain}</span>
                <span
                  className={`rounded-md px-2 py-0.5 text-[10px] font-semibold uppercase ${
                    task.id === 1
                      ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {task.status}
                </span>
              </div>
            </div>

            <p className="mt-2 text-xs text-slate-300">{task.description}</p>
            <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-400">
              <span className="text-slate-500 font-semibold">Stack:</span>
              <span className="font-mono text-sky-300">{task.stack}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
