import JSZip from 'jszip';
import { PROJECT_FILES } from '../data/projectFiles';

export async function downloadProjectZip() {
  const zip = new JSZip();
  const folder = zip.folder('task_1_audio_emotion');

  if (!folder) return;

  // Add all project code files
  PROJECT_FILES.forEach((file) => {
    folder.file(file.name, file.code);
  });

  // Add dummy placeholder for samples directory
  const samplesFolder = folder.folder('data/samples');
  samplesFolder?.file(
    'README.txt',
    'Run `python prepare_data.py --generate-samples` to populate this folder with sample WAV files automatically!'
  );

  // Generate zip file
  const blob = await zip.generateAsync({ type: 'blob' });
  const downloadUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = downloadUrl;
  a.download = 'Growfinix_Task1_Audio_Emotion.zip';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(downloadUrl);
}
