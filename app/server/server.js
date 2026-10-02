/**
 * Node.js Express API Server for Manuscript Layout Region Detection.
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const PROJECT_ROOT = path.resolve(__dirname, '../..');

app.get('/', (req, res) => {
  res.json({
    status: 'online',
    project: 'Manuscript Specific Layout Region Detection',
    framework: 'Node.js Express + Python CV Engine',
    version: '0.1.0',
    target_classes: ['header', 'footer', 'main_text', 'side_text', 'filler']
  });
});

app.get('/api/status', (req, res) => {
  const configPath = path.join(PROJECT_ROOT, 'config', 'config.yaml');
  res.json({
    status: 'ok',
    project_root: PROJECT_ROOT,
    config_exists: fs.existsSync(configPath),
    target_classes: ['header', 'footer', 'main_text', 'side_text', 'filler'],
    cli_script: 'inference.py'
  });
});

app.post('/api/detect', (req, res) => {
  const inputPath = req.body.input || './data/raw/Sample Test Data';
  const outputPath = req.body.output || './outputs';
  const confThreshold = req.body.conf || 0.5;
  const pythonScript = path.join(PROJECT_ROOT, 'inference.py');

  const pythonProcess = spawn('python', [
    pythonScript,
    '--input', inputPath,
    '--output', outputPath,
    '--conf', confThreshold.toString()
  ]);

  let stdoutData = '';
  let stderrData = '';

  pythonProcess.stdout.on('data', (data) => { stdoutData += data.toString(); });
  pythonProcess.stderr.on('data', (data) => { stderrData += data.toString(); });

  pythonProcess.on('close', (code) => {
    if (code === 0) {
      res.json({
        success: true,
        message: 'Manuscript layout region detection completed successfully.',
        stdout: stdoutData,
        input_path: inputPath,
        output_path: outputPath
      });
    } else {
      res.status(500).json({
        success: false,
        error: 'Python inference process failed.',
        code: code,
        stderr: stderrData
      });
    }
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Express Server running on http://localhost:${PORT}`);
});
