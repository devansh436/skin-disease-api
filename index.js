require('dotenv').config();
const express = require('express');
const multer = require('multer');
const path = require('path');
const app = express();
const cors = require('cors');
const { spawn } = require('child_process');

// const port = process.env.PORT || 3000;
const port = 3000;
app.use(cors());
app.use(express.json());

// Configure multer for image storage
const storage = multer.diskStorage({
    destination: 'uploads/',
    filename: (req, file, cb) => {
        cb(null, `${Date.now()}-${file.originalname}`);
    },
});
const upload = multer({
    storage,
    limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed!'));
        }
    },
});

// Create an 'uploads' folder if not exists
const fs = require('fs');
if (!fs.existsSync('uploads')) {
    fs.mkdirSync('uploads');
}

// API to handle image upload
app.post('/upload', upload.single('image'), (req, res) => {
    if (!req.file) return res.status(400).send('No image uploaded');
    res.status(200).json({ message: 'Image uploaded successfully', filename: req.file.filename });
});

// NEW API endpoint for predictions
app.post('/predict', upload.single('image'), (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No image uploaded' });
    
    const imagePath = req.file.path;
    
    // Run Python script to make prediction
    const pythonProcess = spawn('python', ['predict.py', imagePath]);
    
    let prediction = '';
    
    pythonProcess.stdout.on('data', (data) => {
        prediction += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({ error: 'Prediction failed', code });
        }
        
        try {
            const result = JSON.parse(prediction);
            res.json(result);
        } catch (e) {
            res.status(500).json({ 
                error: 'Invalid prediction output', 
                raw: prediction,
                parseError: e.message 
            });
        }
    });
});

// Optional: API to get prediction for an already uploaded image
app.post('/predict-existing', express.json(), (req, res) => {
    const { filename } = req.body;
    
    if (!filename) {
        return res.status(400).json({ error: 'No filename provided' });
    }
    
    const imagePath = path.join(__dirname, 'uploads', filename);
    
    if (!fs.existsSync(imagePath)) {
        return res.status(404).json({ error: 'Image not found' });
    }
    
    // Run Python script to make prediction
    const pythonProcess = spawn('python', ['predict.py', imagePath]);
    
    let prediction = '';
    
    pythonProcess.stdout.on('data', (data) => {
        prediction += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({ error: 'Prediction failed' });
        }
        
        try {
            const result = JSON.parse(prediction);
            res.json(result);
        } catch (e) {
            res.status(500).json({ error: 'Invalid prediction output', raw: prediction });
        }
    });
});

// Global Error Handler
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({ error: err.message });
});

app.listen(port, () => console.log(`Server running on http://localhost:${port}`));