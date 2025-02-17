const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;

// Serve static files from the current directory
app.use(express.static(__dirname));

// Endpoint to list Markdown files
app.get('/api/blogs', (req, res) => {
    const blogsDir = path.join(__dirname, 'blogs');
    fs.readdir(blogsDir, (err, files) => {
        if (err) {
            return res.status(500).json({ error: 'Failed to list blog files' });
        }
        const markdownFiles = files.filter(file => file.endsWith('.md'));
        res.json(markdownFiles);
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
}); 