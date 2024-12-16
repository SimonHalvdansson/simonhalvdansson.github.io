// Initialize Web Worker
let worker;
if (window.Worker) {
    worker = new Worker('worker.js');
} else {
    alert('Your browser does not support Web Workers.');
}

// Global variables to store image channels and norms
let channelR = [];
let channelG = [];
let channelB = [];
let originalL1R = 0;
let originalL1G = 0;
let originalL1B = 0;

// Store the last selected point
window.lastSelectedPoint = null;

// Flag to prevent multiple hover event listeners
let isHoverSetup = false;

// Function to compute the \ell^1 norm of a channel
function computeL1Norm(channel) {
    let sum = 0;
    for (let y = 0; y < channel.length; y++) {
        for (let x = 0; x < channel[0].length; x++) {
            sum += channel[y][x];
        }
    }
    return sum;
}

// Function to draw color data to a canvas (displayed at 128x128)
function drawColor(canvas, dataR, dataG, dataB, maxVal) {
    const ctx = canvas.getContext('2d');
    const width = dataR[0].length;
    const height = dataR.length;
    const imageData = ctx.createImageData(width, height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const r = Math.min(255, Math.floor((dataR[y][x] / maxVal) * 255));
            const g = Math.min(255, Math.floor((dataG[y][x] / maxVal) * 255));
            const b = Math.min(255, Math.floor((dataB[y][x] / maxVal) * 255));
            const index = (y * width + x) * 4;
            imageData.data[index] = r;       // Red
            imageData.data[index + 1] = g;   // Green
            imageData.data[index + 2] = b;   // Blue
            imageData.data[index + 3] = 255; // Alpha
        }
    }
    // Clear the canvas before drawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Put the 128x128 image data onto the canvas
    ctx.putImageData(imageData, 0, 0);
}

// Function to draw G(ω1, ω2) to GCanvas in color
function drawG(G_R, G_G, G_B) {
    const GCanvas = document.getElementById('GCanvas');
    // No additional normalization needed as G_R, G_G, G_B are already normalized to [0,1]
    drawColor(GCanvas, G_R, G_G, G_B, 1); // maxVal is set to 1 since data is already normalized
}

// Function to draw a red circle overlay on the original image
function drawOverlay(x, y, windowSize) {
    const overlayCanvas = document.getElementById('originalOverlayCanvas');
    const ctx = overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    const radius = windowSize / 2;

    ctx.strokeStyle = 'red';
    ctx.lineWidth = 1;

    // Always draw a circle since window function is Gaussian
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.stroke();
}

// Throttle function to limit the rate of function calls
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Function to setup hover event with throttling
function setupHover(originalCanvas) {
    if (isHoverSetup) return; // Prevent multiple listeners
    isHoverSetup = true;

    const originalCanvasElem = originalCanvas;

    // Define the handler outside to ensure it's not recreated
    const handleMouseMove = throttle((event) => {
        const rect = originalCanvasElem.getBoundingClientRect();
        const scaleX = originalCanvasElem.width / rect.width;
        const scaleY = originalCanvasElem.height / rect.height;
        const x = Math.floor((event.clientX - rect.left) * scaleX);
        const y = Math.floor((event.clientY - rect.top) * scaleY);
        if (x >= 0 && x < 128 && y >= 0 && y < 128) {
            const windowSize = getSelectedWindowSize();
            worker.postMessage({
                type: 'computeG',
                payload: {
                    x,
                    y,
                    windowSize,
                    channelR,
                    channelG,
                    channelB
                }
            });
            drawOverlay(x, y, windowSize);

            // Store the last selected point for updates
            window.lastSelectedPoint = { x, y };
        }
    }, 16); // Throttle to once every ~60ms

    originalCanvasElem.addEventListener('mousemove', handleMouseMove);
}

// Function to clear the overlay canvas (not needed since we persist STFT)
function clearOverlay() {
    const overlayCanvas = document.getElementById('originalOverlayCanvas');
    const ctx = overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// Function to get the currently selected window size
function getSelectedWindowSize() {
    const slider = document.getElementById('windowSizeSlider');
    return parseInt(slider.value);
}

// Function to setup window size slider event listener
function setupWindowSizeSlider() {
    const slider = document.getElementById('windowSizeSlider');
    const sliderValueDisplay = document.getElementById('windowSizeValue');

    slider.addEventListener('input', throttle(() => {
        const windowSize = getSelectedWindowSize();
        sliderValueDisplay.textContent = windowSize;

        // Update the currently displayed STFT
        if (window.lastSelectedPoint) {
            const { x, y } = window.lastSelectedPoint;
            worker.postMessage({
                type: 'computeG',
                payload: {
                    x,
                    y,
                    windowSize,
                    channelR,
                    channelG,
                    channelB
                }
            });
            drawOverlay(x, y, windowSize);
        }
    }, 16)); // Throttle to once every ~60ms
}

// Function to setup thumbnail click events
function setupThumbnails() {
    const thumbnails = document.querySelectorAll('.thumbnail');
    thumbnails.forEach(thumbnail => {
        thumbnail.addEventListener('click', () => {
            // Remove 'selected' class from all thumbnails
            thumbnails.forEach(thumb => thumb.classList.remove('selected'));
            // Add 'selected' class to the clicked thumbnail
            thumbnail.classList.add('selected');

            const imageSrc = thumbnail.getAttribute('data-image');
            loadImage(imageSrc);
        });
    });
}

// Function to load and process the image
function loadImage(imagePath = 'image2.png') { // Default image
    const img = new Image();
    img.src = imagePath; // Ensure this image is in the same directory
    img.crossOrigin = "Anonymous"; // Handle CORS if necessary
    img.onload = () => {
        // Draw and resize the image to 128x128
        const originalCanvas = document.getElementById('originalCanvas');
        const oCtx = originalCanvas.getContext('2d');
        oCtx.drawImage(img, 0, 0, 128, 128);

        // Get image data
        const imageData = oCtx.getImageData(0, 0, 128, 128);
        channelR = [];
        channelG = [];
        channelB = [];
        for (let y = 0; y < 128; y++) {
            channelR[y] = [];
            channelG[y] = [];
            channelB[y] = [];
            for (let x = 0; x < 128; x++) {
                const index = (y * 128 + x) * 4;
                // Extract RGB channels
                const r = imageData.data[index];
                const g = imageData.data[index + 1];
                const b = imageData.data[index + 2];
                channelR[y][x] = r;
                channelG[y][x] = g;
                channelB[y][x] = b;
            }
        }

        // Compute the \ell^1 norms of the original image's channels
        originalL1R = computeL1Norm(channelR);
        originalL1G = computeL1Norm(channelG);
        originalL1B = computeL1Norm(channelB);

        // Draw Fourier transform magnitude in color
        // Offload FFT computations to worker
        worker.postMessage({
            type: 'computeFFT',
            payload: {
                channelR,
                channelG,
                channelB,
                originalL1R,
                originalL1G,
                originalL1B
            }
        });

        // **Removed the setupHover call from here**

        // **Select the center point by default and display its STFT**
        const centerX = Math.floor(128 / 2);
        const centerY = Math.floor(128 / 2);
        const windowSize = getSelectedWindowSize();
        worker.postMessage({
            type: 'computeG',
            payload: {
                x: centerX,
                y: centerY,
                windowSize,
                channelR,
                channelG,
                channelB
            }
        });
        drawOverlay(centerX, centerY, windowSize);
        // Store the last selected point
        window.lastSelectedPoint = { x: centerX, y: centerY };
    };
    img.onerror = () => {
        alert(`Failed to load ${imagePath}. Please make sure it is in the correct directory.`);
    };
}

// Handle messages from the worker
worker.onmessage = function(e) {
    const { type, payload } = e.data;
    if (type === 'fftComputed') {
        const { normalizedR, normalizedG, normalizedB } = payload;
        const fourierCanvas = document.getElementById('fourierCanvas');
        drawColor(fourierCanvas, normalizedR, normalizedG, normalizedB, 1); // maxVal is set to 1 since data is already normalized
    } else if (type === 'gComputed') {
        const { G_R, G_G, G_B } = payload;
        drawG(G_R, G_G, G_B);
    }
};

// Initialize the process on window load
window.onload = () => {
    setupThumbnails();
    setupWindowSizeSlider();
    loadImage('image1.png'); // Load the default image
    setupHover(document.getElementById('originalCanvas')); // Setup hover once
};

// Optionally, handle page unload to terminate the worker
window.onunload = () => {
    if (worker) {
        worker.terminate();
    }
};
