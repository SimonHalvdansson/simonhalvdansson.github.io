// worker.js

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

// Complex number class
class Complex {
    constructor(re, im) {
        this.re = re;
        this.im = im;
    }

    add(other) {
        return new Complex(this.re + other.re, this.im + other.im);
    }

    sub(other) {
        return new Complex(this.re - other.re, this.im - other.im);
    }

    mul(other) {
        return new Complex(
            this.re * other.re - this.im * other.im,
            this.re * other.im + this.im * other.re
        );
    }

    magnitude() {
        return Math.sqrt(this.re * this.re + this.im * this.im);
    }

    static exp(theta) {
        return new Complex(Math.cos(theta), Math.sin(theta));
    }
}

// 1D FFT implementation (Cooley-Turkey FFT)
function FFT(signal) {
    const n = signal.length;
    if (n <= 1) return signal;

    if ((n & (n - 1)) !== 0) {
        throw new Error("FFT length must be a power of 2");
    }

    const even = FFT(signal.filter((_, i) => i % 2 === 0));
    const odd = FFT(signal.filter((_, i) => i % 2 !== 0));

    const combined = new Array(n);
    for (let k = 0; k < n / 2; k++) {
        const angle = -2 * Math.PI * k / n;
        const twiddle = Complex.exp(angle);
        const t = twiddle.mul(odd[k]);
        combined[k] = even[k].add(t);
        combined[k + n / 2] = even[k].sub(t);
    }
    return combined;
}

// 2D FFT by applying 1D FFT on rows and then on columns
function FFT2D(imageData) {
    const height = imageData.length;
    const width = imageData[0].length;

    // Apply FFT to each row
    let fftRows = imageData.map(row => FFT(row.map(val => new Complex(val, 0))));

    // Transpose the matrix
    let transposed = [];
    for (let x = 0; x < width; x++) {
        transposed[x] = [];
        for (let y = 0; y < height; y++) {
            transposed[x][y] = fftRows[y][x];
        }
    }

    // Apply FFT to each column
    let fftCols = transposed.map(col => FFT(col));

    // Transpose back to original orientation
    let fft2D = [];
    for (let y = 0; y < height; y++) {
        fft2D[y] = [];
        for (let x = 0; x < width; x++) {
            fft2D[y][x] = fftCols[x][y];
        }
    }

    return fft2D;
}

// Function to compute magnitude with logarithmic scaling and shift the zero frequency to center
function computeMagnitudeShiftedLog(fft2D) {
    const height = fft2D.length;
    const width = fft2D[0].length;
    let magnitude = [];
    let maxMag = 0;

    // Compute magnitude with logarithmic scaling
    for (let y = 0; y < height; y++) {
        magnitude[y] = [];
        for (let x = 0; x < width; x++) {
            const mag = fft2D[y][x].magnitude();
            const scaledMag = Math.log(1 + mag); // Logarithmic scaling
            magnitude[y][x] = scaledMag;
            if (scaledMag > maxMag) maxMag = scaledMag;
        }
    }

    // Prevent division by zero
    if (maxMag === 0) maxMag = 1;

    // Shift the zero frequency to center
    let shifted = [];
    for (let y = 0; y < height; y++) {
        shifted[y] = [];
        for (let x = 0; x < width; x++) {
            let shiftedX = (x + Math.floor(width / 2)) % width;
            let shiftedY = (y + Math.floor(height / 2)) % height;
            shifted[y][x] = magnitude[shiftedY][shiftedX];
        }
    }

    return { shifted, maxMag };
}

// Function to normalize the FFT magnitudes based on \ell^1 norms and prevent overexposure
function normalizeFFT(shiftedR, shiftedG, shiftedB, originalL1R, originalL1G, originalL1B) {
    // Compute \ell^1 norms of the shifted magnitudes
    const shiftedL1R = computeL1Norm(shiftedR);
    const shiftedL1G = computeL1Norm(shiftedG);
    const shiftedL1B = computeL1Norm(shiftedB);

    // Compute scaling factors, handle division by zero
    const scaleR = originalL1R > 0 ? originalL1R / shiftedL1R : 0;
    const scaleG = originalL1G > 0 ? originalL1G / shiftedL1G : 0;
    const scaleB = originalL1B > 0 ? originalL1B / shiftedL1B : 0;

    // Apply scaling factors
    const scaledR = shiftedR.map(row => row.map(val => val * scaleR));
    const scaledG = shiftedG.map(row => row.map(val => val * scaleG));
    const scaledB = shiftedB.map(row => row.map(val => val * scaleB));

    // Find the maximum value across all scaled channels
    let maxVal = 0;
    for (let y = 0; y < scaledR.length; y++) {
        for (let x = 0; x < scaledR[0].length; x++) {
            maxVal = Math.max(maxVal, scaledR[y][x], scaledG[y][x], scaledB[y][x]);
        }
    }

    // Conditional Uniform Scaling: Only scale down if maxVal > 1
    let normalizedR, normalizedG, normalizedB;
    if (maxVal > 1) {
        const scaleFactor = 1 / maxVal;
        normalizedR = scaledR.map(row => row.map(val => val * scaleFactor));
        normalizedG = scaledG.map(row => row.map(val => val * scaleFactor));
        normalizedB = scaledB.map(row => row.map(val => val * scaleFactor));
    } else {
        normalizedR = scaledR;
        normalizedG = scaledG;
        normalizedB = scaledB;
    }

    return { normalizedR, normalizedG, normalizedB };
}

// Function to compute G(ω1, ω2) for a given (x,y) and channel with Gaussian window
function computeG(x, y, grayscale, windowSize) {
    let windowed = [];
    const half = Math.floor(windowSize / 2);
    const width = grayscale[0].length;
    const height = grayscale.length;

    // Generate window function matrix (Gaussian)
    const windowMatrix = generateWindowMatrix(windowSize);

    for (let i = 0; i < height; i++) {
        windowed[i] = [];
        for (let j = 0; j < width; j++) {
            const dy = i - y;
            const dx = j - x;
            if (Math.abs(dy) <= half && Math.abs(dx) <= half) {
                const windowValue = windowMatrix[dy + half][dx + half];
                windowed[i][j] = grayscale[i][j] * windowValue;
            } else {
                windowed[i][j] = 0;
            }
        }
    }

    // Compute 2D FFT of windowed image
    const fft2D = FFT2D(windowed);
    const { shifted, maxMag } = computeMagnitudeShiftedLog(fft2D);

    // Normalize G
    const G = shifted.map(row => row.map(val => val / maxMag));

    return G;
}

// Function to generate window function matrix (Gaussian)
function generateWindowMatrix(windowSize, windowFunction) {
    const half = Math.floor(windowSize / 2);
    const windowMatrix = [];
    let sigma = windowSize / 8;

    for (let dy = -half; dy <= half; dy++) {
        windowMatrix[dy + half] = [];
        for (let dx = -half; dx <= half; dx++) {
            windowMatrix[dy + half][dx + half] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        }
    }

    return windowMatrix;
}

// Function to compute FFT, normalize, and store results
function handleComputeFFT(data) {
    const { channelR, channelG, channelB, originalL1R, originalL1G, originalL1B } = data.payload;

    // Compute 2D FFT for each channel
    const fft2D_R = FFT2D(channelR);
    const fft2D_G = FFT2D(channelG);
    const fft2D_B = FFT2D(channelB);

    // Compute magnitude with logarithmic scaling for each channel
    const { shifted: shiftedR, maxMag: maxMagR } = computeMagnitudeShiftedLog(fft2D_R);
    const { shifted: shiftedG, maxMag: maxMagG } = computeMagnitudeShiftedLog(fft2D_G);
    const { shifted: shiftedB, maxMag: maxMagB } = computeMagnitudeShiftedLog(fft2D_B);

    // Normalize the FFT magnitudes based on \ell^1 norms and prevent overexposure
    const { normalizedR, normalizedG, normalizedB } = normalizeFFT(
        shiftedR,
        shiftedG,
        shiftedB,
        originalL1R,
        originalL1G,
        originalL1B
    );

    // Post the normalized FFT data back to main thread
    postMessage({
        type: 'fftComputed',
        payload: {
            normalizedR,
            normalizedG,
            normalizedB
        }
    });
}

// Function to compute G and send back to main thread
function handleComputeG(data) {
    const { x, y, windowSize, channelR, channelG, channelB } = data.payload;

    const G_R = computeG(x, y, channelR, windowSize);
    const G_G = computeG(x, y, channelG, windowSize);
    const G_B = computeG(x, y, channelB, windowSize);

    // Post the G data back to main thread
    postMessage({
        type: 'gComputed',
        payload: {
            G_R,
            G_G,
            G_B
        }
    });
}

// Handle incoming messages from the main thread
onmessage = function(e) {
    const { type, payload } = e.data;
    if (type === 'computeFFT') {
        handleComputeFFT(e.data);
    } else if (type === 'computeG') {
        handleComputeG(e.data);
    }
};
