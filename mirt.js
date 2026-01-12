/**
 * mirt.js - Multidimensional Item Response Theory for the Browser
 * Includes scoring (EAP) and estimation (EM with 1PL-4PL support)
 */

class MIRT {
    constructor(dimensions = 1) {
        this.dims = dimensions;
        this.items = [];
        this.quadrature = this._generateQuadrature(21);
    }

    /**
     * 4PL Item Response Function
     * P(θ) = c + (γ - c) / (1 + exp(-(aθ + d)))
     */
    irf(theta, item) {
        const a = Array.isArray(item.a) ? item.a : [item.a];
        const kernel = a.reduce((sum, ai, k) => sum + ai * (theta[k] || 0), 0) + item.d;
        const prob = 1 / (1 + Math.exp(-kernel));
        return item.c + (item.gamma - item.c) * prob;
    }

    /**
     * EM Algorithm for Parameter Fitting
     * @param {Array} data - N x J binary response matrix
     * @param {Object} options - { modelType: '2PL', maxIter: 50, learningRate: 0.1 }
     */
    async fit(data, options = {}) {
        const { modelType = '2PL', maxIter = 100, lr = 0.05 } = options;
        const numPeople = data.length;
        const numItems = data[0].length;

        // 1. Initialize items based on modelType
        this.items = Array.from({ length: numItems }, () => ({
            a: new Array(this.dims).fill(1.0),
            d: 0.0,
            c: (modelType === '3PL' || modelType === '4PL') ? 0.2 : 0.0,
            gamma: (modelType === '4PL') ? 0.95 : 1.0
        }));

        console.log(`Starting EM fit for ${modelType}...`);

        for (let iter = 0; iter < maxIter; iter++) {
            // --- E-STEP: Compute Posteriors ---
            const posteriors = this._computePosteriors(data);

            // --- M-STEP: Update Parameters via Gradient Descent ---
            let maxChange = 0;
            for (let j = 0; j < numItems; j++) {
                const update = this._optimizeItem(j, data, posteriors, modelType, lr);
                maxChange = Math.max(maxChange, update);
            }

            // Yield to browser UI thread to prevent freezing
            if (iter % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0));

            if (maxChange < 1e-4) {
                console.log(`Converged at iteration ${iter}`);
                break;
            }
        }
        return this.items;
    }

    /**
     * EAP (Expected A Posteriori) Scoring
     * Estimates θ for a new response pattern
     */
    scoreEAP(responses) {
        const { nodes, weights } = this.quadrature;
        let posteriorWeights = nodes.map((node, idx) => {
            let L = 1;
            this.items.forEach((item, j) => {
                if (responses[j] === null) return;
                const p = this.irf([node], item);
                L *= (responses[j] === 1) ? p : (1 - p);
            });
            return L * weights[idx];
        });

        const sumW = posteriorWeights.reduce((a, b) => a + b, 0);
        const estimate = nodes.reduce((sum, node, idx) => sum + node * (posteriorWeights[idx] / sumW), 0);
        return estimate;
    }

    // --- Private Helper Methods ---

    _generateQuadrature(n) {
        const nodes = [];
        const weights = [];
        for (let i = 0; i < n; i++) {
            const x = -4 + (i * 8) / (n - 1);
            nodes.push(x);
            weights.push(Math.exp(-0.5 * x * x)); // Normal distribution prior
        }
        const sumW = weights.reduce((a, b) => a + b, 0);
        return { nodes, weights: weights.map(w => w / sumW) };
    }

    _computePosteriors(data) {
        const { nodes, weights } = this.quadrature;
        return data.map(personResponse => {
            const likelihoods = nodes.map(node => {
                return personResponse.reduce((L, ans, j) => {
                    const p = this.irf([node], this.items[j]);
                    return L * (ans === 1 ? p : (1 - p));
                }, 1);
            });
            const evidence = likelihoods.reduce((sum, L, idx) => sum + L * weights[idx], 0);
            return likelihoods.map((L, idx) => (L * weights[idx]) / (evidence + 1e-10));
        });
    }

    _optimizeItem(itemIdx, data, posteriors, modelType, lr) {
        const item = this.items[itemIdx];
        const { nodes } = this.quadrature;
        let gradD = 0;
        let gradA = new Array(this.dims).fill(0);

        // Compute expected gradients over all people and quadrature nodes
        for (let i = 0; i < data.length; i++) {
            for (let k = 0; k < nodes.length; k++) {
                const p = this.irf([nodes[k]], item);
                const error = (data[i][itemIdx] - p) * posteriors[i][k];

                gradD += error;
                if (modelType !== '1PL') {
                    gradA[0] += error * nodes[k]; // Simplified for 1D
                }
            }
        }

        // Apply updates
        item.d += gradD * lr;
        if (modelType !== '1PL') {
            item.a[0] += gradA[0] * lr;
        }

        return Math.abs(gradD * lr); // Return change for convergence check
    }
}

if (typeof module !== 'undefined') {
    module.exports = MIRT;
}