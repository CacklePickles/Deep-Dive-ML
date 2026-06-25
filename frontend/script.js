const contentInput = document.getElementById('content-input');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingDiv = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const finalVerdict = document.getElementById('final-verdict');
const finalConfidence = document.getElementById('final-confidence');
const modelsList = document.getElementById('models-list');
const feedbackPhishingBtn = document.getElementById('feedback-phishing');
const feedbackSafeBtn = document.getElementById('feedback-safe');
const feedbackNotSureBtn = document.getElementById('feedback-not-sure');
const feedbackMessage = document.getElementById('feedback-message');

let currentResults = null;
let currentContent = '';
let modelWeights = {};

function extractContent(input) {
    const urlPattern = /(https?:\/\/[^\s]+)/gi;
    const urls = input.match(urlPattern);
    const url = urls && urls.length > 0 ? urls[0] : null;
    const text = input.replace(urlPattern, '').trim();
    return { text: text || input, url };
}

async function checkHealth() {
    try {
        const response = await fetch('http://localhost:5000/api/health');
        const data = await response.json();
        console.log('API Health:', data);
        return data.status === 'ok';
    } catch (error) {
        console.error('API unavailable:', error);
        return false;
    }
}

analyzeBtn.addEventListener('click', async () => {
    const content = contentInput.value.trim();
    
    if (!content) {
        alert('Please enter content to analyze');
        return;
    }
    
    const { text, url } = extractContent(content);
    loadingDiv.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    analyzeBtn.disabled = true;
    
    try {
        const response = await fetch('http://localhost:5000/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                url: url
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentResults = data.results;
            currentResults.analysis_id = data.results.analysis_id;
            currentContent = content;
            displayResults(data.results);
            resultsSection.classList.remove('hidden');
            
            feedbackPhishingBtn.disabled = false;
            feedbackSafeBtn.disabled = false;
            feedbackNotSureBtn.disabled = false;
            
            if (feedbackMessage) {
                feedbackMessage.classList.add('hidden');
                feedbackMessage.style.cssText = 'display: none !important; visibility: hidden !important; opacity: 0 !important;';
            }
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Request error:', error);
        alert('Connection error. Make sure the server is running on http://localhost:5000');
    } finally {
        loadingDiv.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
});

function displayEnsembleVoting(results) {
    const ensembleDiv = document.getElementById('ensemble-voting');
    if (!results.ensemble) {
        ensembleDiv.innerHTML = '<p>Voting information unavailable</p>';
        return;
    }
    
    const ensemble = results.ensemble;
    const phishingScore = ensemble.weighted_phishing_score || 0;
    const safeScore = ensemble.weighted_safe_score || 0;
    const totalWeight = ensemble.total_weight || 1;
    
    const phishingPercent = totalWeight > 0 ? (phishingScore / totalWeight * 100) : 0;
    const safePercent = totalWeight > 0 ? (safeScore / totalWeight * 100) : 0;
    
    ensembleDiv.innerHTML = `
        <div class="ensemble-scores">
            <div class="score-bar-container">
                <div class="score-label">
                    <span>🚨 Phishing</span>
                    <span class="score-value">${phishingScore.toFixed(2)} / ${totalWeight.toFixed(2)} (${phishingPercent.toFixed(1)}%)</span>
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill phishing" style="width: ${phishingPercent}%"></div>
                </div>
            </div>
            <div class="score-bar-container">
                <div class="score-label">
                    <span>✅ Safe</span>
                    <span class="score-value">${safeScore.toFixed(2)} / ${totalWeight.toFixed(2)} (${safePercent.toFixed(1)}%)</span>
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill safe" style="width: ${safePercent}%"></div>
                </div>
            </div>
        </div>
        <div class="ensemble-details">
            <p><strong>Method:</strong> Weighted Voting</p>
            <p><strong>Final Decision:</strong> ${results.is_phishing ? '🚨 PHISHING' : '✅ SAFE'} (${results.confidence.toFixed(1)}% confidence)</p>
        </div>
    `;
}

function displayResults(results) {
    if (results.is_phishing) {
        finalVerdict.textContent = '🚨 PHISHING DETECTED';
        finalVerdict.className = 'verdict phishing';
    } else {
        finalVerdict.textContent = '✅ SAFE';
        finalVerdict.className = 'verdict safe';
    }
    
    let confidenceText = `Confidence: ${results.confidence.toFixed(1)}%`;
    if (results.ensemble) {
        confidenceText += ` | Weighted Voting: Phishing ${results.ensemble.weighted_phishing_score.toFixed(2)} vs Safe ${results.ensemble.weighted_safe_score.toFixed(2)}`;
    }
    finalConfidence.textContent = confidenceText;
    displayEnsembleVoting(results);
    modelsList.innerHTML = '';
    
    const modelNames = {
        'logistic_regression': 'Logistic Regression',
        'naive_bayes': 'Naive Bayes',
        'random_forest': 'Random Forest',
        'svm': 'SVM',
        'svm_tuned': 'SVM Tuned',
        'xgboost': 'XGBoost'
    };
    
    const sortedPredictions = Object.entries(results.predictions).sort((a, b) => {
        const voteA = a[1].weighted_vote || 0;
        const voteB = b[1].weighted_vote || 0;
        return voteB - voteA;
    });
    
    for (const [modelKey, prediction] of sortedPredictions) {
        const modelName = modelNames[modelKey] || modelKey;
        const modelDiv = document.createElement('div');
        modelDiv.className = 'model-result';
        
        if (prediction.error) {
            modelDiv.innerHTML = `
                <div class="model-info">
                    <span class="model-name">${modelName}:</span>
                    <span class="model-status error">❌ ERROR: ${prediction.error}</span>
                </div>
            `;
        } else {
            const status = prediction.is_phishing ? '🚨 PHISHING' : '✅ SAFE';
            const statusClass = prediction.is_phishing ? 'phishing' : 'safe';
            const weight = prediction.weight || 1.0;
            const weightedVote = prediction.weighted_vote || 0;
            const confidence = prediction.confidence || 0;
            const contribution = results.ensemble && results.ensemble.total_weight > 0 
                ? (weightedVote / results.ensemble.total_weight * 100).toFixed(1)
                : '0.0';
            
            modelDiv.innerHTML = `
                <div class="model-info">
                    <span class="model-name">${modelName}</span>
                    <span class="model-status ${statusClass}">${status}</span>
                </div>
                <div class="model-details">
                    <div class="model-metric">
                        <span class="metric-label">Confidence</span>
                        <span class="metric-value">${confidence.toFixed(1)}%</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">Model Weight</span>
                        <span class="metric-value weight-value">${weight.toFixed(2)}</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">Weighted Vote</span>
                        <span class="metric-value vote-value">${weightedVote.toFixed(3)}</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-label">Contribution</span>
                        <span class="metric-value contribution-value">${contribution}%</span>
                    </div>
                </div>
            `;
        }
        
        modelsList.appendChild(modelDiv);
    }
}

async function loadModelWeights() {
    try {
        const response = await fetch('http://localhost:5000/api/models');
        const data = await response.json();
        if (data.weights) {
            modelWeights = data.weights;
            console.log('Model weights loaded:', modelWeights);
        }
    } catch (error) {
        console.error('Error loading model weights:', error);
    }
}

async function sendFeedback(feedbackType) {
    if (!currentResults || !currentResults.predictions) {
        alert('No analysis results available. Please analyze content first.');
        return;
    }
    
    feedbackPhishingBtn.disabled = true;
    feedbackSafeBtn.disabled = true;
    feedbackNotSureBtn.disabled = true;
    
    try {
        const response = await fetch('http://localhost:5000/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                feedback: feedbackType,
                analysis_id: currentResults.analysis_id || null,
                predictions: currentResults.predictions
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (feedbackMessage) {
                if (feedbackType === 'not_sure') {
                    feedbackMessage.textContent = '✓ Feedback received. Thank you!';
                    feedbackMessage.className = 'feedback-message info';
                } else {
                    feedbackMessage.textContent = `✓ Feedback received. Model weights updated based on your input.`;
                    feedbackMessage.className = 'feedback-message success';
                    
                    // Reload model weights
                    await loadModelWeights();
                }
                feedbackMessage.classList.remove('hidden');
                feedbackMessage.style.cssText = 'display: block !important; opacity: 1 !important; visibility: visible !important;';
                setTimeout(() => {
                    if (feedbackMessage) {
                        feedbackMessage.style.cssText = 'opacity: 0 !important; transition: opacity 0.3s ease-out !important;';
                        setTimeout(() => {
                            if (feedbackMessage) {
                                feedbackMessage.classList.add('hidden');
                                feedbackMessage.style.cssText = 'display: none !important; visibility: hidden !important; opacity: 0 !important;';
                            }
                        }, 300);
                    }
                }, 2000);
            }
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
            feedbackPhishingBtn.disabled = false;
            feedbackSafeBtn.disabled = false;
            feedbackNotSureBtn.disabled = false;
        }
    } catch (error) {
        console.error('Feedback error:', error);
        alert('Connection error. Make sure the server is running.');
        feedbackPhishingBtn.disabled = false;
        feedbackSafeBtn.disabled = false;
        feedbackNotSureBtn.disabled = false;
    }
}

feedbackPhishingBtn.addEventListener('click', () => sendFeedback('phishing'));
feedbackSafeBtn.addEventListener('click', () => sendFeedback('safe'));
feedbackNotSureBtn.addEventListener('click', () => sendFeedback('not_sure'));

if (feedbackMessage) {
    feedbackMessage.addEventListener('click', () => {
        if (window.feedbackTimeout) {
            clearTimeout(window.feedbackTimeout);
            window.feedbackTimeout = null;
        }
        feedbackMessage.style.cssText = 'opacity: 0 !important; transition: opacity 0.3s ease-out !important;';
        setTimeout(() => {
            feedbackMessage.classList.add('hidden');
            feedbackMessage.style.cssText = 'display: none !important; visibility: hidden !important; opacity: 0 !important;';
        }, 300);
    });
}

window.addEventListener('DOMContentLoaded', () => {
    checkHealth().then(isHealthy => {
        if (!isHealthy) {
            console.warn('API unavailable. Make sure the server is running.');
        }
    });
    loadModelWeights();
    
    contentInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeBtn.click();
        }
    });
});
