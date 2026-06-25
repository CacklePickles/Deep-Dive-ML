# -*- coding: utf-8 -*-
"""Flask backend for phishing ML models (predict, feedback, analyses API)."""

import os
import sys
import pickle
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer

log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'server.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

from database import Database

models = {}
vectorizer = None
scaler = None
model_weights = {}

db = Database()

def limit_weight_range(weights, min_weight=0.90, max_weight=1.10):
    """Clamp model weights to [min_weight, max_weight], keep mean ~1.0."""
    if not weights:
        return weights

    avg_weight = sum(weights.values()) / len(weights)
    if avg_weight > 0:
        weights = {name: weight / avg_weight for name, weight in weights.items()}

    min_w = min(weights.values())
    max_w = max(weights.values())
    if min_w >= min_weight and max_w <= max_weight:
        return weights

    if max_w != min_w:
        scale = (max_weight - min_weight) / (max_w - min_w)
        offset = min_weight - min_w * scale
        weights = {name: weight * scale + offset for name, weight in weights.items()}
    else:
        avg = (min_weight + max_weight) / 2
        weights = {name: avg for name in weights.keys()}

    avg_weight = sum(weights.values()) / len(weights)
    if avg_weight > 0:
        scale_factor = 1.0 / avg_weight
        scaled_weights = {name: weight * scale_factor for name, weight in weights.items()}
        scaled_min = min(scaled_weights.values())
        scaled_max = max(scaled_weights.values())
        if scaled_min >= min_weight and scaled_max <= max_weight:
            weights = scaled_weights

    return weights

def calculate_model_weights_from_metrics():
    """Load or compute model weights from metrics_table.csv / feedback_data."""
    metrics_path = Path(__file__).parent.parent / 'models' / 'metrics_table.csv'
    feedback_weights_path = Path(__file__).parent.parent / 'feedback_data' / 'model_weights.json'
    
    weights = {}

    if feedback_weights_path.exists():
        try:
            with open(feedback_weights_path, 'r', encoding='utf-8') as f:
                weights = json.load(f)
            logger.info("Weights loaded from feedback_data/model_weights.json")
            weights = limit_weight_range(weights, min_weight=0.90, max_weight=1.10)
            with open(feedback_weights_path, 'w', encoding='utf-8') as f:
                json.dump(weights, f, indent=2, ensure_ascii=False)
            logger.info(f"  Weights after clamp: {weights}")
            return weights
        except Exception as e:
            logger.warning(f"Failed to read feedback weights: {e}")

    model_name_mapping = {
        'Logistic Regression': 'logistic_regression',
        'Naive Bayes': 'naive_bayes',
        'Random Forest': 'random_forest',
        'SVM': 'svm',
        'SVM (Tuned)': 'svm_tuned',
        'XGBoost': 'xgboost'
    }
    
    if metrics_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(metrics_path)
            
            for _, row in df.iterrows():
                csv_name = row['Model']
                code_name = model_name_mapping.get(csv_name)
                
                if code_name:
                    val_f1 = row.get('Val_F1', 0.0)
                    val_roc_auc = row.get('Val_ROC_AUC', 0.0)
                    combined_score = (val_f1 * 0.6 + val_roc_auc * 0.4)
                    weights[code_name] = float(combined_score)

            if weights:
                avg_weight = sum(weights.values()) / len(weights)
                weights = {name: weight / avg_weight for name, weight in weights.items()}
            weights = limit_weight_range(weights, min_weight=0.90, max_weight=1.10)
            logger.info("Weights computed from metrics_table.csv")
        except Exception as e:
            logger.warning(f"Failed to read metrics: {e}")
            import traceback
            traceback.print_exc()
            weights = {}
    else:
        logger.warning("metrics_table.csv not found")
        weights = {}

    all_model_names = ['logistic_regression', 'naive_bayes', 'random_forest', 'svm', 'svm_tuned', 'xgboost']
    for model_name in all_model_names:
        if model_name not in weights:
            weights[model_name] = 1.0
    
    return weights

def load_models():
    """Load pickle models, vectorizer, scaler; set model_weights."""
    global models, vectorizer, scaler, model_weights
    
    models_dir = Path(__file__).parent.parent / 'models'
    preprocessing_dir = Path(__file__).parent.parent / 'preprocessing'
    
    model_files = {
        'logistic_regression': 'logistic_regression.pkl',
        'naive_bayes': 'naive_bayes.pkl',
        'random_forest': 'random_forest.pkl',
        'svm': 'svm.pkl',
        'svm_tuned': 'svm_tuned.pkl',
        'xgboost': 'xgboost.pkl'
    }
    
    logger.info("="*60)
    logger.info("Loading models...")
    logger.info("="*60)

    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                logger.info(f"Loaded: {model_name}")
            except Exception as e:
                logger.error(f"Load error {model_name}: {e}")
        else:
            logger.warning(f"Not found: {model_path}")

    vectorizer_path = preprocessing_dir / 'tfidf_vectorizer.pkl'
    scaler_path = preprocessing_dir / 'url_scaler.pkl'
    
    if vectorizer_path.exists():
        try:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            logger.info("Loaded TF-IDF vectorizer")
        except Exception as e:
            logger.error(f"Vectorizer load error: {e}")

    if scaler_path.exists():
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Loaded URL scaler")
        except Exception as e:
            logger.error(f"Scaler load error: {e}")

    model_weights = calculate_model_weights_from_metrics()
    logger.info(f"Model weights: {model_weights}")

def extract_features(text, url=None):
    """Build text_features and combined_features (text + has_url) for predictors."""
    features = {}

    if vectorizer:
        try:
            text_vector = vectorizer.transform([text])
            text_features_array = text_vector.toarray()
            if text_features_array.shape[0] > 0:
                text_features = text_features_array[0]
            else:
                text_features = np.zeros(text_features_array.shape[1])
            features['text_features'] = text_features.tolist()
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            features['text_features'] = [0.0] * 5000

    import re
    if not url:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls_found = re.findall(url_pattern, text)
        url = urls_found[0] if urls_found else None
    
    has_url = 1.0 if url else 0.0
    if 'text_features' in features:
        combined_features = features['text_features'] + [has_url]
        features['combined_features'] = combined_features
    
    return features

@app.route('/')
def index():
    """Serve frontend."""
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        'status': 'ok',
        'models_loaded': len(models),
        'vectorizer_loaded': vectorizer is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Run ensemble prediction and save analysis."""
    try:
        data = request.json
        text = data.get('text', '')
        url = data.get('url', None)

        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400

        features = extract_features(text, url)
        predictions = {}

        for model_name, model in models.items():
            try:
                if model_name == 'naive_bayes':
                    X = np.array([features['text_features']])
                    if X.shape[1] != model.n_features_in_:
                        logger.warning(f"{model_name}: dimension mismatch, expected {model.n_features_in_}, got {X.shape[1]}")
                        predictions[model_name] = {
                            'is_phishing': False,
                            'confidence': 0.0,
                            'error': f'Dimension mismatch: {model.n_features_in_} vs {X.shape[1]}'
                        }
                        continue
                else:
                    X = np.array([features['combined_features']])
                    if X.shape[1] != model.n_features_in_:
                        logger.warning(f"{model_name}: dimension mismatch, expected {model.n_features_in_}, got {X.shape[1]}")
                        predictions[model_name] = {
                            'is_phishing': False,
                            'confidence': 0.0,
                            'error': f'Dimension mismatch: {model.n_features_in_} vs {X.shape[1]}'
                        }
                        continue

                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
                
                is_phishing = bool(prediction == 1)
                confidence = float(probabilities[1] if len(probabilities) > 1 else probabilities[0]) * 100
                weight = model_weights.get(model_name, 1.0)
                confidence_multiplier = confidence / 100.0
                weighted_vote = weight * confidence_multiplier
                
                predictions[model_name] = {
                    'is_phishing': is_phishing,
                    'confidence': round(confidence, 1),
                    'weight': weight,
                    'weighted_vote': round(weighted_vote, 3)
                }
                
            except Exception as e:
                logger.error(f"Predict error for {model_name}: {e}")
                predictions[model_name] = {
                    'is_phishing': False,
                    'confidence': 0.0,
                    'error': str(e)
                }

        weighted_phishing_score = 0.0
        weighted_safe_score = 0.0
        total_weight = 0.0
        successful_predictions = {}
        
        for model_name, prediction in predictions.items():
            if 'error' in prediction:
                continue
            weight = model_weights.get(model_name, 1.0)
            confidence_multiplier = prediction.get('confidence', 50.0) / 100.0
            weighted_vote = weight * confidence_multiplier
            if prediction.get('is_phishing', False):
                weighted_phishing_score += weighted_vote
            else:
                weighted_safe_score += weighted_vote
            
            total_weight += weight
            successful_predictions[model_name] = prediction

        if total_weight > 0:
            phishing_ratio = weighted_phishing_score / total_weight
            safe_ratio = weighted_safe_score / total_weight
            final_is_phishing = weighted_phishing_score > weighted_safe_score
            if final_is_phishing:
                final_confidence = phishing_ratio * 100
            else:
                final_confidence = safe_ratio * 100
        else:
            final_is_phishing = False
            final_confidence = 0.0

        phishing_votes = sum(1 for p in predictions.values() if p.get('is_phishing', False))
        total_votes = len([p for p in predictions.values() if 'error' not in p])
        ensemble_data = {
            'weighted_phishing_score': round(weighted_phishing_score, 2),
            'weighted_safe_score': round(weighted_safe_score, 2),
            'total_weight': round(total_weight, 2),
            'phishing_votes': phishing_votes,
            'total_votes': total_votes
        }
        
        try:
            analysis_id = db.save_analysis(
                text=text,
                url=url,
                is_phishing=final_is_phishing,
                confidence=round(final_confidence, 1),
                predictions=predictions,
                ensemble=ensemble_data
            )
            logger.info(f"Analysis saved, id={analysis_id}")
        except Exception as e:
            logger.error(f"DB save error: {e}")
            import traceback
            traceback.print_exc()

        return jsonify({
            'success': True,
            'results': {
                'analysis_id': analysis_id if 'analysis_id' in locals() else None,
                'is_phishing': final_is_phishing,
                'confidence': round(final_confidence, 1),
                'predictions': predictions,
                'phishing_votes': phishing_votes,
                'total_votes': total_votes,
                'ensemble': {
                    'weighted_phishing_score': round(weighted_phishing_score, 2),
                    'weighted_safe_score': round(weighted_safe_score, 2),
                    'total_weight': round(total_weight, 2),
                    'method': 'weighted_voting'
                }
            }
        })
        
    except Exception as e:
        logger.error(f"/api/predict error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return loaded models and weights."""
    model_info = {}
    for model_name, model in models.items():
        model_info[model_name] = {
            'loaded': True,
            'n_features': getattr(model, 'n_features_in_', 'unknown'),
            'weight': model_weights.get(model_name, 1.0)
        }
    return jsonify({
        'models': model_info,
        'weights': model_weights,
        'ensemble_method': 'weighted_voting'
    })

def load_feedback_stats():
    """Load model_stats.json; migrate old format to weighted_score if needed."""
    stats_path = Path(__file__).parent.parent / 'feedback_data' / 'model_stats.json'
    if stats_path.exists():
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            for model_name, stat in stats.items():
                if 'weighted_score' not in stat:
                    total = stat.get('total_predictions', 0)
                    correct = stat.get('correct_predictions', 0)
                    old_accuracy = stat.get('accuracy', 0.5)
                    avg_confidence = 0.7
                    if total > 0:
                        correct_bonus = correct * (avg_confidence ** 2)
                        incorrect_penalty = (total - correct) * (avg_confidence ** 2)
                        estimated_score = correct_bonus - incorrect_penalty
                    else:
                        estimated_score = 0.0
                    stats[model_name] = {
                        'total_predictions': total,
                        'weighted_score': estimated_score,
                        'weighted_accuracy': old_accuracy
                    }
            return stats
        except Exception as e:
            logger.warning(f"Failed to read feedback stats: {e}")
    return {}

def save_feedback_stats(stats):
    """Write model_stats.json."""
    stats_path = Path(__file__).parent.parent / 'feedback_data' / 'model_stats.json'
    stats_path.parent.mkdir(exist_ok=True)
    try:
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save feedback stats: {e}")

def save_model_weights(weights):
    """Write model_weights.json."""
    weights_path = Path(__file__).parent.parent / 'feedback_data' / 'model_weights.json'
    weights_path.parent.mkdir(exist_ok=True)
    try:
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(weights, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save weights: {e}")

def update_model_weights_from_feedback(predictions, user_feedback):
    """Update model weights from user feedback (weighted by confidence)."""
    global model_weights
    stats = load_feedback_stats()
    if user_feedback == 'phishing':
        correct_answer = True
    elif user_feedback == 'safe':
        correct_answer = False
    else:
        return

    logger.info(f"Updating weights from feedback: correct={correct_answer}")

    for model_name, prediction in predictions.items():
        if 'error' in prediction:
            continue
        
        if model_name not in stats:
            stats[model_name] = {
                'total_predictions': 0,
                'weighted_score': 0.0,
                'weighted_accuracy': 0.5
            }
        old_score = stats[model_name]['weighted_score']
        stats[model_name]['total_predictions'] += 1
        confidence = prediction.get('confidence', 50.0)
        confidence_normalized = confidence / 100.0
        model_prediction = prediction.get('is_phishing', False)

        if model_prediction == correct_answer:
            bonus = confidence_normalized ** 2
            stats[model_name]['weighted_score'] += bonus
            logger.info(f"  {model_name}: correct (conf {confidence:.1f}%) +{bonus:.4f} -> {stats[model_name]['weighted_score']:.4f}")
        else:
            penalty = confidence_normalized ** 2
            stats[model_name]['weighted_score'] -= penalty
            logger.info(f"  {model_name}: wrong (conf {confidence:.1f}%) -{penalty:.4f} -> {stats[model_name]['weighted_score']:.4f}")

        total = stats[model_name]['total_predictions']
        weighted_score = stats[model_name]['weighted_score']
        if total > 0:
            avg_weighted_score = weighted_score / total
            k = 3.0
            weighted_accuracy = 1.0 / (1.0 + np.exp(-k * avg_weighted_score))
            stats[model_name]['weighted_accuracy'] = float(weighted_accuracy)
        else:
            stats[model_name]['weighted_accuracy'] = 0.5

    save_feedback_stats(stats)
    new_weights = {}
    for model_name, stat in stats.items():
        new_weights[model_name] = stat.get('weighted_accuracy', 0.5)
    if new_weights:
        avg_weight = sum(new_weights.values()) / len(new_weights)
        if avg_weight > 0:
            new_weights = {name: weight / avg_weight for name, weight in new_weights.items()}
    new_weights = limit_weight_range(new_weights, min_weight=0.90, max_weight=1.10)
    all_model_names = ['logistic_regression', 'naive_bayes', 'random_forest', 'svm', 'svm_tuned', 'xgboost']
    for model_name in all_model_names:
        if model_name not in new_weights:
            new_weights[model_name] = 1.0
    for model_name in all_model_names:
        logger.info(f"  {model_name}: {model_weights.get(model_name, 1.0):.4f} -> {new_weights.get(model_name, 1.0):.4f}")
    model_weights = new_weights
    save_model_weights(model_weights)
    logger.info("Weights updated from feedback")

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Store feedback and optionally update model weights."""
    try:
        data = request.json
        feedback_type = data.get('feedback')
        analysis_id = data.get('analysis_id')
        predictions = data.get('predictions', {})

        if feedback_type not in ['phishing', 'safe', 'not_sure']:
            return jsonify({'success': False, 'error': 'Invalid feedback type'}), 400

        if analysis_id:
            try:
                db.update_feedback(analysis_id, feedback_type)
                logger.info(f"Feedback saved for analysis {analysis_id}")
            except Exception as e:
                logger.warning(f"Feedback DB update failed: {e}")

        if feedback_type != 'not_sure' and predictions:
            update_model_weights_from_feedback(predictions, feedback_type)
        
        return jsonify({
            'success': True,
            'message': 'Feedback received',
            'weights': model_weights
        })
        
    except Exception as e:
        logger.error(f"/api/feedback error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyses', methods=['GET'])
def get_analyses():
    """List analyses with optional feedback filter."""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        feedback_filter = request.args.get('feedback', None)
        
        analyses = db.get_analyses(limit=limit, offset=offset, feedback=feedback_filter)
        
        return jsonify({
            'success': True,
            'analyses': analyses,
            'count': len(analyses)
        })
        
    except Exception as e:
        logger.error(f"/api/analyses error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyses/<int:analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Get one analysis by id."""
    try:
        analysis = db.get_analysis(analysis_id)
        
        if not analysis:
            return jsonify({'success': False, 'error': 'Analysis not found'}), 404
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"/api/analyses/<id> error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Return aggregate statistics."""
    try:
        stats = db.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"/api/statistics error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Starting server...")
    logger.info("="*60)
    load_models()
    if not models:
        logger.error("No models loaded.")
        sys.exit(1)
    logger.info(f"Models loaded: {len(models)}. Open http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
