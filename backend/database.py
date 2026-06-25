# -*- coding: utf-8 -*-
"""SQLite database layer. One-to-Many: analyses (header) + model_results (per-model details)."""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Any


class Database:
    """Analyses database (analyses + model_results)."""

    def __init__(self, db_path: Optional[Path] = None):
        """Open DB and ensure schema."""
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'data' / 'analyses.db'

        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()

    def get_connection(self):
        """Return a DB connection with row_factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _has_old_schema(self, cursor) -> bool:
        """True if analyses has is_phishing and no final_verdict."""
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analyses'")
        if not cursor.fetchone():
            return False
        cursor.execute("PRAGMA table_info(analyses)")
        columns = [row[1] for row in cursor.fetchall()]
        return 'is_phishing' in columns and 'final_verdict' not in columns

    def _migrate_from_old_schema(self, cursor):
        """Migrate from analyses/model_votes/ensemble_results to analyses/model_results."""
        cursor.execute('''
            CREATE TABLE analyses_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                url TEXT,
                final_verdict TEXT NOT NULL,
                final_confidence REAL NOT NULL,
                phishing_score REAL NOT NULL,
                safety_score REAL NOT NULL,
                total_weight REAL NOT NULL,
                feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            INSERT INTO analyses_new 
                (id, text, url, final_verdict, final_confidence, phishing_score, safety_score, total_weight, feedback, created_at)
            SELECT 
                a.id,
                a.text,
                a.url,
                CASE WHEN a.is_phishing = 1 THEN 'phishing' ELSE 'safe' END,
                a.confidence,
                COALESCE(e.weighted_phishing_score, 0),
                COALESCE(e.weighted_safe_score, 0),
                COALESCE(e.total_weight, 0),
                a.feedback,
                a.created_at
            FROM analyses a
            LEFT JOIN ensemble_results e ON e.analysis_id = a.id
        ''')
        cursor.execute('''
            CREATE TABLE model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                model_verdict INTEGER NOT NULL,
                confidence REAL NOT NULL,
                weight_impact REAL NOT NULL,
                error TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses_new(id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('''
            INSERT INTO model_results (analysis_id, model_name, model_verdict, confidence, weight_impact, error)
            SELECT analysis_id, model_name, is_phishing, confidence, weight, error
            FROM model_votes
        ''')
        cursor.execute('DROP TABLE IF EXISTS model_votes')
        cursor.execute('DROP TABLE IF EXISTS ensemble_results')
        cursor.execute('DROP TABLE analyses')
        cursor.execute('ALTER TABLE analyses_new RENAME TO analyses')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_results_analysis_id ON model_results(analysis_id)')

    def init_database(self):
        """Create or migrate to analyses + model_results schema."""
        conn = self.get_connection()
        cursor = conn.cursor()

        if self._has_old_schema(cursor):
            self._migrate_from_old_schema(cursor)
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    url TEXT,
                    final_verdict TEXT NOT NULL,
                    final_confidence REAL NOT NULL,
                    phishing_score REAL NOT NULL,
                    safety_score REAL NOT NULL,
                    total_weight REAL NOT NULL,
                    feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    model_verdict INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    weight_impact REAL NOT NULL,
                    error TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
                )
            ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_feedback ON analyses(feedback)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_results_analysis_id ON model_results(analysis_id)')

        conn.commit()
        conn.close()

    def save_analysis(
        self,
        text: str,
        url: Optional[str],
        is_phishing: bool,
        confidence: float,
        predictions: Dict[str, Any],
        ensemble: Dict[str, Any],
    ) -> int:
        """Save one analysis and its model_results. Returns new analysis id."""
        final_verdict = 'phishing' if is_phishing else 'safe'
        phishing_score = float(ensemble.get('weighted_phishing_score', 0.0))
        safety_score = float(ensemble.get('weighted_safe_score', 0.0))
        total_weight = float(ensemble.get('total_weight', 0.0))

        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                '''
                INSERT INTO analyses 
                (text, url, final_verdict, final_confidence, phishing_score, safety_score, total_weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (text, url, final_verdict, confidence, phishing_score, safety_score, total_weight),
            )
            analysis_id = cursor.lastrowid

            for model_name, prediction in predictions.items():
                if 'error' in prediction:
                    cursor.execute(
                        '''
                        INSERT INTO model_results 
                        (analysis_id, model_name, model_verdict, confidence, weight_impact, error)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            analysis_id,
                            model_name,
                            0,
                            0.0,
                            0.0,
                            prediction['error'],
                        ),
                    )
                else:
                    cursor.execute(
                        '''
                        INSERT INTO model_results 
                        (analysis_id, model_name, model_verdict, confidence, weight_impact)
                        VALUES (?, ?, ?, ?, ?)
                        ''',
                        (
                            analysis_id,
                            model_name,
                            1 if prediction.get('is_phishing', False) else 0,
                            prediction.get('confidence', 0.0),
                            prediction.get('weight', 1.0),
                        ),
                    )

            conn.commit()
            return analysis_id

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def update_feedback(self, analysis_id: int, feedback: str) -> bool:
        """Set feedback for an analysis. Returns True if updated."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('UPDATE analyses SET feedback = ? WHERE id = ?', (feedback, analysis_id))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _row_to_analysis_response(self, row: dict, model_rows: List[dict]) -> Dict[str, Any]:
        """Build API-shaped dict (is_phishing, model_votes, ensemble) from DB row + model_results."""
        fv = row.get('final_verdict')
        is_phishing = (fv == 'phishing') if fv is not None else bool(row.get('is_phishing', False))
        confidence = float(row.get('final_confidence', row.get('confidence', 0)))

        out = {
            'id': row['id'],
            'text': row.get('text', ''),
            'url': row.get('url'),
            'is_phishing': is_phishing,
            'confidence': confidence,
            'feedback': row.get('feedback'),
            'created_at': row.get('created_at'),
            'final_verdict': row.get('final_verdict', 'phishing' if is_phishing else 'safe'),
            'final_confidence': confidence,
            'phishing_score': float(row.get('phishing_score', 0)),
            'safety_score': float(row.get('safety_score', 0)),
            'total_weight': float(row.get('total_weight', 0)),
        }
        model_votes = []
        for r in model_rows:
            weight = float(r.get('weight_impact', r.get('weight', 0)))
            conf = float(r.get('confidence', 0))
            model_votes.append({
                'id': r.get('id'),
                'analysis_id': r.get('analysis_id'),
                'model_name': r.get('model_name'),
                'is_phishing': bool(r.get('model_verdict', r.get('is_phishing', 0))),
                'confidence': conf,
                'weight': weight,
                'weighted_vote': round(weight * (conf / 100.0), 4) if conf else 0,
                'error': r.get('error'),
            })
        out['model_votes'] = sorted(model_votes, key=lambda v: v['weighted_vote'], reverse=True)
        out['ensemble'] = {
            'weighted_phishing_score': out['phishing_score'],
            'weighted_safe_score': out['safety_score'],
            'total_weight': out['total_weight'],
        }
        return out

    def get_analysis(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Return one analysis by id with model_results."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT * FROM analyses WHERE id = ?', (analysis_id,))
            row = cursor.fetchone()
            if not row:
                return None
            cursor.execute(
                '''
                SELECT * FROM model_results
                WHERE analysis_id = ?
                ORDER BY weight_impact * (confidence / 100.0) DESC
                ''',
                (analysis_id,),
            )
            model_rows = [dict(r) for r in cursor.fetchall()]
            return self._row_to_analysis_response(dict(row), model_rows)
        finally:
            conn.close()

    def get_analyses(
        self,
        limit: int = 100,
        offset: int = 0,
        feedback: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return analyses list with model_results, optionally filtered by feedback."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            query = 'SELECT * FROM analyses'
            params = []
            if feedback:
                query += ' WHERE feedback = ?'
                params.append(feedback)
            query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            cursor.execute(query, params)
            rows = cursor.fetchall()

            analyses = []
            for row in rows:
                analysis_id = row['id']
                cursor.execute(
                    '''
                    SELECT * FROM model_results
                    WHERE analysis_id = ?
                    ORDER BY weight_impact * (confidence / 100.0) DESC
                    ''',
                    (analysis_id,),
                )
                model_rows = [dict(r) for r in cursor.fetchall()]
                analyses.append(self._row_to_analysis_response(dict(row), model_rows))
            return analyses
        finally:
            conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Aggregate stats: total, feedback counts, result distribution."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            stats = {}
            cursor.execute('SELECT COUNT(*) as count FROM analyses')
            stats['total_analyses'] = cursor.fetchone()['count']

            cursor.execute('SELECT COUNT(*) as count FROM analyses WHERE feedback IS NOT NULL')
            stats['total_feedback'] = cursor.fetchone()['count']

            cursor.execute(
                '''
                SELECT feedback, COUNT(*) as count
                FROM analyses
                WHERE feedback IS NOT NULL
                GROUP BY feedback
                '''
            )
            stats['feedback_distribution'] = {row['feedback']: row['count'] for row in cursor.fetchall()}

            cursor.execute(
                '''
                SELECT final_verdict, COUNT(*) as count
                FROM analyses
                GROUP BY final_verdict
                '''
            )
            stats['result_distribution'] = {'phishing': 0, 'safe': 0}
            for row in cursor.fetchall():
                key = 'phishing' if row['final_verdict'] == 'phishing' else 'safe'
                stats['result_distribution'][key] += row['count']
            return stats
        finally:
            conn.close()
