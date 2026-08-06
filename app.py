#!/usr/bin/env python3
# =========================================================
# ERPSA — Web Interface (Flask)
# =========================================================
# A simple web UI to demonstrate the risk scoring engine.
# Allows users to input risk factor text from two years and
# see the scored results.
# =========================================================

import sys
sys.path.insert(0, '.')

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from text_cleaning import clean_text_preserve_structure
from risk_section_parser import parse_risk_sections
from risk_matcher import match_risk_categories, RiskChangeStatus
from risk_classifier import classify_risk_changes
from step_3_scoring import run_scoring
from sentiment_scorer import score_text_sentiment
from lm_dictionary import get_dictionary


# =========================================================
# Sample Data (for demo mode)
# =========================================================

SAMPLE_CURRENT = """CYBERSECURITY AND DATA PRIVACY THREATS

We face increasingly severe and sophisticated cybersecurity threats that could result in catastrophic data breaches, material financial losses, and irreparable reputational damage. State-sponsored threat actors and organized criminal groups have significantly escalated attacks against retail companies. We may be unable to adequately defend against these threats despite substantial investments in security infrastructure. Any breach could expose us to costly litigation, regulatory penalties, and loss of customer trust that may materially impair our long-term financial performance. The frequency and severity of attempted intrusions has increased substantially over the past twelve months.

INVENTORY MANAGEMENT AND SUPPLY CHAIN DISRUPTION

We face significant risks related to our ability to effectively manage inventory levels in an environment of unprecedented supply chain disruption. Global shipping constraints, port congestion, and labor shortages have materially impaired our logistics operations and may continue to do so. We may be unable to accurately forecast consumer demand, which could result in excess inventory requiring significant markdowns that adversely affect our gross margins, or inventory shortages that impair our ability to serve customers. The financial impact of these disruptions could be substantial and may materially affect our results of operations and financial condition.

GENERAL ECONOMIC CONDITIONS

Our business is subject to the risks arising from adverse changes in domestic and global economic conditions. If economic conditions deteriorate, consumer spending may decline, which could adversely affect our results of operations.

COMPETITION

The retail industry is highly competitive. We compete with other mass merchandisers, department stores, and online retailers on the basis of price, quality, and convenience."""

SAMPLE_PRIOR = """CYBERSECURITY RISKS

We face cybersecurity risks common to companies in our industry. We invest in technology and maintain protocols to protect our systems and customer information from unauthorized access. We continue to monitor emerging threats.

GENERAL ECONOMIC CONDITIONS

Our business is subject to the risks arising from adverse changes in domestic and global economic conditions. If economic conditions deteriorate, consumer spending may decline, which could adversely affect our results of operations.

COMPETITION

The retail industry is highly competitive. We compete with other mass merchandisers, department stores, and online retailers on the basis of price, quality, and convenience."""


# =========================================================
# HTML Template
# =========================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ERPSA - Equity Risk Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1419;
            color: #e1e8ed;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a2332 0%, #0d1b2a 100%);
            border-bottom: 1px solid #2d3748;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 {
            font-size: 24px;
            color: #63b3ed;
            font-weight: 700;
        }
        .header h1 span { color: #a0aec0; font-weight: 400; }
        .header .badge {
            background: #2d3748;
            color: #68d391;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 30px 40px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 30px; }
        .panel {
            background: #1a2332;
            border: 1px solid #2d3748;
            border-radius: 12px;
            padding: 24px;
        }
        .panel h3 {
            color: #a0aec0;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }
        .panel h3 .year { color: #63b3ed; }
        textarea {
            width: 100%;
            height: 280px;
            background: #0f1419;
            border: 1px solid #2d3748;
            border-radius: 8px;
            color: #e1e8ed;
            padding: 16px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            line-height: 1.6;
            resize: vertical;
        }
        textarea:focus { outline: none; border-color: #63b3ed; }
        .controls {
            display: flex;
            gap: 16px;
            align-items: center;
            margin-bottom: 30px;
        }
        .btn {
            padding: 12px 28px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%);
            color: white;
        }
        .btn-primary:hover { background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); transform: translateY(-1px); }
        .btn-secondary {
            background: #2d3748;
            color: #a0aec0;
        }
        .btn-secondary:hover { background: #4a5568; color: #e1e8ed; }
        .ticker-input {
            background: #0f1419;
            border: 1px solid #2d3748;
            border-radius: 8px;
            color: #e1e8ed;
            padding: 10px 16px;
            font-size: 14px;
            width: 100px;
        }
        .ticker-input:focus { outline: none; border-color: #63b3ed; }
        label { color: #a0aec0; font-size: 13px; margin-right: 8px; }

        /* Results */
        .results { display: none; }
        .results.show { display: block; }
        .results-header {
            background: linear-gradient(135deg, #1a365d 0%, #1a2332 100%);
            border: 1px solid #2d3748;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
        }
        .results-header h2 { color: #63b3ed; margin-bottom: 12px; }
        .stats { display: flex; gap: 24px; flex-wrap: wrap; }
        .stat {
            background: #0f1419;
            border-radius: 8px;
            padding: 12px 20px;
            min-width: 140px;
        }
        .stat .value { font-size: 24px; font-weight: 700; color: #e1e8ed; }
        .stat .label { font-size: 11px; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.5px; }

        .risk-card {
            background: #1a2332;
            border: 1px solid #2d3748;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
            border-left: 4px solid #4a5568;
        }
        .risk-card.very-high { border-left-color: #e53e3e; }
        .risk-card.high { border-left-color: #ed8936; }
        .risk-card.medium { border-left-color: #ecc94b; }
        .risk-card.low { border-left-color: #68d391; }
        .risk-card .title { font-size: 15px; font-weight: 600; margin-bottom: 8px; }
        .risk-card .meta { display: flex; gap: 16px; font-size: 12px; color: #a0aec0; }
        .risk-card .probability {
            font-size: 28px;
            font-weight: 700;
            float: right;
            margin-top: -5px;
        }
        .risk-card .probability.very-high { color: #fc8181; }
        .risk-card .probability.high { color: #f6ad55; }
        .risk-card .probability.medium { color: #f6e05e; }
        .risk-card .probability.low { color: #68d391; }
        .risk-card .bar {
            height: 6px;
            background: #2d3748;
            border-radius: 3px;
            margin-top: 12px;
            overflow: hidden;
        }
        .risk-card .bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.8s ease;
        }
        .risk-card .bar-fill.very-high { background: linear-gradient(90deg, #e53e3e, #fc8181); }
        .risk-card .bar-fill.high { background: linear-gradient(90deg, #dd6b20, #f6ad55); }
        .risk-card .bar-fill.medium { background: linear-gradient(90deg, #d69e2e, #f6e05e); }
        .risk-card .bar-fill.low { background: linear-gradient(90deg, #38a169, #68d391); }

        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .status-badge.new { background: #2d3748; color: #fc8181; }
        .status-badge.modified { background: #2d3748; color: #f6ad55; }
        .status-badge.unchanged { background: #2d3748; color: #68d391; }
        .status-badge.removed { background: #2d3748; color: #a0aec0; }

        .signal-detail {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
            font-size: 12px;
        }
        .signal-detail .signal {
            background: #0f1419;
            padding: 6px 10px;
            border-radius: 4px;
        }
        .signal-detail .signal-name { color: #a0aec0; }
        .signal-detail .signal-value { color: #e1e8ed; font-weight: 600; }

        .loading {
            text-align: center;
            padding: 40px;
            color: #63b3ed;
            display: none;
        }
        .loading.show { display: block; }

        .footer {
            text-align: center;
            padding: 30px;
            color: #4a5568;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>ERPSA <span>Equity Risk Predictor & Sentiment Analyzer</span></h1>
        <span class="badge">Signals 1+2 Active</span>
    </div>

    <div class="container">
        <div class="grid">
            <div class="panel">
                <h3>Current Year <span class="year">(FY2022)</span> — Item 1A Risk Factors</h3>
                <textarea id="current-text" placeholder="Paste current year's Item 1A risk factor text here..."></textarea>
            </div>
            <div class="panel">
                <h3>Prior Year <span class="year">(FY2021)</span> — Item 1A Risk Factors</h3>
                <textarea id="prior-text" placeholder="Paste prior year's Item 1A risk factor text here..."></textarea>
            </div>
        </div>

        <div class="controls">
            <label>Ticker:</label>
            <input type="text" class="ticker-input" id="ticker" value="TGT" placeholder="TGT">
            <button class="btn btn-primary" onclick="analyzeRisks()">Analyze Risk Changes</button>
            <button class="btn btn-secondary" onclick="loadSample()">Load Sample Data</button>
        </div>

        <div class="loading" id="loading">
            Analyzing risk factors...
        </div>

        <div class="results" id="results"></div>
    </div>

    <div class="footer">
        ERPSA v0.3 | Signals: Textual Change Magnitude + Sentiment Direction<br>
        Academic basis: Cohen et al. "Lazy Prices" (2020) + Loughran & McDonald (2011)
    </div>

    <script>
        function loadSample() {
            document.getElementById('current-text').value = SAMPLE_CURRENT;
            document.getElementById('prior-text').value = SAMPLE_PRIOR;
            document.getElementById('ticker').value = 'TGT';
        }

        async function analyzeRisks() {
            const currentText = document.getElementById('current-text').value;
            const priorText = document.getElementById('prior-text').value;
            const ticker = document.getElementById('ticker').value || 'UNKNOWN';

            if (!currentText.trim() || !priorText.trim()) {
                alert('Please paste risk factor text for both years (or click "Load Sample Data").');
                return;
            }

            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        current_text: currentText,
                        prior_text: priorText,
                        ticker: ticker,
                    }),
                });
                const data = await response.json();
                displayResults(data);
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
            }
        }

        function displayResults(data) {
            const container = document.getElementById('results');
            const risks = data.risks || [];

            const highPriority = risks.filter(r => r.probability >= 50);
            const actionable = risks.filter(r => r.probability >= 20 && r.probability < 50);
            const low = risks.filter(r => r.probability < 20);

            let html = `
                <div class="results-header">
                    <h2>Risk Analysis: ${data.ticker} (FY${data.current_year} vs FY${data.prior_year})</h2>
                    <div class="stats">
                        <div class="stat"><div class="value">${risks.length}</div><div class="label">Risks Identified</div></div>
                        <div class="stat"><div class="value">${highPriority.length}</div><div class="label">High Priority</div></div>
                        <div class="stat"><div class="value">${actionable.length}</div><div class="label">Actionable</div></div>
                        <div class="stat"><div class="value">${data.avg_score}</div><div class="label">Avg Score</div></div>
                    </div>
                </div>
            `;

            if (highPriority.length > 0) {
                html += '<h3 style="color:#fc8181;margin-bottom:12px;font-size:14px;">HIGH PRIORITY</h3>';
                highPriority.forEach(r => { html += renderRiskCard(r); });
            }
            if (actionable.length > 0) {
                html += '<h3 style="color:#f6e05e;margin:20px 0 12px;font-size:14px;">ACTIONABLE</h3>';
                actionable.forEach(r => { html += renderRiskCard(r); });
            }
            if (low.length > 0) {
                html += '<h3 style="color:#68d391;margin:20px 0 12px;font-size:14px;">LOW RISK / UNCHANGED</h3>';
                low.forEach(r => { html += renderRiskCard(r); });
            }

            container.innerHTML = html;
            container.classList.add('show');
        }

        function renderRiskCard(risk) {
            const level = risk.probability >= 70 ? 'very-high' :
                          risk.probability >= 50 ? 'high' :
                          risk.probability >= 20 ? 'medium' : 'low';
            const statusClass = risk.status.toLowerCase();

            return `
                <div class="risk-card ${level}">
                    <span class="probability ${level}">${risk.probability}%</span>
                    <div class="title">${risk.title}</div>
                    <div class="meta">
                        <span class="status-badge ${statusClass}">${risk.status}</span>
                        <span>Level: ${risk.level}</span>
                    </div>
                    <div class="bar"><div class="bar-fill ${level}" style="width:${risk.probability}%"></div></div>
                    <div class="signal-detail">
                        <div class="signal"><span class="signal-name">Signal 1 (Textual Change):</span> <span class="signal-value">${risk.textual_score}</span></div>
                        <div class="signal"><span class="signal-name">Signal 2 (Sentiment):</span> <span class="signal-value">${risk.sentiment_score}</span></div>
                    </div>
                </div>
            `;
        }

        const SAMPLE_CURRENT = `""" + SAMPLE_CURRENT.replace('`', '\\`').replace('\\n', '\\n') + """`;
        const SAMPLE_PRIOR = `""" + SAMPLE_PRIOR.replace('`', '\\`').replace('\\n', '\\n') + """`;
    </script>
</body>
</html>"""


# =========================================================
# HTTP Request Handler
# =========================================================

class ERPSAHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for the ERPSA web interface."""

    def do_GET(self):
        """Serve the main page."""
        if self.path == '/' or self.path == '':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle analysis requests."""
        if self.path == '/analyze':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')

            try:
                data = json.loads(body)
                result = self._run_analysis(data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def _run_analysis(self, data):
        """Run the full analysis pipeline."""
        current_text = data.get('current_text', '')
        prior_text = data.get('prior_text', '')
        ticker = data.get('ticker', 'UNKNOWN')

        # Clean text
        clean_current = clean_text_preserve_structure(current_text)
        clean_prior = clean_text_preserve_structure(prior_text)

        # Parse into sections
        sections_current = parse_risk_sections(clean_current)
        sections_prior = parse_risk_sections(clean_prior)

        # Match and classify
        matches = match_risk_categories(sections_current, sections_prior)
        change_report = classify_risk_changes(
            matches=matches,
            ticker=ticker,
            current_year=2022,
            prior_year=2021,
            total_current=len(sections_current),
            total_prior=len(sections_prior),
        )

        # Score
        scoring = run_scoring(change_report, verbose=False)

        # Format response
        risks = []
        for r in scoring.risk_scores:
            risks.append({
                'title': r.title[:80],
                'status': r.status.value,
                'probability': round(r.preliminary_probability, 1),
                'level': r.risk_level_label,
                'textual_score': round(r.textual_change_score, 3),
                'sentiment_score': round(r.sentiment_score, 3),
            })

        # Sort by probability descending
        risks.sort(key=lambda x: x['probability'], reverse=True)

        return {
            'ticker': ticker,
            'current_year': 2022,
            'prior_year': 2021,
            'risks': risks,
            'avg_score': round(scoring.risk_scores[0].preliminary_probability if risks else 0, 1) if risks else 0,
            'total_current': len(sections_current),
            'total_prior': len(sections_prior),
        }

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        pass


# =========================================================
# Main
# =========================================================

def main():
    port = 5000
    server = HTTPServer(('0.0.0.0', port), ERPSAHandler)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ERPSA — Equity Risk Predictor & Sentiment Analyzer        ║
║  Web Interface v0.3                                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                            ║
║  Server running at: http://localhost:{port}                 ║
║                                                            ║
║  Signals Active:                                           ║
║    [1] Textual Change Magnitude (Cohen et al. 2020)        ║
║    [2] Sentiment Direction (Loughran & McDonald 2011)      ║
║                                                            ║
║  Press Ctrl+C to stop                                      ║
╚══════════════════════════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
