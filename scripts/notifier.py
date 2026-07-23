# scripts/notifier.py
"""
Email Notification Service for F1CAST

Formats pre-race predictions and post-race comparison reports into responsive HTML emails.
Dispatches emails via Resend API (https://resend.com) when RESEND_API_KEY is configured.
"""

import os
import json
import urllib.request
import urllib.error

RESEND_API_KEY = os.getenv('RESEND_API_KEY')
NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL')
if not NOTIFICATION_EMAIL or not NOTIFICATION_EMAIL.strip():
    NOTIFICATION_EMAIL = 'subscribers@f1cast.com'

SENDER_EMAIL = os.getenv('SENDER_EMAIL')
if not SENDER_EMAIL or not SENDER_EMAIL.strip():
    SENDER_EMAIL = 'F1CAST Predictions <onboarding@resend.dev>'

def send_email(subject, html_content, recipient=None):
    if recipient is None:
        recipient = NOTIFICATION_EMAIL

    api_key = os.getenv('RESEND_API_KEY')
    if not api_key:
        print("[NOTIFIER NOTICE] RESEND_API_KEY environment variable is not configured.")
        print(f"[NOTIFIER DRY-RUN] Would send email '{subject}' to '{recipient}'.")
        return False

    payload = {
        "from": SENDER_EMAIL,
        "to": [recipient] if isinstance(recipient, str) else recipient,
        "subject": subject,
        "html": html_content
    }

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "F1CAST-Pipeline/1.0"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            res_body = json.loads(response.read().decode('utf-8'))
            print(f"[NOTIFIER SUCCESS] Email dispatched successfully! Message ID: {res_body.get('id')}")
            return True
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode('utf-8')
        print(f"[NOTIFIER ERROR] Resend HTTP Error ({e.code}): {err_msg}")
        return False
    except Exception as e:
        print(f"[NOTIFIER ERROR] Failed to dispatch email: {e}")
        return False

def format_prediction_email(season, round_num, gp_name, predictions_list):
    """
    Formations HTML email for Pre-Race Grid Predictions.
    `predictions_list`: list of dicts with keys ['Predicted_Rank', 'Driver', 'Team', 'GridPosition', 'DNF_Risk']
    """
    rows_html = ""
    for driver in sorted(predictions_list, key=lambda x: x['Predicted_Rank']):
        dnf_risk_pct = f"{driver.get('DNF_Risk', 0.0) * 100:.1f}%"
        rows_html += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 10px; text-align: center; font-weight: bold; color: #e74c3c;">#{int(driver['Predicted_Rank'])}</td>
            <td style="padding: 10px; font-weight: bold;">{driver['Driver']}</td>
            <td style="padding: 10px; color: #555;">{driver['Team']}</td>
            <td style="padding: 10px; text-align: center;">P{int(driver['GridPosition'])}</td>
            <td style="padding: 10px; text-align: center; color: #e67e22;">{dnf_risk_pct}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: Arial, sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <div style="background-color: #15151e; padding: 15px; border-radius: 6px; text-align: center; color: #ffffff;">
                <h1 style="margin: 0; font-size: 22px;">🏎️ F1CAST Race Prediction</h1>
                <p style="margin: 5px 0 0 0; color: #e74c3c; font-weight: bold;">{season} Round {round_num} — {gp_name}</p>
            </div>
            <p style="color: #333; margin-top: 20px;">Here is the ML model's predicted finishing order and DNF risk for the upcoming <strong>{gp_name}</strong> based on starting grid positions and rolling driver form:</p>
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <thead>
                    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #ddd; text-align: left;">
                        <th style="padding: 10px; text-align: center;">Pred Rank</th>
                        <th style="padding: 10px;">Driver</th>
                        <th style="padding: 10px;">Team</th>
                        <th style="padding: 10px; text-align: center;">Grid</th>
                        <th style="padding: 10px; text-align: center;">DNF Risk</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <hr style="border: none; border-top: 1px solid #eee; margin: 25px 0;" />
            <p style="font-size: 12px; color: #888; text-align: center;">Automated by F1CAST Machine Learning Pipeline • Powered by XGBoost</p>
        </div>
    </body>
    </html>
    """
    return html

def format_comparison_email(season, round_num, gp_name, comparison_list, spearman_corr):
    """
    Formats HTML email for Post-Race Comparison (Actual vs Predicted).
    `comparison_list`: list of dicts with keys ['Driver', 'FinishPos', 'Predicted_Rank', 'GridPosition', 'Status']
    """
    rows_html = ""
    for driver in sorted(comparison_list, key=lambda x: x['FinishPos']):
        actual_pos = int(driver['FinishPos'])
        pred_rank = int(driver['Predicted_Rank'])
        diff = actual_pos - pred_rank
        
        diff_str = "0" if diff == 0 else (f"+{diff}" if diff > 0 else f"{diff}")
        diff_color = "#27ae60" if abs(diff) <= 2 else ("#e67e22" if abs(diff) <= 5 else "#c0392b")

        rows_html += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 10px; text-align: center; font-weight: bold;">P{actual_pos}</td>
            <td style="padding: 10px; font-weight: bold;">{driver['Driver']}</td>
            <td style="padding: 10px; text-align: center;">#{pred_rank}</td>
            <td style="padding: 10px; text-align: center; color: {diff_color}; font-weight: bold;">{diff_str}</td>
            <td style="padding: 10px; color: #555;">{driver.get('Status', 'Finished')}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: Arial, sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <div style="background-color: #15151e; padding: 15px; border-radius: 6px; text-align: center; color: #ffffff;">
                <h1 style="margin: 0; font-size: 22px;">📊 F1CAST Post-Race Results</h1>
                <p style="margin: 5px 0 0 0; color: #27ae60; font-weight: bold;">{season} Round {round_num} — {gp_name}</p>
            </div>
            
            <div style="background-color: #eafaf1; border-left: 4px solid #27ae60; padding: 12px; margin: 20px 0; border-radius: 4px;">
                <strong style="color: #27ae60; font-size: 16px;">Model Accuracy:</strong> 
                <span style="font-size: 16px; font-weight: bold; color: #2c3e50;">Spearman Rank Correlation = {spearman_corr:.3f}</span>
            </div>

            <p style="color: #333;">Here is the actual race finishing order compared against F1CAST pre-race predictions:</p>
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <thead>
                    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #ddd; text-align: left;">
                        <th style="padding: 10px; text-align: center;">Actual</th>
                        <th style="padding: 10px;">Driver</th>
                        <th style="padding: 10px; text-align: center;">Predicted</th>
                        <th style="padding: 10px; text-align: center;">Diff</th>
                        <th style="padding: 10px;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <hr style="border: none; border-top: 1px solid #eee; margin: 25px 0;" />
            <p style="font-size: 12px; color: #888; text-align: center;">Automated by F1CAST Machine Learning Pipeline • Powered by XGBoost</p>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    test_pred = [
        {'Predicted_Rank': 1, 'Driver': 'VER', 'Team': 'Red Bull Racing', 'GridPosition': 1, 'DNF_Risk': 0.05},
        {'Predicted_Rank': 2, 'Driver': 'NOR', 'Team': 'McLaren', 'GridPosition': 2, 'DNF_Risk': 0.08}
    ]
    email_html = format_prediction_email(2026, 10, 'Belgian GP', test_pred)
    print("Test HTML Email Length:", len(email_html))
