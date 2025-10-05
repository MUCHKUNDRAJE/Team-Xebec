import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
from dotenv import load_dotenv
import os

load_dotenv()

class CMEAlertMailer:
    def __init__(self, sender_email, sender_password):
        """
        Initialize the CME Alert Mailer
        
        Args:
            sender_email (str): Your email address
            sender_password (str): Your email password or app-specific password
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
    
    def should_send_alert(self, impact_level):
        """Check if alert should be sent based on impact level"""
        alert_levels = ["High", "Very High", "Extreme"]
        return impact_level in alert_levels
    
    def get_precautions(self, impact_level):
        """Generate precautions based on impact level"""
        precautions = {
            "High": [
                "Monitor space weather updates regularly",
                "Keep backup power sources ready",
                "Be prepared for possible GPS disruptions",
                "Check emergency communication systems"
            ],
            "Very High": [
                "Activate emergency response plans",
                "Ensure all backup systems are operational",
                "Expect GPS and communication disruptions",
                "Avoid unnecessary travel to polar regions",
                "Keep emergency supplies ready"
            ],
            "Extreme": [
                "IMMEDIATE ACTION: Activate all emergency protocols",
                "Expect widespread power outages",
                "GPS and communication systems may fail completely",
                "Stay home if possible - avoid all travel",
                "Keep flashlights, batteries, and emergency supplies ready",
                "Charge all devices now",
                "Have cash on hand (ATMs may not work)",
                "Fill bathtubs with water for emergency use"
            ]
        }
        return precautions.get(impact_level, [])
    
    def format_datetime(self, dt_string):
        """Format datetime string to readable format"""
        try:
            dt = datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
            return dt.strftime("%B %d, %Y at %I:%M %p UTC")
        except:
            return dt_string
    
    def create_email_body(self, cme_data):
        """Create professional email body"""
        impact_level = cme_data['impact_assessment']['impact_level']
        severity_score = cme_data['impact_assessment']['severity_score']
        
        # Determine urgency and color
        if impact_level == "Extreme":
            urgency_header = "🚨 CRITICAL SOLAR STORM ALERT 🚨"
            header_color = "#b71c1c"
        elif impact_level == "Very High":
            urgency_header = "⚠️ URGENT SOLAR STORM ALERT ⚠️"
            header_color = "#e65100"
        else:
            urgency_header = "⚠️ SOLAR STORM ALERT ⚠️"
            header_color = "#f57c00"
        
        precautions = self.get_precautions(impact_level)
        precautions_html = "".join([f"<li style='margin: 8px 0;'>{p}</li>" for p in precautions])
        
        html_body = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                }}
                .header {{
                    background-color: {header_color};
                    color: white;
                    padding: 25px;
                    text-align: center;
                    font-size: 22px;
                    font-weight: bold;
                }}
                .content {{
                    padding: 25px;
                    background-color: #ffffff;
                }}
                .alert-box {{
                    background-color: #fff8e1;
                    border-left: 5px solid #ffa000;
                    padding: 20px;
                    margin: 20px 0;
                    font-size: 16px;
                }}
                .alert-box strong {{
                    display: block;
                    margin-bottom: 8px;
                    font-size: 18px;
                }}
                .info-box {{
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .info-item {{
                    margin: 10px 0;
                    font-size: 15px;
                }}
                .info-label {{
                    font-weight: bold;
                    color: #555;
                }}
                .precautions {{
                    background-color: #e3f2fd;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 25px 0;
                }}
                .precautions h3 {{
                    color: #1565c0;
                    margin-top: 0;
                    font-size: 20px;
                }}
                .precautions ul {{
                    margin: 15px 0;
                    padding-left: 25px;
                }}
                .what-is {{
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                    font-size: 14px;
                }}
                .footer {{
                    background-color: #f5f5f5;
                    padding: 20px;
                    text-align: center;
                    font-size: 12px;
                    color: #666;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                {urgency_header}
            </div>
            
            <div class="content">
                <div class="alert-box">
                    <strong>⚡ Impact Level: {impact_level}</strong>
                    <strong>🎯 Severity: {severity_score}/10</strong>
                    <strong>📅 Expected Arrival: {self.format_datetime(cme_data['impact_assessment']['estimated_arrival'])}</strong>
                </div>
                
                <p style="font-size: 16px; line-height: 1.8;">
                    A powerful <strong>solar storm</strong> is heading toward Earth and will arrive in approximately <strong>{cme_data['days_from_now']:.1f} days</strong>. This event has the potential to disrupt technology and infrastructure.
                </p>
                
                <div class="info-box">
                    <div class="info-item">
                        <span class="info-label">Storm Speed:</span> {cme_data['cme_parameters']['speed']:.0f} km/s
                    </div>
                    <div class="info-item">
                        <span class="info-label">Geomagnetic Storm:</span> {cme_data['impact_assessment']['geomagnetic_storm_potential']}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Prediction Confidence:</span> {cme_data['metadata']['confidence_score'] * 100:.0f}%
                    </div>
                </div>
                
                <div class="precautions">
                    <h3>🛡️ What You Should Do</h3>
                    <ul>
                        {precautions_html}
                    </ul>
                </div>
                
                <div class="what-is">
                    <strong>What is a Solar Storm?</strong>
                    <p style="margin: 10px 0 0 0;">
                        A solar storm occurs when the Sun releases a massive burst of energy and charged particles. When it reaches Earth, it can disrupt:
                    </p>
                    <ul style="margin: 10px 0 0 0; padding-left: 20px;">
                        <li>Power grids (potential blackouts)</li>
                        <li>GPS and navigation</li>
                        <li>Mobile and internet services</li>
                        <li>Satellites and communications</li>
                    </ul>
                </div>
                
                <p style="font-size: 15px; margin-top: 25px; padding: 15px; background-color: #fff3e0; border-radius: 5px;">
                    <strong>📱 Stay Updated:</strong> Monitor local news and official space weather alerts. This is a developing situation.
                </p>
            </div>
            
            <div class="footer">
                Automated Solar Storm Alert System<br>
                Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p UTC")}
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def send_alert(self, recipient_email, cme_data):
        """
        Send CME alert email
        
        Args:
            recipient_email (str): Recipient's email address
            cme_data (dict): CME prediction data
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        impact_level = cme_data['impact_assessment']['impact_level']
        
        # Check if alert should be sent
        if not self.should_send_alert(impact_level):
            print(f"Impact level '{impact_level}' does not require alert. Email not sent.")
            return False
        
        try:
            # Create message
            message = MIMEMultipart('alternative')
            message['From'] = self.sender_email
            message['To'] = recipient_email
            message['Subject'] = f"🚨 SOLAR STORM ALERT: {impact_level} Impact Expected"
            
            # Create email body
            html_body = self.create_email_body(cme_data)
            message.attach(MIMEText(html_body, 'html'))
            
            # Send email
            print("Connecting to SMTP server...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                print("Logging in...")
                server.login(self.sender_email, self.sender_password)
                print("Sending email...")
                server.send_message(message)
            
            print(f"✅ Alert email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            print(f"❌ Error sending email: {str(e)}")
            return False


# Usage Example
# if __name__ == "__main__":
#     # Your email credentials
#     SENDER_EMAIL = os.getenv("EMAIL")
#     SENDER_PASSWORD =  os.getenv("PASSWORD")
    
#     # Recipient email
#     RECIPIENT_EMAIL = "muchkundthote@gmail.com"
    
#     # CME Data
#     cme_data = {
#         "prediction_id": 1,
#         "time21_5": "2025-10-05T13:20Z",
#         "days_from_now": 0.21,
#         "cme_parameters": {
#             "latitude": -2.78,
#             "longitude": 14.89,
#             "halfAngle": 27.68,
#             "speed": 525.34
#         },
#         "impact_assessment": {
#             "impact_level": "High",  # Change to "High", "Very High", or "Extreme" to test
#             "severity_score": 6,
#             "earth_directed": True,
#             "estimated_arrival": "2025-10-08T20:39Z",
#             "geomagnetic_storm_potential": "Moderate"
#         },
#         "metadata": {
#             "type": "S",
#             "isMostAccurate": True,
#             "catalog": "M2M_CATALOG",
#             "dataLevel": "0",
#             "confidence_score": 0.9,
#             "prediction_method": "Deep-LSTM + RF + GBM Ensemble"
#         }
#     }
    
#     # Create mailer instance
#     mailer = CMEAlertMailer(SENDER_EMAIL, SENDER_PASSWORD)
    
#     # Send alert (only sends if impact level is High, Very High, or Extreme)
#     mailer.send_alert(RECIPIENT_EMAIL, cme_data)