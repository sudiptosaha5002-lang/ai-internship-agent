"""
Email Utilities for AI Internship Finder Agent
==============================================
Handles sending:
1. Verification OTP emails
2. Password reset OTP emails
3. Detailed resume and skills analysis reports

If SMTP environment variables are configured, it sends real emails.
Otherwise, it logs them cleanly to the console for development.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# SMTP Configurations from environment variables
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_SENDER = os.getenv("SMTP_SENDER", SMTP_USER or "noreply@internshipagent.com")

def _send_email(to_email, subject, html_content):
    """Internal helper to send email via SMTP if credentials exist, otherwise log to console."""
    if not SMTP_USER or not SMTP_PASS:
        print("\n" + "="*60)
        print(f"📧 [DEV EMAIL LOG] Sent to: {to_email}")
        print(f"📧 Subject: {subject}")
        print("-"*60)
        # Simple plain-text extraction from html using built-in re to display nicely in logs
        import re
        text_content = re.sub(r'<[^>]+>', '', html_content)
        text_content = re.sub(r'\n\s*\n', '\n', text_content)
        print(text_content.strip())
        print("="*60 + "\n")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_SENDER
        msg["To"] = to_email

        part = MIMEText(html_content, "html")
        msg.attach(part)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_SENDER, to_email, msg.as_string())
        
        print(f"[Email] Successfully sent email to {to_email}: {subject}")
        return True
    except Exception as e:
        print(f"[Email ERROR] Failed to send email to {to_email} via SMTP: {e}")
        return False

def send_otp_email(email, otp):
    """Send verification OTP code during sign-up/login."""
    subject = "🎯 [AI Internship Finder] Verify Your Email Address"
    html_content = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 12px; background: #ffffff;">
        <h2 style="color: #7000ff; text-align: center;">Welcome to AI Internship Finder!</h2>
        <p>Thank you for signing up. Please verify your email address by using the OTP code below:</p>
        <div style="text-align: center; margin: 30px 0;">
            <span style="font-size: 32px; font-weight: bold; letter-spacing: 5px; color: #00f0ff; background: #0f0c1b; padding: 12px 30px; border-radius: 8px; border: 1px solid #7000ff;">
                {otp}
            </span>
        </div>
        <p style="color: #666666; font-size: 14px;">This code is valid for 10 minutes. If you did not request this code, please ignore this email.</p>
        <hr style="border: 0; border-top: 1px solid #eeeeee; margin: 20px 0;">
        <p style="text-align: center; color: #999999; font-size: 12px;">Built with ❤️ by AI Internship Finder Team</p>
    </div>
    """
    return _send_email(email, subject, html_content)

def send_reset_email(email, otp):
    """Send password reset code."""
    subject = "🔑 [AI Internship Finder] Reset Your Password"
    html_content = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 12px; background: #ffffff;">
        <h2 style="color: #7000ff; text-align: center;">Password Reset Request</h2>
        <p>We received a request to reset your password. Use the verification code below to set a new password:</p>
        <div style="text-align: center; margin: 30px 0;">
            <span style="font-size: 32px; font-weight: bold; letter-spacing: 5px; color: #ff0055; background: #1b0c0c; padding: 12px 30px; border-radius: 8px; border: 1px solid #ff0055;">
                {otp}
            </span>
        </div>
        <p style="color: #666666; font-size: 14px;">This code is valid for 10 minutes. If you did not request this, please secure your account.</p>
        <hr style="border: 0; border-top: 1px solid #eeeeee; margin: 20px 0;">
        <p style="text-align: center; color: #999999; font-size: 12px;">Built with ❤️ by AI Internship Finder Team</p>
    </div>
    """
    return _send_email(email, subject, html_content)

def send_analysis_email(user_email, user_name, skills, inferred_role):
    """Send detailed resume parsing and career path analysis report."""
    subject = "📈 [AI Internship Finder] Your AI Career & Resume Analysis Report"
    skills_list = "".join([f"<li style='margin-bottom: 6px;'><b>{sk}</b></li>" for sk in skills]) if skills else "<li>No skills extracted</li>"
    html_content = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 12px; background: #ffffff; color: #333333;">
        <h2 style="color: #7000ff; border-bottom: 2px solid #7000ff; padding-bottom: 10px;">AI Resume Analysis Summary</h2>
        <p>Hello <b>{user_name}</b>,</p>
        <p>Our intelligent agent has successfully parsed and analyzed your resume. Here are the details we've generated for your career copilot profile:</p>
        
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
            <tr style="background: #f8f9fa;">
                <td style="padding: 10px; border: 1px solid #eeeeee; font-weight: bold; width: 30%;">Inferred Target Role:</td>
                <td style="padding: 10px; border: 1px solid #eeeeee; color: #7000ff; font-weight: bold;">{inferred_role}</td>
            </tr>
        </table>

        <h3 style="color: #00f0ff; background: #0f0c1b; padding: 8px 12px; border-radius: 6px; display: inline-block;">Core Skills Extracted:</h3>
        <ul style="padding-left: 20px;">
            {skills_list}
        </ul>

        <p>Our conversational agent is now ready to assist you in finding active internships fitting these exact criteria. You can log in and start chatting with your copilot or take an AI Mock Interview tailored to your profile!</p>
        
        <hr style="border: 0; border-top: 1px solid #eeeeee; margin: 25px 0;">
        <p style="text-align: center; color: #999999; font-size: 12px;">Built with ❤️ by AI Internship Finder Team</p>
    </div>
    """
    return _send_email(user_email, subject, html_content)
