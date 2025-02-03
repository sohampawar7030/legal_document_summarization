import requests
from bs4 import BeautifulSoup
import time
import streamlit as st
import threading
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# Load environment variables from .env file
load_dotenv()

# Google Sheets Configuration
SHEET_ID = '1bZjlA-UJrBhWS2jHlEQ-7nbmDvxpEoKylgxHW51Hhzc'  # Google Sheets ID
RANGE = 'Sheet1!A:D'  # The range where you want to append the data

# Predefined list of URLs to track
TRACKING_URLS = [
    "https://gdpr-info.eu/recitals/no-1/"]

# Event to signal thread termination
stop_event = threading.Event()

# Authenticate Google Sheets API
def authenticate_google_sheets():
    creds = Credentials.from_service_account_file(
        'Credentials.json',
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    service = build('sheets', 'v4', credentials=creds)
    return service

# Append data to Google Sheets
def append_to_google_sheets(service, url, title, content, timestamp):
    values = [
        [url, title, content[:200], timestamp]  # Prepare row to append
    ]
    body = {'values': values}
    try:
        service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range=RANGE,
            valueInputOption="RAW",
            body=body
        ).execute()
        st.write(f"Data appended to Google Sheets at {timestamp}.")
    except Exception as e:
        st.error(f"Error appending to Google Sheets: {e}")

# Send email notification
def send_email_notification(to_email, url, title, content, timestamp):
    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    if not sender_email or not sender_password:
        st.error("Environment variables not loaded. Check your .env file.")
        return

    msg = EmailMessage()
    msg["Subject"] = f"Website Update Notification for {url}"
    msg["From"] = sender_email
    msg["To"] = to_email
    msg.set_content(f"""
    Website: {url}
    Title: {title}
    Content (preview): {content[:200]}...
    Tracked at: {timestamp}
    """)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            st.success(f"Notification email sent to {to_email}")
    except smtplib.SMTPException as e:
        st.error(f"SMTP Error: {e}")

# Fetch website data
def fetch_website_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.strip() if soup.title else 'No title available'
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text.strip() for p in paragraphs]) if paragraphs else 'New Notification available'
        return title, content
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching website data: {e}")
        return "Error occurred", "New notification detected. No content available due to an error."

# Track websites and store updates in Google Sheets
def track_websites(urls, recipient_email, interval=60, max_duration=20*60):
    st.write(f"Started tracking for {recipient_email}")
    service = authenticate_google_sheets()
    last_updates = {}  # To track changes in website content

    start_time = datetime.now()  # Record the start time
    end_time = start_time + timedelta(seconds=max_duration)  # Set end time (20 minutes later)

    while not stop_event.is_set() and datetime.now() < end_time:
        for url in urls:
            title, content = fetch_website_data(url)
            if title and content:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Check for updates
                if url not in last_updates or last_updates[url] != (title, content):
                    last_updates[url] = (title, content)

                    # Append to Google Sheets
                    append_to_google_sheets(service, url, title, content, timestamp)

                    # Send notification email
                    try:
                        send_email_notification(recipient_email, url, title, content, timestamp)
                    except Exception as e:
                        st.error(f"Error sending email notification: {e}")

        # Wait for the next interval or until stop_event is set
        stop_event.wait(interval)

    st.write("Stopped tracking after 20 minutes.")

# Display tracking status
def display_tracking_status():
    st.title("Update Tracking System with Notifications")

    email_input = st.text_input("Enter your email for notifications:")

    # Maintain thread state
    if "tracking_thread" not in st.session_state:
        st.session_state["tracking_thread"] = None

    if email_input:
        # Start tracking
        if st.button("Tracking"):
            if st.session_state["tracking_thread"] is None or not st.session_state["tracking_thread"].is_alive():
                stop_event.clear()  # Clear the stop flag to allow tracking
                thread = threading.Thread(target=track_websites, args=(TRACKING_URLS, email_input), daemon=True)
                thread.start()
                st.session_state["tracking_thread"] = thread
                st.success(f"Notifications will be sent to {email_input}.")
            else:
                st.warning("Tracking Updates is already running.")

        # Stop tracking
        if st.button("Stop Tracking"):
            if st.session_state["tracking_thread"] is not None and st.session_state["tracking_thread"].is_alive():
                stop_event.set()  # Signal the thread to stop
                st.session_state["tracking_thread"].join()  # Wait for the thread to finish
                st.session_state["tracking_thread"] = None
                st.success("Tracking stopped.")
            else:
                st.warning("No active tracking to stop.")

# Main function
def main():
    display_tracking_status()

if __name__ == "__main__":
    main()
