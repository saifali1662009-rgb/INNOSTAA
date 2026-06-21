import os
import re
import json
import smtplib
import imaplib
import email as email_lib
import threading
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

AUTOMATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "automation_data")
CONFIG_FILE    = os.path.join(AUTOMATION_DIR, "email_config.json")
CONTACTS_FILE  = os.path.join(AUTOMATION_DIR, "contacts.json")
SCHEDULE_FILE  = os.path.join(AUTOMATION_DIR, "scheduled_emails.json")
os.makedirs(AUTOMATION_DIR, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"
client       = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are Innostaa's email automation AI. You help users compose, send, read, and reply to emails through voice commands.

You always respond in this exact JSON format — nothing else:
{
  "speak": "what to say to the user",
  "action": "ask | confirm_email | confirm_subject | generate_body | send | send_multiple | schedule | read | reply | save_contact | list_scheduled | done | error",
  "data": {}
}

Actions and their data fields:
- "ask": need more info. Put question in "speak".
- "confirm_email": parsed a spoken email. data: {"email": "normalized@email.com", "for_name": "recipient name"}
- "confirm_subject": confirm subject. data: {"subject": "..."}
- "generate_body": ready to write body for one or more emails. data: {"emails": [{"name":"...", "email":"...", "subject":"...", "description":"..."}]}
- "send": send one email. data: {"to": "email", "subject": "...", "body": "...", "recipient_name": "...", "save_contact": true/false}
- "send_multiple": send multiple emails each with own subject/body. data: {"emails": [{"name":"...", "email":"...", "subject":"...", "body":"..."}], "save_contacts": true/false}
- "schedule": schedule one or more emails for later, each with own subject/body. data: {"emails": [{"name":"...", "email":"...", "subject":"...", "body":"..."}], "send_at": "YYYY-MM-DD HH:MM", "save_contacts": true/false}
- "read": fetch inbox. data: {"count": N}
- "reply": reply to fetched email. data: {"email_index": N, "description": "..."}
- "save_contact": save contact. data: {"name": "...", "email": "..."}
- "list_scheduled": user wants to know pending scheduled emails. data: {}
- "done": finished. 
- "error": problem.

CRITICAL RULES:
1. When user says "no" or corrects something — they are correcting the PREVIOUS answer. Ask them to repeat that specific field only.
2. Normalize all spoken emails: "saif ali 1662009 at the rate gmail dot com" → "saifali1662009@gmail.com", "john dot doe at yahoo dot com" → "john.doe@yahoo.com"
3. ALWAYS confirm each recipient's email and the subject before sending or scheduling.
4. If recipient name matches a known contact, propose that email and confirm.
5. For multiple emails, collect each recipient name, email, subject, and description separately. Each email can have a completely different subject and body.
   Ask about each email one by one if needed, then confirm all before sending.
6. For scheduled emails, parse natural time: "tomorrow at 3pm", "in 2 hours", "tonight at 8". Always confirm the parsed datetime with user.
7. If user says "save their contact" or "remember them" — set save_contact/save_contacts true.
8. Keep spoken responses short and natural. Never expose JSON to user.
9. If command has all info upfront, extract everything and skip unnecessary questions.
10. Today's date and time will be provided in the conversation context.
"""


def _load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        if "email" in cfg:
            cfg["email"] = cfg["email"].strip().replace("mailto:", "").strip("[]() ")
        if "password" in cfg:
            cfg["password"] = cfg["password"].strip()
        return cfg
    return {}

def _save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def _load_contacts():
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_contacts(contacts):
    with open(CONTACTS_FILE, "w") as f:
        json.dump(contacts, f, indent=2)

def _load_schedule():
    if os.path.exists(SCHEDULE_FILE):
        with open(SCHEDULE_FILE, "r") as f:
            return json.load(f)
    return []

def _save_schedule(queue):
    with open(SCHEDULE_FILE, "w") as f:
        json.dump(queue, f, indent=2)

def get_automation_config():
    cfg      = _load_config()
    contacts = _load_contacts()
    return json.dumps({
        "email":    cfg.get("email", ""),
        "password": cfg.get("password", ""),
        "smtp":     cfg.get("smtp", "smtp.gmail.com"),
        "port":     cfg.get("port", "587"),
        "imap":     cfg.get("imap", "imap.gmail.com"),
        "contacts": contacts
    })

def save_automation_config(json_str):
    try:
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
        cfg  = _load_config()
        for key in ["email", "password", "smtp", "port", "imap"]:
            if key in data:
                cfg[key] = data[key]
        _save_config(cfg)
        if "contacts" in data and isinstance(data["contacts"], dict):
            _save_contacts(data["contacts"])
    except Exception as e:
        print(f"[Automation] Config save error: {e}")


def _ai_chat(history):
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            temperature=0.3
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[Automation] AI error: {e}")
        return {"speak": "I had a processing error. Please try again.", "action": "error", "data": {}}


def _send_smtp(cfg, to_email, subject, body):
    try:
        sender   = cfg["email"].strip().replace("mailto:", "").strip("[]() ")
        password = cfg["password"].strip()
        port     = int(str(cfg.get("port", 587)).strip())
        host     = cfg["smtp"].strip()

        msg = MIMEMultipart()
        msg["From"]    = sender
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=15) as server:
                server.login(sender, password)
                server.sendmail(sender, to_email, msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender, password)
                server.sendmail(sender, to_email, msg.as_string())
        return True, "Email sent successfully."
    except smtplib.SMTPAuthenticationError as e:
        return False, f"Authentication failed. {str(e)}"
    except Exception as e:
        return False, f"Failed to send. {str(e)}"


def _fetch_emails(cfg, count=5):
    try:
        mail = imaplib.IMAP4_SSL(cfg.get("imap", "imap.gmail.com"))
        mail.login(cfg["email"], cfg["password"])
        mail.select("inbox")
        _, data = mail.search(None, "ALL")
        ids = data[0].split()
        ids = ids[-count:] if len(ids) >= count else ids
        result = []
        for eid in reversed(ids):
            _, msg_data = mail.fetch(eid, "(RFC822)")
            msg  = email_lib.message_from_bytes(msg_data[0][1])
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")
            result.append({
                "id":      eid.decode(),
                "sender":  msg.get("From", "Unknown"),
                "subject": msg.get("Subject", "No Subject"),
                "body":    body[:600]
            })
        mail.logout()
        return True, result
    except Exception as e:
        return False, str(e)


def _generate_body(recipient, subject, description, sender_name):
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content":
                f"Write a professional email body.\nRecipient: {recipient}\nSender: {sender_name}\n"
                f"Subject: {subject}\nContent to convey: {description}\n\nReturn ONLY the email body text."}],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except:
        return description


def _generate_reply(original_body, description, sender_name):
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content":
                f"Write a professional reply.\nOriginal: \"{original_body[:400]}\"\n"
                f"Reply should say: {description}\nSender: {sender_name}\n\nReturn ONLY the reply body."}],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except:
        return description


def _add_to_schedule(recipients, subject, body, send_at_str, cfg_snapshot):
    queue = _load_schedule()
    entry = {
        "id":         int(time.time() * 1000),
        "recipients": recipients,
        "subject":    subject,
        "body":       body,
        "send_at":    send_at_str,
        "cfg":        cfg_snapshot
    }
    queue.append(entry)
    _save_schedule(queue)
    return entry["id"]


def _run_scheduler(speak=None):
    while True:
        try:
            queue = _load_schedule()
            if not queue:
                time.sleep(30)
                continue
            now       = datetime.now()
            remaining = []
            sent_any  = False
            for entry in queue:
                try:
                    send_at = datetime.strptime(entry["send_at"], "%Y-%m-%d %H:%M")
                except:
                    remaining.append(entry)
                    continue
                if now >= send_at:
                    cfg = entry.get("cfg", _load_config())
                    all_ok = True
                    for r in entry["recipients"]:
                        ok, msg = _send_smtp(cfg, r["email"], entry["subject"], entry["body"])
                        print(f"[Scheduler] Sent to {r['email']}: {msg}")
                        if not ok:
                            all_ok = False
                    sent_any = True
                    if speak and all_ok:
                        names = ", ".join(r["name"] for r in entry["recipients"])
                        speak(f"Scheduled email sent to {names}.")
                else:
                    remaining.append(entry)
            if sent_any:
                _save_schedule(remaining)
        except Exception as e:
            print(f"[Scheduler] Error: {e}")
        time.sleep(30)


_scheduler_started = False

def _ensure_scheduler(speak=None):
    global _scheduler_started
    if not _scheduler_started:
        t = threading.Thread(target=_run_scheduler, args=(speak,), daemon=True)
        t.start()
        _scheduler_started = True


def start(cmd, speak, listen):
    cfg = _load_config()
    _ensure_scheduler(speak)

    if not cfg.get("email") or not cfg.get("password"):
        speak("Your email is not configured. Please open the automation panel from the ribbon and enter your email and app password.")
        return

    contacts     = _load_contacts()
    contacts_json = json.dumps(contacts) if contacts else "{}"
    sender_name  = cfg["email"].split("@")[0]
    now_str      = datetime.now().strftime("%Y-%m-%d %H:%M")

    scheduled    = _load_schedule()
    sched_summary = f"{len(scheduled)} email(s) pending in schedule." if scheduled else "No scheduled emails."

    history = [{
        "role": "user",
        "content": (
            f'User command: "{cmd}"\n'
            f"Current datetime: {now_str}\n"
            f"Sender email: {cfg['email']}\n"
            f"Known contacts: {contacts_json}\n"
            f"Schedule status: {sched_summary}\n"
            "Analyze the command and begin. Extract all info present. Ask only for what is truly missing."
        )
    }]

    fetched_emails = []
    pending        = {}
    max_turns      = 30

    for _ in range(max_turns):
        ai     = _ai_chat(history)
        history.append({"role": "assistant", "content": json.dumps(ai)})

        action = ai.get("action", "ask")
        data   = ai.get("data", {})
        msg    = ai.get("speak", "")

        if action == "done":
            if msg: speak(msg)
            break

        elif action == "error":
            speak(msg or "Something went wrong.")
            break

        elif action == "save_contact":
            name          = data.get("name", "").strip()
            contact_email = data.get("email", "").strip()
            if name and contact_email:
                contacts[name] = contact_email
                _save_contacts(contacts)
                print(f"[Automation] Saved contact: {name} → {contact_email}")
            if msg: speak(msg)
            history.append({"role": "user", "content": f"Contact saved: {name} → {contact_email}. Continue."})

        elif action == "list_scheduled":
            queue = _load_schedule()
            if not queue:
                speak("You have no scheduled emails right now.")
            else:
                lines = [f"{i+1}: to {', '.join(r['name'] for r in e['recipients'])} at {e['send_at']}, subject {e['subject']}" for i, e in enumerate(queue)]
                speak("Scheduled emails: " + ". ".join(lines))
            history.append({"role": "user", "content": f"Schedule listed. {len(queue)} pending. Continue or done."})

        elif action == "read":
            count = int(data.get("count", 3))
            speak(f"Fetching your last {count} emails.")
            ok, result = _fetch_emails(cfg, count)
            if not ok:
                speak(f"Could not fetch emails. {result}")
                break
            fetched_emails = result
            lines = [f"Email {i+1}: from {e['sender']}, subject {e['subject']}" for i, e in enumerate(result)]
            speak(". ".join(lines) + ". Say a number to read the body, or say done.")
            user_in = listen()
            history.append({"role": "user", "content": (
                f"User said: {user_in}\n"
                f"Emails: {json.dumps([{'index': i+1, 'sender': e['sender'], 'subject': e['subject'], 'preview': e['body'][:150]} for i, e in enumerate(result)])}"
            )})

        elif action == "reply":
            if not fetched_emails:
                speak("Fetching inbox first.")
                ok, result = _fetch_emails(cfg, 5)
                if not ok:
                    speak(f"Could not fetch emails. {result}")
                    break
                fetched_emails = result
            idx      = min(max(0, int(data.get("email_index", 1)) - 1), len(fetched_emails) - 1)
            selected = fetched_emails[idx]
            body     = _generate_reply(selected["body"], data.get("description", ""), sender_name)
            match    = re.search(r"<(.+?)>", selected["sender"])
            to_email = match.group(1) if match else selected["sender"].strip()
            subject  = "Re: " + selected["subject"]
            speak(f"Reply to {selected['sender']}. Subject: {subject}. Body preview: {body[:200]}. Say yes to send or no to cancel.")
            user_in = listen()
            if "yes" in user_in.lower():
                ok, message = _send_smtp(cfg, to_email, subject, body)
                speak(message)
            else:
                speak("Reply cancelled.")
            break

        elif action == "generate_body":
            email_specs = data.get("emails", [])
            if not email_specs:
                history.append({"role": "user", "content": "generate_body called but emails list is empty. Ask for recipient details."})
                continue

            speak("Generating emails." if len(email_specs) > 1 else "Generating email.")
            generated = []
            for spec in email_specs:
                body = _generate_body(spec.get("name", "recipient"), spec.get("subject", ""), spec.get("description", ""), sender_name)
                generated.append({
                    "name":    spec.get("name", ""),
                    "email":   spec.get("email", ""),
                    "subject": spec.get("subject", ""),
                    "body":    body
                })

            pending = {"generated": generated}

            preview_lines = [f"Email {i+1} to {g['name']}: subject '{g['subject']}', body preview: {g['body'][:120]}" for i, g in enumerate(generated)]
            speak(". ".join(preview_lines) + ". Say yes to send, schedule it, no to cancel, or tell me what to change.")
            user_in = listen()
            history.append({"role": "user", "content": (
                f"User said: {user_in}\n"
                f"Generated emails: {json.dumps([{'name':g['name'],'email':g['email'],'subject':g['subject'],'body_preview':g['body'][:150]} for g in generated])}\n"
                f"Current datetime: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                "If yes and single → send action. If yes and multiple → send_multiple action with full emails list including bodies. If schedule → schedule action. If changes → ask what to change. If no → done."
            )})

        elif action == "send":
            gen      = pending.get("generated", [])
            to_email = data.get("to") or (gen[0]["email"] if gen else "")
            subject  = data.get("subject") or (gen[0]["subject"] if gen else "")
            body     = data.get("body") or (gen[0]["body"] if gen else "")
            rec_name = data.get("recipient_name") or (gen[0]["name"] if gen else "")
            do_save  = data.get("save_contact", False)

            if not to_email or not subject or not body:
                history.append({"role": "user", "content": "send action missing to/subject/body. Ask for what's missing."})
                continue

            speak("Sending.")
            ok, message = _send_smtp(cfg, to_email, subject, body)
            speak(message)

            if ok and do_save and rec_name and to_email:
                contacts[rec_name] = to_email
                _save_contacts(contacts)
                print(f"[Automation] Auto-saved contact: {rec_name} → {to_email}")
            break

        elif action == "send_multiple":
            emails  = data.get("emails") or pending.get("generated", [])
            do_save = data.get("save_contacts", False)

            if not emails:
                history.append({"role": "user", "content": "send_multiple missing emails list. Ask for what's missing."})
                continue

            results = []
            for e in emails:
                ok, message = _send_smtp(cfg, e["email"], e["subject"], e["body"])
                results.append(f"{e['name']}: {'sent' if ok else 'failed - ' + message}")
                if ok and do_save:
                    contacts[e["name"]] = e["email"]

            if do_save:
                _save_contacts(contacts)

            speak("Done. Results: " + ", ".join(results))
            break

        elif action == "schedule":
            emails   = data.get("emails") or pending.get("generated", [])
            send_at  = data.get("send_at", "")
            do_save  = data.get("save_contacts", False)

            if not emails or not send_at:
                history.append({"role": "user", "content": "schedule action missing emails or send_at. Ask for what's missing."})
                continue

            cfg_snapshot = {k: cfg.get(k) for k in ["email", "password", "smtp", "port", "imap"]}
            for e in emails:
                _add_to_schedule(
                    [{"name": e["name"], "email": e["email"]}],
                    e["subject"], e["body"], send_at, cfg_snapshot
                )
                if do_save:
                    contacts[e["name"]] = e["email"]

            if do_save:
                _save_contacts(contacts)

            names = ", ".join(e["name"] for e in emails)
            speak(f"Scheduled. {'Emails' if len(emails) > 1 else 'Email'} to {names} will be sent at {send_at}.")
            history.append({"role": "user", "content": f"Emails scheduled for {send_at}. Continue or done."})

        else:
            if msg: speak(msg)
            user_in = listen()
            history.append({"role": "user", "content": f"User replied: {user_in}"})