#!/usr/bin/env python
"""Create a Google Doc from a local text/Markdown file using OAuth.

This intentionally uses only Python's standard library so the cluster does not
need google-api-python-client installed.
"""
import argparse
import json
import os
import re
import secrets
import time
import urllib.parse
import urllib.request


DEFAULT_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "~/client_secret_google_oauth.json")
CONFIG_DIR = os.path.expanduser("~/.config/eca-feature-sel")
TOKEN_PATH = os.path.join(CONFIG_DIR, "google_docs_token.json")
STATE_PATH = os.path.join(CONFIG_DIR, "google_docs_oauth_state.json")
SCOPES = ["https://www.googleapis.com/auth/documents"]
REDIRECT_URI = "http://localhost"


def load_client(path):
    with open(os.path.expanduser(path)) as fh:
        data = json.load(fh)
    key = "installed" if "installed" in data else "web"
    info = data[key]
    return {
        "client_id": info["client_id"],
        "client_secret": info["client_secret"],
        "auth_uri": info["auth_uri"],
        "token_uri": info["token_uri"],
    }


def post_json(url, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def get_json(url, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def post_form(url, payload):
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def auth_url(args):
    client = load_client(args.client_secret)
    state = secrets.token_urlsafe(24)
    save_json(STATE_PATH, {"state": state, "created": time.time()})
    query = {
        "client_id": client["client_id"],
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    print(client["auth_uri"] + "?" + urllib.parse.urlencode(query))
    print()
    print("After approving, copy the full redirected localhost URL and run:")
    print("  python scripts/create_google_doc.py create --callback-url 'PASTE_URL_HERE'")


def token_from_callback(args):
    client = load_client(args.client_secret)
    parsed = urllib.parse.urlparse(args.callback_url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "error" in qs:
        raise SystemExit("OAuth returned error: " + qs["error"][0])
    code = qs.get("code", [None])[0]
    state = qs.get("state", [None])[0]
    if not code:
        raise SystemExit("No code= parameter found in callback URL.")
    expected = json.load(open(STATE_PATH)).get("state") if os.path.exists(STATE_PATH) else None
    if expected and state != expected:
        raise SystemExit("OAuth state mismatch; run auth-url again.")
    tok = post_form(client["token_uri"], {
        "code": code,
        "client_id": client["client_id"],
        "client_secret": client["client_secret"],
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    })
    tok["expires_at"] = time.time() + int(tok.get("expires_in", 3600)) - 60
    save_json(TOKEN_PATH, tok)
    return tok


def load_token(args):
    if args.callback_url:
        return token_from_callback(args)
    if not os.path.exists(TOKEN_PATH):
        raise SystemExit("No token found. Run `python scripts/create_google_doc.py auth-url` first.")
    tok = json.load(open(TOKEN_PATH))
    if tok.get("expires_at", 0) > time.time():
        return tok
    refresh = tok.get("refresh_token")
    if not refresh:
        raise SystemExit("Token expired and has no refresh token. Run auth-url again.")
    client = load_client(args.client_secret)
    new = post_form(client["token_uri"], {
        "client_id": client["client_id"],
        "client_secret": client["client_secret"],
        "refresh_token": refresh,
        "grant_type": "refresh_token",
    })
    tok.update(new)
    tok["expires_at"] = time.time() + int(tok.get("expires_in", 3600)) - 60
    save_json(TOKEN_PATH, tok)
    return tok


def infer_title(path, fallback):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
    return fallback


def clean_inline(text):
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("**", "")
    return text


def markdown_blocks(path):
    """Parse enough Markdown to produce a styled Google Doc."""
    blocks = []
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("# "):
                blocks.append({"type": "title", "text": clean_inline(stripped[2:].strip())})
            elif stripped.startswith("## "):
                blocks.append({"type": "heading1", "text": clean_inline(stripped[3:].strip())})
            elif stripped.startswith("### "):
                blocks.append({"type": "heading2", "text": clean_inline(stripped[4:].strip())})
            elif stripped.startswith("- "):
                blocks.append({"type": "bullet", "text": clean_inline(stripped[2:].strip())})
            elif re.match(r"^\d+\.\s+", stripped):
                blocks.append({"type": "number", "text": clean_inline(re.sub(r"^\d+\.\s+", "", stripped))})
            else:
                blocks.append({"type": "body", "text": clean_inline(stripped)})
    return blocks


def render_blocks(blocks):
    text_parts = []
    pos = 1
    rendered = []
    for block in blocks:
        text = block["text"]
        start = pos
        text_parts.append(text + "\n")
        pos += len(text) + 1
        rendered.append({"type": block["type"], "start": start, "end": pos})
    return "".join(text_parts), rendered


def paragraph_style(block_type):
    if block_type == "title":
        return {
            "namedStyleType": "TITLE",
            "spaceAbove": {"magnitude": 0, "unit": "PT"},
            "spaceBelow": {"magnitude": 14, "unit": "PT"},
            "lineSpacing": 100,
        }
    if block_type == "heading1":
        return {
            "namedStyleType": "HEADING_1",
            "spaceAbove": {"magnitude": 18, "unit": "PT"},
            "spaceBelow": {"magnitude": 8, "unit": "PT"},
            "lineSpacing": 100,
        }
    if block_type == "heading2":
        return {
            "namedStyleType": "HEADING_2",
            "spaceAbove": {"magnitude": 12, "unit": "PT"},
            "spaceBelow": {"magnitude": 6, "unit": "PT"},
            "lineSpacing": 100,
        }
    return {
        "namedStyleType": "NORMAL_TEXT",
        "spaceAbove": {"magnitude": 0, "unit": "PT"},
        "spaceBelow": {"magnitude": 6, "unit": "PT"},
        "lineSpacing": 115,
    }


def style_requests(rendered):
    requests = []
    list_ranges = []
    for block in rendered:
        requests.append({
            "updateParagraphStyle": {
                "range": {"startIndex": block["start"], "endIndex": block["end"]},
                "paragraphStyle": paragraph_style(block["type"]),
                "fields": "namedStyleType,spaceAbove,spaceBelow,lineSpacing",
            }
        })
        if block["type"] in {"title", "heading1", "heading2"}:
            requests.append({
                "updateTextStyle": {
                    "range": {"startIndex": block["start"], "endIndex": block["end"] - 1},
                    "textStyle": {
                        "weightedFontFamily": {"fontFamily": "Helvetica"},
                        "fontSize": {"magnitude": 12, "unit": "PT"},
                        "bold": True,
                        "foregroundColor": {
                            "color": {"rgbColor": {"red": 0.12, "green": 0.12, "blue": 0.12}}
                        },
                    },
                    "fields": "weightedFontFamily,fontSize,bold,foregroundColor",
                }
            })
        else:
            requests.append({
                "updateTextStyle": {
                    "range": {"startIndex": block["start"], "endIndex": block["end"] - 1},
                    "textStyle": {
                        "weightedFontFamily": {"fontFamily": "Helvetica"},
                        "fontSize": {"magnitude": 11, "unit": "PT"},
                        "bold": False,
                    },
                    "fields": "weightedFontFamily,fontSize,bold",
                }
            })
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": block["start"], "endIndex": block["end"]},
                    "paragraphStyle": {"alignment": "JUSTIFIED"},
                    "fields": "alignment",
                }
            })
        if block["type"] in {"bullet", "number"}:
            list_ranges.append(block)

    for block in list_ranges:
        preset = "BULLET_DISC_CIRCLE_SQUARE" if block["type"] == "bullet" else "NUMBERED_DECIMAL_ALPHA_ROMAN"
        requests.append({
            "createParagraphBullets": {
                "range": {"startIndex": block["start"], "endIndex": block["end"]},
                "bulletPreset": preset,
            }
        })
    return requests


def replace_document(doc_id, source, title, token):
    blocks = markdown_blocks(source)
    text, rendered = render_blocks(blocks)
    doc = get_json(f"https://docs.googleapis.com/v1/documents/{doc_id}", token)
    body = doc.get("body", {}).get("content", [])
    end = body[-1]["endIndex"] if body else 1
    requests = []
    if end > 2:
        requests.append({"deleteContentRange": {"range": {"startIndex": 1, "endIndex": end - 1}}})
    requests.append({"insertText": {"location": {"index": 1}, "text": text}})
    requests.extend(style_requests(rendered))
    if title:
        requests.append({"updateDocumentStyle": {"documentStyle": {"marginTop": {"magnitude": 54, "unit": "PT"}, "marginBottom": {"magnitude": 54, "unit": "PT"}, "marginLeft": {"magnitude": 54, "unit": "PT"}, "marginRight": {"magnitude": 54, "unit": "PT"}}, "fields": "marginTop,marginBottom,marginLeft,marginRight"}})
    post_json(f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate", {"requests": requests}, token)


def create(args):
    tok = load_token(args)
    source = os.path.expanduser(args.source)
    title = args.title or infer_title(source, "Gene Vocabulary Methodology Summary")
    doc = post_json("https://docs.googleapis.com/v1/documents", {"title": title}, tok["access_token"])
    doc_id = doc["documentId"]
    replace_document(doc_id, source, title, tok["access_token"])
    print(f"https://docs.google.com/document/d/{doc_id}/edit")


def replace(args):
    tok = load_token(args)
    source = os.path.expanduser(args.source)
    title = args.title or infer_title(source, "Gene Vocabulary Methodology Summary")
    replace_document(args.document_id, source, title, tok["access_token"])
    print(f"https://docs.google.com/document/d/{args.document_id}/edit")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--client-secret", default=DEFAULT_CLIENT_SECRET)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auth-url")
    c = sub.add_parser("create")
    c.add_argument("--callback-url")
    c.add_argument("--source", default="docs/gene_vocabulary_methodology_summary.md")
    c.add_argument("--title")
    r = sub.add_parser("replace")
    r.add_argument("document_id")
    r.add_argument("--callback-url")
    r.add_argument("--source", default="docs/gene_vocabulary_methodology_summary.md")
    r.add_argument("--title")
    args = ap.parse_args()
    {"auth-url": auth_url, "create": create, "replace": replace}[args.cmd](args)


if __name__ == "__main__":
    main()
