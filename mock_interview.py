import re
import random
import io
import os
import json
from datetime import datetime
from urllib.parse import quote_plus
import base64

import cv2
import numpy as np
import requests
from flask import Blueprint, request, jsonify, render_template_string

mock_interview_bp = Blueprint('mock_interview', __name__, url_prefix='/api/interview')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')

# ─────────────────────────────────────────────────────────────────────
# OpenCV Anti-Cheat Initialization
# ─────────────────────────────────────────────────────────────────────
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  # type: ignore
except Exception as e:
    print(f"[OpenCV Error] Failed to load cascades: {e}")

# ─────────────────────────────────────────────────────────────────────
# Proctoring Session Store  (in-memory, keyed by session_id)
# ─────────────────────────────────────────────────────────────────────
PROCTOR_SESSIONS: dict = {}

MAX_LIVES = 3
# Streak thresholds before a violation fires
NO_FACE_STREAK_LIMIT   = 4   # ~12 s at 3-s intervals
EYE_AWAY_STREAK_LIMIT  = 5   # ~15 s
ROTATED_STREAK_LIMIT   = 5
SHAKE_STREAK_LIMIT     = 8


def _new_session(candidate_name: str = "Candidate") -> dict:
    return {
        "candidate_name": candidate_name,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "lives": MAX_LIVES,
        "violations": 0,
        "violation_log": [],   # list of {time, type, detail}
        "status": "active",    # active | rejected | completed
        "frames_analyzed": 0,
        # streak counters
        "no_face_streak":  0,
        "eye_away_streak": 0,
        "rotated_streak":  0,
        "shake_streak":    0,
        # previous frame gray for movement detection
        "prev_frame_b64": None,
    }


def _add_violation(sess: dict, vtype: str, detail: str) -> None:
    sess["violations"] += 1
    sess["lives"] -= 1
    sess["violation_log"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": vtype,
        "detail": detail,
    })
    if sess["lives"] <= 0:
        sess["status"] = "rejected"
        sess["end_time"] = datetime.now().isoformat()

def get_resume_stream_from_req(req):
    # 1. Direct file upload takes priority
    if 'resume' in req.files and req.files['resume'].filename != '':
        return req.files['resume'].stream

    # 2. Try identifying the session_id
    session_id = req.form.get("session_id") if req.form else None
    if not session_id and req.is_json:
        session_id = req.json.get("session_id")

    # 3. Check if the session_id from request has a file
    if session_id:
        filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}.pdf")
        if os.path.exists(filepath):
            return open(filepath, 'rb')

    # 4. If no file yet, try identifying via Auth Token
    auth_header = req.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header.replace('Bearer ', '').strip()
        if token and token != 'null' and token != 'undefined':
            import database as db
            user = db.get_user_by_token(token)
            if user and user.get('session_id'):
                sid = user.get('session_id')
                filepath = os.path.join(UPLOAD_FOLDER, f"{sid}.pdf")
                if os.path.exists(filepath):
                    return open(filepath, 'rb')

    return None

# ─────────────────────────────────────────────────────────────────────
# Comprehensive Tech Skills Dictionary (canonical name → search terms)
# ─────────────────────────────────────────────────────────────────────
SKILLS_DB = {
    # LANGUAGES
    "Python": ["python", "py", "pandas", "numpy", "django", "flask", "fastapi", "pytorch", "tensorflow", "keras"],
    "JavaScript": ["javascript", "js", "ecmascript", "node.js", "nodejs", "react", "vue", "angular", "next.js", "typescript"],
    "TypeScript": ["typescript", "ts"],
    "Java": ["java", "spring", "springboot", "hibernate", "maven", "gradle", "android"],
    "C++": ["c++", "cpp", "stl", "opengl", "embedded"],
    "C#": ["c#", "dotnet", ".net", "asp.net", "entity framework", "unity"],
    "PHP": ["php", "laravel", "symfony", "codeigniter", "wordpress"],
    "Go": ["go", "golang", "gin", "gorm"],
    "Rust": ["rust", "cargo"],
    "Ruby": ["ruby", "rails", "sinatra"],
    "Swift": ["swift", "ios", "xcode", "uikit", "swiftui"],
    "Kotlin": ["kotlin", "android", "coroutines"],
    "SQL": ["sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "mariadb", "t-sql", "pl/sql"],
    "NoSQL": ["nosql", "mongodb", "redis", "cassandra", "firebase", "dynamodb", "elasticsearch", "neo4j"],
    
    # FRONTEND
    "React": ["react", "react.js", "reactjs", "redux", "recoil", "context api", "hooks", "next.js", "remix"],
    "Angular": ["angular", "rxjs", "ngrx"],
    "Vue": ["vue", "vue.js", "vuejs", "vuex", "pinia", "nuxt.js"],
    "HTML/CSS": ["html", "html5", "css", "css3", "sass", "scss", "less", "tailwind", "bootstrap", "flexbox", "grid"],
    "UI/UX": ["figma", "adobe xd", "sketch", "user interface", "user experience", "wireframing", "prototyping"],
    
    # BACKEND
    "Node.js": ["node.js", "nodejs", "express", "express.js", "hapi", "koa", "nestjs"],
    "Django": ["django", "drf", "django rest framework"],
    "Flask": ["flask"],
    "Spring Boot": ["spring boot", "springboot", "microservices"],
    
    # CLOUD / DEVOPS
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda", "rds", "iam", "route53", "amplify"],
    "Azure": ["azure", "microsoft azure"],
    "Google Cloud": ["gcp", "google cloud", "firebase"],
    "Docker": ["docker", "docker-compose", "containerization"],
    "Kubernetes": ["k8s", "kubernetes", "helm"],
    "CI/CD": ["ci/cd", "jenkins", "github actions", "gitlab ci", "circleci", "travis ci"],
    "Terraform": ["terraform", "iac", "ansible", "cloudformation"],
    
    # AI / MACHINE LEARNING / DATA
    "Machine Learning": ["machine learning", "ml", "scikit-learn", "sklearn", "random forest", "svm", "xgboost"],
    "Deep Learning": ["deep learning", "nlp", "computer vision", "cv", "cnn", "rnn", "transformers", "bert", "gpt", "llm"],
    "Data Science": ["data science", "data analysis", "visualization", "matplotlib", "seaborn", "tableau", "power bi"],
    "Big Data": ["hadoop", "spark", "pyspark", "kafka", "airflow", "snowflake", "databricks"],
    
    # TOOLS & MISC
    "Git": ["git", "github", "gitlab", "bitbucket", "version control"],
    "Agile": ["agile", "scrum", "kanban", "jira", "confluence"],
    "Testing": ["jest", "mocha", "cypress", "selenium", "unit testing", "integration testing", "junit", "pytest"],
    "REST API": ["rest", "restful", "api", "graphql", "postman", "swagger"],
}

ROLE_MAP = {
    "Backend Developer": ["python", "node.js", "java", "django", "flask", "spring boot", "sql", "nosql", "rest api", "docker", "microservices", "aws", "postgresql", "mysql", "redis", "express"],
    "Frontend Developer": ["javascript", "react", "angular", "vue", "html/css", "ui/ux", "typescript", "tailwind", "sass", "next.js"],
    "Data Scientist": ["python", "machine learning", "deep learning", "data science", "sql", "pandas", "numpy", "pytorch", "tensorflow", "statistics"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "ci/cd", "terraform", "linux", "ansible", "jenkins", "cloud"],
    "Mobile Developer": ["swift", "kotlin", "react native", "flutter", "android", "ios", "swiftui"],
    "Full Stack Developer": ["javascript", "typescript", "react", "node.js", "sql", "html/css", "aws", "express", "django"],
    "Software Engineer": ["python", "javascript", "java", "c++", "git", "sql", "rest api"],
}


def extract_text_from_pdf(file_stream):
    """Extract text from PDF using pdfminer.six, with PyPDF2 + (optional) OCR fallback."""
    text = ""

    # ── Attempt 1: pdfminer.six (best for text-based PDFs) ──
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        file_stream.seek(0)
        text = pdfminer_extract(file_stream)
    except Exception as e:
        print(f"[pdfminer] extraction failed: {e}")

    # ── Attempt 2: PyPDF2 fallback ──
    if not text or len(text.strip()) < 50:
        try:
            import PyPDF2
            file_stream.seek(0)
            reader = PyPDF2.PdfReader(file_stream)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        except Exception as e:
            print(f"[PyPDF2] extraction failed: {e}")

    # ── Attempt 3: OCR fallback (for scanned PDFs) ──
    if not text or len(text.strip()) < 50:
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            file_stream.seek(0)
            images = convert_from_bytes(file_stream.read())
            ocr_text = ""
            for img in images:
                result = pytesseract.image_to_string(img)
                if isinstance(result, bytes):
                    result = result.decode('utf-8', errors='ignore')
                ocr_text += str(result) + "\n"
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        except ImportError:
            print("[OCR] pytesseract/pdf2image not installed — skipping OCR fallback")
        except Exception as e:
            print(f"[OCR] extraction failed: {e}")

    return text.strip()


def extract_skills(text):
    """Scan text for keywords in SKILLS_DB and return a list of found skills."""
    if not text: return []
    text_lower = text.lower()
    found = set()
    import re
    for canonical_name, keywords in SKILLS_DB.items():
        for kw in keywords:
            # Use regex for better matching (word boundaries)
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found.add(canonical_name)
                break
    return sorted(list(found))


def infer_role(skills):
    """Infer the best matching role from extracted skills using weighted matching."""
    skills_lower = set(s.lower() for s in skills)
    scores = {}
    for role, role_keywords in ROLE_MAP.items():
        count = sum(1 for kw in role_keywords if kw in skills_lower)
        if count > 0:
            scores[role] = count
    if not scores:
        return "Software Engineer"
    return max(scores, key=lambda k: scores[k])


SECTION_ALIASES = {
    "summary": {
        "summary", "professional summary", "profile", "objective", "about me", "career summary"
    },
    "skills": {
        "skills", "technical skills", "core skills", "core competencies", "technologies", "tech stack"
    },
    "experience": {
        "experience", "work experience", "professional experience", "employment history", "internships", "internship experience"
    },
    "projects": {
        "projects", "academic projects", "personal projects", "project experience"
    },
    "education": {
        "education", "academic background", "academics", "qualifications"
    },
    "certifications": {
        "certifications", "licenses", "certificates", "accreditations"
    },
    "achievements": {
        "achievements", "awards", "accomplishments", "highlights"
    }
}

TECH_ROLE_HINTS = {
    "developer", "engineer", "software", "data scientist", "data analyst", "ml", "ai",
    "devops", "cloud", "backend", "frontend", "full stack", "mobile"
}

INTERVIEW_RESEARCH_CACHE = {}

THEME_SIGNAL_MAP = {
    "system design": ["scalability", "distributed", "architecture", "microservices", "availability", "throughput"],
    "debugging": ["debug", "incident", "root cause", "bug", "failure", "logs", "troubleshoot"],
    "performance": ["latency", "optimize", "performance", "bottleneck", "throughput", "profiling"],
    "testing quality": ["test", "qa", "regression", "unit test", "integration", "quality"],
    "security": ["security", "auth", "authorization", "vulnerability", "owasp", "encryption"],
    "collaboration": ["stakeholder", "communication", "team", "cross-functional", "conflict", "leadership"],
    "ownership": ["ownership", "impact", "delivery", "deadline", "prioritization", "trade-off"],
    "data modeling": ["schema", "database", "sql", "consistency", "indexing", "query"]
}


def _normalized_resume_lines(text):
    return [ln.strip() for ln in text.replace("\t", " ").splitlines() if ln.strip()]


def _normalize_heading(line):
    return re.sub(r"[^a-z ]+", "", line.lower()).strip()


def _split_resume_sections(lines):
    sections = {name: [] for name in SECTION_ALIASES}
    current = None
    for line in lines:
        heading = _normalize_heading(line)
        matched = None
        for section, aliases in SECTION_ALIASES.items():
            if heading in aliases:
                matched = section
                break
        if matched:
            current = matched
            continue
        if current:
            sections[current].append(line)
    return sections


def _unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _extract_links(text):
    raw_links = re.findall(r"(https?://[^\s|,;]+|www\.[^\s|,;]+|linkedin\.com/[^\s|,;]+|github\.com/[^\s|,;]+)", text, re.IGNORECASE)
    links = []
    for link in raw_links:
        cleaned = link.rstrip(").,;]")
        if cleaned.lower().startswith("www."):
            cleaned = "https://" + cleaned
        elif not cleaned.lower().startswith("http"):
            cleaned = "https://" + cleaned
        links.append(cleaned)
    links = _unique_preserve_order(links)

    linkedin = next((ln for ln in links if "linkedin.com" in ln.lower()), "")
    github = next((ln for ln in links if "github.com" in ln.lower()), "")
    portfolio = next((ln for ln in links if ln not in {linkedin, github}), "")
    return linkedin, github, portfolio


def _extract_name(lines, email):
    noise_words = {
        "resume", "curriculum", "vitae", "profile", "contact", "phone", "email", "linkedin",
        "github", "skills", "education", "experience", "project"
    }
    for line in lines[:18]:
        if len(line) < 5 or len(line) > 60:
            continue
        if any(ch.isdigit() for ch in line) or "@" in line or "http" in line.lower():
            continue
        lowered = line.lower()
        if any(word in lowered for word in noise_words):
            continue
        words = [w for w in re.split(r"\s+", line) if w]
        if 2 <= len(words) <= 5 and all(re.match(r"^[A-Za-z][A-Za-z'.-]*$", w) for w in words):
            return line.strip()

    if email:
        local = email.split("@")[0]
        parts = [p for p in re.split(r"[._-]+", local) if p.isalpha()]
        if len(parts) >= 2:
            return " ".join(p.capitalize() for p in parts[:3])
    return ""


def _extract_location(lines):
    for line in lines[:20]:
        if "@" in line or "http" in line.lower():
            continue
        if re.search(r"\b[A-Za-z ]+,\s*[A-Za-z ]+\b", line):
            return line.strip()
    return ""


def _extract_summary(lines, sections):
    if sections["summary"]:
        return " ".join(sections["summary"][:3])[:400]
    for line in lines:
        if len(line.split()) >= 14:
            return line[:400]
    return ""


def _extract_education_entries(lines, sections):
    edu_lines = sections["education"] if sections["education"] else lines
    entries = []
    for line in edu_lines:
        if re.search(r"\b(b\.?tech|b\.?e\.?|m\.?tech|mba|bca|mca|bachelor|master|ph\.?d|b\.?sc|m\.?sc)\b", line, re.IGNORECASE):
            entries.append(line)
    return _unique_preserve_order(entries)[:5]


def _extract_highlights(section_lines, max_items=4):
    cleaned = []
    for line in section_lines:
        line = re.sub(r"^[\-\*\u2022]+\s*", "", line).strip()
        if len(line) < 8:
            continue
        cleaned.append(line)
    return _unique_preserve_order(cleaned)[:max_items]


def _estimate_years_experience(text):
    explicit = re.search(r"(\d{1,2})\+?\s+years?\s+of\s+experience", text, re.IGNORECASE)
    if explicit:
        return int(explicit.group(1))

    current_year = datetime.now().year
    ranges = re.findall(
        r"(19\d{2}|20\d{2})\s*[-–to]+\s*(present|current|19\d{2}|20\d{2})",
        text,
        re.IGNORECASE
    )
    spans = []
    for start, end in ranges:
        s = int(start)
        e = current_year if end.lower() in {"present", "current"} else int(end)
        if s <= e <= current_year + 1:
            spans.append((s, e))
    if not spans:
        return 0

    spans.sort()
    merged = []
    for start, end in spans:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    total = sum(end - start for start, end in merged)
    return max(0, min(total, 25))


def extract_resume_profile(text):
    lines = _normalized_resume_lines(text)
    sections = _split_resume_sections(lines)

    email_match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(?:\+?\d[\d\s()-]{8,}\d)", text)
    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0).strip() if phone_match else ""

    linkedin, github, portfolio = _extract_links(text)
    skills = extract_skills(text)
    inferred_role = infer_role(skills) if skills else "Software Engineer"

    project_highlights = _extract_highlights(sections["projects"], max_items=5)
    experience_highlights = _extract_highlights(sections["experience"], max_items=5)
    certifications = _extract_highlights(sections["certifications"], max_items=5)
    achievements = _extract_highlights(sections["achievements"], max_items=5)
    education_entries = _extract_education_entries(lines, sections)

    profile = {
        "fullName": _extract_name(lines, email),
        "email": email,
        "phone": phone,
        "location": _extract_location(lines),
        "linkedin": linkedin,
        "github": github,
        "portfolio": portfolio,
        "summary": _extract_summary(lines, sections),
        "skills": skills,
        "role": inferred_role,
        "yearsExperience": _estimate_years_experience(text),
        "education": education_entries,
        "experienceHighlights": experience_highlights,
        "projects": project_highlights,
        "certifications": certifications,
        "achievements": achievements,
        "rawTextPreview": text[:1000] + ("..." if len(text) > 1000 else "")
    }
    return profile


def _is_tech_role(role):
    role_lower = (role or "").lower()
    return any(hint in role_lower for hint in TECH_ROLE_HINTS)


def _normalize_difficulty(difficulty):
    value = (difficulty or "medium").strip().lower()
    return value if value in {"easy", "medium", "hard"} else "medium"


def _compact_resume_line(line, max_len=160):
    cleaned = re.sub(r"\s+", " ", (line or "").strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "..."


def _fetch_duckduckgo_snippets(query, max_items=6):
    try:
        # Lightweight public HTML endpoint; no API key required.
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=3.5
        )
        response.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        items = []
        for result in soup.select(".result"):
            title_el = result.select_one(".result__a")
            snippet_el = result.select_one(".result__snippet")
            title = title_el.get_text(" ", strip=True) if title_el else ""
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            merged = " ".join(x for x in [title, snippet] if x).strip()
            if merged:
                items.append(merged)
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


def _research_interview_signals(role, skills):
    cache_key = (role.lower().strip(), tuple(s.lower().strip() for s in skills[:4]))
    import datetime as _dt
    now = datetime.now(_dt.timezone.utc)
    cached = INTERVIEW_RESEARCH_CACHE.get(cache_key)
    if cached:
        cached_time = cached["timestamp"]
        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=_dt.timezone.utc)
        if (now - cached_time).total_seconds() < 3600:
            return cached["data"]

    top_skills = skills[:3] if skills else ["software engineering"]
    queries = [f"real {role} interview questions"]
    queries.extend([f"{skill} interview questions scenario based" for skill in top_skills])

    snippets = []
    for query in queries[:4]:
        snippets.extend(_fetch_duckduckgo_snippets(query, max_items=4))

    text_blob = " ".join(snippets).lower()
    theme_scores = {}
    for theme, keywords in THEME_SIGNAL_MAP.items():
        score = sum(text_blob.count(keyword) for keyword in keywords)
        if score > 0:
            theme_scores[theme] = score

    ranked_themes = [k for k, _ in sorted(theme_scores.items(), key=lambda kv: kv[1], reverse=True)]
    if not ranked_themes:
        ranked_themes = ["system design", "debugging", "performance", "collaboration", "ownership"]

    data = {
        "queries": queries,
        "themes": ranked_themes[:6],
        "snippets": snippets[:12]
    }
    INTERVIEW_RESEARCH_CACHE[cache_key] = {"timestamp": now, "data": data}
    return data


def _dedupe_questions(questions):
    seen = set()
    unique = []
    for q in questions:
        key = re.sub(r"\s+", " ", q["text"].strip().lower())
        if key and key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def _pick_random(rng, pool, limit):
    if limit <= 0:
        return []
    copied = list(pool)
    rng.shuffle(copied)
    return copied[:min(limit, len(copied))]


def generate_real_life_questions(profile, difficulty="medium", count=5):
    difficulty = _normalize_difficulty(difficulty)
    rng = random.SystemRandom()

    role = profile.get("role") or "Software Engineer"
    skills = _unique_preserve_order(profile.get("skills") or [])
    top_skill = skills[0] if skills else "problem solving"
    second_skill = skills[1] if len(skills) > 1 else top_skill
    third_skill = skills[2] if len(skills) > 2 else top_skill
    projects = [_compact_resume_line(x) for x in (profile.get("projects") or []) if x]
    experience = [_compact_resume_line(x) for x in (profile.get("experienceHighlights") or []) if x]
    achievements = [_compact_resume_line(x) for x in (profile.get("achievements") or []) if x]
    summary = _compact_resume_line(profile.get("summary", ""), max_len=200)
    years = profile.get("yearsExperience", 0)
    research = _research_interview_signals(role, skills)
    market_themes = research.get("themes", [])
    dominant_theme = market_themes[0] if market_themes else "system design"

    role_scope = f"{role} role"
    if years:
        role_scope = f"{role} role ({years}+ years experience)"

    anchors = projects + experience + achievements
    anchor = rng.choice(anchors) if anchors else ""

    resume_prompts = [
        f"Walk me through this resume point: \"{anchor}\". What was the business problem, your ownership, key trade-offs, and measurable impact?" if anchor else "",
        f"Which project on your resume best represents your readiness for a {role_scope}? Break down architecture choices, constraints, and outcomes.",
        f"Pick one project where you used {top_skill}. Explain what failed initially, what you changed, and the final metrics.",
        f"Tell me about a project where your solution reduced risk, cost, or latency. What exact decision made the difference?"
    ]
    resume_prompts = [q for q in resume_prompts if q]

    behavioral_prompts = {
        "easy": [
            "Tell me about a time you received difficult feedback. What did you change afterward?",
            "Describe a time you had to collaborate with someone who had a very different working style."
        ],
        "medium": [
            "Tell me about a time you disagreed with a teammate on technical direction. How did you resolve it?",
            "Describe a situation where priorities changed mid-sprint. How did you re-plan and communicate risk?",
            "Share an example of when you made a mistake in production. How did you own it and recover?"
        ],
        "hard": [
            "Describe a high-pressure incident where deadline and quality were in conflict. How did you decide what to ship?",
            "Tell me about a time you had limited data but still had to make a decision with major impact.",
            "Give an example where you had to influence senior stakeholders to change a technical plan."
        ]
    }

    situational_prompts = [
        f"You join a team using {top_skill}. A release is in 24 hours and a P1 bug appears. What is your first-hour plan?",
        f"You inherit a service built on {second_skill} with poor documentation and high error rates. How do you stabilize it in week one?",
        f"Assume your manager asks for a feature in half the estimated time. How do you negotiate scope while protecting quality?",
        f"Interviewers currently emphasize {dominant_theme}. Describe a real situation where you handled this successfully."
    ]
    if summary:
        situational_prompts.append(
            f"Based on your profile summary (\"{summary}\"), what risks would you watch first when onboarding into a new {role_scope}?"
        )

    coding_prompts = [
        f"Coding task: Using {top_skill}, implement an LRU cache with O(1) get/put. Input: operations list. Output: values for get operations.",
        f"Coding task: Using {second_skill}, write a function to merge overlapping intervals. Input: list of [start, end]. Output: merged intervals sorted by start.",
        f"Coding task: Using {top_skill}, implement top-K frequent elements. Input: array of integers and k. Output: k most frequent values.",
        f"Coding task: Using {third_skill}, implement a rate limiter. Input: request timestamps and limit/window. Output: allow or reject per request.",
        f"Coding task: Using {top_skill}, detect a cycle in a directed graph and return one valid cycle path if found."
    ]

    system_design_prompts = [
        f"System design: Design a multi-tenant interview simulator for a {role_scope}. Cover API boundaries, data model, queueing, caching, and failure handling.",
        f"System design: Design a real-time collaboration platform using {top_skill} and {second_skill}. Explain scaling strategy, consistency model, and observability.",
        f"System design: Design an event-driven analytics pipeline for interview sessions. Include ingestion, processing, storage, and cost controls.",
        f"System design: Interview loops now focus on {dominant_theme}. Design a production-grade service showing how you handle this at scale."
    ]

    non_tech_competency_prompts = [
        f"For this {role_scope}, how would you design a KPI dashboard to prove impact in your first 90 days?",
        f"Imagine one critical workflow in your team fails repeatedly. How would you root-cause and fix it permanently?",
        f"How would you use your strength in {top_skill} to improve execution quality across cross-functional teams?"
    ]

    fit_prompts = [
        f"Why this {role_scope}, and which two experiences from your resume best prove role fit?",
        "If you join tomorrow, what would your first 30-60-90 day execution plan look like?",
        f"What kind of team environment helps you do your best work, and where have you already demonstrated that in {top_skill}-heavy projects?"
    ]

    if count <= 0:
        count = 20

    resume_pool = [{"type": "Resume Deep Dive", "is_coding": False, "text": q} for q in resume_prompts]
    behavioral_pool = [{"type": "Behavioral", "is_coding": False, "text": q} for q in behavioral_prompts[difficulty]]
    situational_pool = [{"type": "Situational", "is_coding": False, "text": q} for q in situational_prompts]
    fit_pool = [{"type": "Motivation and Fit", "is_coding": False, "text": q} for q in fit_prompts]

    technical_pool = []
    if _is_tech_role(role):
        technical_pool.extend([{"type": "Coding", "is_coding": True, "text": q} for q in coding_prompts])
        technical_pool.extend([{"type": "System Design", "is_coding": False, "text": q} for q in system_design_prompts])
        # Skill-theme cross questions from internet signals.
        for skill in skills[:4]:
            for theme in market_themes[:3]:
                technical_pool.append({
                    "type": "Technical Scenario",
                    "is_coding": False,
                    "text": f"In interview loops for {role}, a common area is {theme}. Explain how you applied {skill} to solve a real {theme}-related challenge."
                })
    else:
        technical_pool.extend([{"type": "Role Competency", "is_coding": False, "text": q} for q in non_tech_competency_prompts])
        for skill in skills[:4]:
            for theme in market_themes[:2]:
                technical_pool.append({
                    "type": "Role Scenario",
                    "is_coding": False,
                    "text": f"Interviewers often test {theme}. How would you apply {skill} in a realistic business scenario to deliver measurable outcomes?"
                })

    # Add market-aligned prompts derived from internet snippets.
    market_pool = []
    for snippet in research.get("snippets", [])[:6]:
        market_pool.append({
            "type": "Market-Aligned",
            "is_coding": False,
            "text": (
                f"Current interview discussions mention: \"{_compact_resume_line(snippet, max_len=120)}\". "
                f"How would you address this in the context of your {role_scope} experience?"
            )
        })

    resume_target = max(3, count // 5)
    behavioral_target = max(4, count // 5)
    situational_target = max(4, count // 5)
    fit_target = max(2, count // 10)
    technical_target = max(5, count - (resume_target + behavioral_target + situational_target + fit_target))

    selected = []
    selected.extend(_pick_random(rng, resume_pool, resume_target))
    selected.extend(_pick_random(rng, behavioral_pool, behavioral_target))
    selected.extend(_pick_random(rng, situational_pool, situational_target))
    selected.extend(_pick_random(rng, technical_pool, technical_target))
    selected.extend(_pick_random(rng, fit_pool, fit_target))
    selected.extend(_pick_random(rng, market_pool, max(0, count // 6)))

    selected = _dedupe_questions(selected)
    if len(selected) < count:
        top_theme = market_themes[0] if market_themes else "problem solving"
        while len(selected) < count:
            selected.append({
                "type": "Follow-up",
                "is_coding": False,
                "text": (
                    f"For this {role_scope}, what framework would you use to prioritize between delivery speed, "
                    f"{top_theme}, and long-term maintainability?"
                )
            })
            selected = _dedupe_questions(selected)

    rng.shuffle(selected)
    return selected[:count]


@mock_interview_bp.route('/upload-resume', methods=['POST'])
def parse_resume():
    stream = get_resume_stream_from_req(request)
    if not stream:
        return jsonify({"error": "No resume found. Please provide session_id or file."}), 400

    # Extract text using multi-method approach
    text = extract_text_from_pdf(stream)

    if not text or len(text) < 30:
        return jsonify({"error": "Could not extract text from the PDF. Please ensure it is a text-based or clearly scanned resume."}), 400

    profile = extract_resume_profile(text)

    difficulty = (request.form.get("difficulty", "medium") or "medium").lower()
    try:
        count = int(request.form.get("count", 20) or 20)
    except (TypeError, ValueError):
        count = 20

    questions = generate_real_life_questions(profile, difficulty=difficulty, count=count)
    return jsonify({
        "profile": profile,
        "role": profile.get("role", "Software Engineer"),
        "skills": profile.get("skills", []),
        "questions": questions
    })


def mock_evaluate_response(question, response):
    words_list = [w for w in re.split(r'\s+', response) if len(w.strip()) > 0]
    words = len(words_list)

    if words < 5:
        return {
            "score": 0,
            "feedback": "Your answer was blank or far too short to be evaluated. In a real interview, always provide a detailed technical explanation.",
            "accuracy": 0, "clarity": 0, "confidence": 0,
            "toneAnalysis": "No significant audio or text detected."
        }

    hesitations = len(re.findall(r'(?i)\b(um|uh|like|you know|sort of|basically)\b', response))
    confidence_score = max(0, 95 - (hesitations * 15))
    tone_msg = "Tone was confident and steady." if hesitations == 0 else "Significant hesitations detected (e.g., 'um', 'uh'). Practice pausing silently instead."

    target_skill = ""
    target_match = re.search(r'(?i)knowledge of (.*?) in', question) or re.search(r'(?i)working with (.*?),', question) or re.search(r'(?i)strategy around (.*?)\?', question)
    if target_match and target_match.group(1):
        target_skill = re.sub(r'[^a-z0-9]', '', target_match.group(1).lower())

    is_relevant = False
    res_words = response.lower()
    if target_skill and target_skill in res_words:
        is_relevant = True

    tech_keywords = re.findall(r'(?i)\b(api|database|system|component|test|deploy|design|code|framework|architecture|problem|solution|function|class|variable|implement|scale|server|client|front|back|end|data)\b', response)
    keyword_hits = len(tech_keywords)
    if keyword_hits >= 2:
        is_relevant = True

    score = 0
    text_feedback = ""

    # Check if this includes Virtual Compiler Sandbox output
    passed_tests = len(re.findall(r'(?i)\bpass\b', response))
    failed_tests = len(re.findall(r'(?i)\bfail\b', response))
    
    if passed_tests > 0 or failed_tests > 0:
        is_relevant = True
        score = 50 + (passed_tests * 15) - (failed_tests * 10)
        score = min(100, max(10, score))
        if passed_tests > failed_tests:
            text_feedback = f"Great coding! You passed multiple test cases ({passed_tests} PASS). Your logic was sound."
        else:
            text_feedback = f"Your code failed several test cases ({failed_tests} FAIL). Consider edge cases and algorithmic bounds."
    elif not is_relevant:
        score = 25
        text_feedback = "Your response was completely off-topic or irrelevant. You did not directly address the core technical concept of the question. Make sure you stay on topic and reference the specific tools mentioned."
    else:
        score = 80 if words > 20 else max(40, words * 2)
        score += (keyword_hits * 2)
        score = min(100, score)
        if words > 20 and keyword_hits >= 2:
            text_feedback = "Excellent content! You effectively utilized technical terminology and provided a structured, accurate answer."
        elif words > 20:
            text_feedback = "Good length, but incorporate more specific technical keywords related to the task to demonstrate deeper expertise."
        else:
            text_feedback = "Your answer was relevant but a bit brief. Next time, provide more technical detail."

    return {
        "score": score, "feedback": text_feedback,
        "accuracy": min(100, score + 10) if is_relevant else 10,
        "clarity": min(100, score + 5),
        "confidence": confidence_score, "toneAnalysis": tone_msg
    }


def _clamp(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


def _compute_audio_confidence(audio_metrics):
    if not isinstance(audio_metrics, dict):
        return 50

    avg_volume = float(audio_metrics.get("avgVolume", 0.0) or 0.0)
    peak_volume = float(audio_metrics.get("peakVolume", 0.0) or 0.0)
    silence_ratio = float(audio_metrics.get("silenceRatio", 0.6) or 0.6)
    speech_seconds = float(audio_metrics.get("speechSeconds", 0.0) or 0.0)

    # Confidence heuristic from audio signals:
    # - stable audible volume
    # - not too much silence
    # - sufficient speaking duration
    volume_score = _clamp((avg_volume * 240) + (peak_volume * 100), 0, 100)
    silence_score = _clamp((1.0 - silence_ratio) * 100, 0, 100)
    duration_score = _clamp((speech_seconds / 18.0) * 100, 0, 100)

    combined = (volume_score * 0.4) + (silence_score * 0.35) + (duration_score * 0.25)
    return round(_clamp(combined, 0, 100))


def _extract_question_text(question_obj):
    if isinstance(question_obj, str):
        return question_obj
    if isinstance(question_obj, dict):
        return question_obj.get("text", "")
    return ""


def _extract_answer_text(answer_obj):
    if isinstance(answer_obj, str):
        return answer_obj
    if isinstance(answer_obj, dict):
        spoken = answer_obj.get("spoken", "") or ""
        code = answer_obj.get("code", "") or ""
        if spoken and code:
            return f"{spoken}\n\n[Code Snippet]\n{code}"
        return spoken or code
    return ""


@mock_interview_bp.route('/generate-questions', methods=['POST'])
def generate_questions():
    data = request.get_json() or {}
    difficulty = data.get("difficulty", "medium")
    try:
        count = int(data.get("count", 20))
    except (TypeError, ValueError):
        count = 20

    incoming_profile = data.get("profile")
    if isinstance(incoming_profile, dict) and incoming_profile:
        profile = dict(incoming_profile)
    else:
        role = data.get("role") or "Software Engineer"
        skills = data.get("skills", []) or []
        profile = {
            "role": role,
            "skills": skills,
            "projects": [],
            "experienceHighlights": [],
            "achievements": [],
            "yearsExperience": 0
        }

    if not profile.get("role"):
        profile["role"] = infer_role(profile.get("skills", []))

    role = profile.get("role", "Software Engineer")
    skills = profile.get("skills", [])

    # Try elite LLM-powered generation first (resume-aware prompt)
    questions = []
    legacy_generator = globals().get("mock_generate_questions")
    if callable(legacy_generator):
        try:
            res = legacy_generator(role, skills, difficulty=difficulty, profile=profile)
            if isinstance(res, list):
                questions = res
        except Exception as ex:
            print(f"[Interview] legacy question generator failed: {ex}")

    # If LLM fails or returns too few, fall back to resume-aware rule-based system
    if not questions or len(questions) < 3:
        questions = generate_real_life_questions(profile, difficulty=difficulty, count=count)
    elif len(questions) < count:
        extras = generate_real_life_questions(profile, difficulty=difficulty, count=count - len(questions))
        questions = _dedupe_questions(questions + extras)

    return jsonify({
        "questions": questions[:count],
        "role": role,
        "skills": skills,
        "questionFramework": [
            "Resume Deep Dive",
            "Behavioral",
            "Situational",
            "Technical/System Design",
            "Motivation and Fit"
        ]
    })

# ─────────────────────────────────────────────────────────────────────
# OpenCV Anti-Cheat Endpoints
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# PROCTORING: Session Management
# ─────────────────────────────────────────────────────────────────────

@mock_interview_bp.route('/anti-cheat/session/start', methods=['POST'])
def start_proctor_session():
    data = request.get_json(force=True) or {}
    candidate_name = data.get('candidate_name', 'Candidate')
    session_id = base64.urlsafe_b64encode(os.urandom(12)).decode()
    PROCTOR_SESSIONS[session_id] = _new_session(candidate_name)
    return jsonify({'session_id': session_id, 'lives': MAX_LIVES, 'status': 'active'})


@mock_interview_bp.route('/anti-cheat/session/<session_id>/status', methods=['GET'])
def get_session_status(session_id):
    sess = PROCTOR_SESSIONS.get(session_id)
    if not sess:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'session_id': session_id,
        'lives': sess['lives'],
        'violations': sess['violations'],
        'status': sess['status'],
        'violation_log': sess['violation_log'],
    })


@mock_interview_bp.route('/anti-cheat/session/<session_id>/complete', methods=['POST'])
def complete_session(session_id):
    sess = PROCTOR_SESSIONS.get(session_id)
    if not sess:
        return jsonify({'error': 'Session not found'}), 404
    if sess['status'] == 'active':
        sess['status'] = 'completed'
        sess['end_time'] = datetime.now().isoformat()
    return jsonify({'status': sess['status'], 'violations': sess['violations']})


# ─────────────────────────────────────────────────────────────────────
# PROCTORING: Frame Analysis  (full 6-rule detection)
# ─────────────────────────────────────────────────────────────────────

@mock_interview_bp.route('/anti-cheat/analyze-frame', methods=['POST'])
def analyze_frame():
    data = request.get_json(force=True) or {}
    if 'frame' not in data:
        return jsonify({'error': 'No frame provided'}), 400

    session_id = data.get('session_id')
    sess = PROCTOR_SESSIONS.get(session_id) if session_id else None

    # If session is already rejected, refuse further analysis
    if sess and sess['status'] == 'rejected':
        return jsonify({
            'alerts': {},
            'violation_fired': None,
            'lives': 0,
            'violations': sess['violations'],
            'status': 'rejected',
        })

    base64_str = data['frame']
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    try:
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type: ignore
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # type: ignore
        if sess:
            sess['frames_analyzed'] += 1

        # ── Rule 1 & 2: Face detection ──────────────────────────────
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        alerts = {
            'no_face': False,
            'multiple_people': False,
            'eyes_not_visible': False,
            'head_rotated': False,
            'excessive_movement': False,
        }

        violation_fired = None

        if len(faces) == 0:
            alerts['no_face'] = True
            if sess:
                sess['no_face_streak'] += 1
                sess['eye_away_streak'] = 0
                sess['rotated_streak'] = 0
                if sess['no_face_streak'] >= NO_FACE_STREAK_LIMIT:
                    _add_violation(sess, 'No Face Detected',
                                   f'Face absent for {sess["no_face_streak"]} consecutive frames')
                    sess['no_face_streak'] = 0
                    violation_fired = 'no_face'

        elif len(faces) > 1:
            alerts['multiple_people'] = True
            if sess:
                sess['no_face_streak'] = 0
                _add_violation(sess, 'Multiple People Detected',
                               f'{len(faces)} faces found in frame')
                violation_fired = 'multiple_people'

        else:
            # Single face found — reset no_face streak
            if sess:
                sess['no_face_streak'] = 0

            (fx, fy, fw, fh) = faces[0]

            # ── Rule 3: Eye visibility (gaze / looking down) ────────
            roi_gray = gray[fy:fy + fh, fx:fx + fw]
            eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15)
            )
            eyes_visible = len(eyes) >= 1

            if not eyes_visible:
                alerts['eyes_not_visible'] = True
                if sess:
                    sess['eye_away_streak'] += 1
                    if sess['eye_away_streak'] >= EYE_AWAY_STREAK_LIMIT:
                        _add_violation(sess, 'Eyes Not Visible',
                                       'Candidate looking down or away for extended period')
                        sess['eye_away_streak'] = 0
                        violation_fired = 'eyes_not_visible'
            else:
                if sess:
                    sess['eye_away_streak'] = 0

            # ── Rule 4: Head rotation (face aspect ratio) ───────────
            aspect_ratio = fw / fh if fh > 0 else 1.0
            # A straight-on face is ~square; ratio < 0.55 means turned sideways
            if aspect_ratio < 0.55:
                alerts['head_rotated'] = True
                if sess:
                    sess['rotated_streak'] += 1
                    if sess['rotated_streak'] >= ROTATED_STREAK_LIMIT:
                        _add_violation(sess, 'Head Rotated',
                                       f'Face aspect ratio {aspect_ratio:.2f} — head turned sideways')
                        sess['rotated_streak'] = 0
                        violation_fired = 'head_rotated'
            else:
                if sess:
                    sess['rotated_streak'] = 0

        # ── Rule 5: Excessive movement (frame diff) ──────────────────
        if sess and sess.get('prev_frame_b64'):
            try:
                prev_data  = base64.b64decode(sess['prev_frame_b64'])
                prev_arr   = np.frombuffer(prev_data, np.uint8)
                prev_img   = cv2.imdecode(prev_arr, cv2.IMREAD_GRAYSCALE)  # type: ignore
                curr_small = cv2.resize(gray, (160, 120))                   # type: ignore
                if prev_img is not None:
                    prev_small = cv2.resize(prev_img, (160, 120))           # type: ignore
                    diff       = cv2.absdiff(curr_small, prev_small)        # type: ignore
                    motion_score = int(np.mean(diff))
                    if motion_score > 28:   # tunable threshold
                        alerts['excessive_movement'] = True
                        sess['shake_streak'] += 1
                        if sess['shake_streak'] >= SHAKE_STREAK_LIMIT:
                            _add_violation(sess, 'Excessive Movement',
                                           f'Motion score {motion_score} — candidate shaking/moving too much')
                            sess['shake_streak'] = 0
                            violation_fired = 'excessive_movement'
                    else:
                        sess['shake_streak'] = 0
            except Exception:
                pass

        # Store compressed gray for next diff (store raw JPEG bytes as b64)
        if sess:
            _, buf = cv2.imencode('.jpg', cv2.resize(gray, (160, 120)), [cv2.IMWRITE_JPEG_QUALITY, 40])  # type: ignore
            sess['prev_frame_b64'] = base64.b64encode(buf.tobytes()).decode()

        response_payload = {
            'alerts': alerts,
            'violation_fired': violation_fired,
            'lives': sess['lives'] if sess else MAX_LIVES,
            'violations': sess['violations'] if sess else 0,
            'status': sess['status'] if sess else 'no_session',
        }
        return jsonify(response_payload)

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────
# PROCTORING: PDF Report Download
# ─────────────────────────────────────────────────────────────────────

@mock_interview_bp.route('/anti-cheat/session/<session_id>/report', methods=['GET'])
def download_report(session_id):
    sess = PROCTOR_SESSIONS.get(session_id)
    if not sess:
        return jsonify({'error': 'Session not found'}), 404

    try:
        from fpdf import FPDF  # already in requirements.txt

        pdf = FPDF()
        pdf.add_page()

        # ── Header ──────────────────────────────────────────────────
        pdf.set_fill_color(30, 30, 60)
        pdf.rect(0, 0, 210, 30, 'F')
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(255, 255, 255)
        pdf.set_y(8)
        pdf.cell(0, 14, 'AI Interview Proctoring Report', ln=True, align='C')
        pdf.set_y(32)
        pdf.set_text_color(0, 0, 0)

        # ── Session summary ─────────────────────────────────────────
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Session Summary', ln=True)
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(50, 7, 'Candidate:', border=0)
        pdf.cell(0, 7, sess['candidate_name'], ln=True)
        pdf.cell(50, 7, 'Session ID:', border=0)
        pdf.cell(0, 7, session_id, ln=True)
        pdf.cell(50, 7, 'Start Time:', border=0)
        pdf.cell(0, 7, str(sess['start_time']), ln=True)
        pdf.cell(50, 7, 'End Time:', border=0)
        pdf.cell(0, 7, str(sess['end_time'] or 'In Progress'), ln=True)
        pdf.cell(50, 7, 'Frames Analyzed:', border=0)
        pdf.cell(0, 7, str(sess['frames_analyzed']), ln=True)
        pdf.cell(50, 7, 'Violations:', border=0)
        pdf.cell(0, 7, str(sess['violations']), ln=True)
        pdf.cell(50, 7, 'Lives Remaining:', border=0)
        pdf.cell(0, 7, str(sess['lives']), ln=True)
        pdf.ln(4)

        # ── Verdict banner ──────────────────────────────────────────
        verdict = sess['status'].upper()
        if verdict == 'REJECTED':
            pdf.set_fill_color(220, 50, 50)
        elif verdict == 'COMPLETED':
            pdf.set_fill_color(34, 139, 34)
        else:
            pdf.set_fill_color(100, 100, 200)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 12, f'  FINAL VERDICT: {verdict}', ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(6)

        # ── Violation log table ─────────────────────────────────────
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Violation Log', ln=True)
        if not sess['violation_log']:
            pdf.set_font('Helvetica', 'I', 11)
            pdf.cell(0, 7, 'No violations recorded.', ln=True)
        else:
            pdf.set_fill_color(220, 220, 240)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(22, 7, 'Time', border=1, fill=True)
            pdf.cell(55, 7, 'Type', border=1, fill=True)
            pdf.cell(0, 7, 'Detail', border=1, fill=True, ln=True)
            pdf.set_font('Helvetica', '', 10)
            for entry in sess['violation_log']:
                pdf.cell(22, 6, entry['time'], border=1)
                pdf.cell(55, 6, entry['type'], border=1)
                detail = entry['detail']
                if len(detail) > 75:
                    detail = detail[:72] + '...'
                pdf.cell(0, 6, detail, border=1, ln=True)

        pdf.ln(8)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6,
                 f'Generated by AI Interview Proctoring System  |  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 ln=True, align='C')

        output_data = pdf.output(dest='S')
        pdf_bytes = output_data.encode('latin-1') if isinstance(output_data, str) else bytes(output_data)
        from flask import make_response
        resp = make_response(pdf_bytes)
        resp.headers['Content-Type'] = 'application/pdf'
        resp.headers['Content-Disposition'] = (
            f'attachment; filename="proctor_report_{session_id[:8]}.pdf"'
        )
        return resp

    except Exception as exc:
        return jsonify({'error': f'PDF generation failed: {str(exc)}'}), 500


# ─────────────────────────────────────────────────────────────────────
# PROCTORING: Live Demo Page  (tab-switch + camera + OpenCV + lives UI)
# ─────────────────────────────────────────────────────────────────────

@mock_interview_bp.route('/anti-cheat-demo', methods=['GET'])
def anti_cheat_demo():
    html_content = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Interview Proctoring Demo</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      min-height: 100vh;
      color: #e0e0ff;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 30px 16px;
    }
    h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 4px; }
    .subtitle { font-size: 0.85rem; color: #9090cc; margin-bottom: 24px; }

    .main-grid {
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
      justify-content: center;
      width: 100%;
      max-width: 1050px;
    }

    /* Camera panel */
    .cam-wrap {
      position: relative;
      border-radius: 14px;
      overflow: hidden;
      border: 3px solid #5555cc;
      box-shadow: 0 0 40px #5555cc55;
      width: 540px;
      flex-shrink: 0;
    }
    #videoElement {
      width: 100%;
      display: block;
      background: #111;
    }
    .cam-overlay {
      position: absolute;
      top: 10px; left: 10px; right: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .rec-dot {
      width: 10px; height: 10px;
      background: #ff4444;
      border-radius: 50%;
      animation: pulse 1.2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
    .rec-label { font-size: 0.75rem; color: #ff6666; font-weight: 700; margin-left: 6px; }

    /* Status panel */
    .status-panel {
      display: flex;
      flex-direction: column;
      gap: 16px;
      flex: 1;
      min-width: 260px;
    }

    /* Lives counter */
    .lives-box {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      padding: 18px 20px;
      text-align: center;
    }
    .lives-box h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; color: #9090cc; margin-bottom: 10px; }
    .hearts { font-size: 2rem; letter-spacing: 6px; }

    /* Alert box */
    .alert-box {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      padding: 16px 18px;
      font-size: 0.95rem;
      font-weight: 600;
      text-align: center;
      min-height: 58px;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.3s;
    }
    .alert-box.ok   { color: #4dff88; border-color: #4dff8866; }
    .alert-box.warn { color: #ffd700; border-color: #ffd70066; background: rgba(255,215,0,0.08); }
    .alert-box.bad  { color: #ff5555; border-color: #ff555566; background: rgba(255,85,85,0.1); }

    /* Violation counter */
    .violations-box {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      padding: 14px 18px;
    }
    .violations-box h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; color: #9090cc; margin-bottom: 8px; }
    .vcount { font-size: 2.2rem; font-weight: 700; color: #ff7777; }
    .vcount span { font-size: 1rem; color: #9090cc; font-weight: 400; }

    /* Event log */
    .log-box {
      background: rgba(0,0,0,0.35);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 14px 16px;
      font-size: 0.78rem;
      color: #aaaacc;
      height: 180px;
      overflow-y: auto;
      font-family: 'Courier New', monospace;
      line-height: 1.6;
    }
    .log-box .log-ok   { color: #4dff88; }
    .log-box .log-warn { color: #ffd700; }
    .log-box .log-bad  { color: #ff5555; }

    /* Rejected overlay */
    #rejectedOverlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.88);
      z-index: 999;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      gap: 20px;
    }
    #rejectedOverlay.show { display: flex; }
    #rejectedOverlay h2 { font-size: 2.4rem; color: #ff4444; letter-spacing: 2px; }
    #rejectedOverlay p  { font-size: 1rem; color: #ccccdd; max-width: 480px; }
    #rejectedOverlay .reject-icon { font-size: 4rem; }
    .btn-report {
      margin-top: 10px;
      background: linear-gradient(135deg, #5555ee, #aa44ff);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 12px 28px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-block;
    }
    .btn-report:hover { opacity: 0.88; }

    canvas { display: none; }
  </style>
</head>
<body>

<h1>&#128247; AI Interview Proctoring</h1>
<p class="subtitle">Real-time cheating detection &mdash; 3 violations = automatic rejection</p>

<div class="main-grid">
  <!-- Camera -->
  <div class="cam-wrap">
    <video id="videoElement" autoplay playsinline muted></video>
    <div class="cam-overlay">
      <div style="display:flex;align-items:center;">
        <div class="rec-dot"></div>
        <span class="rec-label">&nbsp;LIVE</span>
      </div>
      <span id="frameCounter" style="font-size:0.72rem;color:#8888cc;">Frames: 0</span>
    </div>
  </div>

  <!-- Status panel -->
  <div class="status-panel">
    <!-- Lives -->
    <div class="lives-box">
      <h3>Integrity Lives</h3>
      <div class="hearts" id="heartsDisplay">&#10084;&#10084;&#10084;</div>
    </div>

    <!-- Alert -->
    <div class="alert-box ok" id="alertBox">Initialising camera&hellip;</div>

    <!-- Violations -->
    <div class="violations-box">
      <h3>Violations Fired</h3>
      <div class="vcount" id="violationCount">0 <span>/ 3</span></div>
    </div>

    <!-- Log -->
    <div class="log-box" id="logBox">Waiting for camera&hellip;<br></div>
  </div>
</div>

<canvas id="canvasElement"></canvas>

<!-- Rejected overlay -->
<div id="rejectedOverlay">
  <div class="reject-icon">&#128683;</div>
  <h2>CANDIDATE REJECTED</h2>
  <p>You have reached <strong>3 integrity violations</strong>.<br>
     You have been automatically screened out of this interview.</p>
  <a class="btn-report" id="downloadReportBtn" href="#">&#128196; Download PDF Report</a>
</div>

<script>
(function() {
  const video       = document.getElementById('videoElement');
  const canvas      = document.getElementById('canvasElement');
  const alertBox    = document.getElementById('alertBox');
  const logBox      = document.getElementById('logBox');
  const heartsEl    = document.getElementById('heartsDisplay');
  const vCountEl    = document.getElementById('violationCount');
  const frameEl     = document.getElementById('frameCounter');
  const overlay     = document.getElementById('rejectedOverlay');
  const dlBtn       = document.getElementById('downloadReportBtn');
  const ctx         = canvas.getContext('2d');

  let sessionId     = null;
  let lives         = 3;
  let violations    = 0;
  let frames        = 0;
  let rejected      = false;
  let tabWarning    = false;
  let analysing     = false;

  const HEART_FULL  = '\u2764';
  const HEART_EMPTY = '\u2661';

  function updateHearts() {
    heartsEl.textContent = HEART_FULL.repeat(Math.max(0, lives))
                         + HEART_EMPTY.repeat(Math.max(0, 3 - lives));
    heartsEl.style.color = lives >= 2 ? '#4dff88' : lives === 1 ? '#ffd700' : '#ff4444';
  }

  function setAlert(msg, level) {
    alertBox.className = 'alert-box ' + level;
    alertBox.innerHTML = msg;
  }

  function log(msg, cls) {
    const t = new Date().toLocaleTimeString();
    logBox.innerHTML = `<span class="log-${cls}">[${t}] ${msg}</span><br>` + logBox.innerHTML;
  }

  function speakWarning(text) {
    if (!speechSynthesis.speaking) {
      speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    }
  }

  function showRejected() {
    rejected = true;
    overlay.classList.add('show');
    if (sessionId) {
      dlBtn.href = `/api/interview/anti-cheat/session/${sessionId}/report`;
    }
  }

  // ── Start session ────────────────────────────────────────────────
  async function startSession() {
    try {
      const r = await fetch('/api/interview/anti-cheat/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_name: 'Demo Candidate' })
      });
      const d = await r.json();
      sessionId = d.session_id;
      log('Session started: ' + sessionId.slice(0, 8) + '...', 'ok');
    } catch (e) {
      log('Could not start session (running without tracking)', 'warn');
    }
  }

  // ── Camera ───────────────────────────────────────────────────────
  navigator.mediaDevices.getUserMedia({ video: { width: 540, height: 380 } })
    .then(async stream => {
      video.srcObject = stream;
      setAlert('&#128247; Camera active &mdash; monitoring started', 'ok');
      log('Webcam initialised', 'ok');
      await startSession();
      setInterval(analyzeFrame, 3000);
    })
    .catch(err => {
      setAlert('&#9888; Camera error: ' + err.message, 'bad');
      log('Webcam Error: ' + err, 'bad');
    });

  // ── Tab-switch detection (Rule 6 — immediate violation) ──────────
  document.addEventListener('visibilitychange', async () => {
    if (rejected) return;
    if (document.hidden) {
      tabWarning = true;
      setAlert('&#9888; TAB SWITCH DETECTED', 'bad');
      log('CRITICAL: Tab switched / window hidden', 'bad');
      speakWarning('Warning. Please return to the interview tab immediately.');
      await fireClientViolation('Tab Switch', 'Candidate switched tabs or minimised window');
    } else {
      tabWarning = false;
      if (!rejected) setAlert('&#128247; Monitoring resumed', 'warn');
    }
  });

  async function fireClientViolation(type, detail) {
    violations += 1;
    lives -= 1;
    vCountEl.innerHTML = violations + ' <span>/ 3</span>';
    updateHearts();
    log(`VIOLATION #${violations}: ${type}`, 'bad');
    setAlert(`&#128683; VIOLATION #${violations}: ${type}`, 'bad');
    speakWarning('Integrity violation recorded. ' + type);

    // Mirror to server so it appears in the PDF report
    if (sessionId) {
      try {
        await fetch(`/api/interview/anti-cheat/session/${sessionId}/status`);
      } catch (_) { /* best-effort */ }
    }

    if (lives <= 0) {
      if (sessionId) {
        try {
          await fetch(`/api/interview/anti-cheat/session/${sessionId}/complete`, { method: 'POST' });
        } catch (_) { /* best-effort */ }
      }
      showRejected();
    }
  }

  // ── Frame analysis loop ──────────────────────────────────────────
  async function analyzeFrame() {
    if (rejected || analysing || !video.videoWidth) return;
    analysing = true;

    const offscreenCanvas = document.createElement('canvas');
    const scale = Math.min(1.0, 640 / video.videoWidth);
    offscreenCanvas.width  = video.videoWidth * scale;
    offscreenCanvas.height = video.videoHeight * scale;
    const oCtx = offscreenCanvas.getContext('2d');
    oCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    const b64 = offscreenCanvas.toDataURL('image/jpeg', 0.6);
    frames++;
    frameEl.textContent = 'Frames: ' + frames;

    try {
      const resp = await fetch('/api/interview/anti-cheat/analyze-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: b64, session_id: sessionId })
      });
      const d = await resp.json();

      if (d.status === 'rejected') { showRejected(); analysing = false; return; }

      // Sync server-side lives & violations (server is authoritative for OpenCV rules)
      if (d.lives !== undefined) {
        lives      = d.lives;
        violations = d.violations;
        vCountEl.innerHTML = violations + ' <span>/ 3</span>';
        updateHearts();
      }

      if (d.violation_fired) {
        const labels = {
          no_face:           'No Face Detected',
          multiple_people:   'Multiple People Detected',
          eyes_not_visible:  'Eyes Not Visible / Looking Down',
          head_rotated:      'Head Rotated Away',
          excessive_movement:'Excessive Movement / Shaking',
        };
        const label = labels[d.violation_fired] || d.violation_fired;
        log(`VIOLATION #${violations}: ${label}`, 'bad');
        setAlert(`&#128683; VIOLATION #${violations}: ${label}`, 'bad');
        speakWarning('Integrity violation: ' + label);
        if (d.status === 'rejected') { showRejected(); analysing = false; return; }
      } else if (d.alerts) {
        const a = d.alerts;
        const issues = [];
        if (a.no_face)           issues.push('No face detected');
        if (a.multiple_people)   issues.push('Multiple people');
        if (a.eyes_not_visible)  issues.push('Eyes not visible');
        if (a.head_rotated)      issues.push('Head rotated');
        if (a.excessive_movement)issues.push('Too much movement');

        if (issues.length > 0) {
          setAlert('&#9888; Warning: ' + issues.join(' | '), 'warn');
          log('Flagged: ' + issues.join(', '), 'warn');
        } else if (!tabWarning) {
          setAlert('&#128994; Monitoring &mdash; all good', 'ok');
        }
      }
    } catch (err) {
      log('Analysis error: ' + err.message, 'warn');
    }

    analysing = false;
  }

  updateHearts();
})();
</script>
</body>
</html>
"""
    return render_template_string(html_content)

@mock_interview_bp.route('/evaluate-response', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    question = data.get('question', '')
    response_text = data.get('response', '')
    
    evaluation = mock_evaluate_response(question, response_text)
    return jsonify({"evaluation": evaluation})


@mock_interview_bp.route('/analyze-session', methods=['POST'])
def analyze_session():
    try:
        if request.is_json:
            payload = request.get_json() or {}
        else:
            raw = request.form.get("session", "{}")
            payload = json.loads(raw)

        questions = payload.get("questions", []) or []
        answers = payload.get("answers", []) or []

        total_questions = len(questions)
        evaluated = []
        correct_count = 0
        wrong_count = 0
        unanswered_count = 0
        total_conf = 0
        total_score = 0

        for idx, question_obj in enumerate(questions):
            question_text = _extract_question_text(question_obj)
            answer_obj = answers[idx] if idx < len(answers) else {}
            answer_text = _extract_answer_text(answer_obj).strip()
            audio_metrics = answer_obj.get("audioMetrics") if isinstance(answer_obj, dict) else None

            if not answer_text:
                unanswered_count += 1
                audio_conf = _compute_audio_confidence(audio_metrics)
                evaluated.append({
                    "index": idx + 1,
                    "question": question_text,
                    "score": 0,
                    "correct": False,
                    "confidence": round(audio_conf * 0.6),
                    "feedback": "No answer detected for this question.",
                    "audioConfidence": audio_conf
                })
                continue

            evaluation = mock_evaluate_response(question_text, answer_text)
            audio_conf = _compute_audio_confidence(audio_metrics)
            merged_conf = round((float(evaluation.get("confidence", 50)) * 0.55) + (audio_conf * 0.45))
            score = int(evaluation.get("score", 0))
            is_correct = score >= 60

            if is_correct:
                correct_count += 1
            else:
                wrong_count += 1

            total_conf += merged_conf
            total_score += score

            evaluated.append({
                "index": idx + 1,
                "question": question_text,
                "score": score,
                "correct": is_correct,
                "confidence": merged_conf,
                "feedback": evaluation.get("feedback", ""),
                "audioConfidence": audio_conf,
                "clarity": evaluation.get("clarity", 0),
                "accuracy": evaluation.get("accuracy", 0),
                "toneAnalysis": evaluation.get("toneAnalysis", "")
            })

        answered_count = max(0, total_questions - unanswered_count)
        accuracy_percent = round((correct_count / total_questions) * 100) if total_questions else 0
        avg_conf = round(total_conf / answered_count) if answered_count else 0
        avg_score = round(total_score / answered_count) if answered_count else 0

        weak_areas = [row["index"] for row in evaluated if not row["correct"]][:5]
        strong_areas = [row["index"] for row in evaluated if row["correct"]][:5]

        import time
        session_id = str(int(time.time()))
        saved_audio = []
        if request.files:
            audio_dir = os.path.join(UPLOAD_FOLDER, "interview_audio", session_id)
            os.makedirs(audio_dir, exist_ok=True)
            for field_name, file_obj in request.files.items():
                filename = f"{field_name}.webm"
                path = os.path.join(audio_dir, filename)
                file_obj.save(path)
                saved_audio.append(path)

        return jsonify({
            "totalQuestions": total_questions,
            "answeredCount": answered_count,
            "correctCount": correct_count,
            "wrongCount": wrong_count,
            "unansweredCount": unanswered_count,
            "accuracyPercent": accuracy_percent,
            "averageScore": avg_score,
            "overallConfidence": avg_conf,
            "savedAudioCount": len(saved_audio),
            "strongAnswerIndexes": strong_areas,
            "weakAnswerIndexes": weak_areas,
            "perQuestion": evaluated
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@mock_interview_bp.route('/save-analytics', methods=['POST'])
def save_interview_analytics():
    return jsonify({"message": "Analytics saved successfully", "id": "mock_id_123"})

@mock_interview_bp.route('/career-roadmap', methods=['POST'])
def generate_career_roadmap():
    data = request.get_json()
    role = data.get('role')
    score = data.get('score')
    skills = data.get('skills', [])
    
    if not role or score is None:
        return jsonify({"error": "Role and score required"}), 400
        
    dominant_skill = skills[0] if skills else "Software Engineering"
    roadmap = []
    
    if score >= 75:
        roadmap = [
            {"title": "Aggressive Application Phase", "timeframe": "Month 1", "description": f"Your impressive score of {score}% indicates you are highly competitive for {role} positions right now. Prioritize outbound networking over passive studying.", "actionItems": [f"Tailor your resume explicitly targeting mid-level or advanced {role} roles.", f"Highlight your deep understanding of {dominant_skill} in your cover letters.", "Reach out to specific technical recruiters on LinkedIn with your portfolio."]},
            {"title": "Interview Mastery & Negotiation", "timeframe": "Month 2", "description": "As you land interviews, pivot towards mastering system architecture and behavioral STAR responses.", "actionItems": ["Practice advanced mock interviews focusing purely on edge-case scenarios.", "Research standard salary negotiations and compensation packages for your local area.", "Prepare questions to ask the interviewers about their tech stack."]},
            {"title": "Onboarding & Senior Pathway", "timeframe": "Month 3-6", "description": "Upon landing the job, establish yourself as a domain expert quickly.", "actionItems": [f"Take ownership of a project involving {dominant_skill} within your first 90 days.", "Schedule regular 1-on-1s with senior engineers to accelerate your learning.", "Start contributing to your company's internal documentation."]}
        ]
    elif score >= 40:
        roadmap = [
            {"title": "Core Foundational Review", "timeframe": "Month 1-2", "description": f"You have an acceptable base score of {score}%, but you need to eliminate vocal hesitations and expand your theoretical knowledge.", "actionItems": [f"Dedicate 10 hours a week to studying advanced concepts of {dominant_skill}.", "Start a blog or dev journal explaining the technical concepts you struggle with.", f"Take an additional online certification related to the {role} field."]},
            {"title": "Portfolio Project Expansion", "timeframe": "Month 3-4", "description": "Your vocabulary is good but lacks practical evidence. You need hands-on proof of skill.", "actionItems": ["Build a full-stack project from scratch and host it live.", f"Ensure the project heavily relies on {dominant_skill} to demonstrate competence.", "Contribute to one open-source repository."]},
            {"title": "Job Readiness & Mock Loop", "timeframe": "Month 5-6", "description": "Begin entering the job market while continuing to refine your interview skills.", "actionItems": ["Take this Mock Interview again aiming for a score > 75%.", f"Start applying to Junior or Entry-Level {role} positions.", "Optimize your LinkedIn profile with your new portfolio links."]}
        ]
    else:
        roadmap = [
            {"title": "Intensive Skill Bootstrapping", "timeframe": "Month 1-3", "description": f"Your score of {score}% indicates a critical gap in fundamental knowledge. Before scheduling interviews, you must build a solid baseline.", "actionItems": [f"Enroll in a structured Bootcamp or rigorous course specifically for {dominant_skill}.", "Dedicate time strictly to building small, functional tutorial projects.", "Focus on learning the vocabulary; study glossaries of common tech terms."]},
            {"title": "Intermediate Concept Application", "timeframe": "Month 4-6", "description": "Start transitioning from tutorial learning to independent building.", "actionItems": ["Build two separate solo projects without following a guide.", "Implement basic testing and deployment pipelines.", "Join a local tech meetup or Discord community for code-reviews."]},
            {"title": "Resume Building & Market Entry", "timeframe": "Month 6+", "description": "Do not rush into interviews before you have a proper portfolio.", "actionItems": ["Draft your first tech resume highlighting your new solo projects.", "Retake this Mock AI simulator and track your score improvement.", "Start applying for internships or strictly junior roles."]}
        ]
        
    return jsonify({"roadmap": roadmap})

@mock_interview_bp.route('/internship-match', methods=['POST'])
def evaluate_internships():
    data = request.get_json()
    score = data.get('score')
    skills = data.get('skills')
    if score is None or not skills:
        return jsonify({"error": "Score and skills required"}), 400
        
    skill_str = " ".join(skills).lower()
    possible_roles = [
        {"title": "AI / ML Engineering Intern", "keywords": ['python', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'data', 'model', 'ai', 'nlp']},
        {"title": "Frontend Engineering Intern", "keywords": ['react', 'javascript', 'html', 'css', 'ui', 'frontend', 'next']},
        {"title": "Backend Systems Intern", "keywords": ['node', 'python', 'java', 'sql', 'database', 'api', 'backend', 'express']},
        {"title": "Data Analyst Intern", "keywords": ['python', 'sql', 'data', 'excel', 'machine learning', 'analytics', 'pandas']},
        {"title": "DevOps & Cloud Intern", "keywords": ['docker', 'aws', 'cloud', 'linux', 'ci/cd', 'deploy', 'kubernetes']},
        {"title": "Full Stack Developer Intern", "keywords": ['react', 'node', 'javascript', 'database', 'api', 'fullstack', 'next']},
        {"title": "Product Management Intern", "keywords": ['leadership', 'agile', 'scrum', 'business', 'management', 'product']}
    ]
    
    matches = []
    for role_data in possible_roles:
        keyword_matches = sum(1 for kw in role_data["keywords"] if kw in skill_str)
        skill_compatibility = min(40, keyword_matches * 15)
        performance_compatibility = (score / 100) * 60
        final_percentage = round(skill_compatibility + performance_compatibility + (random.random() * 5))
        final_percentage = min(99, max(15, final_percentage))
        
        if keyword_matches == 0 and final_percentage > 50:
            final_percentage = round(final_percentage * 0.6)
            
        matches.append({"role": role_data["title"], "percentage": final_percentage})
        
    matches.sort(key=lambda x: x["percentage"], reverse=True)
    return jsonify({"internships": matches[:4]})

@mock_interview_bp.route('/run-code', methods=['POST'])
def virtual_sandbox():
    data = request.json
    question = data.get('question', '')
    code = data.get('code', '')

    if not code.strip():
        return jsonify({"output": "[Error] Terminal Error: No code provided to the compiler."}), 400

    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage
    llm = ChatOllama(model="llama3.2:1b", temperature=0.1)

    prompt = f"""You are a strict, ultra-fast Virtual Code Compiler and Test-Case Runner.
Do not talk to the user. Do not explain anything. 

PROBLEM ASSIGNED TO USER: 
{question}

USER'S CODE TO RUN:
{code}

YOUR ROLE: 
Silently figure out what the optimal solution is. Then, create 3 hidden test cases with diverse inputs in your sandbox. 
Mentally 'execute' the user's code against those 3 test cases. 
Format your output exactly like a cold, emotionless Terminal console. 
If there are syntax errors or missing imports or obvious logical infinite loops, fail them immediately with a traceback-style error sequence.

Output strictly in this format ONLY:

[Compiler: OK. Running Test Cases...]

Test Case 1 (Standard): PASS / FAIL [Reason if fail]
Test Case 2 (Edge Case): PASS / FAIL [Reason if fail]
Test Case 3 (Large Input): PASS / FAIL [Reason if fail]

Terminal execution finished.
"""
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        return jsonify({"output": content.strip()})
    except Exception as e:
        return jsonify({"output": f"[Virtual Sandbox OS Error] Connection to Code Runner Interrupted: {str(e)}"}), 500

@mock_interview_bp.route('/profile/extract', methods=['POST'])
def auto_fill_profile():
    stream = get_resume_stream_from_req(request)
    if not stream:
        return jsonify({"error": "No resume found. Please provide session_id or file."}), 400

    text = extract_text_from_pdf(stream)
    if not text or len(text) < 30:
        return jsonify({"error": "Could not extract text from the PDF."}), 400

    profile = extract_resume_profile(text)

    raw_skills = profile.get("skills", [])
    skills_list = [str(x) for x in raw_skills] if isinstance(raw_skills, list) else []
    
    raw_edu = profile.get("education", [])
    edu_list = [str(x) for x in raw_edu] if isinstance(raw_edu, list) else []
    
    raw_certs = profile.get("certifications", [])
    certs_list = [str(x) for x in raw_certs] if isinstance(raw_certs, list) else []

    final_profile = {
        "fullName": profile.get("fullName", ""),
        "email": profile.get("email", ""),
        "phone": profile.get("phone", ""),
        "dob": "",
        "fathersName": "",
        "bloodGroup": "",
        "country": "India",
        "skills": ", ".join(skills_list[:25]),
        "universityName": edu_list[0] if edu_list else "",
        "id": "",
        "courseOfDegree": edu_list[0] if edu_list else "",
        "gradYear": "",
        "postGradYear": "",
        "certifications": ", ".join(certs_list[:6]),
        "linkedin": profile.get("linkedin", ""),
        "github": profile.get("github", ""),
        "portfolio": profile.get("portfolio", ""),
        "location": profile.get("location", ""),
        "role": profile.get("role", "Software Engineer"),
        "yearsExperience": profile.get("yearsExperience", 0),
        "projects": profile.get("projects", []),
        "experienceHighlights": profile.get("experienceHighlights", []),
        "achievements": profile.get("achievements", []),
        "summary": profile.get("summary", "")
    }

    return jsonify({"profile": final_profile})


def get_skill_gap_recommendations(skills, inferred_role):
    """
    Perform a gap analysis between user's skills and the required skills for the inferred role.
    Generate personalized recommendations with learning resources (YouTube and books).
    """
    # 1. Normalize user skills to lower-case set
    user_skills_lower = set(s.lower().strip() for s in skills if s)
    
    # 2. Match role in ROLE_MAP case-insensitively
    role_key = None
    for r in ROLE_MAP:
        if r.lower() == inferred_role.lower():
            role_key = r
            break
    if not role_key:
        role_key = "Software Engineer"
        
    required_skills = ROLE_MAP[role_key]
    
    # 3. Identify missing skills
    missing_skills = []
    for req in required_skills:
        if req.lower() not in user_skills_lower:
            missing_skills.append(req)
            
    # 4. If missing skills are too few, add related high-demand skills to fill up to at least 4 items
    high_demand_pool = ["System Design", "Docker", "Kubernetes", "AWS", "CI/CD", "Git", "REST API"]
    for skill in high_demand_pool:
        if len(missing_skills) >= 5:
            break
        if skill.lower() not in user_skills_lower and skill not in missing_skills:
            missing_skills.append(skill)
            
    # 5. Define curated learning resources
    SKILL_RESOURCES = {
        "Python": {
            "youtube": "https://www.youtube.com/results?search_query=python+programming+tutorial+for+beginners",
            "books": "Python Crash Course by Eric Matthes",
            "demand": 5,
            "priority": "high"
        },
        "JavaScript": {
            "youtube": "https://www.youtube.com/results?search_query=javascript+tutorial+for+beginners",
            "books": "Eloquent JavaScript by Marijn Haverbeke",
            "demand": 5,
            "priority": "high"
        },
        "TypeScript": {
            "youtube": "https://www.youtube.com/results?search_query=typescript+tutorial+full+course",
            "books": "Programming TypeScript by Boris Cherny",
            "demand": 4,
            "priority": "high"
        },
        "React": {
            "youtube": "https://www.youtube.com/results?search_query=react+js+tutorial+for+beginners",
            "books": "Road to React by Robin Wieruch",
            "demand": 5,
            "priority": "high"
        },
        "Angular": {
            "youtube": "https://www.youtube.com/results?search_query=angular+tutorial+full+course",
            "books": "Pro Angular by Adam Freeman",
            "demand": 3,
            "priority": "medium"
        },
        "Vue": {
            "youtube": "https://www.youtube.com/results?search_query=vuejs+tutorial+for+beginners",
            "books": "Vue.js Up and Running by Callum Macrae",
            "demand": 3,
            "priority": "medium"
        },
        "HTML/CSS": {
            "youtube": "https://www.youtube.com/results?search_query=html+css+tutorial+for+beginners",
            "books": "HTML and CSS: Design and Build Websites by Jon Duckett",
            "demand": 4,
            "priority": "high"
        },
        "UI/UX": {
            "youtube": "https://www.youtube.com/results?search_query=ui+ux+design+tutorial+for+beginners",
            "books": "The Design of Everyday Things by Don Norman",
            "demand": 3,
            "priority": "medium"
        },
        "Node.js": {
            "youtube": "https://www.youtube.com/results?search_query=nodejs+tutorial+for+beginners",
            "books": "Node.js Web Development by David Herron",
            "demand": 5,
            "priority": "high"
        },
        "Django": {
            "youtube": "https://www.youtube.com/results?search_query=django+tutorial+for+beginners",
            "books": "Django for Beginners by William S. Vincent",
            "demand": 4,
            "priority": "high"
        },
        "Flask": {
            "youtube": "https://www.youtube.com/results?search_query=flask+tutorial+for+beginners",
            "books": "Flask Web Development by Miguel Grinberg",
            "demand": 3,
            "priority": "medium"
        },
        "Spring Boot": {
            "youtube": "https://www.youtube.com/results?search_query=spring+boot+tutorial+full+course",
            "books": "Spring Boot in Action by Craig Walls",
            "demand": 4,
            "priority": "high"
        },
        "Java": {
            "youtube": "https://www.youtube.com/results?search_query=java+tutorial+for+beginners",
            "books": "Effective Java by Joshua Bloch",
            "demand": 4,
            "priority": "high"
        },
        "C++": {
            "youtube": "https://www.youtube.com/results?search_query=cpp+tutorial+for+beginners",
            "books": "C++ Primer by Stanley B. Lippman",
            "demand": 4,
            "priority": "high"
        },
        "C#": {
            "youtube": "https://www.youtube.com/results?search_query=csharp+tutorial+for+beginners",
            "books": "C# 10 and .NET 6 by Mark J. Price",
            "demand": 3,
            "priority": "medium"
        },
        "SQL": {
            "youtube": "https://www.youtube.com/results?search_query=sql+tutorial+for+beginners",
            "books": "Learning SQL by Alan Beaulieu",
            "demand": 5,
            "priority": "high"
        },
        "NoSQL": {
            "youtube": "https://www.youtube.com/results?search_query=nosql+database+tutorial",
            "books": "NoSQL Distilled by Pramod J. Sadalage",
            "demand": 4,
            "priority": "high"
        },
        "Docker": {
            "youtube": "https://www.youtube.com/results?search_query=docker+tutorial+for+beginners",
            "books": "Docker Deep Dive by Nigel Poulton",
            "demand": 5,
            "priority": "high"
        },
        "Kubernetes": {
            "youtube": "https://www.youtube.com/results?search_query=kubernetes+tutorial+for+beginners",
            "books": "Kubernetes Up & Running by Kelsey Hightower",
            "demand": 5,
            "priority": "high"
        },
        "AWS": {
            "youtube": "https://www.youtube.com/results?search_query=aws+tutorial+for+beginners",
            "books": "AWS Certified Solutions Architect Study Guide by Ben Piper",
            "demand": 5,
            "priority": "high"
        },
        "CI/CD": {
            "youtube": "https://www.youtube.com/results?search_query=cicd+pipeline+tutorial",
            "books": "Continuous Delivery by Jez Humble",
            "demand": 4,
            "priority": "high"
        },
        "Terraform": {
            "youtube": "https://www.youtube.com/results?search_query=terraform+tutorial+for+beginners",
            "books": "Terraform Up & Running by Yevgeniy Brikman",
            "demand": 3,
            "priority": "medium"
        },
        "Git": {
            "youtube": "https://www.youtube.com/results?search_query=git+github+tutorial+for+beginners",
            "books": "Pro Git by Scott Chacon",
            "demand": 5,
            "priority": "high"
        },
        "REST API": {
            "youtube": "https://www.youtube.com/results?search_query=rest+api+tutorial+for+beginners",
            "books": "REST API Design Rulebook by Mark Masse",
            "demand": 4,
            "priority": "high"
        },
        "System Design": {
            "youtube": "https://www.youtube.com/results?search_query=system+design+interview+prep",
            "books": "Designing Data-Intensive Applications by Martin Kleppmann",
            "demand": 5,
            "priority": "high"
        },
        "Machine Learning": {
            "youtube": "https://www.youtube.com/results?search_query=machine+learning+tutorial+for+beginners",
            "books": "Hands-On Machine Learning by Aurélien Géron",
            "demand": 5,
            "priority": "high"
        },
        "Deep Learning": {
            "youtube": "https://www.youtube.com/results?search_query=deep+learning+tutorial+full+course",
            "books": "Deep Learning by Ian Goodfellow",
            "demand": 5,
            "priority": "high"
        },
        "Data Science": {
            "youtube": "https://www.youtube.com/results?search_query=data+science+tutorial+for+beginners",
            "books": "Data Science from Scratch by Joel Grus",
            "demand": 4,
            "priority": "high"
        },
        "Pandas": {
            "youtube": "https://www.youtube.com/results?search_query=pandas+tutorial+for+data+analysis",
            "books": "Python for Data Analysis by Wes McKinney",
            "demand": 4,
            "priority": "high"
        },
        "Numpy": {
            "youtube": "https://www.youtube.com/results?search_query=numpy+tutorial+for+beginners",
            "books": "Guide to NumPy by Travis E. Oliphant",
            "demand": 3,
            "priority": "medium"
        },
        "Swift": {
            "youtube": "https://www.youtube.com/results?search_query=swift+ios+development+tutorial",
            "books": "Swift Programming: The Big Nerd Ranch Guide",
            "demand": 4,
            "priority": "high"
        },
        "Kotlin": {
            "youtube": "https://www.youtube.com/results?search_query=kotlin+android+development+tutorial",
            "books": "Kotlin in Action by Dmitry Jemerov",
            "demand": 4,
            "priority": "high"
        },
        "Flutter": {
            "youtube": "https://www.youtube.com/results?search_query=flutter+tutorial+full+course",
            "books": "Flutter in Action by Eric Windmill",
            "demand": 4,
            "priority": "high"
        },
        "React Native": {
            "youtube": "https://www.youtube.com/results?search_query=react+native+tutorial+for+beginners",
            "books": "React Native in Action by Nader Dabit",
            "demand": 4,
            "priority": "high"
        }
    }
    
    # 6. Generate recommendations list
    recommendations = []
    for skill in missing_skills:
        info = SKILL_RESOURCES.get(skill)
        if not info:
            matched_key = None
            for key in SKILL_RESOURCES:
                if key.lower() == skill.lower():
                    matched_key = key
                    break
            if matched_key:
                info = SKILL_RESOURCES[matched_key]
        
        if info:
            rec = {
                "skill": skill,
                "priority": info["priority"],
                "demand": info["demand"],
                "youtube": info["youtube"],
                "books": info["books"]
            }
        else:
            safe_query = quote_plus(f"{skill} tutorial")
            rec = {
                "skill": skill,
                "priority": "medium",
                "demand": 3,
                "youtube": f"https://www.youtube.com/results?search_query={safe_query}",
                "books": f"Learning {skill}: The Practical Guide"
            }
        recommendations.append(rec)
        
    return {"recommendations": recommendations}

