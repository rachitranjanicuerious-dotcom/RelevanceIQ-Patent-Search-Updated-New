"""
HTML templates for RelevanceIQ.

Every function returns an HTML string. Rendering is done in app.py
through render_html(), so this module contains markup only.
"""

from pathlib import Path


# ============================================================
# CSS LOADER
# ============================================================

def load_css(path="styles.css"):
    """Read styles.css (next to this file) and wrap it in <style> tags."""

    css_path = Path(__file__).parent / path
    css = css_path.read_text(encoding="utf-8")

    return f"<style>\n{css}\n</style>"


# ============================================================
# LOGIN
# ============================================================

def login_card():
    return """
    <div style="
        max-width:520px;
        margin:70px auto 25px auto;
        background:#FFFFFF;
        border:1px solid #DEDEDE;
        border-top:4px solid #F4512A;
        padding:45px;
        border-radius:4px;
    ">

        <div style="
            font-size:34px;
            font-weight:700;
            color:#071426;
        ">
            iCuerious<span style="color:#F4512A;">™</span>
        </div>

        <div style="
            font-size:11px;
            color:#777777;
            letter-spacing:1px;
            margin-top:-2px;
            margin-bottom:35px;
        ">
            FINDING WHAT MATTERS MOST
        </div>

        <div style="
            font-size:25px;
            font-weight:600;
            color:#071426;
            border-left:3px solid #F4512A;
            padding-left:12px;
            margin-bottom:8px;
        ">
            RelevanceIQ
        </div>

        <div style="
            font-size:13px;
            color:#6B7280;
            margin-bottom:5px;
        ">
            Patent Relevance Analysis
        </div>

    </div>
    """


# ============================================================
# SIDEBAR
# ============================================================

def sidebar_brand():
    return """
    <div class="sidebar-brand">

        <div class="sidebar-brand-main">
            iCuerious<span>™</span>
        </div>

        <div class="sidebar-brand-sub">
            FINDING WHAT MATTERS MOST
        </div>

    </div>
    """


def sidebar_section(label):
    return f"""
    <div class="sidebar-section">
        {label}
    </div>
    """


def sidebar_tool():
    return """
    <div class="sidebar-section">
        Tool
    </div>

    <div style="
        color:#FFFFFF;
        font-size:15px;
        font-weight:600;
    ">
        RelevanceIQ
    </div>

    <div style="
        color:#8997A8;
        font-size:11px;
        margin-top:4px;
        line-height:1.5;
    ">
        Patent relevance and FTO analysis
    </div>
    """


def sidebar_user(username):
    return f"""
    <div class="sidebar-user">

        <div class="sidebar-user-label">
            Logged in as
        </div>

        <div class="sidebar-user-name">
            {username}
        </div>

    </div>
    """


# ============================================================
# HEADER / HERO
# ============================================================

def top_header():
    return """
    <div class="top-header">

        <div>

            <div class="brand-name">
                iCuerious<span>™</span>
            </div>

            <div class="brand-tagline">
                FINDING WHAT MATTERS MOST
            </div>

        </div>

        <div class="product-name">
            RelevanceIQ
        </div>

    </div>
    """


def hero(kicker, title, description):
    return f"""
    <div class="hero">

        <div class="hero-kicker">
            {kicker}
        </div>

        <div class="hero-title">
            {title}
        </div>

        <div class="hero-description">
            {description}
        </div>

    </div>
    """


# ============================================================
# SECTIONS
# ============================================================

def section(number, title, description=None):
    """Numbered section header, with optional description underneath."""

    html = f"""
    <div class="section-header">

        <div class="section-number">
            {number}
        </div>

        <div class="section-title">
            {title}
        </div>

    </div>
    """

    if description:
        html += f"""
    <div class="section-description">
        {description}
    </div>
    """

    return html


# ============================================================
# CARDS
# ============================================================

def info_card(title, value):
    return f"""
    <div class="info-card">

        <div class="info-card-title">
            {title}
        </div>

        <div class="info-card-value">
            {value}
        </div>

    </div>
    """


def status_card(title, text):
    return f"""
    <div class="status-card">

        <div class="status-title">
            {title}
        </div>

        <div class="status-text">
            {text}
        </div>

    </div>
    """


def input_info():
    return """
    <div class="input-info">

        Expected fields include
        <b>Publication Number</b>,
        <b>Title</b>,
        <b>Abstract</b>,
        <b>Independent Claim</b> and
        <b>All Claims</b>.

    </div>
    """


# ============================================================
# FOOTER
# ============================================================

def footer():
    return """
    <div class="footer">

        RelevanceIQ &nbsp; | &nbsp;
        iCuerious Patent Intelligence

        <br><br>

        Patent relevance analysis • FTO support •
        Technology intelligence

    </div>
    """
