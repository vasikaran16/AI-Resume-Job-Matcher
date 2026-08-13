import re
from datetime import datetime

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background: #f8fafc;
}

.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
}

.hero h1 {
    color: white;
    font-size: 2.7rem;
    margin: 0 0 0.5rem 0;
    font-weight: 700;
}

.hero p {
    color: #cbd5e1;
    font-size: 1.05rem;
    margin: 0;
}

/* Cards */
.section-card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    margin-bottom: 1rem;
}

.score-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    min-height: 145px;
    box-shadow: 0 5px 20px rgba(15, 23, 42, 0.06);
}

.score-number {
    color: #0f172a;
    font-size: 3rem;
    font-weight: 800;
    margin: 0.5rem 0;
}

.score-label {
    color: #64748b;
    font-size: 0.9rem;
    font-weight: 700;
}

.score-status {
    color: #475569;
    font-weight: 600;
}

.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem;
    min-height: 145px;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
}

.info-title {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 700;
}

.info-value {
    color: #0f172a;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 12px;
}

/* Skills */
.skill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 20px 0;
}

.skill-badge {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}

.missing-badge {
    display: inline-block;
    background: #fff7ed;
    color: #c2410c;
    border: 1px solid #fed7aa;
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding: 2rem 0 1rem 0;
    font-size: 0.9rem;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 48px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 12px;
}

/* Mobile */
@media (max-width: 768px) {
    .hero {
        padding: 1.5rem;
    }

    .hero h1 {
        font-size: 2rem;
    }

    .hero p {
        font-size: 0.95rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SKILL DATABASE
# ============================================================

SKILLS = [
    "python",
    "java",
    "c++",
    "javascript",
    "typescript",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "computer vision",
    "data science",
    "data analysis",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "keras",
    "opencv",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "xgboost",
    "catboost",
    "random forest",
    "streamlit",
    "flask",
    "fastapi",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "power bi",
    "tableau",
    "html",
    "css",
    "react",
    "rest api",
    "api",
    "nlp",
    "cnn",
    "rnn",
    "lstm",
    "transformers",
]


# ============================================================
# FUNCTIONS
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF."""

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def clean_text(text):
    """Clean text for NLP processing."""

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_skills(text):
    """Extract known technical skills."""

    text = clean_text(text)

    found_skills = []

    for skill in SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            found_skills.append(skill)

    return sorted(set(found_skills))


def calculate_similarity(resume_text, job_description):
    """Calculate TF-IDF cosine similarity."""

    resume_text = clean_text(resume_text)
    job_description = clean_text(job_description)

    documents = [
        resume_text,
        job_description,
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
    )

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2],
    )[0][0]

    return similarity * 100


def calculate_skill_score(resume_skills, job_skills):
    """Calculate technical skill match percentage."""

    if not job_skills:
        return 0

    matched = set(resume_skills).intersection(
        set(job_skills)
    )

    score = (
        len(matched) / len(job_skills)
    ) * 100

    return score


def calculate_final_score(
    similarity_score,
    skill_score,
):
    """Calculate weighted final score."""

    final_score = (
        similarity_score * 0.40
        + skill_score * 0.60
    )

    return round(final_score, 2)


def get_status(score):
    """Return match status."""

    if score >= 80:
        return "Excellent Match", "🟢"

    if score >= 65:
        return "Strong Match", "🟡"

    if score >= 50:
        return "Moderate Match", "🟠"

    return "Needs Improvement", "🔴"


def skill_badges(
    skills,
    badge_class="skill-badge",
):
    """Create HTML skill badges."""

    if not skills:
        return "<p>No skills detected.</p>"

    badges = ""

    for skill in skills:
        badges += (
            f'<span class="{badge_class}">'
            f"{skill}"
            f"</span>"
        )

    return (
        '<div class="skill-container">'
        f"{badges}"
        "</div>"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <h1>📄 AI Resume Matcher</h1>
    <p>
        Analyze your resume against a job description
        using Natural Language Processing and
        technical skill matching.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Analyze Your Resume")

input_col1, input_col2 = st.columns(
    2,
    gap="large",
)


with input_col1:

    st.markdown(
        """
<div class="section-card">
    <h3>📄 Resume</h3>
</div>
""",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload your resume as a PDF",
        type=["pdf"],
    )

    if uploaded_file:
        st.success(
            f"Uploaded: {uploaded_file.name}"
        )


with input_col2:

    st.markdown(
        """
<div class="section-card">
    <h3>💼 Job Description</h3>
</div>
""",
        unsafe_allow_html=True,
    )

    job_description = st.text_area(
        "Paste the job description",
        height=220,
        placeholder=(
            "Paste the complete job description here..."
        ),
        label_visibility="collapsed",
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
):

    if uploaded_file is None:

        st.warning(
            "Please upload your resume PDF."
        )

        st.stop()


    if not job_description.strip():

        st.warning(
            "Please paste a job description."
        )

        st.stop()


    with st.spinner(
        "Analyzing your resume..."
    ):

        try:

            resume_text = extract_text_from_pdf(
                uploaded_file
            )

            if not resume_text.strip():

                st.error(
                    "Could not extract text from "
                    "this PDF. Please use a text-based PDF."
                )

                st.stop()


            resume_skills = extract_skills(
                resume_text
            )

            job_skills = extract_skills(
                job_description
            )


            similarity_score = calculate_similarity(
                resume_text,
                job_description,
            )


            skill_score = calculate_skill_score(
                resume_skills,
                job_skills,
            )


            final_score = calculate_final_score(
                similarity_score,
                skill_score,
            )


            matched_skills = sorted(
                set(resume_skills)
                & set(job_skills)
            )


            missing_skills = sorted(
                set(job_skills)
                - set(resume_skills)
            )


        except Exception as error:

            st.error(
                f"Analysis error: {error}"
            )

            st.stop()


    # ========================================================
    # RESULTS
    # ========================================================

    st.success(
        "Resume analysis completed successfully."
    )

    st.divider()

    st.subheader(
        "📊 Resume Match Analysis"
    )


    status, status_icon = get_status(
        final_score
    )


    # ========================================================
    # SCORE CARDS
    # ========================================================

    score_col1, score_col2, score_col3, score_col4 = st.columns(
        4,
        gap="medium",
    )


    with score_col1:

        st.markdown(
            f"""
<div class="score-card">
    <div class="score-label">OVERALL SCORE</div>
    <div class="score-number">{final_score}%</div>
    <div class="score-status">
        {status_icon} {status}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )


    with score_col2:

        st.markdown(
            f"""
<div class="info-card">
    <div class="info-title">
        TECHNICAL SKILL MATCH
    </div>
    <div class="info-value">
        {skill_score:.1f}%
    </div>
</div>
""",
            unsafe_allow_html=True,
        )


    with score_col3:

        st.markdown(
            f"""
<div class="info-card">
    <div class="info-title">
        MATCHED SKILLS
    </div>
    <div class="info-value">
        {len(matched_skills)}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )


    with score_col4:

        st.markdown(
            f"""
<div class="info-card">
    <div class="info-title">
        MISSING SKILLS
    </div>
    <div class="info-value">
        {len(missing_skills)}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )


    st.write("")


    # ========================================================
    # MATCHED SKILLS
    # ========================================================

    st.markdown(
        "### ✅ Matched Skills"
    )

    st.markdown(
        skill_badges(matched_skills),
        unsafe_allow_html=True,
    )


    st.divider()


    # ========================================================
    # MISSING SKILLS
    # ========================================================

    st.markdown(
        "### ⚠️ Missing Skills"
    )


    if missing_skills:

        st.markdown(
            skill_badges(
                missing_skills,
                "missing-badge",
            ),
            unsafe_allow_html=True,
        )

    else:

        st.success(
            "Your resume contains all detected "
            "technical skills from the job description."
        )


    st.divider()


    # ========================================================
    # SCORE BREAKDOWN
    # ========================================================

    st.markdown(
        "### 📈 Score Breakdown"
    )


    breakdown_col1, breakdown_col2 = st.columns(
        2,
        gap="large",
    )


    with breakdown_col1:

        st.markdown(
            """
<div class="section-card">
    <b>NLP Similarity</b>
    <p>
        Measures overall text similarity between
        your resume and the job description.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.progress(
            min(
                int(similarity_score),
                100,
            )
        )

        st.caption(
            f"{similarity_score:.2f}%"
        )


    with breakdown_col2:

        st.markdown(
            """
<div class="section-card">
    <b>Technical Skill Match</b>
    <p>
        Measures how many detected job skills
        are present in your resume.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.progress(
            min(
                int(skill_score),
                100,
            )
        )

        st.caption(
            f"{skill_score:.2f}%"
        )


    st.divider()


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown(
        "### 💡 Resume Improvement Suggestions"
    )


    if missing_skills:

        st.info(
            "Consider highlighting these skills "
            "if you genuinely have experience with them."
        )

        st.markdown(
            skill_badges(
                missing_skills,
                "missing-badge",
            ),
            unsafe_allow_html=True,
        )


    if final_score < 50:

        st.warning(
            "Your resume has relatively low alignment "
            "with this job. Consider improving your "
            "project descriptions and relevant skills."
        )


    elif final_score < 65:

        st.info(
            "Your resume has moderate alignment. "
            "Strengthen your technical skills section "
            "and add relevant project keywords naturally."
        )


    elif final_score < 80:

        st.success(
            "Your resume has strong alignment. "
            "Focus on measurable project achievements "
            "and clearly presenting your relevant skills."
        )


    else:

        st.success(
            "Excellent alignment! Your resume strongly "
            "matches the detected requirements."
        )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.divider()

    st.markdown(
        "### 📥 Download Analysis"
    )


    report_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )


    report = f"""
AI RESUME & JOB DESCRIPTION MATCHER
====================================

Analysis Date:
{report_time}

Overall Match Score:
{final_score}%

Status:
{status}

NLP Similarity:
{similarity_score:.2f}%

Technical Skill Match:
{skill_score:.2f}%

Matched Skills:
{", ".join(matched_skills) if matched_skills else "None"}

Missing Skills:
{", ".join(missing_skills) if missing_skills else "None"}

Resume Improvement Suggestions:
- Review the missing skills section.
- Highlight relevant skills if you genuinely have experience.
- Add measurable achievements to relevant projects.
- Align project descriptions with the target role.

Note:
This tool provides an automated similarity analysis
and should be used as a supporting tool rather than
a definitive hiring decision.
"""


    st.download_button(
        label="📥 Download Analysis Report",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
**AI Resume & Job Description Matcher**

Built with Python • Streamlit • Scikit-learn • NLP
"""
)