from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.pipeline.optimized_inference import analyze_video_optimized, get_optimization_stats
from src.utils.io import ensure_dir


st.set_page_config(
    page_title="DeepFake Detection | Advanced Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #155e75 50%, #1e3a8a 100%);
        border-radius: 14px;
        padding: 20px 24px;
        color: #f8fafc;
        margin-bottom: 18px;
    }
    .kpi {
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 14px 16px;
        background: #ffffff;
        color: #0f172a;
        min-height: 112px;
    }
    .kpi strong {color:#0f172a;}
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 700;
    }
</style>
""",
    unsafe_allow_html=True,
)


def _risk_badge(risk_level: str) -> str:
    risk = risk_level.lower()
    if risk == "high":
        return "<span class='pill' style='background:#fee2e2;color:#991b1b;'>HIGH RISK</span>"
    if risk == "medium":
        return "<span class='pill' style='background:#fef3c7;color:#92400e;'>MEDIUM RISK</span>"
    return "<span class='pill' style='background:#dcfce7;color:#166534;'>LOW RISK</span>"


def _friendly_reason(reason: str) -> str:
    mapping = {
        "Abnormal blink rhythm profile": "Eye blinking pattern looks unusual for natural speech.",
        "Audio-visual synchronization mismatch": "Lip movement does not line up well with the audio track.",
        "Visual artifact signals are elevated": "Visual texture/lighting patterns look potentially synthetic.",
        "Synthetic/watermark traces detected": "Possible generation traces or watermark-like patterns were found.",
        "Speech-language timing pattern is suspicious": "Speech timing pattern looks machine-like or overly regular.",
        "No single module exceeded suspicious threshold": "No strong red flags were found in any single check.",
    }
    return mapping.get(reason, reason)


def _friendly_module_names() -> dict[str, str]:
    return {
        "blink": "Blink Behavior",
        "lipsync": "Lip-Sync",
        "visual": "Visual Artifacts",
        "watermark": "Watermark/Trace",
        "nlp": "Speech Pattern",
    }


def _to_percent(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))) * 100.0, 1)


def _verdict_text(risk_level: str) -> str:
    risk = risk_level.lower()
    if risk == "high":
        return "Likely Manipulated"
    if risk == "medium":
        return "Needs Manual Review"
    return "Likely Authentic"


def _evidence_quality(payload: dict) -> tuple[str, str]:
    visual = payload["visual"]
    behavioral = payload["behavioral"]
    transcript = payload["transcript"]

    checks = 0
    available = 0

    checks += 1
    available += 1 if visual.get("face_path") else 0

    checks += 1
    available += 1 if visual.get("cnn_confidence") is not None else 0

    checks += 1
    lipsync_available = behavioral.get("lipsync_correlation", 0.0) > 0.01
    available += 1 if lipsync_available else 0

    checks += 1
    available += 1 if transcript.get("method") != "unavailable" else 0

    ratio = available / checks if checks else 0.0
    if ratio >= 0.75 and lipsync_available:
        return "High", "Analysis used strong signal coverage across modules."
    if ratio >= 0.5:
        return "Medium", "Some modules had limited signal quality; interpret with caution."
    return "Low", "Limited usable evidence was available; result confidence is reduced."


st.markdown(
    """
<div class="hero">
  <h2 style="margin:0 0 6px 0;">DeepFake Risk Analyzer</h2>
  <p style="margin:0;opacity:0.95;">Upload a video and get a clear, explainable risk summary with module-by-module evidence.</p>
</div>
""",
    unsafe_allow_html=True,
)

upload = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
processed_dir = Path("data/processed")
upload_dir = ensure_dir(Path("data/uploads"))

if upload is not None:
    save_path = upload_dir / upload.name
    save_path.write_bytes(upload.read())

    media_col, info_col = st.columns([1.6, 1])
    with media_col:
        st.video(str(save_path))
    with info_col:
        st.markdown("### Uploaded File")
        st.write(upload.name)
        st.caption(f"Saved to {save_path}")
        st.caption("Tip: Use clear speaking clips with visible face for better confidence.")

    run_btn = st.button("Run DeepFake Analysis", type="primary")
    if run_btn:
        with st.spinner("Running preprocessing, behavioral, visual, forensic, NLP, and scoring modules..."):
            result = run_phase_four_pipeline(save_path, processed_dir)
            payload = result.to_dict()

        scoring = payload["scoring"]
        module_scores = scoring["module_scores"].copy()
        display_names = _friendly_module_names()

        chart_data = {
            display_names.get(key, key): _to_percent(value)
            for key, value in module_scores.items()
        }

        st.subheader("Final Decision")
        evidence_level, evidence_msg = _evidence_quality(payload)
        risk_col, conf_col, face_col, evidence_col = st.columns([1, 1, 1, 1])
        with risk_col:
            st.markdown(
                "<div class='kpi'><strong>Verdict</strong><br/><br/>"
                + _risk_badge(scoring["risk_level"])
                + f"<div style='margin-top:10px;font-weight:700;'>{_verdict_text(scoring['risk_level'])}</div>"
                + "</div>",
                unsafe_allow_html=True,
            )
        with conf_col:
            st.markdown(
                f"<div class='kpi'><strong>Model Confidence</strong><br/><br/><span style='font-size:1.7rem;font-weight:800;'>{_to_percent(scoring['confidence_score'])}%</span></div>",
                unsafe_allow_html=True,
            )
        with face_col:
            face_detected = payload["visual"]["face_path"] is not None
            st.markdown(
                f"<div class='kpi'><strong>Face Detected</strong><br/><br/><span style='font-size:1.2rem;font-weight:700;'>{'Yes' if face_detected else 'No'}</span></div>",
                unsafe_allow_html=True,
            )
        with evidence_col:
            st.markdown(
                f"<div class='kpi'><strong>Evidence Quality</strong><br/><br/><span style='font-size:1.2rem;font-weight:700;'>{evidence_level}</span></div>",
                unsafe_allow_html=True,
            )

        st.caption(evidence_msg)

        st.subheader("Why This Result")
        if not scoring["reasons"]:
            st.write("- No strong warning signal crossed the decision threshold.")
        for reason in scoring["reasons"]:
            st.write(f"- {_friendly_reason(reason)}")

        st.subheader("Module Breakdown (Higher % means more suspicious)")
        st.bar_chart(chart_data)

        st.subheader("Simple Insights")
        behavioral = payload["behavioral"]
        visual = payload["visual"]
        transcript = payload["transcript"]

        simple_col_1, simple_col_2 = st.columns(2)
        with simple_col_1:
            st.write(f"- Blink events detected: {behavioral['blink_count']}")
            st.write(f"- Lip-sync correlation: {round(behavioral['lipsync_correlation'], 3)}")
            st.write(f"- Speech segments found: {transcript['speech_segments']}")
        with simple_col_2:
            st.write(f"- Lighting asymmetry score: {round(visual['lighting_asymmetry'] or 0.0, 2)}")
            st.write(f"- Sharpness score: {round(visual['sharpness_score'] or 0.0, 2)}")
            st.write(f"- Texture score: {round(visual['texture_score'] or 0.0, 2)}")
            cnn_fake = visual.get("cnn_fake_probability")
            cnn_conf = visual.get("cnn_confidence")
            st.write(
                f"- CNN fake probability: {('Not available' if cnn_fake is None else f'{_to_percent(cnn_fake)}%')}"
            )
            st.write(
                f"- CNN confidence quality: {('Not available' if cnn_conf is None else f'{_to_percent(cnn_conf)}%')}"
            )
            st.write(f"- Transcript method: {transcript['method']}")

        tab1, tab2 = st.tabs(["Readable Summary", "Technical JSON"])
        with tab1:
            st.markdown("#### Non-Technical Summary")
            if scoring["risk_level"].lower() == "high":
                st.warning("This video shows multiple strong signs of manipulation. Review manually before trusting it.")
            elif scoring["risk_level"].lower() == "medium":
                st.info("This video has some suspicious signals. Treat with caution and verify from source.")
            else:
                st.success("This video shows low risk signals in the current checks.")

            if evidence_level == "Low":
                st.warning("Evidence quality is low. Re-test with a clearer clip where face and speech are visible/audible.")

            st.markdown("#### What You Can Do Next")
            st.write("- Check original source and upload context.")
            st.write("- Test another clip from the same source for consistency.")
            st.write("- Compare with known real sample from the same speaker.")

        with tab2:
            # default=str prevents crashes if any Path/object escapes into the payload.
            st.code(json.dumps(payload, indent=2, default=str), language="json")

else:
    st.info("Upload a video file to start analysis.")
