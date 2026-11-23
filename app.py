import html
import io
import os
import tempfile
import zipfile
from fractions import Fraction
from typing import Literal

import geopandas as gpd
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Page Config & Obsidian / Copper / Subtle Gold Styling + Hover Tooltips
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Texas O&G Landman Workbench | Black Mineral Consulting",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Base layout & Obsidian Black canvas */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1420px;
    }
    section[data-testid="stSidebar"] {
        background-color: #0D0D10;
        border-right: 1px solid rgba(184, 115, 51, 0.25);
    }

    /* Hero Banner: Obsidian + Burnished Copper + Subtle Gold Edge */
    .hero-banner {
        background: linear-gradient(135deg, #0E0E11 0%, #171412 55%, #23170F 100%);
        border: 1px solid rgba(184, 115, 51, 0.45);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.65);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .hero-banner:hover {
        border-color: rgba(229, 193, 88, 0.7);
        box-shadow: 0 12px 32px rgba(229, 193, 88, 0.08);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(184, 115, 51, 0.16);
        color: #E5C158;
        border: 1px solid rgba(229, 193, 88, 0.45);
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FAFAFA;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 0.96rem;
        color: #A1A1AA;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }

    /* Interactive Hover-Tooltip Panels & KPI Cards */
    .hover-panel {
        position: relative;
        background: linear-gradient(180deg, #141417 0%, #0F0F12 100%);
        border: 1px solid rgba(184, 115, 51, 0.32);
        border-radius: 11px;
        padding: 0.95rem 1.15rem;
        transition: all 0.22s ease;
        cursor: help;
    }
    .hover-panel:hover {
        border-color: #E5C158;
        box-shadow: 0 0 18px rgba(229, 193, 88, 0.16);
        transform: translateY(-2px);
    }
    .hover-panel::after {
        content: attr(data-tooltip);
        position: absolute;
        left: 50%;
        bottom: calc(100% + 10px);
        transform: translateX(-50%) translateY(4px);
        width: max-content;
        max-width: 280px;
        background: #181614;
        color: #F4F4F5;
        border: 1px solid #E5C158;
        border-radius: 8px;
        padding: 0.55rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 400;
        line-height: 1.38;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.85);
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease, transform 0.2s ease;
        z-index: 9999;
    }
    .hover-panel:hover::after {
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #A1A1AA;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #FAFAFA;
        line-height: 1.2;
    }
    .kpi-accent {
        color: #E5C158;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }

    /* Section Header Bar with Hover Explanation */
    .section-guide {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #141417;
        border-left: 3px solid #B87333;
        border-top: 1px solid rgba(184, 115, 51, 0.25);
        border-right: 1px solid rgba(184, 115, 51, 0.25);
        border-bottom: 1px solid rgba(184, 115, 51, 0.25);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.9rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        cursor: help;
    }
    .section-guide:hover {
        border-color: #E5C158;
        box-shadow: 0 0 16px rgba(229, 193, 88, 0.14);
    }
    .section-guide-title {
        font-weight: 700;
        font-size: 1.02rem;
        color: #FAFAFA;
    }
    .section-guide-hint {
        font-size: 0.78rem;
        color: #E5C158;
        font-weight: 500;
    }
    .section-guide::after {
        content: attr(data-tooltip);
        position: absolute;
        right: 1rem;
        top: calc(100% + 8px);
        max-width: 420px;
        background: #181614;
        color: #F4F4F5;
        border: 1px solid #E5C158;
        border-radius: 8px;
        padding: 0.65rem 0.9rem;
        font-size: 0.82rem;
        line-height: 1.42;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.88);
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease;
        z-index: 9999;
    }
    .section-guide:hover::after {
        opacity: 1;
    }

    /* Streamlit Bordered Containers -> Copper & Subtle Gold Hover */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #121215;
        border: 1px solid rgba(184, 115, 51, 0.3) !important;
        border-radius: 12px !important;
        transition: border-color 0.22s ease, box-shadow 0.22s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(229, 193, 88, 0.65) !important;
        box-shadow: 0 0 20px rgba(229, 193, 88, 0.1);
    }

    /* Navigation Tabs: Obsidian & Copper Active State + Gold Hover */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #121215;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(184, 115, 51, 0.3);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #A1A1AA;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #E5C158;
        background-color: rgba(229, 193, 88, 0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #B87333 0%, #9A5B22 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(229, 193, 88, 0.5) !important;
    }

    /* Primary Copper Buttons with Subtle Gold Hover Glow */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #B87333 0%, #9A5B22 100%);
        border: 1px solid rgba(229, 193, 88, 0.45);
        color: #FFFFFF;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        border-color: #E5C158;
        box-shadow: 0 0 16px rgba(229, 193, 88, 0.32);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hover_kpi(label: str, value: str, subtext: str, tooltip: str):
    safe_tip = html.escape(tooltip, quote=True)
    st.markdown(
        f"""
        <div class="hover-panel" data-tooltip="{safe_tip}">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(value)}</div>
            <div class="kpi-accent">{html.escape(subtext)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, tooltip: str):
    safe_tip = html.escape(tooltip, quote=True)
    st.markdown(
        f"""
        <div class="section-guide" data-tooltip="{safe_tip}">
            <span class="section-guide-title">{html.escape(title)}</span>
            <span class="section-guide-hint">ⓘ Hover for panel guide</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 2. Pydantic Schema for Texas Oil & Gas Instruments
# ---------------------------------------------------------------------------
class Party(BaseModel):
    name: str = Field(..., description="Exact party name as written in the instrument")
    role: Literal["Grantor", "Grantee", "Lessor", "Lessee", "Assignor", "Assignee", "Other"]
    address_or_county: str | None = Field(default=None, description="Recited address, city, state, or county")


class LegalDescription(BaseModel):
    county: str | None = Field(default=None, description="Texas county name")
    abstract_number: str | None = Field(default=None, description="Abstract number if stated")
    survey_name: str | None = Field(default=None, description="Survey name (e.g., Abstract/Survey)")
    block: str | None = Field(default=None, description="Block designation if stated")
    section: str | None = Field(default=None, description="Section or lot number if stated")
    acreage_gross: float | None = Field(default=None, description="Recited gross acreage")
    acreage_net: float | None = Field(default=None, description="Recited net mineral/lease acreage if stated")
    metes_and_bounds_summary: str | None = Field(
        default=None, description="Concise summary of metes-and-bounds or tract callouts"
    )


class DepthAndPughClauses(BaseModel):
    depth_limitation_text: str | None = Field(
        default=None, description="Verbatim depth severance or vertical limit language, if any"
    )
    pugh_clause_present: bool | None = Field(
        default=None, description="True if a Pugh clause or horizontal/vertical severance at lease expiration appears"
    )
    pugh_clause_summary: str | None = Field(
        default=None, description="Summary of Pugh or pooling/unitization severance terms"
    )


class RoyaltyAndTerm(BaseModel):
    royalty_fraction_text: str | None = Field(
        default=None, description="Recited royalty fraction as written (e.g., '1/8', '1/5', '1/4')"
    )
    primary_term_text: str | None = Field(
        default=None, description="Primary term duration and any renewal/extension option text"
    )
    verbatim_granting_clause: str | None = Field(
        default=None, description="Exact granting clause quote for human landman verification"
    )


class TexasOilGasInstrument(BaseModel):
    instrument_type: str = Field(
        ..., description="e.g., Oil and Gas Lease, Memorandum of Lease, Assignment, Mineral Deed, Royalty Deed"
    )
    execution_date: str | None = Field(default=None, description="Date signed/executed (YYYY-MM-DD or as written)")
    recording_date: str | None = Field(default=None, description="Date filed/recorded in county clerk records")
    volume_page_or_instrument_no: str | None = Field(
        default=None, description="Volume/Page or Clerk's File / Instrument Number"
    )
    parties: list[Party] = Field(default_factory=list)
    legal_descriptions: list[LegalDescription] = Field(default_factory=list)
    royalty_and_term: RoyaltyAndTerm = Field(default_factory=RoyaltyAndTerm)
    depth_and_pugh: DepthAndPughClauses = Field(default_factory=DepthAndPughClauses)
    uncertain_fields: list[str] = Field(
        default_factory=list,
        description="List any field names where handwriting, faded ink, or ambiguity made extraction uncertain",
    )


EXTRACTION_SYSTEM_PROMPT = """
You are a precision landman document-extraction assistant reviewing recorded Texas oil and gas instruments.
Extract only what is supported by the provided document pages. Never guess or infer missing dates, abstract numbers, fractions, or clauses.
- Preserve exact names, fractions, dates, and recording references as written.
- Quote granting clauses and depth limitations verbatim so a human landman can audit every call.
- If handwriting, low scan quality, or ambiguous phrasing prevents confident extraction of a field, set that field to null and add its field name to `uncertain_fields`.
- Treat this output as a preliminary run-sheet aid, not a final legal title opinion.
""".strip()


# ---------------------------------------------------------------------------
# 3. Geospatial, Title Math & Helper Functions
# ---------------------------------------------------------------------------
def extract_zip_recursive(zip_bytes: bytes, dest_dir: str):
    """Unpacks a zip file and any inner .zip files (common in RRC bulk downloads)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest_dir)
    for root, _, files in os.walk(dest_dir):
        for fname in files:
            if fname.lower().endswith(".zip"):
                nested_path = os.path.join(root, fname)
                nested_out = os.path.join(root, os.path.splitext(fname)[0])
                os.makedirs(nested_out, exist_ok=True)
                try:
                    with zipfile.ZipFile(nested_path) as nzf:
                        nzf.extractall(nested_out)
                except zipfile.BadZipFile:
                    pass


def inspect_uploaded_archive(file_name: str, file_bytes: bytes) -> tuple[list[str], list[str]]:
    """Returns (shapefile_paths_inside_zip, all_file_extensions) for UI selection."""
    if not file_name.lower().endswith(".zip"):
        return [file_name], [os.path.splitext(file_name)[1].lower()]

    shp_names = []
    exts = set()
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX") or name.endswith("/"):
                continue
            ext = os.path.splitext(name)[1].lower()
            exts.add(ext)
            if ext == ".shp":
                shp_names.append(name)
            elif ext == ".zip":
                try:
                    inner_bytes = zf.read(name)
                    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as izf:
                        for iname in izf.namelist():
                            if not iname.startswith("__MACOSX") and iname.lower().endswith(".shp"):
                                shp_names.append(f"{name} -> {iname}")
                except Exception:
                    pass
    return sorted(shp_names), sorted(exts)


def load_batch_geodata_from_upload(
    uploaded_file,
    selected_layers: list[str] | None = None,
    fallback_crs: str = "EPSG:4267",
    target_crs: str = "EPSG:3083",
) -> gpd.GeoDataFrame:
    """Loads one or multiple selected shapefiles from an upload, aligns CRS, and concatenates them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        if suffix != ".zip":
            raw_path = os.path.join(tmpdir, uploaded_file.name)
            with open(raw_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            gdf = gpd.read_file(raw_path)
            if gdf.crs is None:
                gdf = gdf.set_crs(fallback_crs)
            return gdf.to_crs(target_crs)

        extract_zip_recursive(uploaded_file.getvalue(), tmpdir)

        all_shps = []
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith(".shp") and not fname.startswith("._"):
                    all_shps.append(os.path.join(root, fname))

        if not all_shps:
            raise ValueError("No .shp shapefile found inside this zip.")

        chosen_paths = []
        if selected_layers:
            target_basenames = {os.path.basename(lbl.split("->")[-1].strip()) for lbl in selected_layers}
            chosen_paths = [p for p in all_shps if os.path.basename(p) in target_basenames]

        if not chosen_paths:
            chosen_paths = [all_shps[0]]

        gdfs = []
        for shp_path in chosen_paths:
            part_gdf = gpd.read_file(shp_path)
            if part_gdf.crs is None:
                part_gdf = part_gdf.set_crs(fallback_crs)
            part_gdf = part_gdf.to_crs(target_crs)
            part_gdf["SOURCE_LAYER"] = os.path.basename(shp_path)
            gdfs.append(part_gdf)

        combined = pd.concat(gdfs, ignore_index=True)
        return gpd.GeoDataFrame(combined, geometry="geometry", crs=target_crs)


def run_batch_spatial_join(
    wells_gdf: gpd.GeoDataFrame,
    surveys_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Intersects aligned wells GeoDataFrame with surveys GeoDataFrame."""
    wells_clean = wells_gdf[wells_gdf.geometry.is_valid & ~wells_gdf.geometry.is_empty]
    surveys_clean = surveys_gdf[surveys_gdf.geometry.is_valid & ~surveys_gdf.geometry.is_empty]
    joined = gpd.sjoin(wells_clean, surveys_clean, how="left", predicate="intersects")
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def parse_fraction_or_decimal(val_str: str) -> float:
    """Parses strings like '1/8', '3/16', '0.25', or '25%' into a float."""
    cleaned = val_str.strip()
    if not cleaned:
        return 0.0
    if cleaned.endswith("%"):
        return float(cleaned[:-1].strip()) / 100.0
    if "/" in cleaned:
        return float(Fraction(cleaned))
    return float(cleaned)


def format_texasfile_name(full_name: str) -> str:
    """Converts 'Michael Smith' to 'Smith Michael' for TexasFile name searches."""
    parts = [p.strip() for p in full_name.replace(",", " ").split() if p.strip()]
    corp_tokens = {
        "INC", "LLC", "LP", "LLP", "LTD", "CORP", "CORPORATION", "COMPANY", "CO",
        "ENERGY", "OIL", "GAS", "RESOURCES", "TRUST", "ESTATE", "ENTERPRISE",
        "ENTERPRISES", "PARTNERS", "PARTNERSHIP", "HOLDINGS", "GROUP",
        "PETROLEUM", "PRODUCTION", "EXPLORATION", "MINERALS", "ASSOCIATES",
    }
    if any(p.upper().strip(".") in corp_tokens for p in parts) or len(parts) < 2:
        return full_name.strip()
    return f"{parts[-1]} {' '.join(parts[:-1])}"


def normalize_rrc_api(api_raw: str) -> str:
    """Strips hyphens and leading '42' Texas state code to get the 8-digit RRC GIS Viewer API."""
    digits = "".join(ch for ch in api_raw if ch.isdigit())
    if len(digits) >= 10 and digits.startswith("42"):
        return digits[2:10]
    return digits[:8]


def extract_instrument_with_gemini(
    file_bytes: bytes,
    mime_type: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> TexasOilGasInstrument:
    """Calls Gemini multimodal structured outputs to parse a PDF/image instrument."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            "Extract the structured Texas oil and gas title fields from this recorded instrument.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=TexasOilGasInstrument,
            temperature=0.0,
        ),
    )
    return response.parsed


# ---------------------------------------------------------------------------
# 4. Session State Initialization
# ---------------------------------------------------------------------------
if "crosswalk_df" not in st.session_state:
    st.session_state.crosswalk_df = None
if "layers_joined_count" not in st.session_state:
    st.session_state.layers_joined_count = 0
if "extracted_instruments" not in st.session_state:
    st.session_state.extracted_instruments = []
if "doi_owners" not in st.session_state:
    st.session_state.doi_owners = [
        {"Owner Name": "Example Mineral Owner", "Mineral Fraction": "1/4", "Leased": True, "Lease Royalty": "1/5"}
    ]

# ---------------------------------------------------------------------------
# 5. Sidebar Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Workbench Settings")
    api_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        placeholder="Paste AIza... key here",
        help="Required for AI Deed & Lease extraction in Tab 2.",
    )
    if api_key:
        st.success("AI Extraction Engine Ready", icon="✅")
    else:
        st.info("Add API key above to unlock Tab 2 AI extraction.", icon="🔑")

    model_choice = st.selectbox(
        "Document Vision Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        format_func=lambda m: "Gemini 2.5 Flash (Fast • Free Tier)" if "flash" in m else "Gemini 2.5 Pro (Complex Cursive/Deeds)",
    )

    with st.expander("Advanced GIS Projections", expanded=False):
        fallback_crs = st.text_input("Default Legacy Datum", value="EPSG:4267", help="NAD27 standard for Texas RRC files")
        target_crs = st.text_input("Planar Join Projection", value="EPSG:3083", help="Texas Centric Albers Equal Area")

    st.divider()
    st.markdown("### 🧭 Texas Research Portals")
    st.markdown("- [RRC Public GIS Viewer](https://gis.rrc.texas.gov/GISViewer/)")
    st.markdown("- [RRC Shapefile & Data Downloads](https://www.rrc.texas.gov/resource-center/research/data-sets-available-for-download/)")
    st.markdown("- [TexasFile County Clerk Search](https://www.texasfile.com/)")
    st.markdown("- [Texas GLO Land Grant Search](https://s3.glo.texas.gov/ncu/SCANDOCS/archives_webfiles/arcmaps/webfiles/landgrants/LandGrantsWorklist.html)")

# ---------------------------------------------------------------------------
# 6. Hero Header & Live Hover-Tooltip KPI Strip
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">Black Mineral Consulting • Title &amp; GIS Suite</div>
        <h1 class="hero-title">Texas Oil &amp; Gas Landman Workbench</h1>
        <p class="hero-subtitle">Spatial Well-to-Survey Crosswalks • AI Deed &amp; Lease Extraction • Custom Run Sheets • Division of Interest Math</p>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
well_count = len(st.session_state.crosswalk_df) if st.session_state.crosswalk_df is not None else 0
inst_count = len(st.session_state.extracted_instruments)

with kpi1:
    render_hover_kpi(
        label="Crosswalk Records",
        value=f"{well_count:,}",
        subtext="RRC Wells Mapped",
        tooltip="Total well-to-survey intersections generated in Tab 1. Used to link API numbers to Texas Abstracts and Surveys.",
    )
with kpi2:
    render_hover_kpi(
        label="Layers Joined",
        value=str(st.session_state.layers_joined_count),
        subtext="County Shapefiles",
        tooltip="Number of county well or survey shapefile layers merged in your current spatial crosswalk session.",
    )
with kpi3:
    render_hover_kpi(
        label="Parsed Docs",
        value=str(inst_count),
        subtext="Deeds & Leases",
        tooltip="Count of recorded county clerk PDFs or page scans extracted by AI in Tab 2 and ready for run-sheet export.",
    )
with kpi4:
    render_hover_kpi(
        label="AI Engine",
        value="Connected" if api_key else "Standby",
        subtext="Gemini Vision OCR",
        tooltip="Shows whether your Gemini API key is active in the sidebar to read scanned deeds, leases, and cursive handwriting.",
    )

st.write("")

# ---------------------------------------------------------------------------
# 7. Main Navigation Tabs
# ---------------------------------------------------------------------------
tab_spatial, tab_extract, tab_crossref, tab_lookup, tab_math = st.tabs(
    [
        "🗺️ 1. Spatial Crosswalk",
        "📄 2. AI Deed & Lease Extractor",
        "📋 3. Run Sheet & Audit",
        "🔍 4. API & TexasFile Lookup",
        "🧮 5. NMA / NRI Calculator",
    ]
)

# --- TAB 1: Multi-County Spatial-to-Survey Join ---
with tab_spatial:
    with st.container(border=True):
        render_section_header(
            title="Step 1: Upload RRC Well & Texas Survey Shapefiles",
            tooltip="Spatial Crosswalk Panel: Intersects Texas Railroad Commission well coordinates (NAD27/NAD83) against Texas GLO/RRC Survey polygons to map every API number to its Abstract, Survey, Block, and Section.",
        )
        st.caption("Keep RRC downloads as `.zip` archives — the workbench automatically unpacks nested county folders.")

        col1, col2 = st.columns(2, gap="large")
        wells_selected_shps = []
        surveys_selected_shps = []

        with col1:
            wells_file = st.file_uploader(
                "RRC Wells Archive (.zip or .geojson)",
                type=["zip", "geojson"],
                help="Upload the RRC Well Layers by County zip file.",
            )
            if wells_file is not None:
                shp_list, exts = inspect_uploaded_archive(wells_file.name, wells_file.getvalue())
                if len(shp_list) > 1:
                    join_all_wells = st.checkbox(
                        f"Select all {len(shp_list)} well layers in archive", value=False, key="all_w"
                    )
                    if join_all_wells:
                        wells_selected_shps = shp_list
                    else:
                        default_w = [s for s in shp_list if s.lower().endswith("b.shp")][:1] or [shp_list[0]]
                        wells_selected_shps = st.multiselect(
                            "Choose county well layer(s) (`b.shp` = bottomhole, `s.shp` = surface):",
                            options=shp_list,
                            default=default_w,
                        )
                elif len(shp_list) == 1:
                    wells_selected_shps = shp_list
                    st.success(f"Ready: `{shp_list[0]}`")
                else:
                    st.error(f"No `.shp` shapefile found in `{wells_file.name}` (found: {', '.join(exts)}).")

        with col2:
            surveys_file = st.file_uploader(
                "Texas Survey Archive (.zip or .geojson)",
                type=["zip", "geojson"],
                help="Upload the RRC or Texas GLO Survey Layers zip file.",
            )
            if surveys_file is not None:
                shp_list_s, exts_s = inspect_uploaded_archive(surveys_file.name, surveys_file.getvalue())
                if len(shp_list_s) > 1:
                    poly_layers = [s for s in shp_list_s if s.lower().endswith("p.shp")]
                    join_all_polys = st.checkbox(
                        f"Select all {len(poly_layers) or len(shp_list_s)} survey polygon layers",
                        value=False,
                        key="all_s",
                    )
                    if join_all_polys:
                        surveys_selected_shps = poly_layers or shp_list_s
                    else:
                        default_s = poly_layers[:1] or [shp_list_s[0]]
                        surveys_selected_shps = st.multiselect(
                            "Choose survey polygon layer(s) (usually ends in `p.shp`):",
                            options=shp_list_s,
                            default=default_s,
                        )
                elif len(shp_list_s) == 1:
                    surveys_selected_shps = shp_list_s
                    st.success(f"Ready: `{shp_list_s[0]}`")
                else:
                    st.error(f"No `.shp` shapefile found in `{surveys_file.name}` (found: {', '.join(exts_s)}).")

        st.write("")
        btn_col, _ = st.columns([1, 2])
        with btn_col:
            run_join_clicked = st.button("⚡ Run Spatial Join", type="primary", width="stretch")

        if run_join_clicked:
            if wells_file is None or surveys_file is None:
                st.warning("Upload both a Wells archive and a Survey archive above before running the join.")
            else:
                with st.spinner("Aligning NAD27/NAD83 coordinates and intersecting wells against survey polygons..."):
                    try:
                        wells_gdf = load_batch_geodata_from_upload(
                            wells_file, wells_selected_shps, fallback_crs, target_crs
                        )
                        surveys_gdf = load_batch_geodata_from_upload(
                            surveys_file, surveys_selected_shps, fallback_crs, target_crs
                        )
                        crosswalk_df = run_batch_spatial_join(wells_gdf, surveys_gdf)
                        st.session_state.crosswalk_df = crosswalk_df
                        st.session_state.layers_joined_count = max(
                            len(wells_selected_shps), len(surveys_selected_shps), 1
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Spatial join error: {exc}")

    if st.session_state.crosswalk_df is not None:
        df = st.session_state.crosswalk_df
        with st.container(border=True):
            render_section_header(
                title="Crosswalk Results & Instant Search",
                tooltip="Crosswalk Results Panel: Filter your joined well-to-survey table by API number, Abstract, or Survey name and export the full dataset to CSV.",
            )
            search_col, dl_col = st.columns([2, 1])
            with search_col:
                quick_filter = st.text_input(
                    "Filter crosswalk table (search by API, Abstract, Survey Name, or County):",
                    placeholder="Type e.g. 00132761 or Abstract number...",
                )
            filtered_df = df
            if quick_filter.strip():
                mask = df.astype(str).apply(lambda col: col.str.contains(quick_filter.strip(), case=False, na=False)).any(axis=1)
                filtered_df = df[mask]

            with dl_col:
                st.write("")
                st.write("")
                csv_data = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇️ Download CSV ({len(filtered_df):,} rows)",
                    data=csv_data,
                    file_name="well_to_survey_crosswalk.csv",
                    mime="text/csv",
                    width="stretch",
                )

            st.dataframe(filtered_df.head(500), width="stretch", height=420)
            if len(filtered_df) > 500:
                st.caption(f"Previewing first 500 of {len(filtered_df):,} matching records. Use Download CSV for the complete table.")

# --- TAB 2: Vision-LLM Lease & Deed Extraction ---
with tab_extract:
    with st.container(border=True):
        render_section_header(
            title="Batch Parse Recorded County Clerk PDFs & Page Scans",
            tooltip="AI Document Extractor Panel: Uses multimodal Vision AI to read scanned Oil & Gas Leases, Memorandums, Assignments, and Deeds—extracting parties, recording info, legal descriptions, royalty fractions, and verbatim Pugh/depth clauses.",
        )
        st.caption("Upload one or multiple PDFs/scans. Uncertain handwriting is automatically flagged for curative review.")

        uploaded_docs = st.file_uploader(
            "Drag and drop Oil & Gas Leases, Memorandums, Assignments, or Mineral Deeds",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        b_col, _ = st.columns([1, 2])
        with b_col:
            extract_clicked = st.button(
                "✨ Extract Title Fields",
                type="primary",
                disabled=not uploaded_docs,
                width="stretch",
            )

        if extract_clicked:
            if not api_key:
                st.warning("Enter your Gemini API Key in the left sidebar first.")
            else:
                results = []
                progress = st.progress(0, text="Starting AI document review...")
                for idx, doc in enumerate(uploaded_docs):
                    progress.progress(idx / len(uploaded_docs), text=f"Reading {doc.name}...")
                    try:
                        parsed_obj = extract_instrument_with_gemini(
                            file_bytes=doc.getvalue(),
                            mime_type=doc.type or "application/pdf",
                            api_key=api_key,
                            model_name=model_choice,
                        )
                        results.append({"file_name": doc.name, "data": parsed_obj})
                    except Exception as exc:
                        st.error(f"Could not parse {doc.name}: {exc}")
                progress.progress(1.0, text="Extraction complete!")
                st.session_state.extracted_instruments = results
                st.rerun()

    if st.session_state.extracted_instruments:
        for item in st.session_state.extracted_instruments:
            inst: TexasOilGasInstrument = item["data"]
            with st.container(border=True):
                header_l, header_r = st.columns([3, 1])
                with header_l:
                    st.markdown(f"### 📄 {inst.instrument_type} — `{item['file_name']}`")
                with header_r:
                    st.caption(f"Recording Ref: **{inst.volume_page_or_instrument_no or 'Unstated'}**")

                if inst.uncertain_fields:
                    st.warning(f"Curative Note — Verify handwriting/scan on: **{', '.join(inst.uncertain_fields)}**")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Execution Date", inst.execution_date or "N/A")
                m2.metric("Recording Date", inst.recording_date or "N/A")
                m3.metric("Royalty", inst.royalty_and_term.royalty_fraction_text or "N/A")
                m4.metric("Pugh Clause", "Present" if inst.depth_and_pugh.pugh_clause_present else "None Detected")

                p_col, l_col = st.columns(2)
                with p_col:
                    st.markdown("**Parties**")
                    st.dataframe([p.model_dump() for p in inst.parties], width="stretch", hide_index=True)
                with l_col:
                    st.markdown("**Legal Descriptions**")
                    st.dataframe([ld.model_dump() for ld in inst.legal_descriptions], width="stretch", hide_index=True)

                if inst.depth_and_pugh.depth_limitation_text:
                    st.info(f"**Verbatim Depth Clause:** {inst.depth_and_pugh.depth_limitation_text}")

                if inst.royalty_and_term.verbatim_granting_clause:
                    with st.expander("View Verbatim Granting Clause & Raw JSON"):
                        st.text_area(
                            "Granting Clause",
                            value=inst.royalty_and_term.verbatim_granting_clause,
                            height=110,
                            disabled=True,
                            key=f"grant_{item['file_name']}",
                        )
                        st.json(inst.model_dump())

# --- TAB 3: Custom Run Sheet & Cross-Reference Audit ---
with tab_crossref:
    if not st.session_state.extracted_instruments:
        with st.container(border=True):
            render_section_header(
                title="Client Run-Sheet Builder & Well Cross-Reference",
                tooltip="Run Sheet & Audit Panel: Combines your AI-extracted instruments from Tab 2 with your RRC well crosswalk from Tab 1, flags depth severances or Pugh clauses, and exports a customizable CSV run sheet.",
            )
            st.info(
                "Parse at least one instrument in **Tab 2 (AI Deed & Lease Extractor)** to build your run sheet. "
                "If you also ran **Tab 1 (Spatial Crosswalk)**, matching RRC well counts and APIs will attach automatically."
            )
    else:
        df = st.session_state.crosswalk_df
        abstract_col = (
            next((c for c in df.columns if "ABSTRACT" in c.upper() or "ABST" in c.upper()), None)
            if df is not None
            else None
        )
        api_col = next((c for c in df.columns if "API" in c.upper()), None) if df is not None else None

        run_sheet_rows = []
        for item in st.session_state.extracted_instruments:
            inst: TexasOilGasInstrument = item["data"]
            lessees = ", ".join([p.name for p in inst.parties if p.role in ("Lessee", "Grantee", "Assignee")])
            lessors = ", ".join([p.name for p in inst.parties if p.role in ("Lessor", "Grantor", "Assignor")])

            for ld in inst.legal_descriptions or [LegalDescription()]:
                matching_wells = 0
                sample_apis = ""
                if df is not None and abstract_col and ld.abstract_number:
                    matches = df[df[abstract_col].astype(str).str.strip() == str(ld.abstract_number).strip()]
                    matching_wells = len(matches)
                    if api_col and not matches.empty:
                        sample_apis = ", ".join(matches[api_col].astype(str).head(5).tolist())

                flags = []
                if inst.uncertain_fields:
                    flags.append(f"OCR Check: {', '.join(inst.uncertain_fields)}")
                if inst.depth_and_pugh.depth_limitation_text:
                    flags.append("Depth Severance")
                if inst.depth_and_pugh.pugh_clause_present:
                    flags.append("Pugh Clause")
                if df is not None and matching_wells == 0:
                    flags.append("No RRC Wells Matched in Abstract")

                run_sheet_rows.append(
                    {
                        "Instrument": inst.instrument_type,
                        "Vol/Page or Inst #": inst.volume_page_or_instrument_no,
                        "Exec Date": inst.execution_date,
                        "Rec Date": inst.recording_date,
                        "Grantor / Lessor": lessors,
                        "Grantee / Lessee": lessees,
                        "County": ld.county,
                        "Abstract": ld.abstract_number,
                        "Survey": ld.survey_name,
                        "Block / Sec": f"{ld.block or '-'} / {ld.section or '-'}",
                        "Gross Acres": ld.acreage_gross,
                        "Net Acres": ld.acreage_net,
                        "Royalty": inst.royalty_and_term.royalty_fraction_text,
                        "Primary Term": inst.royalty_and_term.primary_term_text,
                        "Depth Limit": inst.depth_and_pugh.depth_limitation_text,
                        "Pugh Clause": inst.depth_and_pugh.pugh_clause_present,
                        "Matching RRC Wells": matching_wells,
                        "Sample APIs": sample_apis,
                        "Audit Flags": " | ".join(flags) if flags else "Clean",
                        "Source File": item["file_name"],
                    }
                )

        run_sheet_df = pd.DataFrame(run_sheet_rows)

        with st.container(border=True):
            render_section_header(
                title="Client Run-Sheet Template & Export",
                tooltip="Customize which columns appear and in what order before exporting your preliminary title run sheet to CSV.",
            )
            c_left, c_right = st.columns([1, 2])
            with c_left:
                client_project = st.text_input("Prospect / Client Name", value="Texas_Prospect")
            with c_right:
                all_cols = list(run_sheet_df.columns)
                selected_cols = st.multiselect(
                    "Active Run-Sheet Columns (remove or reorder to match client format):",
                    options=all_cols,
                    default=all_cols,
                )

            display_df = run_sheet_df[selected_cols] if selected_cols else run_sheet_df
            st.dataframe(display_df, width="stretch", hide_index=True)

            safe_proj = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in client_project).strip("_") or "RunSheet"
            st.download_button(
                label="⬇️ Export Client Run Sheet (CSV)",
                data=display_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_proj}_run_sheet.csv",
                mime="text/csv",
                type="primary",
            )

# --- TAB 4: API & TexasFile Lookup Helpers ---
with tab_lookup:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        with st.container(border=True):
            render_section_header(
                title="🛢️ RRC Well & API Formatter",
                tooltip="API Lookup Panel: Converts 10- or 14-digit Texas API numbers into the 8-digit format required by the RRC Public GIS Viewer and searches your loaded spatial crosswalk.",
            )
            api_input = st.text_input("Texas API Number", placeholder="e.g. 42-001-32761 or 00132761")
            if api_input:
                clean_api8 = normalize_rrc_api(api_input)
                st.code(clean_api8, language=None)
                st.markdown(f"👉 Paste **`{clean_api8}`** into the [RRC Public GIS Viewer](https://gis.rrc.texas.gov/GISViewer/) search bar.")

                if st.session_state.crosswalk_df is not None:
                    df = st.session_state.crosswalk_df
                    api_col = next((c for c in df.columns if "API" in c.upper()), None)
                    if api_col:
                        hits = df[df[api_col].astype(str).str.contains(clean_api8, na=False)]
                        if not hits.empty:
                            st.success(f"Found {len(hits)} matching record(s) in your loaded crosswalk:")
                            st.dataframe(hits, width="stretch", hide_index=True)
                        else:
                            st.info("Not found in currently loaded crosswalk.")

    with col_b:
        with st.container(border=True):
            render_section_header(
                title="🏛️ TexasFile Search String Builder",
                tooltip="TexasFile Helper Panel: Converts individual mineral owner names into 'LastName FirstName' format (while preserving LLCs/Trusts) and formats Abstract numbers for county clerk searches.",
            )
            party_input = st.text_input("Mineral Owner / Entity Name", placeholder="e.g. Michael Smith or Apache Corporation")
            abstract_input = st.text_input("Abstract Number (Optional)", placeholder="e.g. 28")

            if party_input or abstract_input:
                if party_input:
                    tf_name = format_texasfile_name(party_input)
                    st.markdown("**Formatted Party Query:**")
                    st.code(tf_name, language=None)
                if abstract_input:
                    st.markdown("**Abstract Reference Query:**")
                    st.code(abstract_input.strip(), language=None)
                st.markdown("👉 Run this search on [TexasFile County Records](https://www.texasfile.com/).")

# --- TAB 5: NMA / NRI Ownership Calculator ---
with tab_math:
    with st.container(border=True):
        render_section_header(
            title="Tract & Unit Acreage Parameters",
            tooltip="NMA / NRI Calculator Panel: Calculates exact fractional and decimal Net Mineral Acres (NMA), Royalty Owner Unit NRI, and Lessee Working/Net Revenue Interest for leased and unleased mineral owners.",
        )
        c1, c2 = st.columns(2)
        with c1:
            gross_tract_acres = st.number_input("Gross Tract Acres", min_value=0.0, value=640.0, step=10.0)
        with c2:
            unit_acres = st.number_input(
                "Pooled Unit Acres (keep equal to Gross Tract Acres if unpooled)",
                min_value=0.01,
                value=640.0,
                step=10.0,
            )

        st.markdown("#### Mineral Ownership & Lease Burdens")
        st.caption("Click `+` at the bottom of the table to add mineral owners. Accepts fractions (`1/4`, `3/16`) or decimals (`0.25`).")
        edited_owners = st.data_editor(
            pd.DataFrame(st.session_state.doi_owners),
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Owner Name": st.column_config.TextColumn("Owner Name", required=True),
                "Mineral Fraction": st.column_config.TextColumn("Mineral Interest (e.g. 1/4)", default="1/8"),
                "Leased": st.column_config.CheckboxColumn("Leased?", default=True),
                "Lease Royalty": st.column_config.TextColumn("Lease Royalty (e.g. 1/5)", default="1/5"),
            },
        )

    doi_rows = []
    total_mi = 0.0
    total_nma = 0.0
    total_royalty_nri = 0.0
    total_wi_nri = 0.0

    for _, row in edited_owners.iterrows():
        owner = str(row.get("Owner Name") or "").strip()
        if not owner:
            continue
        try:
            mi = parse_fraction_or_decimal(str(row.get("Mineral Fraction") or "0"))
            roy = parse_fraction_or_decimal(str(row.get("Lease Royalty") or "0")) if row.get("Leased") else 0.0
        except Exception:
            st.error(f"Invalid fraction or decimal for owner **{owner}**.")
            continue

        nma = gross_tract_acres * mi
        tract_factor = gross_tract_acres / unit_acres
        owner_nri = mi * roy * tract_factor if row.get("Leased") else 0.0
        lessee_wi = mi * tract_factor if row.get("Leased") else 0.0
        lessee_nri = mi * (1.0 - roy) * tract_factor if row.get("Leased") else 0.0

        total_mi += mi
        total_nma += nma
        total_royalty_nri += owner_nri
        total_wi_nri += lessee_nri

        doi_rows.append(
            {
                "Owner Name": owner,
                "Status": "Leased" if row.get("Leased") else "Unleased Mineral Owner",
                "Mineral Interest": round(mi, 8),
                "Gross Acres": gross_tract_acres,
                "Net Mineral Acres (NMA)": round(nma, 4),
                "Lease Royalty": round(roy, 8),
                "Owner Unit NRI": round(owner_nri, 8),
                "Lessee Unit WI": round(lessee_wi, 8),
                "Lessee Unit NRI": round(lessee_nri, 8),
            }
        )

    if doi_rows:
        with st.container(border=True):
            render_section_header(
                title="Division of Interest (DOI) Summary",
                tooltip="Division of Interest Panel: Summarizes total accounted mineral ownership, NMA, Royalty NRI, and Lessee NRI with one-click CSV export.",
            )
            doi_df = pd.DataFrame(doi_rows)
            m_a, m_b, m_c, m_d = st.columns(4)
            m_a.metric("Mineral Interest Accounted", f"{total_mi * 100:.4f}%")
            m_b.metric("Total Net Mineral Acres", f"{total_nma:,.4f} NMA")
            m_c.metric("Royalty Owner Unit NRI", f"{total_royalty_nri:.8f}")
            m_d.metric("Lessee Net Revenue (NRI)", f"{total_wi_nri:.8f}")

            if abs(total_mi - 1.0) > 1e-6:
                st.warning(
                    f"Total mineral interest is **{total_mi * 100:.4f}%** "
                    f"({(1.0 - total_mi) * 100:.4f}% remaining unaccounted)."
                )

            st.dataframe(doi_df, width="stretch", hide_index=True)
            st.download_button(
                label="⬇️ Export Division of Interest (CSV)",
                data=doi_df.to_csv(index=False).encode("utf-8"),
                file_name="division_of_interest_nma_nri.csv",
                mime="text/csv",
                type="primary",
            )
