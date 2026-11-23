import os
import tempfile
from typing import Literal

import geopandas as gpd
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Pydantic Schema for Texas Oil & Gas Instruments
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
# 2. Helper Functions
# ---------------------------------------------------------------------------
def load_geodata_from_upload(uploaded_file) -> gpd.GeoDataFrame:
    """Reads an uploaded .zip (shapefile bundle) or .geojson into a GeoDataFrame."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        gdf = gpd.read_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return gdf


def run_spatial_join(
    wells_gdf: gpd.GeoDataFrame,
    surveys_gdf: gpd.GeoDataFrame,
    fallback_crs: str = "EPSG:4267",
    target_crs: str = "EPSG:3083",
) -> pd.DataFrame:
    """Aligns CRS (defaulting missing .prj to NAD27) and intersects wells with surveys."""
    if wells_gdf.crs is None:
        wells_gdf = wells_gdf.set_crs(fallback_crs)
    if surveys_gdf.crs is None:
        surveys_gdf = surveys_gdf.set_crs(fallback_crs)

    wells_proj = wells_gdf.to_crs(target_crs)
    surveys_proj = surveys_gdf.to_crs(target_crs)

    wells_proj = wells_proj[wells_proj.geometry.is_valid & ~wells_proj.geometry.is_empty]
    surveys_proj = surveys_proj[surveys_proj.geometry.is_valid & ~surveys_proj.geometry.is_empty]

    joined = gpd.sjoin(wells_proj, surveys_proj, how="left", predicate="intersects")
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


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
# 3. Streamlit UI Layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Texas O&G Landman Workbench", layout="wide")
st.title("Texas Oil & Gas Landman Workbench")
st.caption("RRC Spatial-to-Survey Crosswalk & Vision-LLM County Deed/Lease Extractor")

# Initialize session state so tabs can share results
if "crosswalk_df" not in st.session_state:
    st.session_state.crosswalk_df = None
if "extracted_instruments" not in st.session_state:
    st.session_state.extracted_instruments = []

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Required for Vision-LLM deed & lease extraction.",
    )
    model_choice = st.selectbox(
        "Extraction Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="Use 2.5-pro for faint handwriting or complex metes-and-bounds.",
    )
    st.divider()
    st.subheader("CRS Defaults")
    fallback_crs = st.text_input("Fallback CRS (if .prj missing)", value="EPSG:4267")
    target_crs = st.text_input("Target Planar CRS", value="EPSG:3083")

tab_spatial, tab_extract, tab_crossref = st.tabs(
    ["1. RRC Spatial Join", "2. Lease / Deed Extractor", "3. Cross-Reference Audit"]
)

# --- TAB 1: Spatial-to-Survey Join ---
with tab_spatial:
    st.subheader("Map RRC Well Layers to Texas Legal Surveys")
    st.markdown(
        "Upload zipped ESRI shapefiles (`.zip` containing `.shp`, `.shx`, `.dbf`, `.prj`) or `.geojson` files."
    )

    col1, col2 = st.columns(2)
    with col1:
        wells_file = st.file_uploader("RRC Wells Layer (.zip or .geojson)", type=["zip", "geojson"])
    with col2:
        surveys_file = st.file_uploader("Texas GLO / RRC Survey Layer (.zip or .geojson)", type=["zip", "geojson"])

    if st.button("Run Spatial Join", type="primary", disabled=not (wells_file and surveys_file)):
        with st.spinner("Reprojecting to common CRS and running spatial intersection..."):
            try:
                wells_gdf = load_geodata_from_upload(wells_file)
                surveys_gdf = load_geodata_from_upload(surveys_file)
                crosswalk_df = run_spatial_join(wells_gdf, surveys_gdf, fallback_crs, target_crs)
                st.session_state.crosswalk_df = crosswalk_df
                st.success(f"Matched {len(crosswalk_df)} well-to-survey records.")
            except Exception as exc:
                st.error(f"Spatial join failed: {exc}")

    if st.session_state.crosswalk_df is not None:
        df = st.session_state.crosswalk_df
        st.dataframe(df, use_container_width=True)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Crosswalk CSV",
            data=csv_data,
            file_name="well_to_survey_crosswalk.csv",
            mime="text/csv",
        )

# --- TAB 2: Vision-LLM Lease & Deed Extraction ---
with tab_extract:
    st.subheader("Extract Run-Sheet Fields from Recorded Instruments")
    uploaded_docs = st.file_uploader(
        "Upload County Clerk PDFs or Page Scans (PNG/JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if st.button("Extract Title Fields", type="primary", disabled=not uploaded_docs):
        if not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar first.")
        else:
            results = []
            progress = st.progress(0)
            for idx, doc in enumerate(uploaded_docs):
                with st.spinner(f"Parsing {doc.name}..."):
                    try:
                        parsed_obj = extract_instrument_with_gemini(
                            file_bytes=doc.getvalue(),
                            mime_type=doc.type or "application/pdf",
                            api_key=api_key,
                            model_name=model_choice,
                        )
                        results.append({"file_name": doc.name, "data": parsed_obj})
                    except Exception as exc:
                        st.error(f"Error parsing {doc.name}: {exc}")
                progress.progress((idx + 1) / len(uploaded_docs))

            st.session_state.extracted_instruments = results

    if st.session_state.extracted_instruments:
        for item in st.session_state.extracted_instruments:
            inst: TexasOilGasInstrument = item["data"]
            with st.expander(
                f"{item['file_name']} — {inst.instrument_type} ({inst.volume_page_or_instrument_no or 'No Vol/Pg'})",
                expanded=True,
            ):
                if inst.uncertain_fields:
                    st.warning(f"Human Review Flagged — Uncertain Fields: {', '.join(inst.uncertain_fields)}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Execution Date", inst.execution_date or "N/A")
                m2.metric("Recording Date", inst.recording_date or "N/A")
                m3.metric("Royalty", inst.royalty_and_term.royalty_fraction_text or "N/A")
                m4.metric("Pugh Clause", "Yes" if inst.depth_and_pugh.pugh_clause_present else "No / Unstated")

                st.markdown("**Parties**")
                st.dataframe([p.model_dump() for p in inst.parties], use_container_width=True)

                st.markdown("**Legal Descriptions**")
                st.dataframe([ld.model_dump() for ld in inst.legal_descriptions], use_container_width=True)

                if inst.depth_and_pugh.depth_limitation_text:
                    st.info(f"**Depth Clause (Verbatim):** {inst.depth_and_pugh.depth_limitation_text}")

                if inst.royalty_and_term.verbatim_granting_clause:
                    st.text_area(
                        "Verbatim Granting Clause",
                        value=inst.royalty_and_term.verbatim_granting_clause,
                        height=100,
                        disabled=True,
                        key=f"grant_{item['file_name']}",
                    )

                with st.expander("Raw Structured JSON"):
                    st.json(inst.model_dump())

# --- TAB 3: Cross-Reference Audit ---
with tab_crossref:
    st.subheader("Cross-Reference RRC Wells Against Extracted Instruments")
    if st.session_state.crosswalk_df is None or not st.session_state.extracted_instruments:
        st.info(
            "Run **1. RRC Spatial Join** and **2. Lease / Deed Extractor** first to cross-reference wells against extracted tracts."
        )
    else:
        df = st.session_state.crosswalk_df
        # Detect likely abstract/survey columns from the spatial join
        abstract_col = next((c for c in df.columns if "ABSTRACT" in c.upper() or "ABST" in c.upper()), None)

        run_sheet_rows = []
        for item in st.session_state.extracted_instruments:
            inst: TexasOilGasInstrument = item["data"]
            lessees = ", ".join([p.name for p in inst.parties if p.role in ("Lessee", "Grantee", "Assignee")])
            lessors = ", ".join([p.name for p in inst.parties if p.role in ("Lessor", "Grantor", "Assignor")])

            for ld in inst.legal_descriptions or [LegalDescription()]:
                matching_wells = 0
                sample_apis = ""
                if abstract_col and ld.abstract_number:
                    matches = df[df[abstract_col].astype(str).str.strip() == str(ld.abstract_number).strip()]
                    matching_wells = len(matches)
                    api_col = next((c for c in df.columns if "API" in c.upper()), None)
                    if api_col and not matches.empty:
                        sample_apis = ", ".join(matches[api_col].astype(str).head(5).tolist())

                run_sheet_rows.append(
                    {
                        "Source File": item["file_name"],
                        "Instrument": inst.instrument_type,
                        "Vol/Page or Inst #": inst.volume_page_or_instrument_no,
                        "Exec Date": inst.execution_date,
                        "Grantor / Lessor": lessors,
                        "Grantee / Lessee": lessees,
                        "County": ld.county,
                        "Abstract": ld.abstract_number,
                        "Survey": ld.survey_name,
                        "Block / Sec": f"{ld.block or '-'} / {ld.section or '-'}",
                        "Royalty": inst.royalty_and_term.royalty_fraction_text,
                        "Primary Term": inst.royalty_and_term.primary_term_text,
                        "Depth Limit": inst.depth_and_pugh.depth_limitation_text,
                        "Pugh Clause": inst.depth_and_pugh.pugh_clause_present,
                        "Matching RRC Wells": matching_wells,
                        "Sample APIs": sample_apis,
                    }
                )

        run_sheet_df = pd.DataFrame(run_sheet_rows)
        st.dataframe(run_sheet_df, use_container_width=True)
        st.download_button(
            label="Export Preliminary Run Sheet (CSV)",
            data=run_sheet_df.to_csv(index=False).encode("utf-8"),
            file_name="preliminary_landman_run_sheet.csv",
            mime="text/csv",
        )
