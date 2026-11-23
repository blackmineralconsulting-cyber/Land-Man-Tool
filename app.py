import io
import os
import tempfile
import zipfile
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
# 2. Geospatial & Helper Functions (Handles Nested RRC Zips!)
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
                # Peek inside nested zip
                try:
                    inner_bytes = zf.read(name)
                    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as izf:
                        for iname in izf.namelist():
                            if not iname.startswith("__MACOSX") and iname.lower().endswith(".shp"):
                                shp_names.append(f"{name} -> {iname}")
                except Exception:
                    pass
    return sorted(shp_names), sorted(exts)


def load_geodata_from_upload(uploaded_file, selected_layer: str | None = None) -> gpd.GeoDataFrame:
    """Unpacks upload (including nested zips) into a temp folder and loads the selected shapefile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        if suffix != ".zip":
            raw_path = os.path.join(tmpdir, uploaded_file.name)
            with open(raw_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            return gpd.read_file(raw_path)

        extract_zip_recursive(uploaded_file.getvalue(), tmpdir)

        # Find all extracted .shp files
        all_shps = []
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith(".shp") and not fname.startswith("._"):
                    all_shps.append(os.path.join(root, fname))

        if not all_shps:
            raise ValueError(
                "No .shp shapefile found inside this zip. Make sure you uploaded an ESRI Shapefile zip (not a PDF or CSV bundle)."
            )

        chosen_path = all_shps[0]
        if selected_layer:
            target_basename = os.path.basename(selected_layer.split("->")[-1].strip())
            for candidate in all_shps:
                if os.path.basename(candidate) == target_basename:
                    chosen_path = candidate
                    break

        return gpd.read_file(chosen_path)


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


def format_texasfile_name(full_name: str) -> str:
    """Converts 'Michael Smith' to 'Smith Michael' for TexasFile name searches."""
    parts = [p.strip() for p in full_name.replace(",", " ").split() if p.strip()]
    corp_tokens = {"INC", "LLC", "LP", "LTD", "CORP", "COMPANY", "CO", "ENERGY", "OIL", "GAS", "RESOURCES", "TRUST", "ESTATE"}
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
# 3. Streamlit UI Layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Texas O&G Landman Workbench", layout="wide")
st.title("Texas Oil & Gas Landman Workbench")
st.caption("RRC Spatial-to-Survey Crosswalk • Vision-LLM Lease Extractor • County & API Research Helpers")

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
    st.divider()
    st.subheader("Quick External Portals")
    st.markdown("- [RRC Public GIS Viewer](https://gis.rrc.texas.gov/GISViewer/)")
    st.markdown("- [RRC Bulk Data Downloads](https://www.rrc.texas.gov/resource-center/research/data-sets-available-for-download/)")
    st.markdown("- [TexasFile County Search](https://www.texasfile.com/)")

tab_spatial, tab_extract, tab_crossref, tab_lookup = st.tabs(
    [
        "1. RRC Spatial Join",
        "2. Lease / Deed Extractor",
        "3. Cross-Reference Audit",
        "4. API & TexasFile Lookup",
    ]
)

# --- TAB 1: Spatial-to-Survey Join ---
with tab_spatial:
    st.subheader("Map RRC Well Layers to Texas Legal Surveys")
    st.markdown(
        "Upload zipped ESRI shapefiles (`.zip` containing `.shp`, `.shx`, `.dbf`, `.prj`) or `.geojson` files."
    )

    col1, col2 = st.columns(2)
    wells_selected_shp = None
    surveys_selected_shp = None

    with col1:
        wells_file = st.file_uploader("RRC Wells Layer (.zip or .geojson)", type=["zip", "geojson"])
        if wells_file is not None:
            shp_list, exts = inspect_uploaded_archive(wells_file.name, wells_file.getvalue())
            if len(shp_list) > 1:
                wells_selected_shp = st.selectbox(
                    "Select Wells shapefile layer (e.g., surface 's' or bottomhole 'b'):",
                    options=shp_list,
                )
            elif len(shp_list) == 1:
                wells_selected_shp = shp_list[0]
                st.success(f"Detected layer: `{wells_selected_shp}`")
            else:
                st.error(f"No `.shp` file found inside `{wells_file.name}` (found extensions: {', '.join(exts)}).")

    with col2:
        surveys_file = st.file_uploader("Texas GLO / RRC Survey Layer (.zip or .geojson)", type=["zip", "geojson"])
        if surveys_file is not None:
            shp_list_s, exts_s = inspect_uploaded_archive(surveys_file.name, surveys_file.getvalue())
            if len(shp_list_s) > 1:
                poly_idx = next((i for i, name in enumerate(shp_list_s) if name.lower().endswith("p.shp")), 0)
                surveys_selected_shp = st.selectbox(
                    "Select Survey polygon layer (usually ends in 'p.shp'):",
                    options=shp_list_s,
                    index=poly_idx,
                )
            elif len(shp_list_s) == 1:
                surveys_selected_shp = shp_list_s[0]
                st.success(f"Detected layer: `{surveys_selected_shp}`")
            else:
                st.error(f"No `.shp` file found inside `{surveys_file.name}` (found extensions: {', '.join(exts_s)}).")

    if st.button("Run Spatial Join", type="primary"):
        if wells_file is None or surveys_file is None:
            st.warning("Please upload both a Wells layer and a Survey layer first.")
        else:
            with st.spinner("Unpacking layers, aligning CRS, and running spatial intersection..."):
                try:
                    wells_gdf = load_geodata_from_upload(wells_file, wells_selected_shp)
                    surveys_gdf = load_geodata_from_upload(surveys_file, surveys_selected_shp)
                    crosswalk_df = run_spatial_join(wells_gdf, surveys_gdf, fallback_crs, target_crs)
                    st.session_state.crosswalk_df = crosswalk_df
                    st.success(f"Matched {len(crosswalk_df):,} well-to-survey records.")
                except Exception as exc:
                    st.error(f"Spatial join failed: {exc}")

    if st.session_state.crosswalk_df is not None:
        df = st.session_state.crosswalk_df
        st.dataframe(df.head(500), use_container_width=True)
        if len(df) > 500:
            st.caption(f"Showing first 500 of {len(df):,} rows in browser. Download the CSV below for the full dataset.")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Crosswalk CSV",
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
        abstract_col = next((c for c in df.columns if "ABSTRACT" in c.upper() or "ABST" in c.upper()), None)
        api_col = next((c for c in df.columns if "API" in c.upper()), None)

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
                    if api_col and not matches.empty:
                        sample_apis = ", ".join(matches[api_col].astype(str).head(5).tolist())

                flags = []
                if inst.uncertain_fields:
                    flags.append(f"OCR Check: {', '.join(inst.uncertain_fields)}")
                if inst.depth_and_pugh.depth_limitation_text:
                    flags.append("Depth Severance")
                if inst.depth_and_pugh.pugh_clause_present:
                    flags.append("Pugh Clause")
                if matching_wells == 0:
                    flags.append("No RRC Wells Matched in Abstract")

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
                        "Audit Flags": " | ".join(flags) if flags else "Clean",
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

# --- TAB 4: API & TexasFile Lookup Helpers ---
with tab_lookup:
    st.subheader("RRC API Lookup & TexasFile Search Generator")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 1. RRC Well / API Lookup")
        api_input = st.text_input("Enter Texas API Number (e.g., 42-389-34567 or 38934567)")
        if api_input:
            clean_api8 = normalize_rrc_api(api_input)
            st.success(f"8-Digit RRC GIS Viewer API: **`{clean_api8}`**")
            st.markdown(
                f"1. Copy **`{clean_api8}`** above.\n"
                f"2. Open the [RRC Public GIS Viewer](https://gis.rrc.texas.gov/GISViewer/) and paste into the top-right API search box."
            )

            if st.session_state.crosswalk_df is not None:
                df = st.session_state.crosswalk_df
                api_col = next((c for c in df.columns if "API" in c.upper()), None)
                if api_col:
                    hits = df[df[api_col].astype(str).str.contains(clean_api8, na=False)]
                    if not hits.empty:
                        st.markdown("**Matching Records in Your Uploaded Crosswalk:**")
                        st.dataframe(hits, use_container_width=True)
                    else:
                        st.info("No matching API found in your currently loaded crosswalk table.")

    with col_b:
        st.markdown("### 2. TexasFile Search String Builder")
        party_input = st.text_input("Party / Mineral Owner Name (e.g., Michael Smith or Apache Corporation)")
        abstract_input = st.text_input("Abstract Number (optional, e.g., 28)")

        if party_input or abstract_input:
            if party_input:
                tf_name = format_texasfile_name(party_input)
                st.code(tf_name, language=None)
                st.caption("Formatted `LastName FirstName` for TexasFile's Name Search box.")
            if abstract_input:
                st.code(abstract_input.strip(), language=None)
                st.caption("Enter into TexasFile's Reference / Legal Search box at the bottom of the county search page.")
            st.markdown("Open [TexasFile County Records Search](https://www.texasfile.com/) to run this query.")
