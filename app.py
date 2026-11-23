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
# 2. Geospatial, Title Math & Helper Functions
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
        "INC", "LLC", "LP", "LTD", "CORP", "COMPANY", "CO", "ENERGY",
        "OIL", "GAS", "RESOURCES", "TRUST", "ESTATE",
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
# 3. Streamlit UI Layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Texas O&G Landman Workbench", layout="wide")
st.title("Texas Oil & Gas Landman Workbench")
st.caption("Multi-County Spatial Crosswalk • Vision-LLM Lease Extractor • Custom Run Sheets • NMA/NRI Math")

if "crosswalk_df" not in st.session_state:
    st.session_state.crosswalk_df = None
if "extracted_instruments" not in st.session_state:
    st.session_state.extracted_instruments = []
if "doi_owners" not in st.session_state:
    st.session_state.doi_owners = [
        {"Owner Name": "Example Mineral Owner", "Mineral Fraction": "1/4", "Leased": True, "Lease Royalty": "1/5"}
    ]

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

tab_spatial, tab_extract, tab_crossref, tab_lookup, tab_math = st.tabs(
    [
        "1. Multi-County Spatial Join",
        "2. Lease / Deed Extractor",
        "3. Custom Run Sheet & Audit",
        "4. API & TexasFile Lookup",
        "5. NMA / NRI Calculator",
    ]
)

# --- TAB 1: Multi-County Spatial-to-Survey Join ---
with tab_spatial:
    st.subheader("Map RRC Well Layers to Texas Legal Surveys (Single or Multi-County Batch)")
    st.markdown(
        "Upload zipped ESRI shapefiles (`.zip` containing one or multiple county `.shp` layers) or `.geojson` files."
    )

    col1, col2 = st.columns(2)
    wells_selected_shps = []
    surveys_selected_shps = []

    with col1:
        wells_file = st.file_uploader("RRC Wells Layer(s) (.zip or .geojson)", type=["zip", "geojson"])
        if wells_file is not None:
            shp_list, exts = inspect_uploaded_archive(wells_file.name, wells_file.getvalue())
            if len(shp_list) > 1:
                join_all_wells = st.checkbox(
                    f"Batch-select all {len(shp_list)} well layers in archive", value=False, key="all_w"
                )
                if join_all_wells:
                    wells_selected_shps = shp_list
                else:
                    default_w = [s for s in shp_list if s.lower().endswith("b.shp")][:1] or [shp_list[0]]
                    wells_selected_shps = st.multiselect(
                        "Select one or more Wells layers (e.g., surface 's.shp' or bottomhole 'b.shp'):",
                        options=shp_list,
                        default=default_w,
                    )
            elif len(shp_list) == 1:
                wells_selected_shps = shp_list
                st.success(f"Detected layer: `{shp_list[0]}`")
            else:
                st.error(f"No `.shp` file found inside `{wells_file.name}` (found: {', '.join(exts)}).")

    with col2:
        surveys_file = st.file_uploader("Texas GLO / RRC Survey Layer(s) (.zip or .geojson)", type=["zip", "geojson"])
        if surveys_file is not None:
            shp_list_s, exts_s = inspect_uploaded_archive(surveys_file.name, surveys_file.getvalue())
            if len(shp_list_s) > 1:
                poly_layers = [s for s in shp_list_s if s.lower().endswith("p.shp")]
                join_all_polys = st.checkbox(
                    f"Batch-select all {len(poly_layers) or len(shp_list_s)} survey polygon layers",
                    value=False,
                    key="all_s",
                )
                if join_all_polys:
                    surveys_selected_shps = poly_layers or shp_list_s
                else:
                    default_s = poly_layers[:1] or [shp_list_s[0]]
                    surveys_selected_shps = st.multiselect(
                        "Select one or more Survey polygon layers (usually ends in 'p.shp'):",
                        options=shp_list_s,
                        default=default_s,
                    )
            elif len(shp_list_s) == 1:
                surveys_selected_shps = shp_list_s
                st.success(f"Detected layer: `{shp_list_s[0]}`")
            else:
                st.error(f"No `.shp` file found inside `{surveys_file.name}` (found: {', '.join(exts_s)}).")

    if st.button("Run Spatial Join", type="primary"):
        if wells_file is None or surveys_file is None:
            st.warning("Please upload both a Wells layer and a Survey layer first.")
        else:
            with st.spinner("Unpacking layers, aligning CRS across selected counties, and running spatial join..."):
                try:
                    wells_gdf = load_batch_geodata_from_upload(
                        wells_file, wells_selected_shps, fallback_crs, target_crs
                    )
                    surveys_gdf = load_batch_geodata_from_upload(
                        surveys_file, surveys_selected_shps, fallback_crs, target_crs
                    )
                    crosswalk_df = run_batch_spatial_join(wells_gdf, surveys_gdf)
                    st.session_state.crosswalk_df = crosswalk_df
                    st.success(
                        f"Matched {len(crosswalk_df):,} well-to-survey records across "
                        f"{len(wells_selected_shps) or 1} well layer(s) and {len(surveys_selected_shps) or 1} survey layer(s)."
                    )
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
                m
