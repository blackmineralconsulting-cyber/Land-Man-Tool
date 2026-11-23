# Texas O&G Landman Workbench

**Black Mineral Consulting — Title & GIS Suite**

A Streamlit workbench for Texas oil & gas land work: mapping Railroad Commission
(RRC) wells to their legal surveys, extracting title data from recorded deeds and
leases with vision AI, building client run sheets, and running division-of-interest
ownership math.

## The five tools

1. **Spatial Crosswalk** — Upload RRC well-layer and Texas GLO/RRC survey-layer
   shapefile archives (nested county zips are unpacked automatically). Wells are
   intersected against survey polygons (NAD27 fallback, Texas Centric Albers
   EPSG:3083 join projection) to produce an API-to-Abstract/Survey crosswalk,
   searchable in-app and exportable to CSV.
2. **AI Deed & Lease Extractor** — Batch-parses recorded county clerk PDFs and
   page scans with Gemini vision models into a strict title schema: parties,
   recording references, legal descriptions, royalty and term, depth limitations
   and Pugh clauses (quoted verbatim), with uncertain handwriting flagged for
   curative review. Requires a Gemini API key (sidebar).
3. **Run Sheet & Audit** — Combines extracted instruments with the well
   crosswalk, attaches matching RRC well counts/APIs by abstract, flags depth
   severances, Pugh clauses, and unmatched abstracts, and exports a
   customizable client run sheet (CSV).
4. **API & TexasFile Lookup** — Converts 10/14-digit Texas API numbers to the
   8-digit RRC GIS Viewer format and searches the loaded crosswalk; formats
   owner names ("Last First") and abstract references for TexasFile searches.
5. **NMA / NRI Calculator** — Division-of-interest math: net mineral acres,
   royalty-owner unit NRI, and lessee working/net revenue interest for leased
   and unleased owners, with CSV export.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

The repo is ready for [Streamlit Community Cloud](https://share.streamlit.io):
point a new app at this repository, branch `main`, file `app.py`. The Gemini
API key is entered in the app's sidebar at runtime (or supplied via the
`GEMINI_API_KEY` environment variable / Streamlit secrets) — no key is stored
in this repository.

*Preliminary title aid only — not a legal title opinion.*
