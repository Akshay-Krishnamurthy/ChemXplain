#!/bin/bash
# ============================================================
# ChemXplain — Setup Script
# Downloads the CReM ChEMBL fragment database from Zenodo.
#
# Usage:
#   chmod +x setup.sh
#   bash setup.sh
# ============================================================

set -e  # Exit immediately on any error

# ── Colors ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   🧬  ChemXplain — Environment Setup                ║"
echo "  ║   Downloading ChEMBL Fragment Database from Zenodo  ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Check for required tools ─────────────────────────────────
if ! command -v wget &> /dev/null; then
    echo -e "${YELLOW}⚠️  wget not found. Trying curl instead...${NC}"
    USE_CURL=true
else
    USE_CURL=false
fi

DB_FILE="chembl22_sa2_hac12.db"
DB_GZ="${DB_FILE}.gz"
ZENODO_URL="https://zenodo.org/records/16909329/files/chembl22_sa2_hac12.db.gz?download=1"

# ── Check if database already exists ────────────────────────
if [ -f "$DB_FILE" ]; then
    echo -e "${GREEN}✅ Database '$DB_FILE' already exists. Skipping download.${NC}"
    echo -e "${GREEN}🚀 You can launch the app: python -m streamlit run app.py${NC}"
    exit 0
fi

# ── Download ─────────────────────────────────────────────────
echo -e "${YELLOW}📥 Downloading ChEMBL22 fragment database (~150 MB)...${NC}"
echo "   Source: Zenodo (DOI: 10.5281/zenodo.16909329)"
echo "   Filter: SA Score ≤ 2, Max heavy atoms ≤ 12 (highest quality fragment space)"
echo ""

if [ "$USE_CURL" = true ]; then
    curl -L "$ZENODO_URL" -o "$DB_GZ" --progress-bar
else
    wget "$ZENODO_URL" -O "$DB_GZ" --progress=bar:force 2>&1
fi

# ── Extract ──────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}📦 Extracting database...${NC}"
gunzip -f "$DB_GZ"

# ── Verify ───────────────────────────────────────────────────
if [ -f "$DB_FILE" ]; then
    SIZE=$(du -sh "$DB_FILE" | cut -f1)
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ Setup Complete!                                  ║${NC}"
    echo -e "${GREEN}║  Database: $DB_FILE ($SIZE)              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}🚀 Launch the app:${NC}"
    echo "   conda activate chemxplain"
    echo "   python -m streamlit run app.py"
else
    echo -e "${RED}❌ Setup failed. '$DB_FILE' not found after extraction.${NC}"
    echo "   Please download manually from:"
    echo "   $ZENODO_URL"
    exit 1
fi
