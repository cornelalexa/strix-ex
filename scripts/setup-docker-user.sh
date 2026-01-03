#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Docker User Group Setup${NC}"
echo "================================"
echo ""

CURRENT_USER="$USER"

if [ -z "$CURRENT_USER" ] || [ "$CURRENT_USER" = "root" ]; then
    echo -e "${RED}❌ Could not determine current user${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo ""
    echo "Please install Docker first:"
    echo "  https://docs.docker.com/engine/install/"
    echo ""
    exit 1
fi

echo -e "${YELLOW}Current user:${NC} $CURRENT_USER"
echo ""
echo "This script will:"
echo -e "  ${GREEN}✓${NC} Add '$CURRENT_USER' to the 'docker' group"
echo -e "  ${GREEN}✓${NC} Enable Docker access without sudo"
echo ""
echo -e "${YELLOW}⚠️  You will need to log out and back in for changes to take effect.${NC}"
echo ""

read -p "Do you want to continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}Adding '$CURRENT_USER' to docker group (requires sudo)...${NC}"
echo ""

if ! getent group docker > /dev/null; then
    echo -e "${BLUE}Creating docker group...${NC}"
    sudo groupadd docker
fi

sudo usermod -aG docker "$CURRENT_USER"

echo -e "${GREEN}✓ Added '$CURRENT_USER' to docker group${NC}"
echo ""

if ! systemctl is-active --quiet docker; then
    echo -e "${BLUE}Starting Docker daemon...${NC}"
    sudo systemctl start docker
    echo -e "${GREEN}✓ Docker daemon started${NC}"
    echo ""
fi

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo -e "  ${YELLOW}1.${NC} Log out and log back in (or run: ${BLUE}newgrp docker${NC})"
echo -e "  ${YELLOW}2.${NC} Verify with: ${BLUE}docker ps${NC}"
echo -e "  ${YELLOW}3.${NC} Run strix: ${BLUE}strix --target ./app${NC}"
echo ""
