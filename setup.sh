#!/usr/bin/env bash
# Automated setup script for Prompt Evolver (Unix/macOS)
# This script guides users through installation and configuration

set -e  # Exit on error

# Colors for output (if terminal supports it)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup
main() {
    print_header "Prompt Evolver Setup"

    echo "This script will help you set up Prompt Evolver by:"
    echo "  1. Checking Python installation"
    echo "  2. Creating a virtual environment"
    echo "  3. Installing dependencies"
    echo "  4. Configuring your LLM backend"
    echo "  5. Running a test to validate setup"
    echo ""

    read -p "Press Enter to continue or Ctrl+C to cancel..."

    # Step 1: Check Python version
    print_header "Step 1: Checking Python Installation"

    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        echo ""
        echo "Please install Python 3.10 or higher from:"
        echo "  macOS: https://www.python.org/downloads/ or use 'brew install python3'"
        echo "  Linux: Use your package manager (apt, yum, dnf, etc.)"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Found Python ${PYTHON_VERSION}"

    # Check if version is >= 3.10
    PYTHON_MAJOR=$(echo ${PYTHON_VERSION} | cut -d. -f1)
    PYTHON_MINOR=$(echo ${PYTHON_VERSION} | cut -d. -f2)

    if [ "${PYTHON_MAJOR}" -lt 3 ] || ([ "${PYTHON_MAJOR}" -eq 3 ] && [ "${PYTHON_MINOR}" -lt 10 ]); then
        print_error "Python 3.10 or higher is required (found ${PYTHON_VERSION})"
        exit 1
    fi

    # Step 2: Create virtual environment
    print_header "Step 2: Creating Virtual Environment"

    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists at .venv"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf .venv
        else
            print_info "Using existing virtual environment"
        fi
    fi

    if [ ! -d ".venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv .venv
        print_success "Virtual environment created"
    fi

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source .venv/bin/activate
    print_success "Virtual environment activated"

    # Step 3: Install dependencies
    print_header "Step 3: Installing Dependencies"

    print_info "Upgrading pip..."
    python -m pip install --upgrade pip --quiet

    print_info "Installing Prompt Evolver dependencies..."
    pip install -r requirements.txt --quiet
    print_success "Dependencies installed"

    # Step 4: Configure LLM backend
    print_header "Step 4: Configuring LLM Backend"

    echo "Select your LLM backend:"
    echo "  1) Local (LM Studio / Ollama running locally)"
    echo "  2) OpenAI API"
    echo "  3) OpenAI-compatible API (custom endpoint)"
    echo "  4) Echo mode (for testing, returns input as output)"
    echo ""
    read -p "Enter your choice (1-4): " -n 1 -r BACKEND_CHOICE
    echo ""

    case $BACKEND_CHOICE in
        1)
            BACKEND_MODE="lmstudio"
            print_info "You selected: Local (LM Studio / Ollama)"
            echo ""
            echo "Default settings:"
            echo "  API URL: http://127.0.0.1:1234"
            echo ""
            read -p "Enter API URL [http://127.0.0.1:1234]: " API_URL
            API_URL=${API_URL:-http://127.0.0.1:1234}

            read -p "Enter model name [mistralai/ministral-3-3b]: " MODEL_NAME
            MODEL_NAME=${MODEL_NAME:-mistralai/ministral-3-3b}
            ;;
        2)
            BACKEND_MODE="openai_compatible"
            print_info "You selected: OpenAI API"
            echo ""
            API_URL="https://api.openai.com/v1"
            read -p "Enter your OpenAI API key: " -s API_KEY
            echo ""

            if [ -z "$API_KEY" ]; then
                print_error "API key is required for OpenAI"
                exit 1
            fi

            echo "export OPENAI_API_KEY=\"$API_KEY\"" >> .venv/bin/activate

            read -p "Enter model name [gpt-4]: " MODEL_NAME
            MODEL_NAME=${MODEL_NAME:-gpt-4}
            ;;
        3)
            BACKEND_MODE="openai_compatible"
            print_info "You selected: OpenAI-compatible API"
            echo ""
            read -p "Enter API URL: " API_URL

            if [ -z "$API_URL" ]; then
                print_error "API URL is required"
                exit 1
            fi

            read -p "Enter API key (leave empty if not required): " -s API_KEY
            echo ""

            if [ ! -z "$API_KEY" ]; then
                echo "export OPENAI_API_KEY=\"$API_KEY\"" >> .venv/bin/activate
            fi

            read -p "Enter model name: " MODEL_NAME
            ;;
        4)
            BACKEND_MODE="echo"
            print_info "You selected: Echo mode (testing)"
            API_URL=""
            MODEL_NAME="echo"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac

    # Create config file
    print_info "Generating configuration file..."

    cat > configs/config.yaml << EOF
# Prompt Evolver Configuration
# Generated by setup script

max_generations: 3
similarity_weight: 0.8
length_weight: 0.2
max_no_improve: 2
leakage_similarity_threshold: 0.45
leakage_ngram_size: 3
leakage_ngram_overlap_threshold: 0.1
min_prompt_tokens: 2
max_prompt_tokens: 200
max_prompt_increase_ratio: 2.0
max_prompt_increase_tokens: 40
min_improvement_attempts: 2

llm_execution:
  mode: ${BACKEND_MODE}
  api_url: ${API_URL}
  timeout_seconds: 30

llm_improvement:
  mode: ${BACKEND_MODE}
  api_url: ${API_URL}
  timeout_seconds: 30

llm_evaluation:
  mode: ${BACKEND_MODE}
  api_url: ${API_URL}
  timeout_seconds: 30
EOF

    print_success "Configuration file created at configs/config.yaml"

    # Create a starter notebook cell content file
    if [ "$MODEL_NAME" != "echo" ]; then
        print_info "Model name to use in notebook: ${MODEL_NAME}"
        echo ""
        echo "Remember to set this in notebooks/PromptEvolver.ipynb:"
        echo ""
        echo "  EXECUTION_MODEL = \"${MODEL_NAME}\""
        echo "  IMPROVEMENT_MODEL = \"${MODEL_NAME}\""
        echo "  EVALUATION_MODEL = \"${MODEL_NAME}\""
        echo ""
    fi

    # Step 5: Validate setup
    print_header "Step 5: Validating Setup"

    print_info "Testing configuration loading..."

    python -c "
import sys
sys.path.insert(0, 'src')
from prompt_evolver.config import load_config

try:
    config = load_config('configs/config.yaml')
    print('Configuration loaded successfully')
    print(f'  Execution mode: {config.llm_execution.mode}')
    print(f'  Max generations: {config.max_generations}')
except Exception as e:
    print(f'Error loading configuration: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        print_success "Configuration validation passed"
    else
        print_error "Configuration validation failed"
        exit 1
    fi

    # Final instructions
    print_header "Setup Complete!"

    print_success "Prompt Evolver is ready to use!"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Activate the virtual environment (when starting a new terminal):"
    echo "   ${GREEN}source .venv/bin/activate${NC}"
    echo ""
    echo "2. Open VS Code and the Jupyter notebook:"
    echo "   ${GREEN}code .${NC}"
    echo "   Then open: ${GREEN}notebooks/PromptEvolver.ipynb${NC}"
    echo ""
    echo "3. Update model names in the notebook (first cell):"
    echo "   ${GREEN}EXECUTION_MODEL = \"${MODEL_NAME}\"${NC}"
    echo ""
    echo "4. Run the cells to start optimizing prompts!"
    echo ""
    echo "For help, see:"
    echo "  - README.md"
    echo "  - docs/overview.md"
    echo "  - docs/quickstart.md"
    echo ""

    if [ "$BACKEND_MODE" = "lmstudio" ] || [ "$BACKEND_MODE" = "openai_compatible" ]; then
        print_warning "Make sure your LLM backend is running at: ${API_URL}"
        echo ""
    fi
}

# Run main function
main
