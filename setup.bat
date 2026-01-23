@echo off
REM Automated setup script for Prompt Evolver (Windows)
REM This script guides users through installation and configuration

setlocal enabledelayedexpansion

echo.
echo ========================================
echo Prompt Evolver Setup
echo ========================================
echo.
echo This script will help you set up Prompt Evolver by:
echo   1. Checking Python installation
echo   2. Creating a virtual environment
echo   3. Installing dependencies
echo   4. Configuring your LLM backend
echo   5. Running a test to validate setup
echo.
pause

REM Step 1: Check Python version
echo.
echo ========================================
echo Step 1: Checking Python Installation
echo ========================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.10 or higher from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version 2>nul | findstr /r "Python 3\.[1-9][0-9]" >nul
if %errorlevel% neq 0 (
    python --version 2>nul | findstr /r "Python 3\.10" >nul
    if %errorlevel% neq 0 (
        echo [ERROR] Python 3.10 or higher is required
        python --version
        pause
        exit /b 1
    )
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python !PYTHON_VERSION!

REM Step 2: Create virtual environment
echo.
echo ========================================
echo Step 2: Creating Virtual Environment
echo ========================================
echo.

if exist ".venv" (
    echo [WARNING] Virtual environment already exists at .venv
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo [INFO] Removing existing virtual environment...
        rmdir /s /q .venv
    ) else (
        echo [INFO] Using existing virtual environment
    )
)

if not exist ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM Step 3: Install dependencies
echo.
echo ========================================
echo Step 3: Installing Dependencies
echo ========================================
echo.

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to upgrade pip
    pause
    exit /b 1
)

echo [INFO] Installing Prompt Evolver dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

REM Step 4: Configure LLM backend
echo.
echo ========================================
echo Step 4: Configuring LLM Backend
echo ========================================
echo.

echo Select your LLM backend:
echo   1) Local (LM Studio / Ollama running locally)
echo   2) OpenAI API
echo   3) OpenAI-compatible API (custom endpoint)
echo   4) Echo mode (for testing, returns input as output)
echo.
set /p BACKEND_CHOICE="Enter your choice (1-4): "

if "!BACKEND_CHOICE!"=="1" (
    set BACKEND_MODE=lmstudio
    echo [INFO] You selected: Local (LM Studio / Ollama)
    echo.
    echo Default settings:
    echo   API URL: http://127.0.0.1:1234
    echo.
    set /p API_URL="Enter API URL [http://127.0.0.1:1234]: "
    if "!API_URL!"=="" set API_URL=http://127.0.0.1:1234

    set /p MODEL_NAME="Enter model name [mistralai/ministral-3-3b]: "
    if "!MODEL_NAME!"=="" set MODEL_NAME=mistralai/ministral-3-3b

) else if "!BACKEND_CHOICE!"=="2" (
    set BACKEND_MODE=openai_compatible
    echo [INFO] You selected: OpenAI API
    echo.
    set API_URL=https://api.openai.com/v1

    set /p API_KEY="Enter your OpenAI API key: "
    if "!API_KEY!"=="" (
        echo [ERROR] API key is required for OpenAI
        pause
        exit /b 1
    )

    REM Add API key to environment
    echo set OPENAI_API_KEY=!API_KEY!>> .venv\Scripts\activate.bat

    set /p MODEL_NAME="Enter model name [gpt-4]: "
    if "!MODEL_NAME!"=="" set MODEL_NAME=gpt-4

) else if "!BACKEND_CHOICE!"=="3" (
    set BACKEND_MODE=openai_compatible
    echo [INFO] You selected: OpenAI-compatible API
    echo.
    set /p API_URL="Enter API URL: "
    if "!API_URL!"=="" (
        echo [ERROR] API URL is required
        pause
        exit /b 1
    )

    set /p API_KEY="Enter API key (leave empty if not required): "
    if not "!API_KEY!"=="" (
        echo set OPENAI_API_KEY=!API_KEY!>> .venv\Scripts\activate.bat
    )

    set /p MODEL_NAME="Enter model name: "

) else if "!BACKEND_CHOICE!"=="4" (
    set BACKEND_MODE=echo
    echo [INFO] You selected: Echo mode (testing)
    set API_URL=
    set MODEL_NAME=echo

) else (
    echo [ERROR] Invalid choice
    pause
    exit /b 1
)

REM Create config file
echo [INFO] Generating configuration file...

(
echo # Prompt Evolver Configuration
echo # Generated by setup script
echo.
echo max_generations: 3
echo similarity_weight: 0.8
echo length_weight: 0.2
echo max_no_improve: 2
echo leakage_similarity_threshold: 0.45
echo leakage_ngram_size: 3
echo leakage_ngram_overlap_threshold: 0.1
echo min_prompt_tokens: 2
echo max_prompt_tokens: 200
echo max_prompt_increase_ratio: 2.0
echo max_prompt_increase_tokens: 40
echo min_improvement_attempts: 2
echo.
echo llm_execution:
echo   mode: !BACKEND_MODE!
echo   api_url: !API_URL!
echo   timeout_seconds: 30
echo.
echo llm_improvement:
echo   mode: !BACKEND_MODE!
echo   api_url: !API_URL!
echo   timeout_seconds: 30
echo.
echo llm_evaluation:
echo   mode: !BACKEND_MODE!
echo   api_url: !API_URL!
echo   timeout_seconds: 30
) > configs\config.yaml

echo [OK] Configuration file created at configs\config.yaml

if not "!MODEL_NAME!"=="echo" (
    echo [INFO] Model name to use in notebook: !MODEL_NAME!
    echo.
    echo Remember to set this in notebooks\PromptEvolver.ipynb:
    echo.
    echo   EXECUTION_MODEL = "!MODEL_NAME!"
    echo   IMPROVEMENT_MODEL = "!MODEL_NAME!"
    echo   EVALUATION_MODEL = "!MODEL_NAME!"
    echo.
)

REM Step 5: Validate setup
echo.
echo ========================================
echo Step 5: Validating Setup
echo ========================================
echo.

echo [INFO] Testing configuration loading...

python -c "import sys; sys.path.insert(0, 'src'); from prompt_evolver.config import load_config; config = load_config('configs/config.yaml'); print('Configuration loaded successfully'); print(f'  Execution mode: {config.llm_execution.mode}'); print(f'  Max generations: {config.max_generations}')"

if %errorlevel% neq 0 (
    echo [ERROR] Configuration validation failed
    pause
    exit /b 1
)

echo [OK] Configuration validation passed

REM Final instructions
echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.

echo [OK] Prompt Evolver is ready to use!
echo.
echo Next steps:
echo.
echo 1. Activate the virtual environment (when starting a new terminal):
echo    .venv\Scripts\activate.bat
echo.
echo 2. Open VS Code and the Jupyter notebook:
echo    code .
echo    Then open: notebooks\PromptEvolver.ipynb
echo.
echo 3. Update model names in the notebook (first cell):
echo    EXECUTION_MODEL = "!MODEL_NAME!"
echo.
echo 4. Run the cells to start optimizing prompts!
echo.
echo For help, see:
echo   - README.md
echo   - docs\overview.md
echo   - docs\quickstart.md
echo.

if "!BACKEND_MODE!"=="lmstudio" (
    echo [WARNING] Make sure your LLM backend is running at: !API_URL!
    echo.
) else if "!BACKEND_MODE!"=="openai_compatible" (
    if not "!API_URL!"=="https://api.openai.com/v1" (
        echo [WARNING] Make sure your LLM backend is running at: !API_URL!
        echo.
    )
)

pause
