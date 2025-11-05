"""
Display utilities for consistent and beautiful terminal output
Provides standardized formatting for all scripts
"""


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text, char='='):
    """Print a colored header with separator lines"""
    width = 80
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.ENDC}\n")


def print_section(text):
    """Print a section title"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}► {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─' * 80}{Colors.ENDC}\n")


def print_subsection(text):
    """Print a subsection title"""
    print(f"\n{Colors.BOLD}▸ {text}{Colors.ENDC}")


def print_success(text):
    """Print a success message"""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_info(text, indent=0):
    """Print an info message with optional indentation"""
    prefix = "  " * indent
    print(f"{prefix}{Colors.CYAN}•{Colors.ENDC} {text}")


def print_warning(text):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {text}")


def print_error(text):
    """Print an error message"""
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")


def print_metric(label, value, unit="", good_threshold=None):
    """
    Print a metric with optional color coding based on threshold
    
    Args:
        label: Metric label
        value: Metric value
        unit: Optional unit (e.g., '%', 'ms')
        good_threshold: If provided, color code based on value >= threshold
    """
    if good_threshold is not None and isinstance(value, (int, float)):
        if value >= good_threshold:
            color = Colors.GREEN
        else:
            color = Colors.YELLOW
        formatted_value = f"{color}{value}{unit}{Colors.ENDC}"
    else:
        formatted_value = f"{value}{unit}"
    
    print(f"  {label}: {formatted_value}")


def print_dict(data, indent=1):
    """Pretty print a dictionary with indentation"""
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{'  ' * indent}{Colors.BOLD}{key}:{Colors.ENDC}")
            print_dict(value, indent + 1)
        else:
            print(f"{'  ' * indent}{key}: {value}")


def print_progress(current, total, prefix="Progress"):
    """Print a simple progress indicator"""
    percent = 100 * (current / float(total))
    bar_length = 40
    filled = int(bar_length * current // total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})", end='', flush=True)
    if current == total:
        print()  # New line when complete


def print_separator(char='─'):
    """Print a simple separator line"""
    print(f"{Colors.BLUE}{char * 80}{Colors.ENDC}")


def print_model_summary(model_name, params, config):
    """Print a formatted model summary"""
    print_subsection(f"{model_name} Architecture")
    print(f"  Total Parameters: {Colors.BOLD}{params:,}{Colors.ENDC}")
    for key, value in config.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
