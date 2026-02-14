
import sys
import os

def setup_paths():
    """
    Robustly setup python paths to ensure gnn_learning package is found.
    This handles differences between local development and Streamlit Cloud deployment.
    """
    # Current file directory (app/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Potential source directories to check
    # 1. ../src relative to app/ (Development/Standard structure)
    # 2. ./src relative to CWD (Streamlit Cloud root)
    # 3. ./ relative to CWD (if src is not used)
    
    candidates = [
        os.path.abspath(os.path.join(current_dir, '../src')),  # ../src
        os.path.abspath(os.path.join(os.getcwd(), 'src')),     # ./src
        os.getcwd(),                                           # ./
    ]
    
    # print("DEBUG: Path configuration starting...")
    # print(f"DEBUG: Current file: {__file__}")
    # print(f"DEBUG: Current dir: {current_dir}")
    # print(f"DEBUG: Working dir: {os.getcwd()}")
    
    src_found = False
    for path in candidates:
        # Check if 'gnn_learning' exists in this path
        package_path = os.path.join(path, 'gnn_learning')
        if os.path.isdir(package_path) or os.path.isfile(package_path + '.py'):
            # print(f"DEBUG: Found gnn_learning at {package_path}")
            if path not in sys.path:
                sys.path.insert(0, path)
                # print(f"DEBUG: Inserted {path} to sys.path")
            src_found = True
            break
    
    if not src_found:
        # Fallback: Just add ../src and hope for the best
        default_src = os.path.abspath(os.path.join(current_dir, '../src'))
        if default_src not in sys.path:
            sys.path.insert(0, default_src)
            # print(f"DEBUG: Fallback - Inserted {default_src} to sys.path")

    # Debug imports
    try:
        import gnn_learning
        # print(f"DEBUG: Successfully imported gnn_learning from {gnn_learning.__file__}")
        
        # Check data module specifically since that was failing
        try:
            from gnn_learning import data
            # print(f"DEBUG: Successfully imported gnn_learning.data from {data.__file__}")
        except ImportError as e:
            # print(f"DEBUG: WARNING - Imported gnn_learning but failed to import data: {e}")
            pass
            
    except ImportError as e:
        # print(f"DEBUG: ERROR - Failed to import gnn_learning: {e}")
        # print(f"DEBUG: sys.path: {sys.path}")
        pass
