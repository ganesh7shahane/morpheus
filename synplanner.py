"""
Retrosynthetic Planning Script using SynPlanner

This script provides functions for retrosynthetic route planning for molecules.
It uses the SynPlanner library for MCTS-based retrosynthesis.

Usage:
    python synplanner.py <smiles> [--output <output_file>] [--max_routes <n>]
"""

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from itertools import count, islice

# Check if SynPlanner is available
SYNPLANNER_AVAILABLE = False
try:
    from synplan.utils.loading import (
        download_all_data,
        load_building_blocks,
        load_reaction_rules,
        load_policy_function,
        load_evaluation_function,
    )
    from synplan.utils.config import TreeConfig, RolloutEvaluationConfig
    from synplan.chem.utils import mol_from_smiles, safe_canonicalization
    from synplan.mcts.tree import Tree
    from synplan.utils.visualisation import get_route_svg
    SYNPLANNER_AVAILABLE = True
except ImportError:
    pass

# CGRtools for SMILES canonicalization (consistent with SynPlanner)
CGRTOOLS_AVAILABLE = False
try:
    from CGRtools import smiles as read_cgr_smiles
    from CGRtools.containers.molecule import MoleculeContainer
    CGRTOOLS_AVAILABLE = True
except ImportError:
    pass

# RDKit for loading SDF files
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    pass

# Global variables for cached data
_data_folder: Optional[Path] = None
_building_blocks = None
_building_blocks_id_map: Dict[str, str] = {}  # SMILES -> ID mapping
_reaction_rules = None
_reaction_rules_path: Optional[Path] = None
_policy_function = None
_evaluation_function = None
_tree_config = None

# List of required large files that are downloaded by SynPlanner
# These files are in .gitignore and will be downloaded on first run
REQUIRED_SYNPLANNER_FILES = [
    "uspto/weights/ranking_policy_network.ckpt",
    "uspto/weights/filtering_policy_network.ckpt",
    "uspto/uspto_reaction_rules.pickle",
    "building_blocks/building_blocks_em_sa_ln.smi",
]


def check_synplanner_data_complete(data_folder: Path) -> bool:
    """
    Check if all required SynPlanner data files exist.
    Returns True if all files are present, False otherwise.
    """
    for rel_path in REQUIRED_SYNPLANNER_FILES:
        full_path = data_folder / rel_path
        if not full_path.exists():
            return False
    return True


def ensure_synplanner_data(data_folder: Optional[Path] = None) -> Path:
    """
    Ensure SynPlanner data is downloaded and return the data folder path.
    
    This function checks if all required large files exist in the synplan_data folder.
    If any files are missing, it downloads them using SynPlanner's download_all_data function.
    
    These large files (>50MB) are excluded from Git via .gitignore and are downloaded
    automatically when someone clones the repo and runs the application for the first time.
    
    Files downloaded:
    - uspto/weights/ranking_policy_network.ckpt (~157MB)
    - uspto/weights/filtering_policy_network.ckpt (~298MB)
    - uspto/uspto_standardized.smi (~610MB)
    - uspto/uspto_standardized.zip (~113MB)
    - building_blocks/building_blocks_em_sa_ln.sdf (~303MB)
    - tutorial/ranking_policy_training/ranking_policy_network.ckpt (~144MB)
    - chembl/molecules_for_filtering_policy_training.smi (~52MB)
    """
    global _data_folder
    
    if data_folder is None:
        # Default to synplan_data folder in the same directory as this script
        data_folder = Path(__file__).parent / "synplan_data"
    else:
        data_folder = Path(data_folder).resolve()
    
    # Check if data folder exists and has all required files
    needs_download = False
    if not data_folder.exists():
        needs_download = True
        print(f"SynPlanner data folder not found at {data_folder}")
    elif not check_synplanner_data_complete(data_folder):
        needs_download = True
        print(f"Some SynPlanner data files are missing in {data_folder}")
    
    if needs_download:
        print(f"Downloading SynPlanner data to {data_folder}...")
        print("This may take a while as the files are large (>1GB total)...")
        print("These files are excluded from Git and will be downloaded once per machine.")
        download_all_data(save_to=data_folder)
        print("Download complete!")
    
    _data_folder = data_folder
    return data_folder


def ensure_decompressed_data_files():
    """
    Ensure that compressed data files (.gz) are decompressed to their original form.
    
    The repository stores compressed versions of large data files to reduce Git size.
    This function decompresses them on first run if the uncompressed versions don't exist.
    
    Compressed files in data/:
    - filtered_chembl_activities.csv.gz -> filtered_chembl_activities.csv
    - whole_filtered_chembl_with_smiles.csv.gz -> whole_filtered_chembl_with_smiles.csv
    - building_blocks_em_sa_ln_with_ids.sdf.gz -> building_blocks_em_sa_ln_with_ids.sdf
    """
    data_folder = Path(__file__).parent / "data"
    
    # List of files that have compressed versions committed to Git
    compressed_files = [
        "filtered_chembl_activities.csv",
        "whole_filtered_chembl_with_smiles.csv",
        "building_blocks_em_sa_ln_with_ids.sdf",
    ]
    
    for filename in compressed_files:
        original_path = data_folder / filename
        compressed_path = data_folder / f"{filename}.gz"
        
        # If original doesn't exist but compressed does, decompress it
        if not original_path.exists() and compressed_path.exists():
            print(f"Decompressing {compressed_path.name}...")
            try:
                with gzip.open(compressed_path, 'rb') as f_in:
                    with open(original_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"  Created {original_path.name}")
            except Exception as e:
                print(f"  Error decompressing {compressed_path.name}: {e}")


def load_building_blocks_with_ids(sdf_path: Path) -> tuple:
    """
    Load building blocks from an SDF file and extract both SMILES and IDs.
    Uses CGRtools canonicalization for consistency with SynPlanner.
    
    Returns:
        building_blocks: frozenset of canonical SMILES (for tree search)
        smiles_to_id: dict mapping canonical SMILES to building block IDs
    """
    global _building_blocks_id_map
    
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required to load building blocks from SDF")
    
    building_blocks_set = set()
    smiles_to_id = {}
    
    suppl = Chem.SDMolSupplier(str(sdf_path))
    
    for mol in suppl:
        if mol is None:
            continue
        try:
            # Get the ID from the molecule
            mol_id = mol.GetProp("ID") if mol.HasProp("ID") else mol.GetProp("_Name")
            
            # Get SMILES using RDKit
            rdkit_smiles = Chem.MolToSmiles(mol)
            
            # Convert to CGRtools canonical SMILES for consistency with SynPlanner
            if CGRTOOLS_AVAILABLE:
                try:
                    cgr_mol = read_cgr_smiles(rdkit_smiles)
                    cgr_mol = safe_canonicalization(cgr_mol)
                    canonical_smiles = str(cgr_mol)
                except:
                    canonical_smiles = rdkit_smiles
            else:
                canonical_smiles = rdkit_smiles
            
            building_blocks_set.add(canonical_smiles)
            smiles_to_id[canonical_smiles] = mol_id
            
            # Also store original SMILES mapping
            if rdkit_smiles != canonical_smiles:
                smiles_to_id[rdkit_smiles] = mol_id
                
        except Exception as e:
            continue
    
    _building_blocks_id_map = smiles_to_id
    print(f"Loaded {len(building_blocks_set)} building blocks with {len(smiles_to_id)} ID mappings")
    
    return frozenset(building_blocks_set), smiles_to_id


def get_building_block_id(smiles: str) -> Optional[str]:
    """Get the building block ID for a given SMILES string."""
    global _building_blocks_id_map
    
    if smiles in _building_blocks_id_map:
        return _building_blocks_id_map[smiles]
    
    # Try CGRtools canonicalization
    if CGRTOOLS_AVAILABLE:
        try:
            cgr_mol = read_cgr_smiles(smiles)
            cgr_mol = safe_canonicalization(cgr_mol)
            canonical = str(cgr_mol)
            if canonical in _building_blocks_id_map:
                return _building_blocks_id_map[canonical]
        except:
            pass
    
    # Try RDKit canonical SMILES
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                canonical = Chem.MolToSmiles(mol)
                if canonical in _building_blocks_id_map:
                    return _building_blocks_id_map[canonical]
        except:
            pass
    
    return None


def initialize_synplanner(
    data_folder: Optional[Path] = None,
    building_blocks_sdf_path: Optional[Path] = None,
    max_iterations: int = 300,
    max_time: int = 120,
    max_depth: int = 9,
) -> bool:
    """
    Initialize SynPlanner components (building blocks, reaction rules, policy, etc.).
    """
    global _building_blocks, _reaction_rules, _reaction_rules_path
    global _policy_function, _evaluation_function, _tree_config, _data_folder
    
    if not SYNPLANNER_AVAILABLE:
        print("SynPlanner is not installed. Please install it with: pip install synplan")
        return False
    
    try:
        # Ensure SynPlanner data is downloaded (large files not in Git)
        data_folder = ensure_synplanner_data(data_folder)
        
        # Ensure compressed data files are decompressed
        ensure_decompressed_data_files()
        
        # Load building blocks
        if building_blocks_sdf_path is not None:
            building_blocks_sdf_path = Path(building_blocks_sdf_path)
            
            # Handle .sdf.gz files directly
            if str(building_blocks_sdf_path).endswith('.sdf.gz'):
                if building_blocks_sdf_path.exists():
                    # Decompress to a .sdf file
                    decompressed_path = Path(str(building_blocks_sdf_path)[:-3])  # Remove .gz
                    if not decompressed_path.exists():
                        print(f"Decompressing {building_blocks_sdf_path.name}...")
                        with gzip.open(building_blocks_sdf_path, 'rb') as f_in:
                            with open(decompressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    building_blocks_sdf_path = decompressed_path
            
            # Also check for decompressed version if .sdf was passed
            if not building_blocks_sdf_path.exists() and str(building_blocks_sdf_path).endswith('.sdf'):
                gz_path = Path(str(building_blocks_sdf_path) + '.gz')
                if gz_path.exists():
                    # Decompress the file
                    print(f"Decompressing {gz_path.name}...")
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(building_blocks_sdf_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
            if building_blocks_sdf_path.exists():
                _building_blocks, _ = load_building_blocks_with_ids(building_blocks_sdf_path)
            else:
                print(f"Building blocks SDF not found at {building_blocks_sdf_path}, using default")
                building_blocks_sdf_path = None
        
        if building_blocks_sdf_path is None:
            # Try default SDF path first (decompressed version)
            default_sdf_path = Path(__file__).parent / "data" / "building_blocks_em_sa_ln_with_ids.sdf"
            if default_sdf_path.exists():
                _building_blocks, _ = load_building_blocks_with_ids(default_sdf_path)
            else:
                # Fall back to original SMI file from downloaded data
                building_blocks_path = data_folder.joinpath("building_blocks/building_blocks_em_sa_ln.smi")
                if not building_blocks_path.exists():
                    print(f"Building blocks not found")
                    return False
                _building_blocks = load_building_blocks(building_blocks_path, standardize=False)
                print(f"Loaded {len(_building_blocks)} building blocks from SMI file (no IDs)")
        
        # Load reaction rules
        _reaction_rules_path = data_folder.joinpath("uspto/uspto_reaction_rules.pickle")
        if not _reaction_rules_path.exists():
            print(f"Reaction rules not found at {_reaction_rules_path}")
            return False
        _reaction_rules = load_reaction_rules(_reaction_rules_path)
        print(f"Loaded {len(_reaction_rules)} reaction rules")
        
        # Load policy function
        policy_network_path = data_folder.joinpath("uspto/weights/ranking_policy_network.ckpt")
        if not policy_network_path.exists():
            print(f"Policy network not found at {policy_network_path}")
            return False
        _policy_function = load_policy_function(weights_path=policy_network_path)
        print("Loaded policy network")
        
        # Create tree configuration
        _tree_config = TreeConfig(
            search_strategy="expansion_first",
            max_iterations=max_iterations,
            max_time=max_time,
            max_depth=max_depth,
            min_mol_size=1,
            init_node_value=0.5,
            ucb_type="uct",
            c_ucb=0.1,
        )
        
        # Create evaluation function
        eval_config = RolloutEvaluationConfig(
            policy_network=_policy_function,
            reaction_rules=_reaction_rules,
            building_blocks=_building_blocks,
            min_mol_size=_tree_config.min_mol_size,
            max_depth=_tree_config.max_depth,
        )
        _evaluation_function = load_evaluation_function(eval_config)
        print("Initialized evaluation function")
        
        return True
        
    except Exception as e:
        print(f"Error initializing SynPlanner: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# SVG Rendering with Building Block IDs (from notebook)
# =============================================================================

def render_svg_with_labels(pred, columns, box_colors, smiles_to_id=None):
    """
    Renders an SVG representation of a retrosynthetic route with building block ID labels.
    """
    if not CGRTOOLS_AVAILABLE:
        return None
        
    x_shift = 0.0
    c_max_x = 0.0
    c_max_y = 0.0
    render = []
    cx = count()
    cy = count()
    arrow_points = {}
    label_positions = []  # Store label positions for building blocks
    
    for ms in columns:
        heights = []
        for m in ms:
            m.clean2d()
            min_x = min(x for x, y in m._plane.values()) - x_shift
            min_y = min(y for x, y in m._plane.values())
            m._plane = {n: (x - min_x, y - min_y) for n, (x, y) in m._plane.items()}
            max_x = max(x for x, y in m._plane.values())
            c_max_x = max(c_max_x, max_x)
            arrow_points[next(cx)] = [x_shift, max_x]
            heights.append(max(y for x, y in m._plane.values()))

        x_shift = c_max_x + 5.0
        y_shift = sum(heights) + 3.0 * (len(heights) - 1)
        c_max_y = max(c_max_y, y_shift)
        y_shift /= 2.0
        
        for m, h in zip(ms, heights):
            m._plane = {n: (x, y - y_shift) for n, (x, y) in m._plane.items()}
            
            max_x = max(x for x, y in m._plane.values()) + 0.9
            min_x = min(x for x, y in m._plane.values()) - 0.6
            max_y = -(max(y for x, y in m._plane.values()) + 0.45)
            min_y = -(min(y for x, y in m._plane.values()) - 0.45)
            x_delta = abs(max_x - min_x)
            y_delta = abs(max_y - min_y)
            
            box = (
                f'<rect x="{min_x}" y="{max_y}" rx="{y_delta * 0.1}" ry="{y_delta * 0.1}" '
                f'width="{x_delta}" height="{y_delta}" stroke="black" stroke-width=".0025" '
                f'fill="{box_colors[m.meta["status"]]}" fill-opacity="0.30"/>'
            )
            
            # Store position for building block labels
            if m.meta.get("status") == "instock" and smiles_to_id:
                smiles = str(m)
                bb_id = smiles_to_id.get(smiles) or get_building_block_id(smiles)
                if bb_id:
                    label_x = (min_x + max_x) / 2
                    label_y = min_y + 0.3  # Position below the box
                    label_positions.append((label_x, label_y, bb_id))
            
            arrow_points[next(cy)].append(y_shift - h / 2.0)
            y_shift -= h + 3.0
            depicted_molecule = list(m.depict(embedding=True))[:3]
            depicted_molecule.append(box)
            render.append(depicted_molecule)

    # Calculate mid-X coordinate for arrows
    graph = {}
    for s, p in pred:
        try:
            graph[s].append(p)
        except KeyError:
            graph[s] = [p]
    for s, ps in graph.items():
        mid_x = float("-inf")
        for p in ps:
            s_min_x, s_max, s_y = arrow_points[s][:3]
            p_min_x, p_max, p_y = arrow_points[p][:3]
            p_max += 1
            mid = p_max + (s_min_x - p_max) / 3
            mid_x = max(mid_x, mid)
        for p in ps:
            arrow_points[p].append(mid_x)

    config = MoleculeContainer._render_config
    font_size = config["font_size"]
    font125 = 1.25 * font_size
    width = c_max_x + 4.0 * font_size
    height = c_max_y + 3.5 * font_size
    box_y = height / 2.0
    
    svg = [
        f'<svg width="{0.6 * width:.2f}cm" height="{0.6 * height:.2f}cm" '
        f'viewBox="{-font125:.2f} {-box_y:.2f} {width:.2f} '
        f'{height:.2f}" xmlns="http://www.w3.org/2000/svg" version="1.1">',
        '  <defs>\n    <marker id="arrow" markerWidth="10" markerHeight="10" '
        'refX="0" refY="3" orient="auto">\n      <path d="M0,0 L0,6 L9,3"/>\n    </marker>\n  </defs>',
    ]

    # Draw arrows
    for s, p in pred:
        s_min_x, s_max, s_y = arrow_points[s][:3]
        p_min_x, p_max, p_y = arrow_points[p][:3]
        p_max += 1
        mid_x = arrow_points[p][-1]
        arrow = (
            f'  <polyline points="{p_max:.2f} {p_y:.2f}, {mid_x:.2f} {p_y:.2f}, '
            f'{mid_x:.2f} {s_y:.2f}, {s_min_x - 1.:.2f} {s_y:.2f}" '
            f'fill="none" stroke="black" stroke-width=".04" marker-end="url(#arrow)"/>'
        )
        if p_y != s_y:
            arrow += f'  <circle cx="{mid_x}" cy="{p_y}" r="0.1"/>'
        svg.append(arrow)
    
    # Draw molecules
    for atoms, bonds, masks, box in render:
        molecule_svg = MoleculeContainer._graph_svg(
            atoms, bonds, masks, -font125, -box_y, width, height
        )
        molecule_svg.insert(1, box)
        svg.extend(molecule_svg)
    
    # Add building block ID labels
    for label_x, label_y, bb_id in label_positions:
        svg.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" '
            f'text-anchor="middle" font-size="0.35" font-family="sans-serif" '
            f'fill="#006400" font-weight="bold">{bb_id}</text>'
        )
    
    svg.append("</svg>")
    return "\n".join(svg)


def get_route_svg_with_bb_ids(tree, node_id, smiles_to_id=None):
    """
    Visualizes the retrosynthetic route with building block IDs labeled.
    """
    if not CGRTOOLS_AVAILABLE:
        # Fall back to standard SVG without labels
        return get_route_svg(tree, node_id)
    
    if node_id not in tree.winning_nodes:
        return None
    
    nodes = tree.route_to_node(node_id)
    
    # Use the provided map or the global one
    if smiles_to_id is None:
        smiles_to_id = _building_blocks_id_map
    
    # Set up node types for different box colors
    for n in nodes:
        for precursor in n.new_precursors:
            precursor.molecule.meta["status"] = (
                "instock"
                if precursor.is_building_block(tree.building_blocks)
                else "mulecule"
            )
    nodes[0].curr_precursor.molecule.meta["status"] = "target"
    
    box_colors = {
        "target": "#98EEFF",
        "mulecule": "#F0AB90",
        "instock": "#9BFAB3",
    }

    # Build columns
    columns = [
        [nodes[0].curr_precursor.molecule],
        [x.molecule for x in nodes[1].new_precursors],
    ]
    pred = {x: 0 for x in range(1, len(columns[1]) + 1)}
    
    cx = [
        n
        for n, x in enumerate(nodes[1].new_precursors, 1)
        if not x.is_building_block(tree.building_blocks)
    ]
    size = len(cx)
    nodes_iter = iter(nodes[2:])
    cy = count(len(columns[1]) + 1)
    
    while size:
        layer = []
        for s in islice(nodes_iter, size):
            n = cx.pop(0)
            for x in s.new_precursors:
                layer.append(x)
                m = next(cy)
                if not x.is_building_block(tree.building_blocks):
                    cx.append(m)
                pred[m] = n
        size = len(cx)
        columns.append([x.molecule for x in layer])

    columns = [columns[::-1] for columns in columns[::-1]]
    
    pred = tuple(
        (abs(source - len(pred)), abs(target - len(pred)))
        for target, source in pred.items()
    )
    
    return render_svg_with_labels(pred, columns, box_colors, smiles_to_id)


def plan_synthesis(
    smiles: str,
    max_routes: int = 4,
    return_svg: bool = True,
) -> Dict[str, Any]:
    """
    Plan retrosynthetic routes for a given SMILES string.
    """
    global _building_blocks, _reaction_rules, _policy_function, _evaluation_function, _tree_config
    
    if not SYNPLANNER_AVAILABLE:
        return {
            'success': False,
            'solved': False,
            'routes': [],
            'error': 'SynPlanner is not installed'
        }
    
    # Initialize if not already done
    if _building_blocks is None or _reaction_rules is None:
        if not initialize_synplanner():
            return {
                'success': False,
                'solved': False,
                'routes': [],
                'error': 'Failed to initialize SynPlanner'
            }
    
    try:
        # Parse the target molecule
        target_molecule = mol_from_smiles(smiles, clean2d=True, standardize=True, clean_stereo=True)
        if target_molecule is None:
            return {
                'success': False,
                'solved': False,
                'routes': [],
                'error': f'Could not parse SMILES: {smiles}'
            }
        
        # Create the search tree
        tree = Tree(
            target=target_molecule,
            config=_tree_config,
            reaction_rules=_reaction_rules,
            building_blocks=_building_blocks,
            expansion_function=_policy_function,
            evaluation_function=_evaluation_function,
        )
        
        # Run the search
        tree_solved = False
        for solved, node_id in tree:
            if solved:
                tree_solved = True
        
        # Collect routes
        routes = []
        if tree_solved and hasattr(tree, 'winning_nodes'):
            for n, node_id in enumerate(tree.winning_nodes):
                if n >= max_routes:
                    break
                
                route_info = {
                    'node_id': node_id,
                    'score': tree.route_score(node_id),
                }
                
                if return_svg:
                    try:
                        # Try custom SVG with BB IDs first
                        svg = get_route_svg_with_bb_ids(tree, node_id, _building_blocks_id_map)
                        if svg is None:
                            # Fall back to standard SVG
                            svg = get_route_svg(tree, node_id)
                        route_info['svg'] = svg
                    except Exception as e:
                        # Fall back to standard SVG on error
                        try:
                            route_info['svg'] = get_route_svg(tree, node_id)
                        except Exception as e2:
                            route_info['svg'] = None
                            route_info['svg_error'] = str(e2)
                
                routes.append(route_info)
        
        return {
            'success': True,
            'solved': tree_solved,
            'routes': routes,
            'num_iterations': tree.curr_iteration if hasattr(tree, 'curr_iteration') else None,
            'target_smiles': smiles,
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'solved': False,
            'routes': [],
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def get_route_image(route_svg: str) -> bytes:
    """Convert route SVG to PNG bytes."""
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=route_svg.encode('utf-8'))
        return png_bytes
    except ImportError:
        print("cairosvg is required for PNG conversion. Install with: pip install cairosvg")
        return None
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return None


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Retrosynthetic planning using SynPlanner'
    )
    parser.add_argument(
        'smiles',
        type=str,
        help='Target molecule SMILES string'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--max_routes', '-n',
        type=int,
        default=4,
        help='Maximum number of routes to return (default: 4)'
    )
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=300,
        help='Maximum MCTS iterations (default: 300)'
    )
    parser.add_argument(
        '--max_time',
        type=int,
        default=120,
        help='Maximum search time in seconds (default: 120)'
    )
    parser.add_argument(
        '--data_folder',
        type=str,
        default=None,
        help='Path to SynPlanner data folder'
    )
    parser.add_argument(
        '--building_blocks_sdf',
        type=str,
        default=None,
        help='Path to SDF file with building blocks and IDs'
    )
    parser.add_argument(
        '--no_svg',
        action='store_true',
        help='Do not include SVG visualizations'
    )
    
    args = parser.parse_args()
    
    # Initialize with custom parameters
    if not initialize_synplanner(
        data_folder=args.data_folder,
        building_blocks_sdf_path=args.building_blocks_sdf,
        max_iterations=args.max_iterations,
        max_time=args.max_time,
    ):
        sys.exit(1)
    
    # Run planning
    result = plan_synthesis(
        args.smiles,
        max_routes=args.max_routes,
        return_svg=not args.no_svg,
    )
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))
    
    if result['success'] and result['solved']:
        print(f"\n✓ Found {len(result['routes'])} synthesis routes")
        sys.exit(0)
    elif result['success']:
        print("\n✗ No synthesis route found")
        sys.exit(0)
    else:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
