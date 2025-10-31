import os
import subprocess
import tempfile
import logging
from typing import Optional, Dict, List, Union, Tuple

class VMDLAMMPSConverter:
    """
    A class for converting PDB files to LAMMPS data files using VMD.
    This class uses VMD's Topo Tools plugin to create LAMMPS data files with
    full molecular topology information (bonds, angles, etc.).
    """
    
    def __init__(self,
                 vmd_path: Optional[str] = None,
                 working_dir: Optional[str] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the converter.
        
        Args:
            vmd_path: Path to the VMD executable. If None, tries to find it in PATH.
            working_dir: Directory to store temporary files and output. If None, uses a temp directory.
            log_level: Logging level (default: logging.INFO).
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Find VMD executable
        self.vmd_path = vmd_path or self._find_vmd()
        if not self.vmd_path:
            raise ValueError("VMD executable not found. Please provide the path to VMD.")
            
        # Setup working directory
        if working_dir:
            self.working_dir = working_dir
            os.makedirs(working_dir, exist_ok=True)
            self.temp_dir = None
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.working_dir = self.temp_dir.name
            
        self.logger.info(f"Initialized VMD-LAMMPS converter. Working directory: {self.working_dir}")
        
    def __del__(self):
        """Clean up temporary directory if created."""
        if self.temp_dir:
            self.temp_dir.cleanup()
            
    def _find_vmd(self) -> Optional[str]:
        """
        Try to find the VMD executable in the system PATH.
        
        Returns:
            Path to VMD executable if found, None otherwise.
        """
        # Common names/locations for the VMD executable
        vmd_names = ["vmd", "vmd.exe"]
        
        for name in vmd_names:
            try:
                path = subprocess.check_output(["which", name],
                                              stderr=subprocess.DEVNULL,
                                              text=True).strip()
                if path:
                    return path
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
        # Check common installation directories
        common_paths = [
            "/usr/local/bin/vmd",
            "/opt/vmd/bin/vmd",
            "C:\\Program Files\\VMD\\vmd.exe",
            "C:\\Program Files (x86)\\VMD\\vmd.exe",
            "/Applications/VMD 1.9.4a57-arm64-Rev12.app/Contents/MacOS/vmd", # Common macOS path
        ]
        
        for path in common_paths:
            if os.path.isfile(path):
                return path
                
        return None
        
    def convert(self,
                pdb_file: str,
                output_file: Optional[str] = None,
                box_dimensions: Optional[Union[float, Tuple[float, float, float]]] = None,
                options: Dict[str, Union[str, bool, int]] = None) -> str:
        """
        Convert a PDB file to a LAMMPS data file using VMD.
        
        Args:
            pdb_file: Path to the input PDB file.
            output_file: Path for the output LAMMPS data file. If None, generates a name.
            box_dimensions: Either a single float for cubic box (e.g. 40.0 for 40Å³) or 
                           a tuple of three floats (x, y, z) for non-cubic box dimensions.
            options: Dictionary of conversion options:
                - autobonds: Whether to automatically generate bonds (default: False)
                - retypebonds: Whether to retype bonds (default: True)
                - guessangles: Whether to guess angles (default: True)
                - guess_dihedrals: Whether to guess dihedrals (default: False)
                - guess_impropers: Whether to guess impropers (default: False)
                - style: LAMMPS data file style (default: 'full')
                - atom_style: LAMMPS atom style (default: 'full')
                - center_system: Center molecules in the box (default: True)
                
        Returns:
            Path to the generated LAMMPS data file.
        """
        if not os.path.isfile(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
            
        # Set default options
        default_options = {
            "autobonds": False,
            "retypebonds": True,
            "guessangles": True,
            "guess_dihedrals": False,
            "guess_impropers": False,
            "style": "full",
            "atom_style": "full",
            "center_system": True
        }
        
        # Update with user options
        if options:
            default_options.update(options)
            
        options = default_options
        
        # If box dimensions not provided, try to extract from filename
        if box_dimensions is None:
            filename = os.path.basename(pdb_file)
            # Try to extract box size from filename (e.g., "nacl_water_0_5M_40A.pdb")
            import re
            match = re.search(r'(\d+)A', filename)
            if match:
                box_size = float(match.group(1))
                self.logger.info(f"Extracted box size from filename: {box_size}Å")
                box_dimensions = box_size
            else:
                # Default to a reasonable size if we can't determine it
                box_dimensions = 40.0
                self.logger.warning(f"Box dimensions not provided and couldn't be extracted from filename. "
                                   f"Using default: {box_dimensions}Å cubic box")
            
        # Generate output filename if not provided
        if not output_file:
            base_name = os.path.splitext(os.path.basename(pdb_file))[0]
            output_file = os.path.join(self.working_dir, f"{base_name}.lammps")
            
        # Create VMD script with box dimensions
        script_path = self._create_vmd_script(pdb_file, output_file, options, box_dimensions)
        
        # Run VMD with the script
        self.logger.info(f"Converting {pdb_file} to LAMMPS data file {output_file}...")
        self._run_vmd_script(script_path)
        
        # Check if the output file was created
        if not os.path.isfile(output_file):
            raise RuntimeError(f"Failed to generate LAMMPS data file: {output_file}")
            
        self.logger.info(f"Successfully created LAMMPS data file: {output_file}")
        return output_file
        
    def _create_vmd_script(self,
                          pdb_file: str,
                          output_file: str,
                          options: Dict[str, Union[str, bool, int]],
                          box_dimensions: Union[float, Tuple[float, float, float]]) -> str:
        """
        Create a VMD script to convert PDB to LAMMPS data file.
        
        Args:
            pdb_file: Path to the input PDB file.
            output_file: Path for the output LAMMPS data file.
            options: Dictionary of conversion options.
            box_dimensions: Box dimensions as float (cubic) or tuple (x, y, z).
            
        Returns:
            Path to the created VMD script.
        """
        # Create script filename
        script_path = os.path.join(self.working_dir, "convert_to_lammps.tcl")
        
        # Handle box dimensions
        if isinstance(box_dimensions, (int, float)):
            # Cubic box
            box_x = box_y = box_z = float(box_dimensions)
        elif isinstance(box_dimensions, (list, tuple)) and len(box_dimensions) == 3:
            # Non-cubic box
            box_x, box_y, box_z = [float(dim) for dim in box_dimensions]
        else:
            raise ValueError(f"Invalid box dimensions: {box_dimensions}. "
                            f"Must be a single float for cubic box or tuple of 3 floats for non-cubic box.")
            
        # Create script content
        script_content = [
            "# VMD script to convert PDB to LAMMPS data file",
            "package require topotools",
            "",
            f"mol new \"{os.path.abspath(pdb_file)}\" "
            f"autobonds {'yes' if options['autobonds'] else 'no'} waitfor all",
            "",
            "# Set box dimensions",
            f"molinfo top set a {{$box_x 0.0 0.0}}".replace("$box_x", str(box_x)),
            f"molinfo top set b {{0.0 $box_y 0.0}}".replace("$box_y", str(box_y)),
            f"molinfo top set c {{0.0 0.0 $box_z}}".replace("$box_z", str(box_z)),
            "",
        ]
        
        # Add centering if requested
        if options.get("center_system", True):
            script_content.extend([
                "# Center the system in the box",
                "set all [atomselect top all]",
                "set center [measure center $all]",
                "$all moveby [vecscale -1.0 $center]  # Center at origin",
                f"$all moveby [list {box_x/2} {box_y/2} {box_z/2}]  # Center in box",
                ""
            ])
            
        # Add topology processing commands
        if options["retypebonds"]:
            script_content.append("topo retypebonds")
            
        if options["guessangles"]:
            script_content.append("topo guessangles")
            
        if options["guess_dihedrals"]:
            script_content.append("topo guessdihedrals")
            
        if options["guess_impropers"]:
            script_content.append("topo guessimpropers")
        
        # Add the write command
        script_content.append(
            f"topo writelammpsdata \"{os.path.abspath(output_file)}\" "
            f"{options['style']} {{'atom_style' {options['atom_style']}}}"
        )
        
        # Add exit command
        script_content.append("exit")
        
        # Write script to file
        with open(script_path, 'w') as f:
            f.write('\n'.join(script_content))
            
        return script_path
        
    def _run_vmd_script(self, script_path: str) -> None:
        """
        Run a VMD script.
        
        Args:
            script_path: Path to the VMD script.
        """
        cmd = [self.vmd_path, "-dispdev", "text", "-e", script_path]
        
        self.logger.debug(f"Running VMD command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.logger.debug(result.stdout)
            
            if result.stderr:
                self.logger.warning(f"VMD warnings or errors:\n{result.stderr}")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"VMD execution failed: {e}")
            self.logger.error(f"VMD stderr: {e.stderr}")
            raise RuntimeError("VMD execution failed. See log for details.")
            
    def calculate_box_from_pdb(self, pdb_file: str, padding: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate appropriate box dimensions from the PDB file by analyzing atom positions.
        
        Args:
            pdb_file: Path to the PDB file.
            padding: Extra space to add around the system in Ångstroms.
            
        Returns:
            Tuple of (x_size, y_size, z_size) for box dimensions.
        """
        # Create a temporary script to calculate box dimensions
        script_path = os.path.join(self.working_dir, "calc_box_size.tcl")
        output_path = os.path.join(self.working_dir, "box_size.txt")
        
        script_content = [
            "package require topotools",
            f"mol new {pdb_file} waitfor all",
            "set all [atomselect top all]",
            "set minmax [measure minmax $all]",
            "set min_coords [lindex $minmax 0]",
            "set max_coords [lindex $minmax 1]",
            f"set padding {padding}",
            "set xsize [expr [lindex $max_coords 0] - [lindex $min_coords 0] + $padding]",
            "set ysize [expr [lindex $max_coords 1] - [lindex $min_coords 1] + $padding]",
            "set zsize [expr [lindex $max_coords 2] - [lindex $min_coords 2] + $padding]",
            f"set outfile [open {output_path} w]",
            "puts $outfile \"$xsize $ysize $zsize\"",
            "close $outfile",
            "exit"
        ]
        
        with open(script_path, 'w') as f:
            f.write('\n'.join(script_content))
            
        # Run the script
        self._run_vmd_script(script_path)
        
        # Read the calculated box dimensions
        with open(output_path, 'r') as f:
            box_dims = f.read().strip().split()
            x_size, y_size, z_size = map(float, box_dims)
            
        self.logger.info(f"Calculated box dimensions: {x_size}×{y_size}×{z_size}Å")
        return (x_size, y_size, z_size)
        
    def post_process_data_file(self, data_file: str, options: Dict[str, any] = None) -> str:
        """
        Post-process the LAMMPS data file if needed (e.g., fix atom types, adjust parameters).
        
        Args:
            data_file: Path to the LAMMPS data file.
            options: Dictionary of post-processing options.
            
        Returns:
            Path to the processed LAMMPS data file.
        """
        # This is a placeholder for any post-processing needed
        if not options:
            return data_file
            
        # Create a new file name
        base_name, ext = os.path.splitext(data_file)
        processed_file = f"{base_name}_processed{ext}"
        
        # Read the original file
        with open(data_file, 'r') as f:
            lines = f.readlines()
            
        # Process the file (this is just an example, implement as needed)
        # This would need to be customized based on specific requirements
        if "adjust_mass" in options:
            # Example: adjust masses of specific atom types
            pass
            
        if "rename_atom_types" in options:
            # Example: rename atom types
            pass
            
        # Write the processed file
        with open(processed_file, 'w') as f:
            f.writelines(lines)
            
        return processed_file
    
    def add_box_to_pdb(self, input_pdb: str, box_dimensions: Union[float, Tuple[float, float, float]]) -> str:
        """
        Add box dimensions to a PDB file by adding a CRYST1 record.
        
        Args:
            input_pdb: Path to the input PDB file.
            box_dimensions: Box dimensions as float (cubic) or tuple (x, y, z).
            
        Returns:
            Path to the modified PDB file.
        """
        # Process box dimensions
        if isinstance(box_dimensions, (int, float)):
            # Cubic box
            box_x = box_y = box_z = float(box_dimensions)
        elif isinstance(box_dimensions, (list, tuple)) and len(box_dimensions) == 3:
            # Non-cubic box
            box_x, box_y, box_z = [float(dim) for dim in box_dimensions]
        else:
            raise ValueError("Box dimensions must be a single value for cubic box or tuple of 3 values.")
        
        # Create output filename
        output_pdb = os.path.join(
            self.working_dir, 
            f"{os.path.splitext(os.path.basename(input_pdb))[0]}_with_box.pdb"
        )
        
        # Create CRYST1 record
        cryst_line = f"CRYST1{box_x:8.3f}{box_y:8.3f}{box_z:8.3f}  90.00  90.00  90.00 P 1           1\n"
        
        # Add to the PDB file
        with open(input_pdb, 'r') as f_in:
            content = f_in.readlines()
            
        with open(output_pdb, 'w') as f_out:
            f_out.write(cryst_line)
            f_out.writelines(content)
            
        self.logger.info(f"Created PDB with box dimensions: {output_pdb}")
        return output_pdb
