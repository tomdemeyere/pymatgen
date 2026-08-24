from __future__ import annotations

import itertools
import re
from itertools import islice
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.io.lobster.future.constants import LOBSTER_ORBITALS
from pymatgen.io.lobster.future.core import LobsterFile
from pymatgen.io.lobster.future.utils import parse_orbital_from_text
from pymatgen.io.lobster.future.versioning import version_processor
from pymatgen.io.vasp.outputs import VolumetricData

if TYPE_CHECKING:
    from typing import Literal

    from numpy import floating

    from pymatgen.core.structure import Structure
    from pymatgen.io.lobster.future.types import LobsterMatrixData
    from pymatgen.util.typing import PathLike


class Wavefunction(LobsterFile):
    """Parser for wave function files from LOBSTER.

    Reads wave function files and creates VolumetricData objects.

    Attributes:
        grid (tuple[int, int, int]): Grid for the wave function [Nx+1, Ny+1, Nz+1].
        points (list[tuple[float, float, float]]): Points in real space.
        reals (list[float]): Real parts of the wave function.
        imaginaries (list[float]): Imaginary parts of the wave function.
        distances (list[float]): Distances to the first point in the wave function file.
        structure (Structure): Structure object associated with the calculation.
    """

    def __init__(
        self,
        filename: PathLike,
        structure: Structure,
        process_immediately: bool = True,
        lobster_version: str | None = None,
    ) -> None:
        """Initialize the Wavefunction parser.

        Args:
            filename (PathLike): The wavecar file from LOBSTER.
            structure (Structure): The Structure object.
            process_immediately (bool): Whether to parse the file immediately. Defaults to True.
        """
        super().__init__(
            filename,
            process_immediately=process_immediately,
            lobster_version=lobster_version,
        )

        self.structure = structure

    @version_processor()
    def parse_file(
        self,
    ) -> None:
        """Parse wave function file.

        Reads the wave function file and extracts grid, points, real and imaginary parts,
        and distances.

        Raises:
            ValueError: If the number of real or imaginary parts does not match the expected grid size.
        """
        lines_generator = self.iterate_lines()

        line_parts = next(lines_generator).split()

        self.grid: tuple[int, int, int] = [
            int(line_parts[7]),
            int(line_parts[8]),
            int(line_parts[9]),
        ]
        n_points = self.grid[0] * self.grid[1] * self.grid[2]

        self.points = np.empty((n_points, 3), dtype=np.float64)
        self.distances = np.empty(n_points, dtype=np.float64)
        self.reals = np.empty(n_points, dtype=np.float64)
        self.imaginaries = np.empty(n_points, dtype=np.float64)

        i = 0
        for line in lines_generator:
            line_parts = line.split()

            if len(line_parts) >= 6:
                self.points[i] = (
                    float(line_parts[0]),
                    float(line_parts[1]),
                    float(line_parts[2]),
                )
                self.distances[i] = float(line_parts[3])
                self.reals[i] = float(line_parts[4])
                self.imaginaries[i] = float(line_parts[5])
                i += 1

        if (
            len(self.reals) != self.grid[0] * self.grid[1] * self.grid[2]
            or len(self.imaginaries) != self.grid[0] * self.grid[1] * self.grid[2]
        ):
            raise ValueError("Something went wrong while reading the file")

    def set_volumetric_data(self, grid: tuple[int, int, int], structure: Structure) -> None:
        """Create VolumetricData instances for real, imaginary, and density parts.

        Args:
            grid (tuple[int, int, int]): Grid on which wavefunction was calculated.
            structure (Structure): Structure object.

        Raises:
            ValueError: If the wavefunction file does not contain all relevant points.
        """
        Nx = grid[0] - 1
        Ny = grid[1] - 1
        Nz = grid[2] - 1
        a = structure.lattice.matrix[0]
        b = structure.lattice.matrix[1]
        c = structure.lattice.matrix[2]
        new_x = []
        new_y = []
        new_z = []
        new_real = []
        new_imaginary = []
        new_density = []

        for runner, (x, y, z) in enumerate(itertools.product(range(Nx + 1), range(Ny + 1), range(Nz + 1))):
            x_here = x / float(Nx) * a[0] + y / float(Ny) * b[0] + z / float(Nz) * c[0]
            y_here = x / float(Nx) * a[1] + y / float(Ny) * b[1] + z / float(Nz) * c[1]
            z_here = x / float(Nx) * a[2] + y / float(Ny) * b[2] + z / float(Nz) * c[2]

            if x != Nx and y != Ny and z != Nz:
                if (
                    not np.isclose(self.points[runner][0], x_here, 1e-3)
                    and not np.isclose(self.points[runner][1], y_here, 1e-3)
                    and not np.isclose(self.points[runner][2], z_here, 1e-3)
                ):
                    raise ValueError(
                        "The provided wavefunction from Lobster does not contain all relevant"
                        " points. "
                        "Please use a line similar to: printLCAORealSpaceWavefunction kpoint 1 "
                        "coordinates 0.0 0.0 0.0 coordinates 1.0 1.0 1.0 box bandlist 1 "
                    )

                new_x.append(x_here)
                new_y.append(y_here)
                new_z.append(z_here)

                new_real.append(self.reals[runner])
                new_imaginary.append(self.imaginaries[runner])
                new_density.append(self.reals[runner] ** 2 + self.imaginaries[runner] ** 2)

        self.final_real = np.reshape(new_real, [Nx, Ny, Nz])
        self.final_imaginary = np.reshape(new_imaginary, [Nx, Ny, Nz])
        self.final_density = np.reshape(new_density, [Nx, Ny, Nz])

        self.volumetricdata_real = VolumetricData(structure, {"total": self.final_real})
        self.volumetricdata_imaginary = VolumetricData(structure, {"total": self.final_imaginary})
        self.volumetricdata_density = VolumetricData(structure, {"total": self.final_density})

    def get_volumetricdata_real(self) -> VolumetricData:
        """Get VolumetricData object for the real part of the wave function.

        Returns:
            VolumetricData: Real part volumetric data.
        """
        if not hasattr(self, "volumetricdata_real"):
            self.set_volumetric_data(self.grid, self.structure)
        return self.volumetricdata_real

    def get_volumetricdata_imaginary(self) -> VolumetricData:
        """Get VolumetricData object for the imaginary part of the wave function.

        Returns:
            VolumetricData: Imaginary part volumetric data.
        """
        if not hasattr(self, "volumetricdata_imaginary"):
            self.set_volumetric_data(self.grid, self.structure)
        return self.volumetricdata_imaginary

    def get_volumetricdata_density(self) -> VolumetricData:
        """Get VolumetricData object for the density part of the wave function.

        Returns:
            VolumetricData: Density volumetric data.
        """
        if not hasattr(self, "volumetricdata_density"):
            self.set_volumetric_data(self.grid, self.structure)
        return self.volumetricdata_density

    def write_file(
        self,
        filename: PathLike = "WAVECAR.vasp",
        part: Literal["real", "imaginary", "density"] = "real",
    ) -> None:
        """Save the wave function in a file readable by VESTA.

        Args:
            filename (PathLike): Output file name. Defaults to "WAVECAR.vasp".
            part (Literal["real", "imaginary", "density"]): Which part to save. Defaults to "real".

        Raises:
            ValueError: If the specified part is not "real", "imaginary", or "density".
        """
        if not (
            hasattr(self, "volumetricdata_real")
            and hasattr(self, "volumetricdata_imaginary")
            and hasattr(self, "volumetricdata_density")
        ):
            self.set_volumetric_data(self.grid, self.structure)

        if part == "real":
            self.volumetricdata_real.write_file(filename)
        elif part == "imaginary":
            self.volumetricdata_imaginary.write_file(filename)
        elif part == "density":
            self.volumetricdata_density.write_file(filename)
        else:
            raise ValueError('part can be only "real" or "imaginary" or "density"')

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """"""
        instance = super().from_dict(d)

        instance.points = np.asarray(instance.points, dtype=np.float64)
        instance.distances = np.asarray(instance.distances, dtype=np.float64)
        instance.reals = np.asarray(instance.reals, dtype=np.float64)
        instance.imaginaries = np.asarray(instance.imaginaries, dtype=np.float64)

        return instance


class MadelungEnergies(LobsterFile):
    """Parser for MadelungEnergies.lobster files.

    Attributes:
        madelung_energies_mulliken (float): Madelung energy (Mulliken).
        madelung_energies_loewdin (float): Madelung energy (Loewdin).
        ewald_splitting (float): Ewald splitting parameter.
    """

    @version_processor()
    def parse_file(self) -> None:
        """Parse MadelungEnergies.lobster file.

        Extracts the Ewald splitting parameter and Madelung energies.

        Returns:
            None
        """
        line = self.lines[5]

        line_parts = line.split()

        self.ewald_splitting = float(line_parts[0])
        self.madelung_energies_mulliken = float(line_parts[1])
        self.madelung_energies_loewdin = float(line_parts[2])

    @classmethod
    def get_default_filename(cls) -> str:
        """Get the default filename for MadelungEnergies.

        Returns:
            str: Default filename.
        """
        return "MadelungEnergies.lobster"


class SitePotentials(LobsterFile):
    """Parser for SitePotentials.lobster files.

    Attributes:
        centers (list[str]): Atom centers.
        site_potentials_mulliken (list[float]): Mulliken site potentials.
        site_potentials_loewdin (list[float]): Loewdin site potentials.
        madelung_energies_mulliken (float): Madelung energy (Mulliken).
        madelung_energies_loewdin (float): Madelung energy (Loewdin).
        ewald_splitting (float): Ewald splitting parameter.
    """

    @version_processor()
    def parse_file(self) -> None:
        """Parse SitePotentials.lobster file.

        Extracts site potentials, Madelung energies, and Ewald splitting parameter.

        Returns:
            None
        """
        self.centers = []
        self.site_potentials_mulliken = []
        self.site_potentials_loewdin = []

        for line in self.iterate_lines():
            if ewald_splitting := re.search(r"splitting parameter\s+(\S+)", line):
                self.ewald_splitting = float(ewald_splitting.group(1))

            if madelung_energies := re.search(r"Madelung Energy \(eV\)\s*(\S+)\s+(\S+)", line):
                self.madelung_energies_mulliken = float(madelung_energies.group(1))
                self.madelung_energies_loewdin = float(madelung_energies.group(2))

            if data := re.search(r"(\d+)\s+([a-zA-Z]{1,2})\s+(\S+)\s+(\S+)", line):
                data = data.groups()
                self.centers.append(data[1] + data[0])
                self.site_potentials_mulliken.append(float(data[2]))
                self.site_potentials_loewdin.append(float(data[3]))

    @classmethod
    def get_default_filename(cls) -> str:
        """Get the default filename for SitePotentials.

        Returns:
            str: Default filename.
        """
        return "SitePotentials.lobster"


def get_orb_from_str(orbs: list[str]) -> tuple[str, list[tuple[int, Orbital]]]:
    """Get Orbitals from string representations.

    Args:
        orbs (list[str]): Orbitals, e.g. ["2p_x", "3s"].

    Returns:
        tuple[str, list[tuple[int, Orbital]]]: Orbital label and list of orbitals.
    """
    orbitals = [(int(orb[0]), Orbital(LOBSTER_ORBITALS.index(orb[1:]))) for orb in orbs]

    orb_label = ""
    for iorb, orbital in enumerate(orbitals):
        if iorb == 0:
            orb_label += f"{orbital[0]}{orbital[1].name}"
        else:
            orb_label += f"-{orbital[0]}{orbital[1].name}"

    return orb_label, orbitals


class LobsterMatrices(LobsterFile):
    """Parser for LOBSTER matrix files.

    Attributes:
        matrix_type (str): Type of matrix (hamilton, coefficient, transfer, overlap).
        centers (list[str]): Atom centers.
        orbitals (list[str]): Orbitals.
        matrices (LobsterMatrixData): Matrix data for each k-point and spin.
        efermi (float): Fermi energy (for Hamilton matrices).
    """

    matrix_types: ClassVar[set[str]] = {
        "hamilton",
        "coefficient",
        "transfer",
        "overlap",
    }

    def __init__(
        self,
        filename: PathLike | None = None,
        matrix_type: str | None = None,
        efermi: float | None = None,
        process_immediately: bool = True,
        lobster_version: str | None = None,
    ) -> None:
        """Initialize LOBSTER matrices parser.

        Args:
            filename: Path to the matrix file
            matrix_type: Type of matrix. If None, inferred from filename
            efermi: Fermi level in eV (required for Hamilton matrices)
            process_immediately: Whether to parse the file immediately
        """
        super().__init__(
            filename=filename,
            process_immediately=False,
            lobster_version=lobster_version,
        )

        self.efermi = efermi

        self.matrix_type = matrix_type or self.get_matrix_type()

        if self.matrix_type == "hamilton" and self.efermi is None:
            raise ValueError("Fermi energy (eV) required for Hamilton matrices")

        if process_immediately:
            self.parse_file()

    def get_matrix_type(self) -> str:
        """Infer matrix type from filename.

        Returns:
            str: Matrix type.
        """
        name = str(self.filename).lower()

        for matrix_type in self.matrix_types:
            if matrix_type in name:
                return matrix_type

        raise ValueError(f"Cannot infer matrix type from filename: {self.filename}")

    @version_processor()
    def parse_file(self) -> None:
        """Parse matrix data and set instance attributes.

        The file is read twice. The first pass records the line number at which each
        matrix block starts, together with the k-points and the basis function labels.
        The second pass streams every block straight into ``self.matrices``, an array
        of shape ``(n_spins, n_kpoints, n_basis_functions, n_bands)``.

        Returns:
            None
        """
        self.kpoints: list[tuple[float, ...]] = []
        self.basis_functions: list[str] = []
        self.spins: list[Spin] = []

        spin_indices: dict[str, int] = {}
        kpoint_indices: dict[str, int] = {}
        block_positions: list[tuple[int, int, bool, int]] = []
        matrix_dimension = 0

        current_spin_index = 0
        current_kpoint_index = 0
        is_imaginary_part = False

        line_iterator = self.iterate_lines()
        line_number = -1

        for line in line_iterator:
            line_number += 1
            stripped_line = line.strip()
            lowered_line = stripped_line.lower()

            if stripped_line.startswith("basisfunction"):
                band_count = len(re.findall(r"band\s+\d+", line)) or len(stripped_line.split()) - 1

                if matrix_dimension and band_count != matrix_dimension:
                    raise ValueError(
                        f"Expected {matrix_dimension} bands per block, found {band_count} at line {line_number}."
                    )
                matrix_dimension = band_count

                block_positions.append(
                    (current_spin_index, current_kpoint_index, is_imaginary_part, line_number + 1)
                )

                if not self.basis_functions:  # labels are identical in every block
                    for _ in range(matrix_dimension):
                        self.basis_functions.append(next(line_iterator).split()[0])
                        line_number += 1

            elif lowered_line.endswith("real parts"):
                is_imaginary_part = False

            elif lowered_line.endswith("imag parts"):
                is_imaginary_part = True

            elif "kpoint" in lowered_line and (header_match := re.search(r"(?:(?:spin\s+)?(?P<spin>\d+)\s+)?kpoint\s+(?P<kpoint>\d+)(?:\s+at\s+(?P<coordinates>.*))?", line, re.IGNORECASE)):
                spin_label = "1" if self.matrix_type == "overlap" else (header_match["spin"] or "1")

                if spin_label not in spin_indices:
                    spin_indices[spin_label] = len(spin_indices)
                    self.spins.append(Spin.up if spin_label == "1" else Spin.down)
                current_spin_index = spin_indices[spin_label]

                kpoint_label = header_match["kpoint"]

                if kpoint_label not in kpoint_indices:
                    kpoint_indices[kpoint_label] = len(kpoint_indices)
                    coordinates = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", header_match["coordinates"] or "")
                    self.kpoints.append(tuple(float(coordinate) for coordinate in coordinates))
                current_kpoint_index = kpoint_indices[kpoint_label]

        if not block_positions:
            raise ValueError("Could not find any matrix block in the file.")

        self.matrices = np.zeros(
            (len(self.spins), len(self.kpoints), matrix_dimension, matrix_dimension), dtype=np.complex128
        )

        line_iterator = self.iterate_lines()
        consumed_lines = 0

        for spin_index, kpoint_index, is_imaginary_part, first_data_line in block_positions:
            for _ in range(first_data_line - consumed_lines):
                next(line_iterator)

            values = np.loadtxt(
                islice(line_iterator, matrix_dimension),
                usecols=range(1, matrix_dimension + 1),
            )
            consumed_lines = first_data_line + matrix_dimension

            block = self.matrices[spin_index, kpoint_index]

            if is_imaginary_part:
                block.imag = values
            else:
                block.real = values

    @classmethod
    def get_default_filename(cls) -> str:
        """Get the default filename for the LobsterMatrices class.

        Returns:
            str: Default filename.
        """
        return "hamiltonMatrices.lobster"

    def as_dict(self) -> dict[str, Any]:
        """Serialize object to a dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the object.
        """
        dictionary = super().as_dict()

        for kpoint in dictionary["matrices"]:
            for spin in dictionary["matrices"][kpoint]:
                matrix_data = dictionary["matrices"][kpoint][spin]
                dictionary["matrices"][kpoint][spin] = {
                    "real": matrix_data.real,
                    "imag": matrix_data.imag,
                }

        return dictionary

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Deserialize from dictionary."""
        instance = super().from_dict(d)

        for kpoint in instance.matrices:
            for spin in instance.matrices[kpoint]:
                matrix_data = instance.matrices[kpoint][spin]
                instance.matrices[kpoint][spin] = np.asarray(matrix_data["real"]) + 1j * np.asarray(matrix_data["imag"])

        return instance


class POLARIZATION(LobsterFile):
    """Parser for POLARIZATION.lobster file.

    Attributes:
        rel_mulliken_pol_vector (dict[str, float]): Relative Mulliken polarization vector.
        rel_loewdin_pol_vector (dict[str, float]): Relative Loewdin polarization vector.
    """

    @version_processor()
    def parse_file(self) -> None:
        """Parse POLARIZATION.lobster file.

        Returns:
            None
        """
        self.rel_mulliken_pol_vector = {}
        self.rel_loewdin_pol_vector = {}

        for line in islice(self.iterate_lines(), 3, None):
            cleanlines = [idx for idx in line.split(" ") if idx != ""]
            if cleanlines and len(cleanlines) == 3:
                self.rel_mulliken_pol_vector[cleanlines[0]] = float(cleanlines[1])
                self.rel_loewdin_pol_vector[cleanlines[0]] = float(cleanlines[2])
            if cleanlines and len(cleanlines) == 4:
                self.rel_mulliken_pol_vector[cleanlines[0].replace(":", "")] = cleanlines[1].replace("\u03bc", "u")
                self.rel_loewdin_pol_vector[cleanlines[2].replace(":", "")] = cleanlines[3].replace("\u03bc", "u")

    @classmethod
    def get_default_filename(cls) -> str:
        """Get the default filename for the Polarization class.

        Returns:
            str: Default filename.
        """
        return "POLARIZATION.lobster"


class BWDF(LobsterFile):
    """Parser for BWDF.lobster/BWDFCOHP.lobster files.

    Attributes:
        centers (NDArray): Bond length centers for the distribution.
        bwdf (dict[Literal[1, -1], NDArray]): Bond weighted distribution function.
        bin_width (float): Bin width used for computing the distribution by LOBSTER.
    """

    is_cohp: ClassVar[bool] = False

    def __init__(
        self,
        filename: PathLike | None = None,
        process_immediately: bool = True,
        lobster_version: str | None = None,
    ) -> None:
        """
        Args:
            filename (PathLike): The BWDF file from LOBSTER, typically "BWDF.lobster"
                or "BWDFCOHP.lobster".
        """
        self.bwdf = {}
        self.centers = np.array([])
        self.data = np.array([[]])

        super().__init__(
            filename=filename,
            process_immediately=process_immediately,
            lobster_version=lobster_version,
        )

    @version_processor()
    def parse_file(self) -> None:
        """Parse BWDF.lobster/BWDFCOHP.lobster file.

        Returns:
            None
        """
        self.bwdf = {}
        self.data = np.genfromtxt(self.iterate_lines(), dtype=float, skip_header=1)

        self.process_data_into_bwdf_centers()

    def process_data_into_bwdf_centers(self) -> None:
        """Process data into bwdf and centers.

        Returns:
            None
        """
        self.centers = self.data[:, 0]
        self.bwdf[Spin.up] = self.data[:, 1]

        if self.data.shape[1] > 2:
            self.bwdf[Spin.down] = self.data[:, 2]

    @classmethod
    def get_default_filename(cls) -> str:
        """Get the default filename for the BWDF class.

        Returns:
            str: Default filename.
        """
        return "BWDF.lobster"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Deserialize object from dictionary produced by `as_dict`.

        Args:
            d (dict[str, Any]): Dictionary representation of the object.

        Returns:
            Self: Deserialized BWDF object.
        """
        instance = super().from_dict(d)

        instance.data = np.asarray(instance.data, dtype=np.float64)
        instance.process_data_into_bwdf_centers()

        return instance


class BWDFCOHP(BWDF):
    """Parser for BWDFCOHP.lobster files.

    Returns:
        None
    """

    @classmethod
    def get_default_filename(cls) -> str:
        """Get the default filename for the BWDFCOHP class.

        Returns:
            str: Default filename.
        """
        return "BWDFCOHP.lobster"
