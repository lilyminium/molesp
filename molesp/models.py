import pickle
import typing

from pydantic import BaseModel, Extra
from openff.units import unit
from openff.units.openmm import from_openmm

def unify(quantity):
    if not isinstance(quantity, unit.Quantity):
        return from_openmm(quantity)


class MolESPModel(BaseModel):
    class Config:
        extra = Extra.forbid


class Surface(MolESPModel):

    vertices: typing.List[float]
    indices: typing.List[int]

    @classmethod
    def from_openff(cls, molecule):
        import numpy as np
        from openff.recharge.utilities.toolkits import VdWRadiiType, compute_vdw_radii
        from molesp.cli._cli import compute_surface

        assert len(molecule.conformers) == 1
        conformer = unify(molecule.conformers[0])

        vdw_radii = compute_vdw_radii(molecule, radii_type=VdWRadiiType.Bondi)
        radii = (
            np.array([[radii] for radii in vdw_radii.m_as(unit.angstrom)])
            * unit.angstrom
        )

        vertices, indices = compute_surface(
            molecule, conformer, radii, 1.4, 0.2 * unit.angstrom
        )
        return cls(
            vertices=vertices.flatten().tolist(),
            indices=indices.flatten().tolist(),
        )

    def to_grid(self, with_units: bool = True):
        import numpy as np

        arr = np.array(self.vertices).reshape(-1, 3)
        if with_units:
            arr = arr * unit.angstrom
        return arr



class ESPMolecule(MolESPModel):

    mapped_smiles: str
    atomic_numbers: typing.List[int]

    conformer: typing.List[float]

    surface: Surface
    esp: typing.Dict[str, typing.List[float]]

    def to_pickle(self, path: str):
        with open(str(path), "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, path: str):
        with open(str(path), "rb") as file:
            return pickle.load(file)

    @classmethod
    def from_openff(
        cls,
        molecule
    ):
        if not molecule.n_conformers:
            molecule.generate_conformers(n_conformers=1)
        assert molecule.n_conformers == 1, "molecule must have exactly one conformer"

        conformer = unify(molecule.conformers[0])
        surface = Surface.from_openff(molecule)
        atomic_numbers = [atom.atomic_number for atom in molecule.atoms]
        coordinates = conformer.m_as(unit.angstrom).flatten().tolist()
        obj = cls(
            mapped_smiles=molecule.to_smiles(mapped=True),
            atomic_numbers=atomic_numbers,
            conformer=coordinates,
            surface=surface,
            esp={}
        )
        return obj
    
    @classmethod
    def from_qcportal_results(
        cls,
        qc_result: "qcportal.models.ResultRecord",
        qc_molecule: "qcelemental.models.Molecule",
        qc_keyword_set: "qcportal.models.KeywordSet",
    ):
        import numpy as np
        from openff.toolkit.topology.molecule import Molecule
        
        # Convert the OE molecule to a QC molecule and extract the conformer of
        # interest.
        molecule = Molecule.from_qcschema(
            qc_molecule.dict(encoding="json"), allow_undefined_stereo=True
        )
        obj = cls.from_openff(molecule)
        obj.compute_esp_from_qcportal_results(
            qc_result, qc_molecule, qc_keyword_set
        )
        return obj
    
    
    def compute_esp_from_qcportal_results(
        self,
        qc_result: "qcportal.models.ResultRecord",
        qc_molecule: "qcelemental.models.Molecule",
        qc_keyword_set: "qcportal.models.KeywordSet",
        name: str = "QC ESP",
    ):
        from openff.units import unit
        from openff.recharge.esp.qcresults import compute_esp
        from qcelemental.models.results import WavefunctionProperties
        from openff.recharge.esp import ESPSettings
        from openff.recharge.grids._grids import MSKGridSettings
        from openff.recharge.esp.qcresults import reconstruct_density, _parse_pcm_input

        # Compute and store the ESP and electric field for each result.
        if qc_result.wavefunction is None:
            raise ValueError(qc_result.id)

        # Retrieve the wavefunction and use it to reconstruct the electron density.
        wavefunction = WavefunctionProperties(
            **qc_result.get_wavefunction(
                ["scf_eigenvalues_a", "scf_orbitals_a", "basis", "restricted"]
            ),
            **qc_result.wavefunction["return_map"],
        )
        density = reconstruct_density(wavefunction, qc_result.properties.calcinfo_nalpha)

        # Retrieve the ESP settings from the record.
        enable_pcm = "pcm" in qc_keyword_set.values()

        esp_settings = ESPSettings(
            basis=qc_result.basis,
            method=qc_result.method,
            grid_settings=MSKGridSettings(),
            pcm_settings=(
                None
                if not enable_pcm
                else _parse_pcm_input(qc_keyword_set.values["pcm__input"])
            ),
        )

        esp, _ = compute_esp(
            qc_molecule,
            density,
            esp_settings,
            self.surface.to_grid(),
            compute_field=False
        )
        esp_values = esp.m_as(unit.hartree / unit.e).flatten().tolist()
        self.esp[name] = esp_values

    
    @staticmethod
    def _compute_esp_from_coordinates_and_charges(
        atom_coordinates,
        grid_coordinates,
        charges,
    ):
        import numpy as np
        ke = 1 / (4 * np.pi * unit.epsilon_0)

        displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N x M x 3
        distance = (displacement ** 2).sum(axis=-1) ** 0.5  # N x M
        inv_distance = 1 / distance
        esp = ke * (inv_distance @ charges)  # N

        esp_values = esp.m_as(unit.hartree / unit.e).flatten().tolist()
        return esp_values
    

    def compute_esp_from_charges(
        self,
        charges: typing.List[float],
        name: str = "ESP"
    ):
        import numpy as np
        ke = 1 / (4 * np.pi * unit.epsilon_0)

        charges = np.asarray(charges) * unit.elementary_charge
        atom_coordinates = np.asarray(self.conformer).reshape((-1, 3)) * unit.angstrom
        grid_coordinates = self.surface.to_grid(with_units=True)
        esp_values = self._compute_esp_from_coordinates_and_charges(
            atom_coordinates,
            grid_coordinates,
            charges
        )
        self.esp[name] = esp_values

    def to_openff(self):
        import numpy as np
        from openff.toolkit.topology.molecule import Molecule, unit

        molecule = Molecule.from_mapped_smiles(self.mapped_smiles, allow_undefined_stereo=True)
        conformer = np.asarray(self.conformer).reshape((-1, 3))
        molecule._conformers = [conformer * unit.angstrom]
        return molecule

    def compute_esp_from_forcefield(
        self,
        forcefield: str,
        name: str = "FF ESP"
    ):
        import numpy as np
        from openff.toolkit.typing.engines.smirnoff import ForceField
        import openmm
        from openff.units.openmm import from_openmm
        from openmm import unit as openmm_unit
        from .utilities import get_charges_from_forcefield


        molecule = self.to_openff()
        forcefield = ForceField(forcefield, allow_cosmetic_attributes=True)
        system = forcefield.create_openmm_system(molecule.to_topology())
        charges = get_charges_from_forcefield(molecule, forcefield)
        charges = np.asarray(charges) * unit.elementary_charge

        integrator = openmm.VerletIntegrator(0.1 * openmm_unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName("Reference")
        
        openmm_context = openmm.Context(system, integrator, platform)
        
        # account for vsites
        n_particles = system.getNumParticles() - molecule.n_atoms
        dummy_positions = np.zeros((n_particles, 3))
        actual_positions = unify(molecule.conformers[0]).m_as(unit.nanometer)
        positions = np.concatenate((actual_positions, dummy_positions))
        
        openmm_context.setPositions(positions)
        openmm_context.computeVirtualSites()
        atom_coordinates = openmm_context.getState(getPositions=True).getPositions(asNumpy=True)
        atom_coordinates = unify(atom_coordinates)
        assert atom_coordinates.shape == (system.getNumParticles(), 3)

        grid_coordinates = self.surface.to_grid(with_units=True)

        esp_values = self._compute_esp_from_coordinates_and_charges(
            atom_coordinates.reshape((-1, 3)),
            grid_coordinates.reshape((-1, 3)),
            charges.flatten(),
        )
        self.esp[name] = esp_values
