import contextlib
import os.path
import typing

if typing.TYPE_CHECKING:
    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff import ForceField


@contextlib.contextmanager
def set_env(**environ):

    old_environ = dict(os.environ)
    os.environ.update(environ)

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def get_charges_from_forcefield(
    molecule: "Molecule",
    forcefield: "ForceField",
):
    """Compute the partial charges of a molecule using a force field.

    Parameters
    ----------
    molecule
        The molecule to compute the charges for.
    forcefield
        The force field to use to compute the charges.

    Returns
    -------
        The partial charges of the molecule.
    """

    # for now raise an error if there's an Interchange involved
    # https://github.com/openforcefield/openff-interchange/issues/792
    # Creating an Interchange means virtual site charges replace 
    # existing electrostatics instead of incrementing them
    import openmm
    import numpy as np

    topology = molecule.to_topology()
    # check vsites
    handler = forcefield.get_parameter_handler("VirtualSites")
    if any(
        molecule.chemical_environment_matches(parameter.smirks)
        for parameter in handler.parameters
    ):
        if hasattr(forcefield, "create_interchange"):
            raise NotImplementedError("Interchange not supported")
    
    system = forcefield.create_openmm_system(topology)

    charges = []
    nbforces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    nbforce = nbforces[0]
    for i in range(nbforce.getNumParticles()):
        charge, sigma, epsilon = nbforce.getParticleParameters(i)
        charges.append(float(charge.value_in_unit(charge.unit)))
    return np.array(charges)