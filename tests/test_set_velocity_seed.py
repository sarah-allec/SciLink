"""set_velocity_seed: stamp a distinct RNG seed into a LAMMPS deck so replicas
of the same system are independent samples."""

from scilink.skills.molecular_dynamics.lammps.lammps import set_velocity_seed


def test_replaces_literal_integer_seed():
    deck = "minimize 1e-6 1e-8 1000 10000\nvelocity all create 298.15 12345 dist gaussian\nrun 1000\n"
    out = set_velocity_seed(deck, 777)
    assert "velocity all create 298.15 777 dist gaussian" in out
    assert "12345" not in out


def test_replaces_placeholder_seed():
    deck = "velocity all create 298.15 {seed} dist gaussian\n"
    assert "create 298.15 42 dist gaussian" in set_velocity_seed(deck, 42)
    deck2 = "velocity all create 298.15 ${seed} dist gaussian\n"
    assert "create 298.15 42 dist gaussian" in set_velocity_seed(deck2, 42)


def test_keeps_trailing_keywords_and_temperature():
    deck = "velocity all create 350.0 999 mom yes rot yes dist gaussian\n"
    out = set_velocity_seed(deck, 5)
    assert "velocity all create 350.0 5 mom yes rot yes dist gaussian" in out


def test_sets_every_velocity_create():
    deck = ("velocity all create 300 111 dist gaussian\n"
            "velocity mobile create 300 222 dist gaussian\n")
    out = set_velocity_seed(deck, 8)
    assert out.count("create 300 8 ") == 2
    assert "111" not in out and "222" not in out


def test_no_velocity_create_is_unchanged():
    deck = "read_data system.data\nfix 1 all npt temp 300 300 0.1\nrun 1000\n"
    assert set_velocity_seed(deck, 3) == deck


def test_seed_coerced_to_int():
    deck = "velocity all create 300 111 dist gaussian\n"
    assert "create 300 9 " in set_velocity_seed(deck, 9.0)
