import os

import pytest

pytest.importorskip("mujoco")

from Test.verify_xml import REQUIRED_FILES, resolve_project_dir, validate_scene_xml


def test_scene_xml_files_exist_and_loadable():
    project_dir = resolve_project_dir()
    assert os.path.isdir(project_dir)

    ok, model, missing_files = validate_scene_xml(project_dir)

    assert missing_files == []
    assert ok is True
    assert model is not None
    assert model.nq > 0
    assert model.nv > 0
    assert model.nbody > 0


def test_required_files_constant_matches_directory():
    project_dir = resolve_project_dir()
    existing_files = set(os.listdir(project_dir))

    for filename in REQUIRED_FILES:
        assert filename in existing_files
