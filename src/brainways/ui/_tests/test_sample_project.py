from brainways.ui.brainways_ui import BrainwaysUI


def test_open_sample_project(app: BrainwaysUI):
    assert app._project is None
    app.viewer.open_sample(plugin="brainways", sample="sample_project")
    assert app._project is not None


def test_open_sample_project_annotated(app: BrainwaysUI):
    assert app._project is None
    app.viewer.open_sample(plugin="brainways", sample="sample_project_annotated")
    assert app._project is not None


def test_open_sample_project_twice(app: BrainwaysUI):
    assert app._project is None
    app.viewer.open_sample(plugin="brainways", sample="sample_project")
    app.viewer.open_sample(plugin="brainways", sample="sample_project")
    assert app._project is not None
