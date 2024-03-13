from napari_brainways.brainways_ui import BrainwaysUI


def test_open_sample_project(app: BrainwaysUI):
    assert app.project is None
    app.viewer.open_sample(plugin="napari-brainways", sample="sample_project")
    assert app.project is not None


def test_open_sample_project_annotated(app: BrainwaysUI):
    assert app.project is None
    app.viewer.open_sample(plugin="napari-brainways", sample="sample_project_annotated")
    assert app.project is not None


def test_open_sample_project_twice(app: BrainwaysUI):
    assert app.project is None
    app.viewer.open_sample(plugin="napari-brainways", sample="sample_project")
    app.viewer.open_sample(plugin="napari-brainways", sample="sample_project")
    assert app.project is not None
