# # tmp_path is a pytest fixture
# def test_reader(napari_viewer: Viewer, monkeypatch):
#     monkeypatch.setattr(BrainwaysUI, "open_project_async", Mock())
#     package = Path(importlib_resources.files("brainways"))
#     sample_data_dir = str(package / "resources/sample_data/")

#     # try to read it back in
#     reader = get_reader(sample_data_dir)
#     assert callable(reader)

#     # make sure we're delivering the right format
#     layer_data_list = reader(sample_data_dir)

#     assert layer_data_list == [(None,)]
#     BrainwaysUI.open_project_async.assert_called()


# def test_get_reader_pass():
#     reader = get_reader("fake.file")
#     assert reader is None
