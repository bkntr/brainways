from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas


def test_bounding_box(mock_atlas: BrainwaysAtlas):
    box = mock_atlas.bounding_box(0)
    assert box == (0, 0, mock_atlas.shape[1], mock_atlas.shape[2])
