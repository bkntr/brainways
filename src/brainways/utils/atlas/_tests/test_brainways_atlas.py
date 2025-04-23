from unittest.mock import MagicMock, patch

from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas


def test_bounding_box(mock_atlas: BrainwaysAtlas):
    box = mock_atlas.bounding_box(0)
    assert box == (0, 0, mock_atlas.shape[1], mock_atlas.shape[2])


def test_brainways_atlas_singleton():
    with (
        patch.object(BrainwaysAtlas, "_instances", new={}),
        patch.object(BrainwaysAtlas, "_atlas_obj_cache", new={}),
    ):
        atlas_name = "whs_sd_rat_39um"
        mock_bg_atlas = MagicMock()
        mock_bg_atlas.atlas_name = atlas_name

        exclude_regions_1 = [1, 2]
        exclude_regions_2 = [3, 4]

        with patch(
            "brainways.utils.atlas.brainways_atlas.BrainGlobeAtlas",
            return_value=mock_bg_atlas,
        ):
            # Instance 1: Using string name and exclude_regions_1
            instance1 = BrainwaysAtlas(atlas_name, exclude_regions=exclude_regions_1)
            # Instance 2: Using the same string name and exclude_regions_1 again
            instance2 = BrainwaysAtlas(atlas_name, exclude_regions=exclude_regions_1)
            # Instance 3: Using atlas object and exclude_regions_1
            instance3 = BrainwaysAtlas(mock_bg_atlas, exclude_regions=exclude_regions_1)
            # Instance 4: Using string name but different exclude_regions
            instance4 = BrainwaysAtlas(atlas_name, exclude_regions=exclude_regions_2)
            # Instance 5: Using a different atlas name (simulate with a different string)
            instance5 = BrainwaysAtlas(
                "different_atlas_name", exclude_regions=exclude_regions_1
            )
            # Instance 6: Using string name and exclude_regions=None
            instance6 = BrainwaysAtlas(atlas_name, exclude_regions=None)
            # Instance 7: Using string name and exclude_regions=None again
            instance7 = BrainwaysAtlas(atlas_name, exclude_regions=None)

            assert (
                instance1 is instance2
            )  # Same name, same exclude_regions -> same instance
            assert (
                instance1 is instance3
            )  # Same atlas object (name), same exclude_regions -> same instance
            assert (
                instance1 is not instance4
            )  # Same name, different exclude_regions -> different instance
            assert (
                instance1 is not instance5
            )  # Different name, same exclude_regions -> different instance
            assert (
                instance6 is instance7
            )  # Same name, None exclude_regions -> same instance
            assert (
                instance1 is not instance6
            )  # Same name, different exclude_regions ([1,2] vs None) -> different instance


def test_brainglobe_atlas_instance_cache():
    with (
        patch.object(BrainwaysAtlas, "_instances", new={}),
        patch.object(BrainwaysAtlas, "_atlas_obj_cache", new={}),
    ):
        atlas_name = "whs_sd_rat_39um"
        other_atlas_name = "other_atlas"
        mock_bg_atlas_1 = MagicMock()
        mock_bg_atlas_1.atlas_name = atlas_name
        mock_bg_atlas_2 = MagicMock()
        mock_bg_atlas_2.atlas_name = atlas_name
        mock_bg_atlas_other = MagicMock()
        mock_bg_atlas_other.atlas_name = other_atlas_name

        # Patch BrainGlobeAtlas to return mock_bg_atlas_1 for atlas_name, mock_bg_atlas_other for other_atlas_name
        def atlas_factory(name, check_latest=False):
            if name == atlas_name:
                return mock_bg_atlas_1
            elif name == other_atlas_name:
                return mock_bg_atlas_other
            raise ValueError("Unexpected atlas name")

        with patch(
            "brainways.utils.atlas.brainways_atlas.BrainGlobeAtlas",
            side_effect=atlas_factory,
        ):
            # First instance: triggers BrainGlobeAtlas(atlas_name)
            instance1 = BrainwaysAtlas(atlas_name, exclude_regions=[1])
            # Second instance: uses the same atlas_name, should reuse mock_bg_atlas_1
            instance2 = BrainwaysAtlas(atlas_name, exclude_regions=[1])
            # Third instance: pass a different object with the same atlas_name, should still reuse mock_bg_atlas_1
            instance3 = BrainwaysAtlas(mock_bg_atlas_2, exclude_regions=[1])
            # Fourth instance: same atlas_name, different exclude_regions
            instance4 = BrainwaysAtlas(atlas_name, exclude_regions=[2])
            # Fifth instance: same atlas_name, exclude_regions=None
            instance5 = BrainwaysAtlas(atlas_name, exclude_regions=None)
            # Sixth instance: different atlas name
            instance6 = BrainwaysAtlas(other_atlas_name, exclude_regions=[1])
            # Seventh instance: pass the other atlas object directly
            instance7 = BrainwaysAtlas(mock_bg_atlas_other, exclude_regions=[1])

            # All should share the same brainglobe_atlas instance for atlas_name
            assert instance1.brainglobe_atlas is mock_bg_atlas_1
            assert instance2.brainglobe_atlas is mock_bg_atlas_1
            assert instance3.brainglobe_atlas is mock_bg_atlas_1
            assert instance4.brainglobe_atlas is mock_bg_atlas_1
            assert instance5.brainglobe_atlas is mock_bg_atlas_1

            # The instance for the other atlas name should use the other mock
            assert instance6.brainglobe_atlas is mock_bg_atlas_other
            assert instance7.brainglobe_atlas is mock_bg_atlas_other
